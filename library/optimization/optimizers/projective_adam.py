import logging
from collections.abc import Iterable
from typing import Literal

import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import (
    copy_stochastic_,
    resolve_state_storage_dtype,
)
from library.utils.compile_env import prepare_windows_compiler_env_for_torch_compile


logger = logging.getLogger(__name__)


NS_COEFFS = [
    (8.287212018145622, -23.59588651909882, 17.300387312530923),
    (4.107059111542197, -2.9478499167379084, 0.54484310829266),
    (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
    (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
    (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
    (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
    (1.875, -1.25, 0.375),
]

def _resolve_ortho_dtype(ortho_dtype: str | torch.dtype | None) -> torch.dtype:
    if ortho_dtype is None:
        return torch.float32
    if isinstance(ortho_dtype, str):
        return getattr(torch, ortho_dtype.split(".")[-1])
    return ortho_dtype


def reshape_to_2d(grad: torch.Tensor) -> torch.Tensor:
    """Reshape a tensor to 2D for matrix operations."""
    if grad.ndim > 2:
        return grad.reshape(len(grad), -1)
    if grad.ndim < 2:
        return grad.reshape(1, -1)
    return grad


@torch.no_grad()
def orthogonalize(matrix: torch.Tensor, ortho_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Orthogonalize a matrix via 5th order Newton-Schulz iteration."""
    if ortho_dtype is not None:
        original_dtype = matrix.dtype
        matrix = matrix.to(ortho_dtype)
    else:
        original_dtype = None

    transpose = matrix.shape[0] < matrix.shape[1]
    if transpose:
        matrix = matrix.T

    identity = torch.eye(matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
    for a, b, c in NS_COEFFS:
        matrix = matrix / torch.linalg.norm(matrix).clamp_min_(1e-8)
        gram = matrix.T @ matrix
        matrix = matrix @ (a * identity + b * gram + c * gram @ gram)

    if transpose:
        matrix = matrix.T
    if original_dtype is not None:
        matrix = matrix.to(original_dtype)
    return matrix


class ProjectiveAdam(Optimizer):
    r"""ProjectiveAdam: An Adam-based optimizer with selectable geometric projections.

    This optimizer maps gradients onto a geometric manifold using one of several
    projection types, tracks momentum on that manifold, and reconstructs the
    update via inverse projection.

    Supported Projections:
        - 'stereographic': Maps R^n -> S^n via stereographic projection from the south pole.
        - 'gnomonic': Maps R^n -> Hemisphere via central/gnomonic projection.
        - 'hyperbolic': Maps R^n -> Poincaré Ball (Hyperbolic space) via tanh scaling.
    
    Arguments:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float]): Coefficients for EMAs (default: (0.95, 0.999)).
        eps (float): Numerical stability term (default: 1e-16).
        weight_decay (float): Weight decay coefficient (default: 0.0).
        projection (str): Projection type: 'stereographic', 'gnomonic', 'hyperbolic' (default: 'gnomonic').
        input_norm (bool): Normalize RMS by last 2D dimension if True, otherwise tensor-wise RMS (default: True).
        normuon (bool): Use NorMuon update scaling (default: True).
        use_compile (bool): Use torch.compile on orthogonalization for faster execution (default: True).
        ortho_dtype (str): Data type for Newton-Schulz orthogonalization (default: None (torch.float32)).
        stochastic_fp (bool): Use stochastic rounding for half-precision (default: True).
    """


    PROJECTION_TYPES = ("stereographic", "gnomonic", "hyperbolic")

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.95, 0.999),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
        projection: Literal["stereographic", "gnomonic", "hyperbolic"] = "gnomonic",
        input_norm: bool = True,
        normuon: bool = True,
        use_compile: bool = True,
        ortho_dtype: str | torch.dtype | None = None,
        stochastic_fp: bool = True,
        sync_chunk_size: int = 128,
        state_storage_dtype: str | torch.dtype = torch.bfloat16,
        state_storage_device: str | torch.device = "cpu",
        **kwargs,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if projection not in self.PROJECTION_TYPES:
            raise ValueError(f"Invalid projection type: {projection}. Choose from {self.PROJECTION_TYPES}")

        for key in kwargs:
            logger.warning("Unrecognized optimizer argument '%s'. It will be ignored.", key)

        final_state_dtype = resolve_state_storage_dtype(state_storage_dtype)
        resolved_ortho_dtype = _resolve_ortho_dtype(ortho_dtype)

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "projection": projection,
            "input_norm": input_norm,
            "normuon": normuon,
            "use_compile": use_compile,
            "ortho_dtype": resolved_ortho_dtype,
            "stochastic_fp": stochastic_fp,
            "sync_chunk_size": sync_chunk_size,
            "state_storage_dtype": final_state_dtype,
            "state_storage_device": state_storage_device,
        }
        super().__init__(params, defaults)

        self.sync_chunk_size = sync_chunk_size
        self.state_storage_dtype = final_state_dtype
        self.state_storage_device = state_storage_device
        if use_compile:
            prepare_windows_compiler_env_for_torch_compile()
        self.ortho_func = torch.compile(orthogonalize, mode="reduce-overhead") if use_compile else orthogonalize

    def __str__(self) -> str:
        return "ProjectiveAdam"

    @staticmethod
    def _get_compute_device(parameter_device: torch.device) -> torch.device:
        if parameter_device.type == "cpu" and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return parameter_device

    def _initialize_state_tensor(
        self,
        reference: torch.Tensor,
        *,
        use_reference_shape: bool = True,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if use_reference_shape:
            state_tensor = torch.zeros_like(
                reference,
                dtype=self.state_storage_dtype,
                device=self.state_storage_device,
                memory_format=torch.preserve_format,
            )
        else:
            assert shape is not None
            state_tensor = torch.zeros(
                shape,
                dtype=self.state_storage_dtype,
                device=self.state_storage_device,
            )

        if self.state_storage_device == "cpu" and torch.cuda.is_available():
            state_tensor = state_tensor.pin_memory()
        return state_tensor

    @staticmethod
    def _copy_parameter_back(parameter: torch.Tensor, parameter_fp32: torch.Tensor, stochastic_fp: bool) -> None:
        destination = parameter_fp32.to(parameter.device, non_blocking=parameter.device.type == "cuda")
        if parameter.dtype == torch.bfloat16 and stochastic_fp:
            copy_stochastic_(parameter.data, destination)
        else:
            parameter.data.copy_(destination, non_blocking=parameter.device.type == "cuda")

    def _copy_state_back(self, state_tensor: torch.Tensor, compute_tensor: torch.Tensor, stochastic_fp: bool) -> None:
        destination = compute_tensor.to(state_tensor.device, non_blocking=state_tensor.device.type == "cuda")
        if self.state_storage_dtype == torch.bfloat16 and stochastic_fp:
            copy_stochastic_(state_tensor, destination)
        else:
            state_tensor.copy_(destination, non_blocking=state_tensor.device.type == "cuda")

    @staticmethod
    def _stereographic_project(grad: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Element-wise Stereographic Projection from South Pole.
        Maps each element g_i -> (y_i, z_i) on a 2D circle.

        proj(g_i) = (2*g_i / (g_i^2 + 1), (g_i^2 - 1) / (g_i^2 + 1))
        """
        del eps
        grad_sq = grad.pow(2)
        denom = grad_sq + 1.0
        return (2.0 * grad) / denom, (grad_sq - 1.0) / denom

    @staticmethod
    def _stereographic_inverse(y: torch.Tensor, z: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Inverse Stereographic Projection. Maps S^n -> R^n.

        inv(y, z) = y / (1 - z)
        """
        return y / (1.0 - z).clamp_min(eps)

    @staticmethod
    def _gnomonic_project(grad: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Element-wise Gnomonic (Central) Projection.
        Maps each element g_i -> (y_i, z_i) on the hemisphere.

        proj(g_i) = (g_i / sqrt(1 + g_i^2), 1 / sqrt(1 + g_i^2))
        """
        del eps
        inv_sqrt = torch.rsqrt(1.0 + grad.pow(2))
        return grad * inv_sqrt, inv_sqrt

    @staticmethod
    def _gnomonic_inverse(y: torch.Tensor, z: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Inverse Gnomonic Projection. Maps Hemisphere -> R^n.

        inv(y, z) = y / z
        """
        return y / z.clamp_min(eps)

    @staticmethod
    def _hyperbolic_project(grad: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Element-wise Hyperbolic (Poincaré) Projection.

        proj(g_i) = tanh(g_i), z = |tanh(g_i)|
        """
        del eps
        y = torch.tanh(grad)
        return y, y.abs()

    @staticmethod
    def _hyperbolic_inverse(y: torch.Tensor, z: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Inverse Hyperbolic Projection. Maps D^1 -> R^1 via arctanh.

        inv(y_i) = arctanh(clamp(y_i))
        """
        del z
        y_clamped = y.clamp(min=-1.0 + eps, max=1.0 - eps)
        return torch.atanh(y_clamped)

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            stochastic_fp = group["stochastic_fp"]

            if group["projection"] == "stereographic":
                project_fn = self._stereographic_project
                inverse_fn = self._stereographic_inverse
            elif group["projection"] == "gnomonic":
                project_fn = self._gnomonic_project
                inverse_fn = self._gnomonic_inverse
            elif group["projection"] == "hyperbolic":
                project_fn = self._hyperbolic_project
                inverse_fn = self._hyperbolic_inverse
            else:
                raise ValueError(f"Unknown projection: {group['projection']}")

            for index, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    continue

                grad = parameter.grad.data
                if grad.is_sparse:
                    raise RuntimeError("ProjectiveAdam does not support sparse gradients")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_y"] = self._initialize_state_tensor(parameter.data)
                    state["exp_avg_z"] = self._initialize_state_tensor(parameter.data)
                    if parameter.ndim >= 1 and group["normuon"]:
                        grad_2d = reshape_to_2d(grad)
                        state["normuon_second_momentum"] = self._initialize_state_tensor(
                            grad_2d,
                            use_reference_shape=False,
                            shape=(grad_2d.shape[0], 1),
                        )

                state["step"] += 1
                step = state["step"]

                compute_device = self._get_compute_device(parameter.device)
                non_blocking = compute_device.type == "cuda"

                exp_avg_y = state["exp_avg_y"].detach().to(compute_device, non_blocking=non_blocking, dtype=torch.float32)
                exp_avg_z = state["exp_avg_z"].detach().to(compute_device, non_blocking=non_blocking, dtype=torch.float32)
                grad_fp32 = grad.detach().to(compute_device, non_blocking=non_blocking, dtype=torch.float32)
                parameter_fp32 = parameter.detach().to(compute_device, non_blocking=non_blocking, dtype=torch.float32)

                normuon_second_momentum = None
                if parameter_fp32.ndim >= 1 and group["normuon"]:
                    normuon_second_momentum = state["normuon_second_momentum"].detach().to(
                        compute_device,
                        non_blocking=non_blocking,
                        dtype=torch.float32,
                    )

                if parameter_fp32.ndim >= 1 and group["input_norm"]:
                    grad_work_2d = reshape_to_2d(grad_fp32)
                    grad_work_2d.div_(grad_work_2d.pow(2).mean(dim=-1, keepdim=True).sqrt_().clamp_min_(eps)).clamp_(-step, step)
                    grad_fp32 = grad_work_2d.view_as(parameter_fp32)
                else:
                    grad_fp32.div_(grad_fp32.pow(2).mean().sqrt_().clamp_min_(eps)).clamp_(-step, step)

                y, z = project_fn(grad_fp32, eps)
                exp_avg_y.mul_(beta1).add_(y, alpha=1 - beta1)
                exp_avg_z.mul_(beta2).add_(z, alpha=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                current_y = exp_avg_y / bias_correction1
                current_z = exp_avg_z / bias_correction2
                update = inverse_fn(current_y, current_z, eps)

                if parameter_fp32.ndim >= 1:
                    full_step_2d = reshape_to_2d(update)
                    q_matrix = self.ortho_func(full_step_2d, ortho_dtype=group["ortho_dtype"])

                    if group["normuon"]:
                        assert normuon_second_momentum is not None
                        q_norm = q_matrix.norm(dim=(-2, -1), keepdim=True)
                        variance_mean = torch.mean(q_matrix * q_matrix, dim=-1, keepdim=True)
                        normuon_second_momentum.lerp_(variance_mean, 1 - beta2)
                        step_size = normuon_second_momentum.div(bias_correction2).sqrt().clamp_min_(eps)
                        q_matrix.div_(step_size)
                        q_norm_new = q_matrix.norm(dim=(-2, -1), keepdim=True)
                        q_matrix = q_matrix * (q_norm / q_norm_new.clamp_min(eps))

                    final_step = q_matrix.view_as(parameter_fp32)
                    final_step.mul_((update * final_step).sum())
                else:
                    final_step = update

                cautious_mask = (grad_fp32 * final_step > 0).to(final_step.dtype)
                cautious_mask.div_(cautious_mask.mean().clamp_min_(1e-3))
                final_step.mul_(cautious_mask)

                if weight_decay != 0.0:
                    parameter_fp32.add_(parameter_fp32, alpha=-lr * weight_decay)
                parameter_fp32.add_(final_step, alpha=-lr)

                self._copy_parameter_back(parameter, parameter_fp32, stochastic_fp)
                self._copy_state_back(state["exp_avg_y"], exp_avg_y, stochastic_fp)
                self._copy_state_back(state["exp_avg_z"], exp_avg_z, stochastic_fp)
                if parameter.ndim >= 1 and group["normuon"]:
                    assert normuon_second_momentum is not None
                    self._copy_state_back(state["normuon_second_momentum"], normuon_second_momentum, stochastic_fp)

                if compute_device.type == "cuda" and (index + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize(compute_device)

            if any(parameter.device.type == "cuda" for parameter in group["params"]) or (
                torch.cuda.is_available() and any(parameter.device.type == "cpu" for parameter in group["params"])
            ):
                compute_device = self._get_compute_device(group["params"][0].device)
                if compute_device.type == "cuda":
                    torch.cuda.synchronize(compute_device)

        return loss
