from collections.abc import Iterable

import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import copy_stochastic_


NS_COEFFS = [
    (8.287212018145622, -23.59588651909882, 17.300387312530923),
    (4.107059111542197, -2.9478499167379084, 0.54484310829266),
    (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
    (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
    (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
    (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
    (1.875, -1.25, 0.375),
]


def reshape_to_2d(grad: torch.Tensor) -> torch.Tensor:
    """Reshape a tensor to 2D for matrix operations."""
    if grad.ndim > 2:
        return grad.reshape(len(grad), -1)
    if grad.ndim < 2:
        return grad.reshape(1, -1)
    return grad


def _summary_tensor(parameter: torch.Tensor) -> torch.Tensor:
    if parameter.ndim == 0:
        return parameter.reshape(1)
    return parameter.mean(dim=-1, keepdim=True)


def _mean_square_keepdim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.pow(2).reshape(1)
    return tensor.pow(2).mean(dim=-1, keepdim=True)


def _norm_keepdim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.abs().reshape(1)
    return tensor.norm(dim=-1, keepdim=True)


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


@torch.no_grad()
def sanger_update(samples: torch.Tensor, basis: torch.Tensor, lr: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Single step of Sanger's Rule (Generalized Oja's rule) for online PCA."""
    samples_norm = samples / samples.norm().clamp_min(1e-8)
    projected = samples_norm @ basis
    basis_update = samples_norm.T @ projected - basis @ torch.triu(projected.T @ projected)
    basis_new = basis + lr * basis_update
    projected_new = samples @ basis_new
    return basis_new, projected_new


class WiwiOpt(Optimizer):
    r"""WiwiOpt (V1.1).

    A gradient descent optimizer that combines several stabilization and
    acceleration techniques to produce high-signal stable parameter updates.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float, float] | tuple[float, float] = (0.95, 0.995, 0.99),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
        normuon: bool = True,
        use_compile: bool = True,
        ortho_dtype: str | torch.dtype | None = None,
        stochastic_fp: bool = True,
        dynamic_lr: bool = True,
        dynamic_lr_boost: bool = True,
        egd: bool = True,
        egd_oja: bool = True,
        **kwargs,
    ):
        del kwargs

        if len(betas) == 2:
            betas = (betas[0], betas[0], betas[1])
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if ortho_dtype is None:
            resolved_ortho_dtype = torch.float32
        elif isinstance(ortho_dtype, str):
            resolved_ortho_dtype = getattr(torch, ortho_dtype.split(".")[-1])
        else:
            resolved_ortho_dtype = ortho_dtype

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "normuon": normuon,
            "use_compile": use_compile,
            "ortho_dtype": resolved_ortho_dtype,
            "stochastic_fp": stochastic_fp,
            "dynamic_lr": dynamic_lr,
            "dynamic_lr_boost": dynamic_lr_boost,
            "egd": egd,
            "egd_oja": egd_oja,
        }
        super().__init__(params, defaults)

        self.ortho_func = torch.compile(orthogonalize, mode="reduce-overhead") if use_compile else orthogonalize
        if egd:
            if egd_oja:
                self.egd_func = torch.compile(sanger_update, mode="reduce-overhead") if use_compile else sanger_update
            else:
                self.egd_func = torch.compile(torch.svd_lowrank, mode="reduce-overhead") if use_compile else torch.svd_lowrank

    def __str__(self) -> str:
        return "WiwiOpt"

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
            beta1, beta2, beta3 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            stochastic_fp = group["stochastic_fp"]
            egd = group["egd"]
            egd_oja = group["egd_oja"]
            dynamic_lr = group["dynamic_lr"]
            dynamic_lr_boost = group["dynamic_lr_boost"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("WiwiOpt does not support sparse gradients")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["polyak"] = torch.ones_like(_summary_tensor(parameter), memory_format=torch.preserve_format)
                    state["accum"] = torch.ones_like(_summary_tensor(parameter), memory_format=torch.preserve_format)
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    if dynamic_lr:
                        state["delta_ema"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                        state["delta_norm_ema"] = torch.zeros_like(_summary_tensor(parameter), memory_format=torch.preserve_format)
                    if parameter.ndim >= 1 and group["normuon"]:
                        grad_2d = reshape_to_2d(grad)
                        state["normuon_second_momentum"] = torch.zeros(grad_2d.shape[0], 1, device=parameter.device, dtype=parameter.dtype)

                state["step"] += 1
                step = state["step"]

                polyak = state["polyak"]
                accum = state["accum"]
                exp_avg = state["exp_avg"]
                normuon_second_momentum = state.get("normuon_second_momentum")

                use_stochastic = stochastic_fp and parameter.dtype == torch.bfloat16
                if use_stochastic:
                    parameter_work = parameter.detach().to(torch.float32)
                    grad_work = grad.detach().to(torch.float32)
                    polyak_work = polyak.detach().to(torch.float32)
                    accum_work = accum.detach().to(torch.float32)
                    exp_avg_work = exp_avg.detach().to(torch.float32)
                    if dynamic_lr:
                        delta_ema_work = state["delta_ema"].detach().to(torch.float32)
                        delta_norm_ema_work = state["delta_norm_ema"].detach().to(torch.float32)
                    if normuon_second_momentum is not None:
                        normuon_work = normuon_second_momentum.detach().to(torch.float32)
                else:
                    parameter_work = parameter.detach()
                    grad_work = grad.detach()
                    polyak_work = polyak.detach()
                    accum_work = accum.detach()
                    exp_avg_work = exp_avg.detach()
                    if dynamic_lr:
                        delta_ema_work = state["delta_ema"].detach()
                        delta_norm_ema_work = state["delta_norm_ema"].detach()
                    if normuon_second_momentum is not None:
                        normuon_work = normuon_second_momentum.detach()

                poly_beta1 = (beta1**step - beta1) / (beta1**step - 1.0)
                poly_beta2 = (beta2**step - beta2) / (beta2**step - 1.0)
                poly_beta3 = (beta3**step - beta3) / (beta3**step - 1.0)

                grad_rms = _mean_square_keepdim(grad_work)
                accum_work.lerp_(grad_rms, 1.0 - poly_beta1)
                grad_work.div_(accum_work.sqrt().clamp_min_(eps)).clamp_(-step, step)

                if egd and parameter_work.ndim >= 2:
                    grad_work_2d = reshape_to_2d(grad_work)
                    m_dim, n_dim = grad_work_2d.shape
                    current_rank = min(128, m_dim, n_dim)

                    if current_rank > 0:
                        if egd_oja:
                            if "oja_basis" not in state:
                                track_u = m_dim < n_dim
                                feature_dim = m_dim if track_u else n_dim
                                basis = torch.randn(feature_dim, current_rank, device=parameter_work.device, dtype=parameter_work.dtype)
                                basis, _ = torch.linalg.qr(basis)
                                state["oja_basis"] = basis

                            track_u = m_dim < n_dim
                            oja_basis_work = state["oja_basis"]
                            if use_stochastic:
                                oja_basis_work = oja_basis_work.detach().float()

                            oja_input = grad_work_2d.T if track_u else grad_work_2d
                            if use_stochastic:
                                oja_input = oja_input.float()

                            try:
                                oja_basis_work, projected = self.egd_func(oja_input, oja_basis_work, 1.0 - poly_beta1)
                                if track_u:
                                    v_basis = projected / projected.norm(dim=0, keepdim=True).clamp_min_(eps)
                                    grad_precond = oja_basis_work @ v_basis.T
                                else:
                                    u_basis = projected / projected.norm(dim=0, keepdim=True).clamp_min_(eps)
                                    grad_precond = u_basis @ oja_basis_work.T

                                if use_stochastic:
                                    grad_precond = grad_precond.to(parameter_work.dtype)
                                    copy_stochastic_(state["oja_basis"], oja_basis_work)
                                else:
                                    state["oja_basis"].copy_(oja_basis_work)

                                grad_work = grad_precond.view_as(parameter_work)
                            except RuntimeError:
                                pass
                        else:
                            try:
                                original_dtype = grad_work_2d.dtype
                                grad_f32 = grad_work_2d.float()
                                u_matrix, singular_values, _ = self.egd_func(grad_f32, q=current_rank)
                                u_matrix = u_matrix.to(original_dtype)
                                singular_values = singular_values.to(original_dtype)
                                singular_values = torch.maximum(
                                    singular_values,
                                    torch.tensor(eps, device=singular_values.device, dtype=singular_values.dtype),
                                )
                                aux = (u_matrix * (1.0 / singular_values).unsqueeze(0)) @ u_matrix.mT
                                grad_work = (aux @ grad_work_2d).view_as(parameter_work)
                            except RuntimeError:
                                pass

                polyak_work.lerp_(_mean_square_keepdim(grad_work - exp_avg_work), weight=1.0 - poly_beta2)
                exp_avg_work.lerp_(grad_work, weight=1.0 - poly_beta1)
                effective_grad = grad_work.lerp(exp_avg_work, weight=poly_beta1).div(polyak_work.sqrt().clamp_min_(eps))

                if parameter_work.ndim >= 1:
                    full_step_2d = reshape_to_2d(effective_grad)
                    q_matrix = self.ortho_func(full_step_2d, ortho_dtype=group["ortho_dtype"])

                    if group["normuon"]:
                        assert normuon_second_momentum is not None
                        v_norm = q_matrix.norm(dim=(-2, -1), keepdim=True)
                        v_mean = torch.mean(q_matrix * q_matrix, dim=-1, keepdim=True)
                        normuon_work.lerp_(v_mean, 1 - poly_beta2)
                        step_size = normuon_work.sqrt().clamp_min_(eps)
                        q_matrix.div_(step_size)
                        v_norm_new = q_matrix.norm(dim=(-2, -1), keepdim=True)
                        q_matrix = q_matrix * (v_norm / v_norm_new.clamp_min(eps))

                    final_step = q_matrix.view_as(parameter_work)
                    final_step.mul_((effective_grad * final_step).sum())
                else:
                    final_step = effective_grad

                cautious_mask = (grad_work * final_step > 0).to(final_step.dtype)
                cautious_mask.div_(cautious_mask.mean().clamp_min_(1e-3))
                final_step.mul_(cautious_mask)

                if dynamic_lr:
                    if step > 1:
                        alignment_ratio = _norm_keepdim(delta_ema_work) / delta_norm_ema_work.clamp_min(eps)
                        if dynamic_lr_boost:
                            update_ratio = delta_norm_ema_work.atan2(delta_ema_work.abs()).mul_(1.27323954474)
                            lr_adjustment = alignment_ratio * update_ratio
                        else:
                            lr_adjustment = alignment_ratio
                    else:
                        lr_adjustment = torch.ones_like(_summary_tensor(parameter_work))

                    final_step.mul_(lr_adjustment)
                    current_norm = _norm_keepdim(final_step)
                    delta_ema_work.lerp_(final_step, 1.0 - poly_beta3)
                    delta_norm_ema_work.lerp_(current_norm, 1.0 - poly_beta3)

                if weight_decay != 0.0:
                    parameter_work.add_(parameter_work * lr_adjustment if dynamic_lr else parameter_work, alpha=-lr * weight_decay)
                parameter_work.add_(final_step, alpha=-lr)

                if use_stochastic:
                    copy_stochastic_(polyak, polyak_work)
                    copy_stochastic_(accum, accum_work)
                    copy_stochastic_(exp_avg, exp_avg_work)
                    if dynamic_lr:
                        copy_stochastic_(state["delta_ema"], delta_ema_work)
                        copy_stochastic_(state["delta_norm_ema"], delta_norm_ema_work)
                    copy_stochastic_(parameter, parameter_work)
                    if normuon_second_momentum is not None:
                        copy_stochastic_(normuon_second_momentum, normuon_work)
                else:
                    polyak.copy_(polyak_work)
                    accum.copy_(accum_work)
                    exp_avg.copy_(exp_avg_work)
                    if dynamic_lr:
                        state["delta_ema"].copy_(delta_ema_work)
                        state["delta_norm_ema"].copy_(delta_norm_ema_work)
                    parameter.copy_(parameter_work)
                    if normuon_second_momentum is not None:
                        normuon_second_momentum.copy_(normuon_work)

        return loss
