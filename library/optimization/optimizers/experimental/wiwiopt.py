# WiwiOpt from https://github.com/Clybius/Personalized-Optimizers by Clybius

from collections.abc import Iterable
from typing import Literal

import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import copy_stochastic_
from library.utils.compile_env import prepare_windows_compiler_env_for_torch_compile


# Newton-Schulz iteration coefficients for orthogonalization
# From https://kexue.fm/archives/11059
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


def _mean_last_dim_keepdim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1)
    return tensor.mean(dim=-1, keepdim=True)


def _norm_last_dim_keepdim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.abs().reshape(1)
    return tensor.norm(dim=-1, keepdim=True)


@torch.no_grad()
def orthogonalize(matrix: torch.Tensor, num_ns_steps: int = len(NS_COEFFS), ortho_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Orthogonalize a matrix via 5th order Newton-Schulz iteration."""
    original_dtype = matrix.dtype
    if ortho_dtype is not None:
        matrix = matrix.to(ortho_dtype)

    transpose = matrix.shape[0] < matrix.shape[1]
    if transpose:
        matrix = matrix.T

    identity = torch.eye(matrix.shape[1], dtype=matrix.dtype, device=matrix.device)

    for a, b, c in NS_COEFFS[:num_ns_steps]:
        matrix = matrix / torch.linalg.norm(matrix).clamp_min_(1e-8)
        gram = matrix.T @ matrix
        matrix = matrix @ (a * identity + b * gram + c * gram @ gram)

    if transpose:
        matrix = matrix.T

    if ortho_dtype is not None:
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


@torch.no_grad()
def past_update(samples: torch.Tensor, basis: torch.Tensor, inverse_covariance: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch Projection Approximation Subspace Tracking (PAST) update."""
    projected = samples @ basis
    covariance = projected.T @ projected

    rank = basis.shape[1]
    identity = torch.eye(rank, device=samples.device, dtype=samples.dtype)
    inverse_covariance_new = torch.linalg.solve(beta * identity + inverse_covariance @ covariance, inverse_covariance)
    inverse_covariance_new = (inverse_covariance_new + inverse_covariance_new.T) * 0.5

    basis_new = basis + (samples.T @ projected - basis @ covariance) @ inverse_covariance_new
    projected_new = samples @ basis_new
    return basis_new, projected_new, inverse_covariance_new


@torch.no_grad()
def _approx_sq_grad(exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor, eps: float = 1e-16) -> torch.Tensor:
    """CAME-style factorized denominator computation."""
    row_factor = (exp_avg_sq_row + eps).sqrt().unsqueeze(-1)
    col_factor = ((exp_avg_sq_col + eps) / (exp_avg_sq_col.mean(dim=0, keepdim=True) + eps)).unsqueeze(-2).sqrt()
    return row_factor * col_factor


class WiwiOpt(Optimizer):
    r"""
    WiwiOpt (V1.3 with CAME-style factorization).

    A gradient descent optimizer that combines several stabilization & acceleration techniques to produce
    high-signal stable parameter updates.

    WiwiOpt works by:
    1. RMS-based gradient normalization: Incoming gradients are normalized
       by a polynomial-decay EMA of their per-row RMS, preventing exploding
       or vanishing gradient magnitudes.
    2. Egalitarian Gradient Descent (EGD) preconditioning: For 2D+
       parameters, a low-rank SVD approximation is used to precondition the
       gradient, equalizing contribution across singular directions.
    3. Polynomial-schedule momentum: Momentum and accumulation use
       polynomial schedules instead of fixed betas, providing smoothing that
       naturally increases over early training.
    4. Newton-Schulz orthogonalization (Muon): The effective gradient is
       orthogonalized via Newton-Schulz iteration for multi-dimensional
       parameters, producing direction-pure updates.
    5. NorMuon scaling: After orthogonalization, the update is re-scaled
       using a tracked second-moment estimate to maintain consistent update
       magnitudes, then re-projected to preserve the original norm.
    6. Projection re-scaling: The orthogonalized step is re-scaled by its
       projection onto the un-orthogonalized effective gradient, preserving
       meaningful magnitude information.
    7. Cautious masking: Updates are masked so that only components
       agreeing in sign with the raw gradient are kept, preventing
       counterproductive steps.
    8. Dynamic learning rate: Per-parameter learning rate adjustment based
       on the alignment between the EMA of parameter deltas and the EMA of
       their norms, optionally boosted by an ``atan2``-based scaling factor.
    9. CAME-style factorized variance tracking: Uses row-wise AND column-wise
       variance estimates for more accurate gradient normalization.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-4,
        betas: tuple[float, float, float] | tuple[float, float] = (0.95, 0.995, 0.99),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
        weight_decay_rate: float = 1.0,
        normuon: bool = True,
        use_compile: bool = True,
        ortho_dtype: str | torch.dtype | None = None,
        stochastic_fp: bool = True,
        dynamic_lr: bool = True,
        dynamic_lr_boost: bool = True,
        egd: bool = True,
        egd_oja: bool = True,
        egd_method: Literal["past", "oja", "svd"] = "past",
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
            resolved_ortho_dtype = torch.bfloat16
        elif isinstance(ortho_dtype, str):
            resolved_ortho_dtype = getattr(torch, ortho_dtype.split(".")[-1])
        else:
            resolved_ortho_dtype = ortho_dtype

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "weight_decay_rate": weight_decay_rate,
            "normuon": normuon,
            "use_compile": use_compile,
            "ortho_dtype": resolved_ortho_dtype,
            "stochastic_fp": stochastic_fp,
            "dynamic_lr": dynamic_lr,
            "dynamic_lr_boost": dynamic_lr_boost,
            "egd": egd,
            "egd_oja": egd_oja,
            "egd_method": egd_method,
        }

        if use_compile:
            prepare_windows_compiler_env_for_torch_compile()
        self.ortho_func = torch.compile(orthogonalize, mode="reduce-overhead") if use_compile else orthogonalize
        self.oja_func = None
        self.past_func = None
        self.svd_func = None
        if egd:
            if egd_method == "oja" or (egd_method is None and egd_oja):
                self.oja_func = torch.compile(sanger_update, mode="reduce-overhead") if use_compile else sanger_update
            elif egd_method == "past":
                self.past_func = torch.compile(past_update, mode="reduce-overhead") if use_compile else past_update
            elif egd_method == "svd" or (egd_method is None and not egd_oja):
                self.svd_func = torch.compile(torch.svd_lowrank, mode="reduce-overhead") if use_compile else torch.svd_lowrank

        if use_compile:
            try:
                import torch._inductor.config as inductor_config

                inductor_config.triton.cudagraph_skip_dynamic_graphs = True
                inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None
            except (ImportError, AttributeError):
                pass

        super().__init__(params, defaults)

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
            weight_decay_rate = group["weight_decay_rate"]
            stochastic_fp = group["stochastic_fp"]
            egd = group["egd"]
            egd_oja = group["egd_oja"]
            egd_method = group["egd_method"]
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
                    state["accum"] = torch.ones_like(_summary_tensor(parameter), memory_format=torch.preserve_format)
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    if parameter.ndim >= 2:
                        state["exp_avg_sq_row"] = torch.zeros(parameter.shape[:-1], device=parameter.device, dtype=parameter.dtype)
                        col_shape = parameter.shape[:-2] + parameter.shape[-1:]
                        state["exp_avg_sq_col"] = torch.zeros(col_shape, device=parameter.device, dtype=parameter.dtype)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    if dynamic_lr:
                        state["delta_ema"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                        state["delta_norm_ema"] = torch.zeros_like(_summary_tensor(parameter), memory_format=torch.preserve_format)
                    if parameter.ndim >= 1 and group["normuon"]:
                        grad_2d = reshape_to_2d(grad)
                        state["normuon_second_momentum"] = torch.zeros(grad_2d.shape[0], 1, device=parameter.device, dtype=parameter.dtype)

                state["step"] += 1
                step = state["step"]

                accum = state["accum"]
                exp_avg = state["exp_avg"]

                use_stochastic = stochastic_fp and parameter.dtype == torch.bfloat16

                parameter_work = parameter.detach()
                grad_work = grad.detach()
                accum_work = accum.detach()
                exp_avg_work = exp_avg.detach()

                exp_avg_sq_row_work = None
                exp_avg_sq_col_work = None
                exp_avg_sq_work = None
                delta_ema_work = None
                delta_norm_ema_work = None
                normuon_work = None

                if use_stochastic:
                    parameter_work = parameter_work.to(torch.float32)
                    grad_work = grad_work.to(torch.float32)
                    accum_work = accum_work.to(torch.float32)
                    exp_avg_work = exp_avg_work.to(torch.float32)

                if parameter.ndim >= 2:
                    exp_avg_sq_row_work = state["exp_avg_sq_row"].detach()
                    exp_avg_sq_col_work = state["exp_avg_sq_col"].detach()
                    if use_stochastic:
                        exp_avg_sq_row_work = exp_avg_sq_row_work.to(torch.float32)
                        exp_avg_sq_col_work = exp_avg_sq_col_work.to(torch.float32)
                else:
                    exp_avg_sq_work = state["exp_avg_sq"].detach()
                    if use_stochastic:
                        exp_avg_sq_work = exp_avg_sq_work.to(torch.float32)

                if dynamic_lr:
                    delta_ema_work = state["delta_ema"].detach()
                    delta_norm_ema_work = state["delta_norm_ema"].detach()
                    if use_stochastic:
                        delta_ema_work = delta_ema_work.to(torch.float32)
                        delta_norm_ema_work = delta_norm_ema_work.to(torch.float32)

                if parameter.ndim >= 1 and group["normuon"]:
                    normuon_work = state["normuon_second_momentum"].detach()
                    if use_stochastic:
                        normuon_work = normuon_work.to(torch.float32)

                poly_beta1 = (beta1**step - beta1) / (beta1**step - 1.0)
                poly_beta2 = (beta2**step - beta2) / (beta2**step - 1.0)
                poly_beta3 = (beta3**step - beta3) / (beta3**step - 1.0)

                grad_rms = grad_work.pow(2)
                grad_rms = _mean_last_dim_keepdim(grad_rms)
                accum_work.lerp_(grad_rms, weight=1.0 - poly_beta1)
                grad_work.div_(accum_work.sqrt().clamp_min_(eps)).clamp_(-step, step)

                if egd and parameter_work.ndim >= 2:
                    grad_work_2d = reshape_to_2d(grad_work)
                    m_dim, n_dim = grad_work_2d.shape
                    current_rank = min(128, m_dim, n_dim)

                    if current_rank > 0:
                        is_online = (egd_method in {"oja", "past"}) or (egd_method is None and egd_oja)
                        if is_online:
                            if "oja_basis" not in state:
                                track_u = m_dim < n_dim
                                feature_dim = m_dim if track_u else n_dim
                                basis = torch.randn(feature_dim, current_rank, device=parameter_work.device, dtype=torch.float32)
                                basis, _ = torch.linalg.qr(basis)
                                state["oja_basis"] = basis
                                if egd_method == "past":
                                    state["inv_cov"] = torch.eye(current_rank, device=parameter_work.device, dtype=torch.float32) * 0.1

                            track_u = m_dim < n_dim
                            oja_basis_work = state["oja_basis"].detach().float()
                            oja_input = grad_work_2d.T if track_u else grad_work_2d
                            oja_input = oja_input.float()

                            try:
                                projected = None
                                if egd_method == "past" and self.past_func is not None:
                                    inv_cov_work = state["inv_cov"].detach().float()
                                    past_beta = max(poly_beta1, 0.99)
                                    oja_basis_work, projected, inv_cov_work = self.past_func(oja_input, oja_basis_work, inv_cov_work, past_beta)
                                    state["inv_cov"].copy_(inv_cov_work)
                                elif egd_method == "oja" and self.oja_func is not None:
                                    oja_basis_work, projected = self.oja_func(oja_input, oja_basis_work, 1.0 - poly_beta1)

                                if projected is not None:
                                    basis_norm = oja_basis_work / oja_basis_work.norm(dim=0, keepdim=True).clamp_min_(eps)
                                    projected_norm = projected / projected.norm(dim=0, keepdim=True).clamp_min_(eps)
                                    if track_u:
                                        grad_precond = basis_norm @ projected_norm.T
                                    else:
                                        grad_precond = projected_norm @ basis_norm.T

                                    state["oja_basis"].copy_(oja_basis_work)
                                    grad_work = grad_precond.to(parameter_work.dtype).view_as(parameter_work)
                            except RuntimeError:
                                pass
                        else:
                            try:
                                original_dtype = grad_work_2d.dtype
                                grad_f32 = grad_work_2d.float()
                                if self.svd_func is not None:
                                    u_matrix, singular_values, _ = self.svd_func(grad_f32, q=current_rank)
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

                grad_err = grad_work - exp_avg_work
                if parameter_work.ndim >= 2:
                    grad_err_sq = grad_err.pow(2)
                    exp_avg_sq_row_work.lerp_(_mean_last_dim_keepdim(grad_err_sq).squeeze(-1), weight=1.0 - poly_beta2)
                    if grad_err_sq.ndim > 2:
                        exp_avg_sq_col_work.lerp_(grad_err_sq.mean(dim=-2), weight=1.0 - poly_beta2)
                    else:
                        exp_avg_sq_col_work.lerp_(grad_err_sq.mean(dim=0), weight=1.0 - poly_beta2)
                    denominator = _approx_sq_grad(exp_avg_sq_row_work, exp_avg_sq_col_work, eps)
                else:
                    exp_avg_sq_work.lerp_(grad_err.pow(2), weight=1.0 - poly_beta2)
                    denominator = exp_avg_sq_work.sqrt().clamp_min_(eps)

                exp_avg_work.lerp_(grad_work, weight=1.0 - poly_beta1)
                effective_grad = grad_work.clone()
                effective_grad.lerp_(exp_avg_work, weight=poly_beta1)
                effective_grad.div_(denominator)

                if parameter_work.ndim >= 1:
                    full_step_2d = reshape_to_2d(effective_grad)
                    q_matrix = self.ortho_func(full_step_2d, ortho_dtype=group["ortho_dtype"])

                    if group["normuon"] and normuon_work is not None:
                        v_norm = q_matrix.norm(dim=(-2, -1), keepdim=True)
                        v_mean = torch.mean(q_matrix * q_matrix, dim=-1, keepdim=True)
                        normuon_work.lerp_(v_mean, 1.0 - poly_beta2)
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

                lr_adjustment = torch.ones_like(_summary_tensor(parameter_work))
                if dynamic_lr and delta_ema_work is not None and delta_norm_ema_work is not None:
                    if step > 1:
                        alignment_ratio = _norm_last_dim_keepdim(delta_ema_work) / delta_norm_ema_work.clamp_min(eps)
                        if dynamic_lr_boost:
                            update_ratio = delta_norm_ema_work.atan2(delta_ema_work.abs()).mul_(1.27323954474)
                            lr_adjustment = alignment_ratio * update_ratio
                        else:
                            lr_adjustment = alignment_ratio

                    final_step.mul_(lr_adjustment)
                    current_norm = _norm_last_dim_keepdim(final_step)
                    delta_ema_work.lerp_(final_step, weight=1.0 - poly_beta3)
                    delta_norm_ema_work.lerp_(current_norm, weight=1.0 - poly_beta3)

                if weight_decay != 0.0:
                    weight_decay_multiplier = weight_decay_rate**step
                    parameter_mid = torch.where(parameter_work * final_step > 0, parameter_work, torch.zeros_like(parameter_work))
                    decay_source = parameter_mid * lr_adjustment if dynamic_lr else parameter_mid
                    parameter_work.add_(decay_source, alpha=-lr * weight_decay * weight_decay_multiplier)
                parameter_work.add_(final_step, alpha=-lr)

                if use_stochastic:
                    copy_stochastic_(accum, accum_work)
                    copy_stochastic_(exp_avg, exp_avg_work)
                    if parameter.ndim >= 2:
                        copy_stochastic_(state["exp_avg_sq_row"], exp_avg_sq_row_work)
                        copy_stochastic_(state["exp_avg_sq_col"], exp_avg_sq_col_work)
                    else:
                        copy_stochastic_(state["exp_avg_sq"], exp_avg_sq_work)
                    if dynamic_lr and delta_ema_work is not None and delta_norm_ema_work is not None:
                        copy_stochastic_(state["delta_ema"], delta_ema_work)
                        copy_stochastic_(state["delta_norm_ema"], delta_norm_ema_work)
                    copy_stochastic_(parameter, parameter_work)
                    if normuon_work is not None:
                        copy_stochastic_(state["normuon_second_momentum"], normuon_work)
                else:
                    accum.copy_(accum_work)
                    exp_avg.copy_(exp_avg_work)
                    if parameter.ndim >= 2:
                        state["exp_avg_sq_row"].copy_(exp_avg_sq_row_work)
                        state["exp_avg_sq_col"].copy_(exp_avg_sq_col_work)
                    else:
                        state["exp_avg_sq"].copy_(exp_avg_sq_work)
                    if dynamic_lr and delta_ema_work is not None and delta_norm_ema_work is not None:
                        state["delta_ema"].copy_(delta_ema_work)
                        state["delta_norm_ema"].copy_(delta_norm_ema_work)
                    parameter.copy_(parameter_work)
                    if normuon_work is not None:
                        state["normuon_second_momentum"].copy_(normuon_work)

        return loss
