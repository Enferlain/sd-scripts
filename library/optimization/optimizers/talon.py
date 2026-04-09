import logging
import math
import os
import shutil

import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import copy_stochastic_


NS_COEFFS = [
    (3.5318, -4.7911, 1.9388),
    (3.3274, -4.0557, 1.5782),
    (3.0809, -3.5160, 1.3464),
    (2.7476, -2.8484, 1.0775),
    (2.2948, -2.0951, 0.7895),
    (2.1535, -1.8338, 0.6869),
]


@torch.no_grad()
def orthogonalize(matrix: torch.Tensor, num_ns_steps: int = len(NS_COEFFS), ortho_dtype=None, adaptive: bool = False):
    if ortho_dtype is not None:
        orig_dtype = matrix.dtype
        matrix = matrix.to(ortho_dtype)
    if adaptive:
        matrix_orig = matrix.clone()
    transpose = matrix.shape[0] < matrix.shape[1]
    if transpose:
        matrix = matrix.T
    matrix = matrix / (torch.linalg.norm(matrix) + 1e-20)
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        gram = matrix.T @ matrix
        ident = torch.eye(gram.shape[0], dtype=matrix.dtype, device=matrix.device)
        matrix = matrix @ (a * ident + b * gram + c * gram @ gram)
    if transpose:
        matrix = matrix.T
    if adaptive:
        matrix = torch.einsum("ij,ij,ab->ab", matrix_orig.type_as(matrix), matrix, matrix)
    if ortho_dtype is not None:
        matrix = matrix.to(orig_dtype)
    return matrix


@torch.no_grad()
def block_matmul(p1, q1, r1, p2, q2, r2):
    return p1 @ p2 + q1 @ q2.T, p1 @ q2 + q1 @ r2, q1.T @ q2 + r1 @ r2


@torch.no_grad()
def newton_schulz_iter(p, q, r, a: float, b: float, c: float):
    p2, q2, r2 = block_matmul(p, q, r, p, q, r)
    p4, q4, r4 = block_matmul(p2, q2, r2, p2, q2, r2)
    ident_p = a * torch.eye(p.shape[0], dtype=p.dtype, device=p.device)
    ident_r = a * torch.eye(r.shape[0], dtype=r.dtype, device=r.device)
    ppoly = ident_p + b * p2 + c * p4
    qpoly = b * q2 + c * q4
    rpoly = ident_r + b * r2 + c * r4
    return block_matmul(p, q, r, ppoly, qpoly, rpoly)


@torch.no_grad()
def orthogonalize_blockwise(weights: torch.Tensor, ortho_dtype=torch.float32, num_ns_steps: int = len(NS_COEFFS)):
    orig_dtype = weights.dtype
    m, n = weights.shape
    ident_m = torch.eye(m, device=weights.device)
    ident_n = torch.eye(n, device=weights.device)
    norm = 1 + torch.linalg.norm(weights)
    p = (ident_m / (norm + 1e-12)).to(ortho_dtype)
    q = (weights / (norm + 1e-12)).to(ortho_dtype)
    r = (ident_n / (norm + 1e-12)).to(ortho_dtype)
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        p, q, r = newton_schulz_iter(p, q, r, a=a, b=b, c=c)
    return p.to(orig_dtype), q.to(orig_dtype), r.to(orig_dtype)


def _spectral_hardcap_blockwise(weights: torch.Tensor, sigma_max=1.0, ortho_dtype=torch.float32, num_ns_steps: int = len(NS_COEFFS), adaptive: bool = False):
    def _util(local_weights: torch.Tensor):
        if adaptive:
            weights_orig = local_weights.clone()
        transpose = local_weights.shape[0] > local_weights.shape[1]
        if transpose:
            local_weights = local_weights.T
        orig_dtype = local_weights.dtype
        local_weights = local_weights.to(ortho_dtype)
        p, q, _ = orthogonalize_blockwise(local_weights, ortho_dtype, num_ns_steps)
        result = q + p @ local_weights
        if transpose:
            result = result.T
        if adaptive:
            result = torch.einsum("ij,ij,ab->ab", weights_orig.type_as(result), result, result)
        return result.to(orig_dtype)

    return sigma_max * _util(weights / sigma_max)


def _spectral_clip(weights: torch.Tensor, sigma_min: float = -1.0, sigma_max: float = 1.0, ortho_dtype=torch.float32, num_ns_steps: int = len(NS_COEFFS), adaptive: bool = False):
    if adaptive:
        weights_orig = weights.clone()
    flip = weights.shape[0] > weights.shape[1]
    if flip:
        weights = weights.T
    orig_dtype = weights.dtype
    weights = weights.to(ortho_dtype)
    ortho_weights = orthogonalize(weights, num_ns_steps)
    eye_m = torch.eye(weights.shape[0], dtype=weights.dtype, device=weights.device)
    result = (
        0.5
        * (
            (sigma_min + sigma_max) * eye_m
            + (sigma_min * ortho_weights - weights) @ orthogonalize(sigma_min * ortho_weights - weights, num_ns_steps).T
            - (sigma_max * ortho_weights - weights) @ orthogonalize(sigma_max * ortho_weights - weights, num_ns_steps).T
        )
        @ ortho_weights
    )
    if flip:
        result = result.T
    if adaptive:
        result = torch.einsum("ij,ij,ab->ab", weights_orig.type_as(result), result, result)
    return result.to(orig_dtype)


@torch.no_grad()
def batch_project(matrix: torch.Tensor, project_fn):
    matrix_shape = matrix.shape[-2:]
    flattened = matrix.reshape(-1, *matrix_shape)
    projected = torch.vmap(project_fn)(flattened)
    return projected.reshape(matrix.shape) / len(flattened)


@torch.no_grad()
def spectral_clip_func(weights: torch.Tensor, sigma_min: float = -1.0, sigma_max: float = 1.0, ortho_dtype=torch.float32, num_ns_steps: int = len(NS_COEFFS), adaptive: bool = False):
    return batch_project(weights, lambda value: _spectral_clip(value, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive))


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def spectral_clip_compiled_func(weights: torch.Tensor, sigma_min: float = -1.0, sigma_max: float = 1.0, ortho_dtype=torch.float32, num_ns_steps: int = len(NS_COEFFS), adaptive: bool = False):
    return batch_project(weights, lambda value: _spectral_clip(value, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive))


@torch.no_grad()
def spectral_hardcap_func(weights: torch.Tensor, sigma_min: float = -1.0, sigma_max: float = 1.0, ortho_dtype=torch.float32, num_ns_steps: int = len(NS_COEFFS), adaptive: bool = False):
    del sigma_min
    return batch_project(weights, lambda value: _spectral_hardcap_blockwise(value, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive))


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def spectral_hardcap_compiled_func(weights: torch.Tensor, sigma_min: float = -1.0, sigma_max: float = 1.0, ortho_dtype=torch.float32, num_ns_steps: int = len(NS_COEFFS), adaptive: bool = False):
    del sigma_min
    return batch_project(weights, lambda value: _spectral_hardcap_blockwise(value, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive))


@torch.no_grad()
def orthogonalize_func(weights: torch.Tensor, sigma_min: float = -1.0, sigma_max: float = 1.0, ortho_dtype=torch.float32, num_ns_steps: int = len(NS_COEFFS), adaptive: bool = False):
    del sigma_min, sigma_max
    return batch_project(weights, lambda value: orthogonalize(value, num_ns_steps=num_ns_steps, ortho_dtype=ortho_dtype, adaptive=adaptive))


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def orthogonalize_compiled_func(weights: torch.Tensor, sigma_min: float = -1.0, sigma_max: float = 1.0, ortho_dtype=torch.float32, num_ns_steps: int = len(NS_COEFFS), adaptive: bool = False):
    del sigma_min, sigma_max
    return batch_project(weights, lambda value: orthogonalize(value, num_ns_steps=num_ns_steps, ortho_dtype=ortho_dtype, adaptive=adaptive))


@torch.no_grad()
def separate_frequencies(grad: torch.Tensor, cutoff_freq_ratio: float = 0.1):
    if not 0.0 <= cutoff_freq_ratio <= 1.0:
        raise ValueError("cutoff_freq_ratio must be between 0.0 and 1.0")
    if cutoff_freq_ratio == 1.0:
        return grad.clone(), torch.zeros_like(grad)
    if cutoff_freq_ratio == 0.0:
        return torch.zeros_like(grad), grad.clone()

    grad_fft = torch.fft.fftn(grad)
    grad_fft_shifted = torch.fft.fftshift(grad_fft)
    center_indices = [size // 2 for size in grad.shape]
    min_dim_size = min(grad.shape)
    cutoff_radius = int(min_dim_size * cutoff_freq_ratio / 2)
    grid_coords = torch.meshgrid(*[torch.arange(size, device=grad.device) for size in grad.shape], indexing="ij")
    dist_from_center_sq = torch.zeros_like(grad, dtype=torch.float32)
    for i, center_idx in enumerate(center_indices):
        dist_from_center_sq += (grid_coords[i] - center_idx) ** 2
    low_pass_mask = dist_from_center_sq <= cutoff_radius**2
    low_freq_fft_shifted = grad_fft_shifted * low_pass_mask
    low_freq_fft = torch.fft.ifftshift(low_freq_fft_shifted)
    low_freq_component = torch.fft.ifftn(low_freq_fft).real
    return low_freq_component, grad - low_freq_component


@torch.no_grad()
def freq_sep_func(weights: torch.Tensor, cutoff_freq_ratio: float = 0.1):
    return separate_frequencies(weights, cutoff_freq_ratio=cutoff_freq_ratio)


def filter_grad(grad, fft_alpha: float = 1.0):
    grad_freq = torch.fft.fftn(grad, dim=list(range(grad.dim())))
    freq_dims = [torch.fft.fftfreq(size, device=grad.device) for size in grad.shape]
    shifted_freq_dims = [torch.fft.ifftshift(dim) for dim in freq_dims]
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing="ij"))
    max_radius = 0.5 * math.sqrt(len(grad.shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    filter_weights = torch.exp(-fft_alpha * (radius**2))
    filtered_grad_freq = grad_freq * filter_weights
    return torch.fft.ifftn(filtered_grad_freq, dim=list(range(grad.dim()))).real


def _can_use_compiled_spectral_helpers() -> bool:
    nvcc_path = shutil.which("nvcc")
    return bool(torch.cuda.is_available() and nvcc_path and os.access(nvcc_path, os.X_OK))


class TALON(Optimizer):
    r"""
    TALON: Temporal Adaptation via Level and Orientation Normalization.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float, float] = (0.9, 0.99, 1.0 - 1e-7),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.995,
        denom_atan2: bool = True,
        separate_frequencies: float = 0.0,
        highfreq_mult: float = 0.1,
        lowpass_grad: float = 0.0,
        invariant: bool = False,
        spectral_clip: bool = True,
        spectral_clip_compile: bool = True,
        spectral_min: float = -1.0,
        spectral_max: float = 1.0,
        spectral_adaptive: bool = False,
        signscale_power: float = 1.0,
        orthograd: bool = False,
        stochastic_fp: bool = True,
        **kwargs,
    ):
        for key in kwargs:
            logging.warning("Optimizer argument '%s' passed into TALON. It will be ignored.", key)

        self._init_lr = lr
        use_compiled_helper = spectral_clip_compile and _can_use_compiled_spectral_helpers()
        if spectral_clip:
            if spectral_min < -1000:
                self.clip_func = spectral_hardcap_compiled_func if use_compiled_helper else spectral_hardcap_func
            elif spectral_min == 0 and spectral_max == 0:
                self.clip_func = orthogonalize_compiled_func if use_compiled_helper else orthogonalize_func
            else:
                self.clip_func = spectral_clip_compiled_func if use_compiled_helper else spectral_clip_func

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decay_rate": weight_decay_rate,
            "denom_atan2": denom_atan2,
            "separate_frequencies": separate_frequencies,
            "highfreq_mult": highfreq_mult,
            "lowpass_grad": lowpass_grad,
            "invariant": invariant,
            "spectral_clip": spectral_clip,
            "spectral_clip_compile": spectral_clip_compile,
            "spectral_min": spectral_min,
            "spectral_max": spectral_max,
            "spectral_adaptive": spectral_adaptive,
            "signscale_power": signscale_power,
            "orthograd": orthograd,
            "stochastic_fp": stochastic_fp,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "TALON"

    @torch.no_grad()
    def orthograd_atan2sin(self, param, grad):
        weights = param.view(-1)
        grad_flat = grad.view(-1)
        dot_product = torch.dot(weights, grad_flat).atan2_(torch.dot(weights, weights))
        sin_dot_product = torch.sin(dot_product)
        grad_atansin = grad_flat.to(dtype=torch.float32, copy=True).atan().sin_()
        grad_atancos = grad_flat.to(dtype=torch.float32, copy=True).atan().cos_()
        grad_orth = grad_atansin.sub(weights.atan().sin_(), alpha=sin_dot_product).div(grad_atancos)
        grad_orth_scaled = grad_orth.mul(grad_flat.norm(2).div_(grad_orth.norm(2).clamp_min_(1e-16)))
        grad.copy_(grad_orth_scaled.view_as(grad))

    @torch.no_grad()
    def invariance(self, grad, degrad=None):
        if degrad is None:
            grad_atansin = grad.atan().sin_()
            grad_atancos = grad.atan().cos_()
            return grad_atansin, grad_atancos
        return grad.atan2(degrad).mul_(1.27323954474)

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1

            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            step = group["step"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                grad = param.grad.data

                if len(state) == 0:
                    state["value_momentum"] = torch.ones_like(param.data)
                    state["stage2_emasq"] = torch.ones_like(param.data)
                    state["sign_momentum"] = torch.zeros_like(grad)

                param_fp32 = param.detach().clone()
                value_momentum = state["value_momentum"].detach().clone()
                stage2_emasq = state["stage2_emasq"].detach().clone()
                sign_momentum = state["sign_momentum"].detach().clone()

                if param.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    grad = grad.to(torch.float32)
                    value_momentum = state["value_momentum"].detach().clone().to(torch.float32)
                    stage2_emasq = state["stage2_emasq"].detach().clone().to(torch.float32)
                    sign_momentum = state["sign_momentum"].detach().clone().to(torch.float32)
                    param_fp32 = param.detach().clone().to(torch.float32)

                slow_beta1 = (betas[1] ** step - betas[1]) / (betas[1] ** step - 1.0)
                slow_beta2 = (betas[2] ** step - betas[2]) / (betas[2] ** step - 1.0)
                clip_lambda = step**0.25
                rms = grad.pow(2).mean().sqrt_().clamp_min_(1)
                grad = grad.div(rms)

                if group["orthograd"] and param_fp32.data.nelement() > 1:
                    self.orthograd_atan2sin(param_fp32, grad)

                sign_momentum = sign_momentum.lerp(grad.sign(), weight=1.0 - betas[0])
                grad = torch.where(grad.abs() > 255, grad.mul(255 / grad.abs()), grad)

                if grad.ndim > 0 and group["lowpass_grad"] != 0:
                    grad = filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs().mul_(grad.sign())

                value_momentum = value_momentum.mul(slow_beta1).add_(grad.abs(), alpha=1 - slow_beta1)

                if grad.ndim > 0 and group["spectral_clip"]:
                    if grad.ndim > 1:
                        c_t = self.clip_func(
                            value_momentum,
                            sigma_min=group["spectral_min"],
                            sigma_max=group["spectral_max"],
                            adaptive=group["spectral_adaptive"],
                        )
                    else:
                        c_t = self.clip_func(
                            value_momentum.view(len(value_momentum), -1),
                            sigma_min=group["spectral_min"],
                            sigma_max=group["spectral_max"],
                            adaptive=group["spectral_adaptive"],
                        ).view(value_momentum.shape)
                else:
                    c_t = value_momentum

                if group["invariant"] and c_t.nelement() > 0:
                    c_t, degrad = self.invariance(c_t)
                else:
                    degrad = None

                if group["denom_atan2"]:
                    full_step = c_t.atan2(stage2_emasq.sqrt()).mul_(1.27323954474)
                else:
                    stage2_denom = torch.clamp(stage2_emasq.sqrt(), 1e-16)
                    full_step = c_t.div(stage2_denom).clamp_(-clip_lambda, clip_lambda)

                stage2_emasq = stage2_emasq.mul(slow_beta2).addcmul_(grad, grad, value=1 - slow_beta2)

                if group["invariant"] and degrad is not None and grad.nelement() > 0:
                    full_step = self.invariance(full_step, degrad)

                if grad.ndim > 0 and group["separate_frequencies"] != 0:
                    low_freq_grad, high_freq_grad = freq_sep_func(
                        full_step, cutoff_freq_ratio=group["separate_frequencies"]
                    )
                    full_step = low_freq_grad + high_freq_grad.mul(group["highfreq_mult"])

                full_step = full_step.mul(
                    sign_momentum.abs().pow_(group["signscale_power"]).mul_(sign_momentum.sign())
                )

                if weight_decay != 0:
                    full_step = full_step.add(
                        param_fp32.data, alpha=weight_decay * weight_decay_rate**group["step"]
                    )

                param_fp32.data.add_(full_step, alpha=-lr)
                if param.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    copy_stochastic_(state["value_momentum"], value_momentum)
                    copy_stochastic_(state["stage2_emasq"], stage2_emasq)
                    copy_stochastic_(state["sign_momentum"], sign_momentum)
                    copy_stochastic_(param, param_fp32)
                else:
                    state["value_momentum"].copy_(value_momentum)
                    state["stage2_emasq"].copy_(stage2_emasq)
                    state["sign_momentum"].copy_(sign_momentum)
                    param.copy_(param_fp32)
        return loss
