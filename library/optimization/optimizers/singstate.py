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
    return _spectral_clip(weights, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive)


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def spectral_clip_compiled_func(weights: torch.Tensor, sigma_min: float = -1.0, sigma_max: float = 1.0, ortho_dtype=torch.float32, num_ns_steps: int = len(NS_COEFFS), adaptive: bool = False):
    return _spectral_clip(weights, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive)


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
    return orthogonalize(weights, num_ns_steps=num_ns_steps, ortho_dtype=ortho_dtype, adaptive=adaptive)


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def orthogonalize_compiled_func(weights: torch.Tensor, sigma_min: float = -1.0, sigma_max: float = 1.0, ortho_dtype=torch.float32, num_ns_steps: int = len(NS_COEFFS), adaptive: bool = False):
    del sigma_min, sigma_max
    return orthogonalize(weights, num_ns_steps=num_ns_steps, ortho_dtype=ortho_dtype, adaptive=adaptive)


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
    grad_freq = torch.fft.fftn(grad, norm="ortho")
    freq_dims = [torch.fft.fftfreq(size, device=grad.device) for size in grad.shape]
    shifted_freq_dims = [torch.fft.ifftshift(dim) for dim in freq_dims]
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing="ij"))
    max_radius = 0.5 * math.sqrt(len(grad.shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    filter_weights = torch.exp(-fft_alpha * (radius**2))
    filtered_grad_freq = grad_freq * filter_weights
    return torch.fft.ifftn(filtered_grad_freq, norm="ortho").real


def sym(matrix):
    return 0.5 * (matrix + matrix.T)


def project_to_stiefel_tangent_space(point, delta_point):
    return delta_point - point @ sym(point.T @ delta_point)


def steepest_descent_stiefel_manifold_heuristic(weights, grad, num_steps: int = 3):
    assert num_steps > 0, "Number of steps must be positive"
    a_star = grad
    for _ in range(num_steps):
        a_star = project_to_stiefel_tangent_space(weights, a_star)
        a_star = orthogonalize(a_star)
    return a_star


def _can_use_compiled_spectral_helpers() -> bool:
    nvcc_path = shutil.which("nvcc")
    return bool(torch.cuda.is_available() and nvcc_path and os.access(nvcc_path, os.X_OK))


class SingState(Optimizer):
    r"""
    SingState cuts through noise by decoupling the gradient sign and magnitude into different momentum states.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.995,
        spectral_clip: bool = False,
        spectral_clip_compile: bool = True,
        spectral_clip_dtype=None,
        spectral_min: float = -1.0,
        spectral_max: float = 1.0,
        spectral_adaptive: bool = False,
        lowpass_grad: float = 1.0,
        stochastic_fp: bool = True,
        **kwargs,
    ):
        for key in kwargs:
            logging.warning("Optimizer argument '%s' passed into SingState. It will be ignored.", key)

        self._init_lr = lr

        use_compiled_helper = spectral_clip_compile and _can_use_compiled_spectral_helpers()
        if spectral_clip:
            if spectral_min == 0 and spectral_max == 0:
                self.clip_func = orthogonalize_compiled_func if use_compiled_helper else orthogonalize_func
            else:
                self.clip_func = spectral_clip_compiled_func if use_compiled_helper else spectral_clip_func

        if spectral_clip_dtype is None:
            spectral_clip_dtype = torch.float32
        if isinstance(spectral_clip_dtype, str):
            dtype_name = spectral_clip_dtype.split(".")[-1]
            spectral_clip_dtype = getattr(torch, dtype_name)

        defaults = {
            "lr": lr,
            "beta": beta,
            "weight_decay": weight_decay,
            "weight_decay_rate": weight_decay_rate,
            "spectral_clip": spectral_clip,
            "spectral_clip_compile": spectral_clip_compile,
            "spectral_clip_dtype": spectral_clip_dtype,
            "spectral_min": spectral_min,
            "spectral_max": spectral_max,
            "spectral_adaptive": spectral_adaptive,
            "lowpass_grad": lowpass_grad,
            "stochastic_fp": stochastic_fp,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "SingState"

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
            beta = group["beta"]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                grad = param.grad.data
                dimcount = grad.ndim

                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(grad)

                param_fp32 = param.detach().clone()
                momentum = state["momentum"].detach().clone()

                if param.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    grad = grad.to(torch.float32)
                    momentum = state["momentum"].detach().clone().to(torch.float32)
                    param_fp32 = param.detach().clone().to(torch.float32)

                if dimcount > 0:
                    grad = filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs().mul_(grad.sign())

                grad = grad.clamp(-group["step"], group["step"])
                denom = momentum.abs()
                momentum = momentum.lerp(grad.sign(), weight=1.0 - beta)
                c_t = grad.abs().lerp(momentum.abs(), weight=beta)

                if dimcount >= 2 and group["spectral_clip"]:
                    if dimcount > 2:
                        c_t_2d = c_t.reshape(len(c_t), -1)
                    else:
                        c_t_2d = c_t
                    flip = c_t_2d.shape[0] > c_t_2d.shape[1]
                    if flip:
                        c_t_2d = c_t_2d.T
                    c_t_2d = self.clip_func(
                        c_t_2d,
                        sigma_min=group["spectral_min"],
                        sigma_max=group["spectral_max"],
                        adaptive=group["spectral_adaptive"],
                        ortho_dtype=group["spectral_clip_dtype"],
                    )
                    if flip:
                        c_t_2d = c_t_2d.T
                    full_step = c_t_2d.view_as(c_t).atan2(denom).mul_(1.27323954474)
                else:
                    full_step = c_t.atan2(denom).mul_(1.27323954474)

                nesterov_direction = grad.sign().lerp_(momentum, weight=beta)
                full_step = full_step.mul(nesterov_direction)

                if weight_decay != 0:
                    full_step = full_step.add(param_fp32.data, alpha=weight_decay * weight_decay_rate**group["step"])
                param_fp32.data.add_(full_step, alpha=-lr)

                if param.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(param, param_fp32)
                else:
                    state["momentum"].copy_(momentum)
                    param.copy_(param_fp32)
        return loss
