import logging
import math

import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import copy_stochastic_


logger = logging.getLogger(__name__)


NS_COEFFS = [
    (3.5318, -4.7911, 1.9388),
    (3.3274, -4.0557, 1.5782),
    (3.0809, -3.5160, 1.3464),
    (2.7476, -2.8484, 1.0775),
    (2.2948, -2.0951, 0.7895),
    (2.1535, -1.8338, 0.6869),
]


@torch.no_grad()
def orthogonalize(
    matrix: torch.Tensor,
    *,
    ortho_dtype: torch.dtype | None = None,
    adaptive: bool = False,
) -> torch.Tensor:
    """Orthogonalize a matrix via 5th order Newton-Schulz iteration."""
    if ortho_dtype is not None:
        original_dtype = matrix.dtype
        matrix = matrix.to(ortho_dtype)
    else:
        original_dtype = None

    if adaptive:
        matrix_original = matrix.clone()
    else:
        matrix_original = None

    transpose = matrix.shape[0] < matrix.shape[1]
    if transpose:
        matrix = matrix.T

    matrix = matrix / (torch.linalg.norm(matrix) + 1e-20)
    for a, b, c in NS_COEFFS:
        gram = matrix.T @ matrix
        identity = torch.eye(gram.shape[0], dtype=matrix.dtype, device=matrix.device)
        matrix = matrix @ (a * identity + b * gram + c * gram @ gram)

    if transpose:
        matrix = matrix.T
    if adaptive and matrix_original is not None:
        matrix = torch.einsum("ij,ij,ab->ab", matrix_original.type_as(matrix), matrix, matrix)
    if original_dtype is not None:
        matrix = matrix.to(original_dtype)
    return matrix


def _spectral_clip(
    matrix: torch.Tensor,
    *,
    sigma_min: float = -1.0,
    sigma_max: float = 1.0,
    ortho_dtype: torch.dtype = torch.float32,
    adaptive: bool = False,
) -> torch.Tensor:
    if adaptive:
        matrix_original = matrix.clone()
    else:
        matrix_original = None

    original_dtype = matrix.dtype
    matrix = matrix.to(ortho_dtype)
    orthogonal = orthogonalize(matrix)
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    result = 0.5 * (
        (sigma_min + sigma_max) * identity
        + (sigma_min * orthogonal - matrix) @ orthogonalize(sigma_min * orthogonal - matrix).T
        - (sigma_max * orthogonal - matrix) @ orthogonalize(sigma_max * orthogonal - matrix).T
    ) @ orthogonal
    if adaptive and matrix_original is not None:
        result = torch.einsum("ij,ij,ab->ab", matrix_original.type_as(result), result, result)
    return result.to(original_dtype)


@torch.no_grad()
def spectral_clip_func(
    matrix: torch.Tensor,
    *,
    sigma_min: float = -1.0,
    sigma_max: float = 1.0,
    ortho_dtype: torch.dtype | None = None,
    adaptive: bool = False,
) -> torch.Tensor:
    if ortho_dtype is None:
        ortho_dtype = torch.float32
    return _spectral_clip(matrix, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, adaptive=adaptive)


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def spectral_clip_compiled_func(
    matrix: torch.Tensor,
    *,
    sigma_min: float = -1.0,
    sigma_max: float = 1.0,
    ortho_dtype: torch.dtype | None = None,
    adaptive: bool = False,
) -> torch.Tensor:
    if ortho_dtype is None:
        ortho_dtype = torch.float32
    return _spectral_clip(matrix, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, adaptive=adaptive)


@torch.no_grad()
def filter_grad(grad: torch.Tensor, fft_alpha: float = 1.0) -> torch.Tensor:
    grad_freq = torch.fft.fftn(grad, norm="ortho")
    freq_dims = [torch.fft.fftfreq(size, device=grad.device) for size in grad.shape]
    shifted_freq_dims = [torch.fft.ifftshift(freq) for freq in freq_dims]
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing="ij"))
    max_radius = 0.5 * math.sqrt(len(grad.shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    filter_weights = torch.exp(-fft_alpha * (radius**2))
    filtered_grad_freq = grad_freq * filter_weights
    modified_grad = torch.fft.ifftn(filtered_grad_freq, norm="ortho")
    return modified_grad.real


class FFTDescent(Optimizer):
    r"""
    FFTDescent: ***TEMPORARY NAME***

    Pre-conditions gradients with FFT low-pass filtering, tracks magnitude momentum, and optionally applies
    spectral clipping before the atan2-scaled update.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta: float = 0.95,
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.995,
        spectral_clip: bool = True,
        spectral_clip_compile: bool = True,
        spectral_clip_dtype=None,
        spectral_min: float = -1.0,
        spectral_max: float = 1.0,
        spectral_adaptive: bool = False,
        lowpass_grad: float = 1.0,
        sign_momentum: float = 0.9,
        stochastic_fp: bool = True,
        **kwargs,
    ):
        for key in kwargs:
            logger.warning("Optimizer argument '%s' passed into FFTDescent. It will be ignored.", key)

        self._init_lr = lr

        if spectral_clip:
            self.clip_func = spectral_clip_compiled_func if spectral_clip_compile else spectral_clip_func

        if isinstance(spectral_clip_dtype, str):
            spectral_clip_dtype = getattr(torch, spectral_clip_dtype.split(".")[-1])

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
            "sign_momentum": sign_momentum,
            "stochastic_fp": stochastic_fp,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "FFTDescent"

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
            step = group["step"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                grad = param.grad.data
                dimcount = grad.ndim

                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(grad)
                    if group["sign_momentum"] != 0:
                        state["sign_momentum"] = torch.zeros_like(grad)

                param_fp32 = param.detach().clone()
                momentum = state["momentum"].detach().clone()
                if group["sign_momentum"] != 0:
                    sign_momentum = state["sign_momentum"].detach().clone()

                if param.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    grad = grad.to(torch.float32)
                    momentum = state["momentum"].detach().clone().to(torch.float32)
                    if group["sign_momentum"] != 0:
                        sign_momentum = state["sign_momentum"].detach().clone().to(torch.float32)
                    param_fp32 = param.detach().clone().to(torch.float32)

                if dimcount > 0:
                    grad = filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs().mul_(grad.sign())

                if step == 1:
                    grad.clamp_(-0, 0)

                if group["sign_momentum"] != 0:
                    momentum = momentum.mul(beta).add_(grad.abs(), alpha=1.0 - beta)
                else:
                    momentum = momentum.mul(beta).add_(grad, alpha=1.0 - beta)

                if group["sign_momentum"] != 0:
                    sign_momentum = sign_momentum.mul(group["sign_momentum"]).add_(
                        grad.sign(),
                        alpha=1 - group["sign_momentum"],
                    )
                    current_tensor = grad.abs().lerp(momentum, weight=beta)
                else:
                    current_tensor = grad.lerp(momentum, weight=beta)

                if dimcount >= 2 and group["spectral_clip"]:
                    if dimcount > 2:
                        current_tensor_2d = current_tensor.reshape(len(current_tensor), -1)
                    else:
                        current_tensor_2d = current_tensor

                    flip = current_tensor_2d.shape[0] > current_tensor_2d.shape[1]
                    if flip:
                        current_tensor_2d = current_tensor_2d.T

                    full_step = self.clip_func(
                        current_tensor_2d,
                        sigma_min=group["spectral_min"],
                        sigma_max=group["spectral_max"],
                        adaptive=group["spectral_adaptive"],
                        ortho_dtype=group["spectral_clip_dtype"],
                    )

                    if flip:
                        full_step = full_step.T

                    full_step = full_step.view_as(current_tensor).atan2(momentum.abs()).mul_(1.27323954474)
                else:
                    full_step = current_tensor.atan2(momentum.abs()).mul_(1.27323954474)

                if group["sign_momentum"] != 0:
                    full_step = full_step.mul(sign_momentum)

                if weight_decay != 0:
                    full_step = full_step.add(param_fp32.data, alpha=weight_decay * weight_decay_rate**group["step"])

                param_fp32.data.add_(full_step, alpha=-lr)

                if param.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    copy_stochastic_(state["momentum"], momentum)
                    if group["sign_momentum"] != 0:
                        copy_stochastic_(state["sign_momentum"], sign_momentum)
                    copy_stochastic_(param, param_fp32)
                else:
                    state["momentum"].copy_(momentum)
                    if group["sign_momentum"] != 0:
                        state["sign_momentum"].copy_(sign_momentum)
                    param.copy_(param_fp32)

        return loss
