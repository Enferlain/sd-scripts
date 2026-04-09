import math

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
        matrix = (matrix_original * matrix).sum() * matrix
    if original_dtype is not None:
        matrix = matrix.to(original_dtype)
    return matrix


@torch.no_grad()
def orthogonalize_func(
    matrix: torch.Tensor,
    *,
    sigma_min: float = -1.0,
    sigma_max: float = 1.0,
    ortho_dtype: torch.dtype | None = torch.float32,
    adaptive: bool = False,
) -> torch.Tensor:
    del sigma_min, sigma_max
    return orthogonalize(matrix, ortho_dtype=ortho_dtype, adaptive=adaptive)


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def orthogonalize_compiled_func(
    matrix: torch.Tensor,
    *,
    sigma_min: float = -1.0,
    sigma_max: float = 1.0,
    ortho_dtype: torch.dtype | None = torch.float32,
    adaptive: bool = False,
) -> torch.Tensor:
    del sigma_min, sigma_max
    return orthogonalize(matrix, ortho_dtype=ortho_dtype, adaptive=adaptive)


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


def create_gaussian_mask(
    shape: tuple[int, ...],
    sigma: float = 1.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    freq_dims = [torch.fft.fftfreq(size, device=device) for size in shape]
    shifted_freq_dims = [torch.fft.ifftshift(freq) for freq in freq_dims]
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing="ij"))
    max_radius = 0.5 * math.sqrt(len(shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    filter_weights = torch.exp(-sigma * (radius**2))
    return filter_weights


def similarity_fft(grad: torch.Tensor, prev_grad: torch.Tensor, sigma: float = 0.0) -> torch.Tensor:
    grad_freq = torch.fft.fftn(grad, norm="ortho")
    prev_grad_freq = torch.fft.fftn(prev_grad, norm="ortho")
    grad_shifted = torch.fft.fftshift(grad_freq)
    prev_shifted = torch.fft.fftshift(prev_grad_freq)
    agreement_mask = grad_shifted.abs() * prev_shifted.abs().conj()
    mask_max = torch.max(agreement_mask.abs())
    if mask_max > 1e-16:
        agreement_mask = agreement_mask / mask_max
    new_grad_fft = grad_shifted * agreement_mask.real
    if sigma != 0:
        new_grad_fft = new_grad_fft * create_gaussian_mask(grad.shape, sigma=sigma, device=grad.device)
    new_grad_fft = torch.fft.ifftshift(new_grad_fft)
    new_grad = torch.fft.ifftn(new_grad_fft, norm="ortho").real
    return new_grad


class SCGOpt(Optimizer):
    r"""
    SCGOpt: Sign-based Centralized Gradient Optimization.

    Separates momentum states into full gradient and centralized gradient for smoother and faster descent, with a few
    extra features for boosting and stabilizing descent.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float, float] = (0.95, 0.9999999, 0.9999999),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.998,
        centralization: float = 1.0,
        spectral_clip: bool = False,
        spectral_adaptive: bool = True,
        spectral_clip_compile: bool = True,
        spectral_clip_dtype=None,
        adaptive: bool = True,
        adaptive_min: float = -1.0,
        adaptive_max: float = 1.0,
        use_sign: bool = True,
        lowpass_grad: float = 0.0,
        sim_match: bool = False,
        cautious_min: float = 0.0,
        stochastic_fp: bool = True,
    ):
        self._init_lr = lr

        if spectral_clip:
            self.clip_func = orthogonalize_compiled_func if spectral_clip_compile else orthogonalize_func

        if spectral_clip_dtype is None:
            spectral_clip_dtype = torch.float32
        elif isinstance(spectral_clip_dtype, str):
            spectral_clip_dtype = getattr(torch, spectral_clip_dtype.split(".")[-1])

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decay_rate": weight_decay_rate,
            "centralization": centralization,
            "spectral_clip": spectral_clip,
            "spectral_adaptive": spectral_adaptive,
            "spectral_clip_compile": spectral_clip_compile,
            "spectral_clip_dtype": spectral_clip_dtype,
            "adaptive": adaptive,
            "adaptive_min": adaptive_min,
            "adaptive_max": adaptive_max,
            "use_sign": use_sign,
            "lowpass_grad": lowpass_grad,
            "sim_match": sim_match,
            "cautious_min": cautious_min,
            "stochastic_fp": stochastic_fp,
        }

        super().__init__(params, defaults)

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
            beta, beta2, beta3 = group["betas"]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            centralization = group["centralization"]

            step = group["step"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                grad = param.grad.data
                dimcount = grad.ndim

                if len(state) == 0:
                    state["denom"] = torch.ones_like(grad).mean() if group["use_sign"] else torch.ones_like(grad)
                    state["value_momentum"] = torch.zeros_like(grad)
                    state["centralized_momentum"] = torch.zeros_like(grad)

                param_fp32 = param.detach().clone()
                denom = state["denom"].detach().clone()
                value_momentum = state["value_momentum"].detach().clone()
                centralized_momentum = state["centralized_momentum"].detach().clone()

                if param.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    grad = grad.to(torch.float32)
                    denom = state["denom"].detach().clone().to(torch.float32)
                    value_momentum = state["value_momentum"].detach().clone().to(torch.float32)
                    centralized_momentum = state["centralized_momentum"].detach().clone().to(torch.float32)
                    param_fp32 = param.detach().clone().to(torch.float32)

                slow_beta2 = (beta2**step - beta2) / (beta2**step - 1.0)
                slow_beta3 = (beta3**step - beta3) / (beta3**step - 1.0)

                grad = grad.clamp(-step, step)

                if group["use_sign"]:
                    grad = grad.sign()
                else:
                    if dimcount > 0 and group["lowpass_grad"] != 0:
                        grad = filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs().mul_(grad.sign())

                    if group["adaptive"]:
                        rms = grad.pow(2).mean().sqrt_().clamp_min_(1e-16)
                    else:
                        rms = grad.pow(2).mean().sqrt_().clamp_min_(1.0)
                    grad = grad.div(rms)

                current_denom = denom.sqrt()

                centralized_grad = grad.sub(value_momentum, alpha=centralization)
                centralized_momentum = centralized_momentum.lerp(centralized_grad, weight=1.0 - beta)
                value_momentum = value_momentum.lerp(grad, weight=1.0 - slow_beta2)
                exp_avg = centralized_grad.lerp(centralized_momentum, weight=beta).add_(
                    grad.lerp(value_momentum, weight=slow_beta2),
                    alpha=centralization,
                )

                denom = denom.lerp(
                    centralized_grad.pow(2).mean() if group["use_sign"] else centralized_grad.pow(2),
                    weight=1.0 - slow_beta3,
                )

                if dimcount > 0 and group["sim_match"] and not group["use_sign"]:
                    exp_avg = similarity_fft(exp_avg, grad)

                if dimcount >= 1 and group["spectral_clip"]:
                    if dimcount > 2:
                        exp_avg_2d = exp_avg.reshape(len(exp_avg), -1)
                    elif dimcount < 2:
                        exp_avg_2d = exp_avg.reshape(1, -1)
                    else:
                        exp_avg_2d = exp_avg

                    flip = exp_avg_2d.shape[0] > exp_avg_2d.shape[1]
                    if flip:
                        exp_avg_2d = exp_avg_2d.T

                    exp_avg_2d = self.clip_func(
                        exp_avg_2d,
                        sigma_min=0.0,
                        sigma_max=0.0,
                        adaptive=group["spectral_adaptive"],
                        ortho_dtype=group["spectral_clip_dtype"],
                    )

                    if flip:
                        exp_avg_2d = exp_avg_2d.T

                    exp_avg = exp_avg_2d.view_as(exp_avg)

                scale_factor_mask = torch.where(
                    grad * exp_avg > 0,
                    torch.ones_like(exp_avg),
                    torch.ones_like(exp_avg) * group["cautious_min"],
                ).to(exp_avg.dtype)
                scale_factor_mask = scale_factor_mask.div(scale_factor_mask.mean().clamp_min_(1e-3))

                full_step = exp_avg.mul(scale_factor_mask).atan2(current_denom).mul_(1.27323954474)
                if group["adaptive"] and dimcount > 0:
                    scale_factor = (exp_avg * full_step).sum().clamp(group["adaptive_min"], group["adaptive_max"])
                    full_step = scale_factor * full_step

                if weight_decay != 0:
                    full_step = full_step.add(param_fp32.data, alpha=weight_decay * weight_decay_rate**group["step"])

                param_fp32.data.add_(full_step, alpha=-lr)

                if param.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    copy_stochastic_(state["denom"], denom)
                    copy_stochastic_(state["value_momentum"], value_momentum)
                    copy_stochastic_(state["centralized_momentum"], centralized_momentum)
                    copy_stochastic_(param, param_fp32)
                else:
                    state["denom"].copy_(denom)
                    state["value_momentum"].copy_(value_momentum)
                    state["centralized_momentum"].copy_(centralized_momentum)
                    param.copy_(param_fp32)

        return loss
