# OCGOpt from https://github.com/Clybius/Personalized-Optimizers by Clybius

import logging
import math

import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import copy_stochastic_, filter_grad, resolve_state_storage_dtype


logger = logging.getLogger(__name__)

# Original Spectral Clipping code by leloykun (https://leloykun.github.io/ponder/spectral-clipping/ https://github.com/leloykun/spectral_clip)

"""
@misc{cesista2025spectralclipping,
  author = {Franz Louis Cesista},
  title = {"Fast, Numerically Stable, and Auto-Differentiable Spectral Clipping Via Newton-Schulz Iteration"},
  year = {2025},
  url = {http://leloykun.github.io/ponder/spectral-clipping/},
}
"""

"""
NS_COEFFS = [
    (3.5318, -4.7911, 1.9388),
    (3.3274, -4.0557, 1.5782),
    (3.0809, -3.5160, 1.3464),
    (2.7476, -2.8484, 1.0775),
    (2.2948, -2.0951, 0.7895),
    (2.1535, -1.8338, 0.6869),
]
"""

# New coeffs from https://kexue.fm/archives/11059, may enable later.
NS_COEFFS = [
    (8.287212018145622, -23.59588651909882, 17.300387312530923),
    (4.107059111542197, -2.9478499167379084, 0.54484310829266),
    (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
    (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
    (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
    (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
    (1.875, -1.25, 0.375),
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

    for a, b, c in NS_COEFFS:
        matrix = matrix / torch.linalg.norm(matrix).clamp_min_(1e-8)
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


class OCGOpt(Optimizer):
    r"""
    OCGOpt: Orthogonal Centralized Gradient Optimization.

    Separates momentum states into full gradient and centralized gradient for smoother and faster descent.
    Featuring orthogonalization, RMS normalization, cautious stepping, and dual-normed adaptive update magnitudes.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float, float):
            Coefficient used for computing the centralized momentum, full gradient momentum (used for centering), and the long-term squared running average (default: 0.95, 0.9999999, 0.9999999).
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step - Visualization: https://www.desmos.com/calculator/ipgbjovebr - (default: 0.995).
        centralization (float):
            Subtract the full gradient momentum from the current gradient at this ratio (default: 1.0).
        spectral_adaptive (bool):
            Adapt the result of spectral clipping to adapt to the scale of the gradients - https://github.com/leloykun/adaptive-muon (default: True).
        spectral_clip_compile (bool):
            Compile the spectral clip function (Highly recommended for a large speed increase) (default: True).
        spectral_clip_dtype (torch.dtype in string format):
            Sets the dtype of spectral clipping calculation. Recommended to use torch.float32 (or leave at default of None) (default: None, which results in torch.float32).
        adaptive (bool):
            Scale the full step to the momentumized average gradient, always utilizes RMS normalization on the gradient if True, otherwise caps RMS at 1.0 (default: True).
        adaptive_min (float):
            Minimum multiplier for the adaptive scale (default: -1.0).
        adaptive_max (float):
            Maximum multiplier for the adaptive scale (default: 1.0).
        input_norm (bool):
            Normalizes with RMS on the input feature dimensions instead of utilizing gradient-wise RMS normalization (default: False).
        lowpass_grad (float):
            Pre-conditions the gradient via a low-pass filter that maintains the direction of the gradient. Higher = stronger filtering, 0 = disabled (default: 0.0).
        sim_match (bool):
            Filters the frequencies of the running average with the gradient of the current step's frequencies (default: False).
        cautious_min (float):
            A value other than 1.0 will utilize cautious-stepping. At 0.0, this zeros out parts of the momentum which don't correlate with the current gradient's direction. 0.5 will halve it instead (default: 0.0).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
        kahan_summation (bool):
            Utilize Kahan Summation for the parameter update. This maintains a high-precision error buffer to effectively effectively double the precision of the accumulation step. 
            Excellent for bfloat16 training, compatible with stochastic rounding. (default: False).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float, float] = (0.95, 0.9999999, 0.9999999),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.995,
        centralization: float = 1.0,
        spectral_adaptive: bool = True,
        spectral_clip_compile: bool = True,
        spectral_clip_dtype=None,  # Can be set to torch.bfloat16, torch.float16, torch.float32, or even torch.float64 if you're insane in the membrane.
        adaptive: bool = True,
        adaptive_min: float = -1.0,
        adaptive_max: float = 1.0,
        input_norm: bool = False,
        lowpass_grad: float = 0.0,
        sim_match: bool = False,
        cautious_min: float = 0.0,
        stochastic_fp: bool = True,
        kahan_summation: bool = False,
        sync_chunk_size: int = 128,
        state_storage_dtype: str | torch.dtype = torch.bfloat16,
        state_storage_device: str | torch.device = "cpu",
        **kwargs,
    ):
        for key in kwargs:
            logger.warning("Unrecognized optimizer argument '%s'. It will be ignored.", key)

        final_dtype = resolve_state_storage_dtype(state_storage_dtype)

        self.sync_chunk_size = sync_chunk_size
        self.state_storage_dtype = final_dtype
        self.state_storage_device = state_storage_device
        self._init_lr = lr

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
            "spectral_adaptive": spectral_adaptive,
            "spectral_clip_compile": spectral_clip_compile,
            "spectral_clip_dtype": spectral_clip_dtype,
            "adaptive": adaptive,
            "adaptive_min": adaptive_min,
            "adaptive_max": adaptive_max,
            "input_norm": input_norm,
            "lowpass_grad": lowpass_grad,
            "sim_match": sim_match,
            "cautious_min": cautious_min,
            "stochastic_fp": stochastic_fp,
            "kahan_summation": kahan_summation,
            "sync_chunk_size": sync_chunk_size,
            "state_storage_dtype": final_dtype,
            "state_storage_device": state_storage_device,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def reset(self):
        pass

    @staticmethod
    def _get_compute_device(parameter_device: torch.device) -> torch.device:
        if parameter_device.type == "cpu" and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return parameter_device

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        cuda_available = torch.cuda.is_available()

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1

            lr = group["lr"]
            beta, beta2, beta3 = group["betas"]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            centralization = group["centralization"]
            stochastic_fp = group["stochastic_fp"]
            kahan_sum = group["kahan_summation"]

            step = group["step"]

            for i, param in enumerate(group["params"]):
                if param.grad is None:
                    continue

                state = self.state[param]
                grad = param.grad.data
                dimcount = grad.ndim
                use_kahan = kahan_sum and param.dtype in {torch.bfloat16, torch.float16}

                if len(state) == 0:
                    if dimcount < 1:
                        state["denom"] = torch.ones_like(
                            param.data,
                            dtype=self.state_storage_dtype,
                            device=self.state_storage_device,
                        )
                    state["value_momentum"] = torch.zeros_like(
                        param.data,
                        dtype=self.state_storage_dtype,
                        device=self.state_storage_device,
                    )
                    state["centralized_momentum"] = torch.zeros_like(
                        param.data,
                        dtype=self.state_storage_dtype,
                        device=self.state_storage_device,
                    )
                    if use_kahan:
                        kahan_dtype = torch.float32 if self.state_storage_device == "cpu" else self.state_storage_dtype
                        state["kahan_comp"] = torch.zeros_like(
                            param.data,
                            dtype=kahan_dtype,
                            device=self.state_storage_device,
                        )
                    if self.state_storage_device == "cpu" and cuda_available:
                        if dimcount < 1:
                            state["denom"] = state["denom"].pin_memory()
                        state["value_momentum"] = state["value_momentum"].pin_memory()
                        state["centralized_momentum"] = state["centralized_momentum"].pin_memory()
                        if use_kahan:
                            state["kahan_comp"] = state["kahan_comp"].pin_memory()

                if use_kahan and "kahan_comp" not in state:
                    state["kahan_comp"] = torch.zeros_like(
                        param.data,
                        dtype=torch.float32,
                        device=self.state_storage_device,
                    )
                    if self.state_storage_device == "cpu" and cuda_available:
                        state["kahan_comp"] = state["kahan_comp"].pin_memory()

                compute_device = self._get_compute_device(param.device)
                non_blocking = compute_device.type == "cuda"

                if dimcount < 1:
                    denom = state["denom"].to(compute_device, non_blocking=non_blocking, dtype=torch.float32)
                value_momentum = state["value_momentum"].to(compute_device, non_blocking=non_blocking, dtype=torch.float32)
                centralized_momentum = state["centralized_momentum"].to(
                    compute_device, non_blocking=non_blocking, dtype=torch.float32
                )
                grad = grad.to(torch.float32).to(compute_device, non_blocking=non_blocking)
                param_fp32 = param.to(compute_device, dtype=torch.float32, non_blocking=non_blocking)

                slow_beta2 = (beta2**step - beta2) / (beta2**step - 1.0)
                slow_beta3 = (beta3**step - beta3) / (beta3**step - 1.0)

                grad = grad.clamp(-step, step)

                if dimcount > 0 and group["lowpass_grad"] != 0:
                    grad = filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs().mul_(grad.sign())

                if dimcount >= 1 and group["input_norm"]:
                    if dimcount > 2:
                        grad_2d = grad.reshape(len(grad), -1)
                    elif dimcount < 2:
                        grad_2d = grad.reshape(1, -1)
                    else:
                        grad_2d = grad

                    rms = grad_2d.pow(2).mean(dim=1, keepdim=True).sqrt_().clamp_min_(1e-16)
                    grad = grad_2d.div(rms).view_as(grad)
                else:
                    rms = grad.pow(2).mean().sqrt_().clamp_min_(1e-16)
                    grad = grad.div(rms)

                if dimcount < 1:
                    current_denom = denom.sqrt()

                centralized_grad = grad.sub(value_momentum, alpha=centralization)
                centralized_momentum = centralized_momentum.lerp(centralized_grad, weight=1.0 - beta)
                value_momentum = value_momentum.lerp(grad, weight=1.0 - slow_beta2)

                exp_avg = centralized_grad.lerp(centralized_momentum, weight=beta).add_(
                    grad.lerp(value_momentum, weight=slow_beta2),
                    alpha=centralization,
                )

                if dimcount < 1:
                    denom = denom.lerp(centralized_grad.pow(2), weight=1.0 - slow_beta3)

                if dimcount > 0 and group["sim_match"]:
                    exp_avg = similarity_fft(exp_avg, grad)

                if dimcount >= 1:
                    if dimcount > 2:
                        exp_avg_2d = exp_avg.reshape(len(exp_avg), -1)
                    elif dimcount < 2:
                        exp_avg_2d = exp_avg.reshape(1, -1)
                    else:
                        exp_avg_2d = exp_avg

                    flip = exp_avg_2d.shape[0] < exp_avg_2d.shape[1]
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

                    full_step = exp_avg_2d.view_as(exp_avg)
                    full_step = full_step.div(full_step.pow(2).mean().sqrt_().clamp_min_(1.0))
                else:
                    full_step = exp_avg.atan2(current_denom).mul_(1.27323954474)

                scale_factor_mask = torch.where(
                    grad * full_step > 0,
                    torch.ones_like(full_step),
                    torch.ones_like(full_step) * group["cautious_min"],
                ).to(full_step.dtype)
                scale_factor_mask = scale_factor_mask.div(scale_factor_mask.mean().clamp_min_(1e-3))
                full_step = full_step.mul(scale_factor_mask)

                if group["adaptive"]:
                    if dimcount >= 1 and group["input_norm"]:
                        if dimcount > 2:
                            full_step_2d = full_step.reshape(len(full_step), -1)
                            exp_avg_2d = exp_avg.reshape(len(exp_avg), -1)
                        elif dimcount < 2:
                            full_step_2d = full_step.reshape(1, -1)
                            exp_avg_2d = exp_avg.reshape(1, -1)
                        else:
                            full_step_2d = full_step
                            exp_avg_2d = exp_avg

                        scale_factor = (exp_avg_2d * full_step_2d).sum(dim=1, keepdim=True).clamp(
                            group["adaptive_min"], group["adaptive_max"]
                        )

                        full_step = (full_step_2d * scale_factor).view_as(full_step)
                    else:
                        scale_factor = (exp_avg * full_step).sum().clamp(group["adaptive_min"], group["adaptive_max"])
                        full_step = scale_factor * full_step

                if weight_decay != 0:
                    full_step = full_step.add(param_fp32.data, alpha=weight_decay * weight_decay_rate**group["step"])

                update_step = full_step.mul(-lr)

                if use_kahan:
                    kahan_comp = state["kahan_comp"].to(compute_device, non_blocking=non_blocking, dtype=torch.float32)
                    update_step = update_step.sub(kahan_comp)
                    param_fp32.add_(update_step)

                    if stochastic_fp and param.dtype == torch.bfloat16:
                        param_simulated = param_fp32.clone()
                        noise = torch.randint_like(param_simulated, dtype=torch.int32, low=0, high=(1 << 16))
                        param_simulated_int = param_simulated.view(dtype=torch.int32)
                        param_simulated_int.add_(noise)
                        param_simulated_int.bitwise_and_(-65536)

                        new_comp = param_simulated.sub(param_fp32)

                        if param.device.type == "cpu":
                            param.data.copy_(param_simulated)
                        else:
                            param.data.copy_(param_simulated, non_blocking=True)
                    else:
                        param_rounded = param_fp32.to(param.dtype)
                        param_rounded_f32 = param_rounded.to(torch.float32)

                        new_comp = param_rounded_f32.sub(param_fp32)

                        if param.device.type == "cpu":
                            param.data.copy_(param_rounded)
                        else:
                            param.data.copy_(param_rounded, non_blocking=True)

                    if self.state_storage_dtype == torch.bfloat16:
                        copy_stochastic_(state["kahan_comp"], new_comp)
                    else:
                        state["kahan_comp"].copy_(new_comp, non_blocking=non_blocking)
                else:
                    param_fp32.data.add_(update_step)

                    if param.device.type == "cpu":
                        if param.dtype == torch.bfloat16 and stochastic_fp:
                            copy_stochastic_(param.data, param_fp32)
                        else:
                            param.data.copy_(param_fp32)
                    else:
                        if param.dtype == torch.bfloat16 and stochastic_fp:
                            copy_stochastic_(param, param_fp32)
                        else:
                            param.data.copy_(param_fp32, non_blocking=True)

                if self.state_storage_dtype == torch.bfloat16:
                    if dimcount < 1:
                        copy_stochastic_(state["denom"], denom)
                    copy_stochastic_(state["value_momentum"], value_momentum)
                    copy_stochastic_(state["centralized_momentum"], centralized_momentum)
                else:
                    if dimcount < 1:
                        state["denom"].copy_(denom, non_blocking=non_blocking)
                    state["value_momentum"].copy_(value_momentum, non_blocking=non_blocking)
                    state["centralized_momentum"].copy_(centralized_momentum, non_blocking=non_blocking)

                if compute_device.type == "cuda" and (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize(compute_device)

            compute_device = self._get_compute_device(group["params"][0].device)
            if compute_device.type == "cuda":
                torch.cuda.synchronize(compute_device)

        return loss
