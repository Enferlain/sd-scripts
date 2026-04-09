import logging
import math

import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import copy_stochastic_


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


def _resolve_state_storage_dtype(state_storage_dtype: str | torch.dtype) -> torch.dtype:
    if not isinstance(state_storage_dtype, str):
        return state_storage_dtype

    normalized_dtype = state_storage_dtype.strip().lower()
    if normalized_dtype == "float32":
        return torch.float32
    if normalized_dtype == "float16":
        return torch.float16
    if normalized_dtype == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16


@torch.no_grad()
def orthogonalize(matrix: torch.Tensor, ortho_dtype: torch.dtype | None = None, adaptive: bool = False) -> torch.Tensor:
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
    ortho_dtype: torch.dtype,
    adaptive: bool = False,
) -> torch.Tensor:
    return orthogonalize(matrix, ortho_dtype=ortho_dtype, adaptive=adaptive)


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def orthogonalize_compiled_func(
    matrix: torch.Tensor,
    *,
    ortho_dtype: torch.dtype,
    adaptive: bool = False,
) -> torch.Tensor:
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


def create_gaussian_mask(shape: tuple[int, ...], sigma: float = 1.0, device: str | torch.device = "cpu") -> torch.Tensor:
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


class OAGOpt(Optimizer):
    r"""
    OAGOpt: Orthogonal Adaptive Gradient Optimization.

    Two scalars, one momentum state, fully free descent. Featuring Muon's
    orthogonalization, RMS normalization, cautious stepping, and dual-normed
    adaptive update magnitudes.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float, float] = (0.95, 0.99, 0.999),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.995,
        spectral_adaptive: bool = True,
        spectral_clip_compile: bool = True,
        spectral_clip_dtype=None,
        adaptive: bool = True,
        adaptive_min: float = -1.0,
        adaptive_max: float = 1.0,
        input_norm: bool = True,
        lowpass_grad: float = 0.0,
        sim_match: bool = False,
        cautious_min: float = 0.0,
        sgd_nesterov: bool = True,
        stochastic_fp: bool = True,
        sync_chunk_size: int = 128,
        state_storage_dtype: str | torch.dtype = torch.bfloat16,
        state_storage_device: str | torch.device = "cpu",
        **kwargs,
    ):
        for key in kwargs:
            logger.warning("Unrecognized optimizer argument '%s'. It will be ignored.", key)

        final_dtype = _resolve_state_storage_dtype(state_storage_dtype)

        self.sync_chunk_size = sync_chunk_size
        self.state_storage_dtype = final_dtype
        self.state_storage_device = state_storage_device
        self._init_lr = lr

        if spectral_clip_dtype is None:
            spectral_clip_dtype = torch.float32
        elif isinstance(spectral_clip_dtype, str):
            spectral_clip_dtype = getattr(torch, spectral_clip_dtype.split(".")[-1])

        self.clip_func = orthogonalize_compiled_func if spectral_clip_compile else orthogonalize_func

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decay_rate": weight_decay_rate,
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
            "sgd_nesterov": sgd_nesterov,
            "stochastic_fp": stochastic_fp,
            "sync_chunk_size": sync_chunk_size,
            "state_storage_dtype": final_dtype,
            "state_storage_device": state_storage_device,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "OAGOpt"

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
            step = group["step"]

            for i, param in enumerate(group["params"]):
                if param.grad is None:
                    continue

                state = self.state[param]
                grad = param.grad.data
                dimcount = grad.ndim

                if len(state) == 0:
                    state["denom"] = torch.tensor(1.0, dtype=self.state_storage_dtype, device=self.state_storage_device)
                    state["ratio"] = torch.tensor(1.0, dtype=self.state_storage_dtype, device=self.state_storage_device)
                    state["value_momentum"] = torch.zeros_like(grad, dtype=self.state_storage_dtype, device=self.state_storage_device)
                    if self.state_storage_device == "cpu" and cuda_available:
                        state["denom"] = state["denom"].pin_memory()
                        state["ratio"] = state["ratio"].pin_memory()
                        state["value_momentum"] = state["value_momentum"].pin_memory()

                compute_device = self._get_compute_device(param.device)
                non_blocking = compute_device.type == "cuda"

                denom = state["denom"].to(compute_device, non_blocking=non_blocking, dtype=torch.float32)
                ratio = state["ratio"].to(compute_device, non_blocking=non_blocking, dtype=torch.float32)
                value_momentum = state["value_momentum"].to(compute_device, non_blocking=non_blocking, dtype=torch.float32)
                grad = grad.to(torch.float32).to(compute_device, non_blocking=non_blocking)
                param_fp32 = param.to(compute_device, dtype=torch.float32, non_blocking=non_blocking)

                slow_beta2 = (beta2**step - beta2) / (beta2**step - 1.0)
                slow_beta3 = (beta3**step - beta3) / (beta3**step - 1.0)

                grad = grad.clamp(-step, step)

                if dimcount > 0 and group["lowpass_grad"] != 0:
                    grad = filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs().mul_(grad.sign())

                if dimcount >= 1 and group["input_norm"]:
                    grad_2d = grad.reshape(len(grad), -1) if dimcount != 2 else grad
                    rms = grad_2d.pow(2).mean(dim=1, keepdim=True).sqrt_().clamp_min_(1e-16)
                    grad = grad_2d.div(rms).view_as(grad)
                else:
                    rms = grad.pow(2).mean().sqrt_().clamp_min_(1e-16)
                    grad = grad.div(rms)

                current_denom = denom.sqrt()

                if group["sgd_nesterov"]:
                    value_momentum = value_momentum.mul(beta).add_(grad)
                    exp_avg = value_momentum.mul(beta).add_(grad).mul(1.0 - beta)
                else:
                    value_momentum = value_momentum.lerp(grad, weight=1.0 - beta)
                    exp_avg = grad.lerp(value_momentum, weight=beta)

                if dimcount > 0 and group["sim_match"]:
                    exp_avg = similarity_fft(exp_avg, grad)

                if dimcount >= 1:
                    exp_avg_2d = exp_avg.reshape(len(exp_avg), -1) if dimcount != 2 else exp_avg
                    flip = exp_avg_2d.shape[0] < exp_avg_2d.shape[1]
                    if flip:
                        exp_avg_2d = exp_avg_2d.T
                    exp_avg_2d = self.clip_func(
                        exp_avg_2d,
                        ortho_dtype=group["spectral_clip_dtype"],
                        adaptive=group["spectral_adaptive"],
                    )
                    if flip:
                        exp_avg_2d = exp_avg_2d.T

                    full_step = exp_avg_2d.view_as(exp_avg)
                    denom = denom.lerp(full_step.pow(2).mean(), weight=1.0 - slow_beta2)
                    full_step = full_step.div(current_denom.clamp_min(1.0))
                else:
                    denom = denom.lerp(exp_avg.pow(2), weight=1.0 - slow_beta2)
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
                        full_step_2d = full_step.reshape(len(full_step), -1) if dimcount != 2 else full_step
                        exp_avg_2d = exp_avg.reshape(len(exp_avg), -1) if dimcount != 2 else exp_avg
                        scale_factor = (exp_avg_2d * full_step_2d).sum(dim=1, keepdim=True).clamp(
                            group["adaptive_min"], group["adaptive_max"]
                        )
                        full_step = (full_step_2d * scale_factor).view_as(full_step)
                    else:
                        scale_factor = (exp_avg * full_step).sum().clamp(group["adaptive_min"], group["adaptive_max"])
                        full_step = scale_factor * full_step

                if weight_decay != 0:
                    full_step = full_step.add(param_fp32, alpha=weight_decay * weight_decay_rate**group["step"])

                grad_norm = full_step.norm().clamp_min_(1e-16).div(param.numel())
                step_size = lr * (grad_norm.atan2(ratio.sqrt()).mul_(1.27323954474))
                ratio = ratio.lerp(grad_norm.pow(2), weight=1.0 - slow_beta3)

                param_fp32.add_(full_step, alpha=-step_size)

                if param.device.type == "cpu":
                    if param.dtype == torch.bfloat16 and group["stochastic_fp"]:
                        copy_stochastic_(param.data, param_fp32)
                    else:
                        param.data.copy_(param_fp32)
                else:
                    if param.dtype == torch.bfloat16 and group["stochastic_fp"]:
                        copy_stochastic_(param, param_fp32)
                    else:
                        param.data.copy_(param_fp32, non_blocking=True)

                if self.state_storage_dtype == torch.bfloat16 and group["stochastic_fp"]:
                    copy_stochastic_(state["denom"], denom)
                    copy_stochastic_(state["ratio"], ratio)
                    copy_stochastic_(state["value_momentum"], value_momentum)
                else:
                    state["denom"].copy_(denom, non_blocking=True)
                    state["ratio"].copy_(ratio, non_blocking=True)
                    state["value_momentum"].copy_(value_momentum, non_blocking=True)

                if compute_device.type == "cuda" and (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize(compute_device)

            if any(p.device.type == "cuda" for p in group["params"]) or (cuda_available and any(p.device.type == "cpu" for p in group["params"])):
                compute_device = self._get_compute_device(group["params"][0].device)
                if compute_device.type == "cuda":
                    torch.cuda.synchronize(compute_device)

        return loss
