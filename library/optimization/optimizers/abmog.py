# ABMOG from https://github.com/Clybius/Personalized-Optimizers by Clybius

import collections
import math

import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import copy_stochastic_, filter_grad
from library.utils.compile_env import can_use_compiled_cuda_helpers

# Original Spectral Clipping code by leloykun (https://leloykun.github.io/ponder/spectral-clipping/ https://github.com/leloykun/spectral_clip)

"""
@misc{cesista2025spectralclipping,
  author = {Franz Louis Cesista},
  title = {"Fast, Numerically Stable, and Auto-Differentiable Spectral Clipping Via Newton-Schulz Iteration"},
  year = {2025},
  url = {http://leloykun.github.io/ponder/spectral-clipping/},
}
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
    num_ns_steps: int = len(NS_COEFFS),
    ortho_dtype=None,
    adaptive: bool = False,
) -> torch.Tensor:
    """Orthogonalize a matrix via 5th order Newton-Schulz iteration."""
    if ortho_dtype is not None:
        orig_dtype = matrix.dtype
        matrix = matrix.to(ortho_dtype)
    if adaptive:
        matrix_orig = matrix.clone()
    transpose = matrix.shape[0] < matrix.shape[1]
    if transpose:
        matrix = matrix.T
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        matrix = matrix / (torch.linalg.norm(matrix).clamp_min_(1e-8))
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
def orthogonalize_func(
    weights: torch.Tensor,
    sigma_min: float = -1.0,
    sigma_max: float = 1.0,
    ortho_dtype=torch.float32,
    num_ns_steps: int = len(NS_COEFFS),
    adaptive: bool = False,
):
    del sigma_min, sigma_max
    return orthogonalize(weights, num_ns_steps=num_ns_steps, ortho_dtype=ortho_dtype, adaptive=adaptive)


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def orthogonalize_compiled_func(
    weights: torch.Tensor,
    sigma_min: float = -1.0,
    sigma_max: float = 1.0,
    ortho_dtype=torch.float32,
    num_ns_steps: int = len(NS_COEFFS),
    adaptive: bool = False,
):
    del sigma_min, sigma_max
    return orthogonalize(weights, num_ns_steps=num_ns_steps, ortho_dtype=ortho_dtype, adaptive=adaptive)

def create_gaussian_mask(shape, sigma: float = 1.0, device="cpu"):
    freq_dims = [torch.fft.fftfreq(size, device=device) for size in shape]
    shifted_freq_dims = [torch.fft.ifftshift(dim) for dim in freq_dims]
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing="ij"))
    max_radius = 0.5 * math.sqrt(len(shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    return torch.exp(-sigma * (radius**2))


def similarity_fft(grad, prev_grad, sigma: float = 0.0):
    grad_freq = torch.fft.fftn(grad, norm="ortho")
    prev_grad_freq = torch.fft.fftn(prev_grad, norm="ortho")
    grad_freq_shifted = torch.fft.fftshift(grad_freq)
    prev_grad_freq_shifted = torch.fft.fftshift(prev_grad_freq)
    agreement_mask = grad_freq_shifted.abs() * prev_grad_freq_shifted.abs().conj()
    mask_max = torch.max(agreement_mask.abs())
    if mask_max > 1e-16:
        agreement_mask /= mask_max
    new_grad_fft = grad_freq_shifted * agreement_mask.real
    if sigma != 0:
        gaussian_mask = create_gaussian_mask(grad.shape, sigma=sigma, device=grad.device)
        new_grad_fft = new_grad_fft * gaussian_mask
    new_grad_fft = torch.fft.ifftshift(new_grad_fft)
    return torch.fft.ifftn(new_grad_fft, norm="ortho").real


def reshape_to_2d(grad):
    if grad.ndim > 2:
        return grad.reshape(len(grad), -1)
    if grad.ndim < 2:
        return grad.reshape(1, -1)
    return grad


def _can_use_compiled_spectral_helpers() -> bool:
    return can_use_compiled_cuda_helpers()


class ABMOG(Optimizer):
    r"""
    ABMOG: Adams-Bashforth-Moulton Orthogonal Gradient

    A Muon-styled optimizer which incorporates an Adams-Bashforth predictor and Adams-Moulton corrector
    step to refine the gradient based on its history, accelerating convergence. Now includes bonus goodies (bcos, cautious, dual-norm gradient)

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float, float):
            Coefficient used for computing the Nesterov-styled momentum, the long-term squared mean running average, and the running average grad norm for the adaptive learning-rate ratio (default: 0.95, 0.99, 0.999).
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step - Visualization: https://www.desmos.com/calculator/ipgbjovebr - (default: 0.995).
        spectral_adaptive (bool):
            Adapt the result of spectral clipping to adapt to the scale of the gradients - https://github.com/leloykun/adaptive-muon (default: True).
        spectral_clip_compile (bool):
            Compile the spectral clip function (Highly recommended for a large speed increase) (default: True).
        spectral_clip_dtype (torch.dtype in string format):
            Sets the dtype of spectral clipping calculation. Recommended to use torch.float32 (or leave at default of None) (default: None, which results in torch.float32).
        adaptive (bool):
            Scale the full step to the momentumized average gradient (default: True).
        adaptive_min (float):
            Minimum multiplier for the adaptive scale (default: -1.0).
        adaptive_max (float):
            Maximum multiplier for the adaptive scale (default: 1.0).
        input_norm (bool):
            Normalizes with RMS on the input feature dimensions instead of utilizing gradient-wise RMS normalization (default: True).
        lowpass_grad (float):
            Pre-conditions the gradient via a low-pass filter that maintains the direction of the gradient. Higher = stronger filtering, 0 = disabled (default: 0.0).
        bcos (bool):
            Uses a conditional estimator from facebookresearch's bcos as the denominator - https://github.com/facebookresearch/bcos (default: True).
        cautious_min (float):
            A value other than 1.0 will utilize cautious-stepping. At 0.0, this zeros out parts of the momentum which don't correlate with the current gradient's direction. 0.5 will halve it instead (default: 0.0).
        sgd_nesterov (bool):
            Utilizes SGD-like Nesterov momentum instead of current-gradient-focused momentum (default: True).
        abm_order (int):
            Order of the Adams-Bashforth-Moulton method. Uses abm_order gradients. Set abm_order to 1 to disable ABM extrapolation. (default: 4).
        abm_k (int):
            Do an Adams-Bashforth-Moulton extrapolation every abm_k steps. (default: 5).
        abm_cpu_storage (bool):
            Store ABM gradient history on CPU to save VRAM. (default: True).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.95, 0.99),
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
        bcos: bool = True,
        cautious_min: float = 0.0,
        sgd_nesterov: bool = True,
        abm_order: int = 4,
        abm_k: int = 5,
        abm_cpu_storage: bool = True,
        stochastic_fp: bool = True,
        sync_chunk_size: int = 128,
        state_storage_dtype: str | torch.dtype = torch.bfloat16,
        state_storage_device: str | torch.device = "cpu",
    ):
        if isinstance(state_storage_dtype, str):
            normalized_str_dtype = state_storage_dtype.strip().lower()
            if normalized_str_dtype == "float32":
                final_dtype = torch.float32
            elif normalized_str_dtype == "float16":
                final_dtype = torch.float16
            elif normalized_str_dtype == "bfloat16":
                final_dtype = torch.bfloat16
            else:
                final_dtype = torch.bfloat16
        else:
            final_dtype = state_storage_dtype

        self.sync_chunk_size = sync_chunk_size
        self.state_storage_dtype = final_dtype
        self.state_storage_device = state_storage_device
        self._init_lr = lr
        use_compiled_helper = spectral_clip_compile and _can_use_compiled_spectral_helpers()
        self.clip_func = orthogonalize_compiled_func if use_compiled_helper else orthogonalize_func

        if spectral_clip_dtype is None:
            spectral_clip_dtype = torch.float32
        if isinstance(spectral_clip_dtype, str):
            dtype_name = spectral_clip_dtype.split(".")[-1]
            spectral_clip_dtype = getattr(torch, dtype_name)

        # Coefficients for Adams-Bashforth (Predictor)
        # k=1 to 9. History is [g_n, g_{n-1}, ...]
        self.ab_coeffs = {
            1: [1.0],
            2: [1.5, -0.5],
            3: [23 / 12, -16 / 12, 5 / 12],
            4: [55 / 24, -59 / 24, 37 / 24, -9 / 24],
            5: [1901 / 720, -2774 / 720, 2616 / 720, -1274 / 720, 251 / 720],
            6: [4277 / 1440, -7923 / 1440, 9982 / 1440, -7298 / 1440, 2877 / 1440, -475 / 1440],
            7: [198721 / 60480, -447288 / 60480, 705549 / 60480, -688256 / 60480, 407139 / 60480, -134472 / 60480, 19087 / 60480],
            8: [434241 / 120960, -1152169 / 120960, 2183877 / 120960, -2664477 / 120960, 2102243 / 120960, -1041723 / 120960, 295767 / 120960, -36799 / 120960],
            9: [14097241 / 3628800, -43448842 / 3628800, 98223681 / 3628800, -145788142 / 3628800, 143531169 / 3628800, -92956942 / 3628800, 38162241 / 3628800, -9124282 / 3628800, 959281 / 3628800],
            10: [29579241 / 7257600, -104829331 / 7257600, 276985582 / 7257600, -491429182 / 7257600, 608822461 / 7257600, -520448951 / 7257600, 296222582 / 7257600, -107198731 / 7257600, 22254361 / 7257600, -2043851 / 7257600],
        }

        # Coefficients for Adams-Moulton (Corrector)
        # k=1 to 9. History is [g_{n+1}_pred, g_n, g_{n-1}, ...]
        self.am_coeffs = {
            1: [1.0],  # Using predicted gradient only as corrector
            2: [0.5, 0.5],
            3: [5 / 12, 8 / 12, -1 / 12],
            4: [9 / 24, 19 / 24, -5 / 24, 1 / 24],
            5: [251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720],
            6: [475 / 1440, 1427 / 1440, -798 / 1440, 482 / 1440, -173 / 1440, 27 / 1440],
            7: [19087 / 60480, 65112 / 60480, -46461 / 60480, 37504 / 60480, -20211 / 60480, 6312 / 60480, -863 / 60480],
            8: [36799 / 120960, 139849 / 120960, -121797 / 120960, 123133 / 120960, -88547 / 120960, 41499 / 120960, -11351 / 120960, 1375 / 120960],
            9: [2043851 / 7257600, 8648118 / 7257600, -8526441 / 7257600, 10049438 / 7257600, -8507853 / 7257600, 4899438 / 7257600, -1818321 / 7257600, 392958 / 7257600, -37867 / 7257600],
            10: [37867 / 1451520, 196967 / 1451520, -214753 / 1451520, 290177 / 1451520, -289063 / 1451520, 199367 / 1451520, -89533 / 1451520, 24047 / 1451520, -2953 / 1451520],
        }

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
            "bcos": bcos,
            "cautious_min": cautious_min,
            "sgd_nesterov": sgd_nesterov,
            "stochastic_fp": stochastic_fp,
            "abm_order": abm_order,
            "abm_k": abm_k,
            "abm_cpu_storage": abm_cpu_storage,
            "sync_chunk_size": sync_chunk_size,
            "state_storage_dtype": final_dtype,
            "state_storage_device": state_storage_device,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "ABMOG"

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
            beta, beta2 = group["betas"][0], group["betas"][1]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            abm_order, abm_k = group["abm_order"], group["abm_k"]
            abm_cpu_storage = group["abm_cpu_storage"]
            step = group["step"]
            used_cuda = False

            for i, param in enumerate(group["params"]):
                if param.grad is None:
                    continue

                state = self.state[param]
                device = param.device
                grad = param.grad.data

                # State initialization
                if len(state) == 0:
                    if self.state_storage_device == "cpu":
                        if not group["bcos"]:
                            state["denom"] = torch.tensor(
                                1.0,
                                dtype=self.state_storage_dtype,
                                device=self.state_storage_device,
                            ).pin_memory()
                        state["value_momentum"] = torch.zeros_like(
                            param.data,
                            dtype=self.state_storage_dtype,
                            device=self.state_storage_device,
                        ).pin_memory()
                    else:
                        if not group["bcos"]:
                            state["denom"] = torch.tensor(
                                1.0,
                                dtype=self.state_storage_dtype,
                                device=self.state_storage_device,
                            )
                        state["value_momentum"] = torch.zeros_like(
                            param.data,
                            dtype=self.state_storage_dtype,
                            device=self.state_storage_device,
                        )
                    if abm_order > 1:
                        # Use a deque to efficiently manage fixed-size history
                        state["p_history"] = collections.deque(maxlen=abm_order)

                # ========= Asynchronously queue all operations for this parameter =========
                # Determine target GPU device for computation
                if device.type == "cpu":
                    compute_device = (
                        torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else device
                    )
                else:
                    compute_device = device
                used_cuda = used_cuda or compute_device.type == "cuda"

                # 1. Queue Host-to-Device copy
                if not group["bcos"]:
                    denom = state["denom"].to(compute_device, non_blocking=True, dtype=torch.float32)
                value_momentum = state["value_momentum"].to(compute_device, non_blocking=True, dtype=torch.float32)
                grad = grad.to(torch.float32).to(compute_device, non_blocking=True)
                param_fp32 = param.to(compute_device, dtype=torch.float32, non_blocking=True)

                # Fast-to-slow beta (0 @ step 1, 0.5 @ step 2, 0.6667... @ step 3, repeating to a max of beta2)
                slow_beta2 = (beta2**step - beta2) / (beta2**step - 1.0)
                
                # ADOPT-style clamp to prevent overshooting at the beginning
                grad = grad.clamp(-step, step)

                # Optional low-passing of gradient
                if grad.ndim > 0 and group["lowpass_grad"] != 0:
                    grad = filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs().mul_(grad.sign())

                # Normalize the gradient per-channel (input_norm=True + dim > 0) or per-tensor
                if grad.ndim >= 1 and group["input_norm"]:
                    grad_2d = reshape_to_2d(grad)
                    rms = grad_2d.pow(2).mean(dim=1, keepdim=True).sqrt_().clamp_min_(1e-16)
                    grad = grad_2d.div(rms).view_as(grad)
                else:
                    rms = grad.pow(2).mean().sqrt_().clamp_min_(1e-16)
                    grad = grad.div(rms)

                # SGD-Like Nesterov or Adam-like Nesterov
                if group["sgd_nesterov"]:
                    value_momentum = value_momentum.mul(beta).add_(grad)
                    exp_avg = value_momentum.mul(beta).add_(grad).mul(1.0 - beta)
                else:
                    value_momentum = value_momentum.lerp(grad, weight=1.0 - beta)
                    exp_avg = grad.lerp(value_momentum, weight=beta)

                if not group["bcos"]:
                    current_denom = denom.sqrt()

                # Muon-styled spectral norming, with scalar denominator
                if grad.ndim >= 1:
                    exp_avg_2d = reshape_to_2d(exp_avg)
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
                    if not group["bcos"]:
                        denom = denom.lerp(full_step.pow(2).mean(), weight=1.0 - slow_beta2)
                    else:
                        current_denom = (
                            (3 * beta**2 - 2 * beta**3) * full_step.square()
                            + (1 - beta) ** 2 * grad.detach().square()
                            + 2 * beta * (1 - beta) ** 2 * full_step * grad.detach()
                        ).mean().sqrt()
                    full_step = full_step.div(current_denom.clamp_min(1.0))
                else:
                    if not group["bcos"]:
                        denom = denom.lerp(exp_avg.pow(2), weight=1.0 - slow_beta2)
                    else:
                        current_denom = (
                            (3 * beta**2 - 2 * beta**3) * exp_avg.square()
                            + (1 - beta) ** 2 * grad.detach().square()
                            + 2 * beta * (1 - beta) ** 2 * exp_avg * grad.detach()
                        ).mean().sqrt()
                    full_step = exp_avg.atan2(current_denom).mul_(1.27323954474)

                scale_factor_mask = torch.where(
                    grad * full_step > 0,
                    torch.ones_like(full_step),
                    torch.ones_like(full_step) * group["cautious_min"],
                ).to(full_step.dtype)
                scale_factor_mask = scale_factor_mask.div(scale_factor_mask.mean().clamp_min_(1e-3))
                
                # Cautious masking
                full_step = full_step.mul(scale_factor_mask)

                # Dual-norm gradient
                if group["adaptive"]:
                    if grad.ndim >= 1 and group["input_norm"]:
                        if grad.ndim > 2:
                            full_step_2d = full_step.reshape(len(full_step), -1)
                            exp_avg_2d = exp_avg.reshape(len(exp_avg), -1)
                        elif grad.ndim < 2:
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
                        scale_factor = (exp_avg * full_step).sum().clamp(
                            group["adaptive_min"], group["adaptive_max"]
                        )
                        full_step = scale_factor * full_step

                if weight_decay != 0:
                    param_fp32 = param_fp32.mul(1 - lr * weight_decay * weight_decay_rate**group["step"])

                # Add step
                param_fp32.add_(full_step, alpha=-lr)

                if abm_order > 1 and step % abm_k == 0:
                    # Store history on CPU
                    storage_device = "cpu" if abm_cpu_storage else param_fp32.device

                    # Add current grad to history (left side is newest)
                    state["p_history"].appendleft(param_fp32.detach().to(storage_device))
                    history = list(state["p_history"])
                    current_k = len(history)
                    
                    # Wait for history buffer to fill at least once
                    if current_k > 1:
                        # Bring history to calculation device
                        history_compute = [hist.to(compute_device) for hist in history]

                        # Predictor (Adams-Bashforth)
                        ab_c = self.ab_coeffs[current_k]
                        p_pred = torch.zeros_like(history_compute[0])
                        for hist, coeff in zip(history_compute, ab_c, strict=False):
                            p_pred.add_(hist, alpha=coeff)

                        # Corrector (Adams-Moulton)
                        am_c = self.am_coeffs[current_k]

                        # Use predicted grad as proxy for g_{n+1}
                        corrector_hist = [p_pred, *history_compute[:-1]]

                        p_corrected = torch.zeros_like(history_compute[0])
                        for hist, coeff in zip(corrector_hist, am_c, strict=False):
                            p_corrected.add_(hist, alpha=coeff)

                        param_fp32.copy_(p_corrected)

                # 3. Queue Device-to-Host copy
                # only use stochastic rounding if using bf16
                if device.type == "cpu":
                    if param.dtype == torch.bfloat16:
                        copy_stochastic_(param.data, param_fp32)
                    else:
                        param.data.copy_(param_fp32.to(device))
                else:
                    # Original GPU path
                    if param.dtype == torch.bfloat16:
                        copy_stochastic_(param, param_fp32)
                    else:
                        param.data.copy_(param_fp32, non_blocking=True)

                if self.state_storage_dtype == torch.bfloat16:
                    if not group["bcos"]:
                        copy_stochastic_(state["denom"], denom)
                    copy_stochastic_(state["value_momentum"], value_momentum)
                else:
                    if not group["bcos"]:
                        state["denom"].copy_(denom.to(state["denom"].device), non_blocking=True)
                    state["value_momentum"].copy_(
                        value_momentum.to(state["value_momentum"].device),
                        non_blocking=True,
                    )

                # ========= Check if we need to synchronize =========
                # We synchronize after processing a chunk of parameters.
                # The (i + 1) ensures we sync after the 1st, 2nd, ... chunk.
                if used_cuda and (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization to handle the last partial chunk
            # This ensures all operations for the group are complete before exiting.
            if used_cuda:
                torch.cuda.synchronize()

        return loss
