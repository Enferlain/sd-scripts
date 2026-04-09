import collections
import math
import os
import shutil

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


@torch.no_grad()
def orthogonalize(
    matrix: torch.Tensor,
    num_ns_steps: int = len(NS_COEFFS),
    ortho_dtype=None,
    adaptive: bool = False,
) -> torch.Tensor:
    """Orthogonalize a matrix via Newton-Schulz iteration."""
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


def filter_grad(grad, fft_alpha: float = 1.0):
    grad_freq = torch.fft.fftn(grad, norm="ortho")
    freq_dims = [torch.fft.fftfreq(size, device=grad.device) for size in grad.shape]
    shifted_freq_dims = [torch.fft.ifftshift(dim) for dim in freq_dims]
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing="ij"))
    max_radius = 0.5 * math.sqrt(len(grad.shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    filter_weights = torch.exp(-fft_alpha * (radius**2))
    filtered_grad_freq = grad_freq * filter_weights
    modified_grad = torch.fft.ifftn(filtered_grad_freq, norm="ortho")
    return modified_grad.real


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
    nvcc_path = shutil.which("nvcc")
    return bool(torch.cuda.is_available() and nvcc_path and os.access(nvcc_path, os.X_OK))


class ABMOG(Optimizer):
    r"""
    ABMOG: Adams-Bashforth-Moulton Orthogonal Gradient.

    A Muon-styled optimizer which incorporates an Adams-Bashforth predictor and Adams-Moulton corrector
    step to refine the gradient based on its history, accelerating convergence.
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
        self.am_coeffs = {
            1: [1.0],
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
                        state["p_history"] = collections.deque(maxlen=abm_order)

                if device.type == "cpu":
                    compute_device = (
                        torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else device
                    )
                else:
                    compute_device = device
                used_cuda = used_cuda or compute_device.type == "cuda"

                if not group["bcos"]:
                    denom = state["denom"].to(compute_device, non_blocking=True, dtype=torch.float32)
                value_momentum = state["value_momentum"].to(compute_device, non_blocking=True, dtype=torch.float32)
                grad = grad.to(torch.float32).to(compute_device, non_blocking=True)
                param_fp32 = param.to(compute_device, dtype=torch.float32, non_blocking=True)

                slow_beta2 = (beta2**step - beta2) / (beta2**step - 1.0)
                grad = grad.clamp(-step, step)

                if grad.ndim > 0 and group["lowpass_grad"] != 0:
                    grad = filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs().mul_(grad.sign())

                if grad.ndim >= 1 and group["input_norm"]:
                    grad_2d = reshape_to_2d(grad)
                    rms = grad_2d.pow(2).mean(dim=1, keepdim=True).sqrt_().clamp_min_(1e-16)
                    grad = grad_2d.div(rms).view_as(grad)
                else:
                    rms = grad.pow(2).mean().sqrt_().clamp_min_(1e-16)
                    grad = grad.div(rms)

                if group["sgd_nesterov"]:
                    value_momentum = value_momentum.mul(beta).add_(grad)
                    exp_avg = value_momentum.mul(beta).add_(grad).mul(1.0 - beta)
                else:
                    value_momentum = value_momentum.lerp(grad, weight=1.0 - beta)
                    exp_avg = grad.lerp(value_momentum, weight=beta)

                if not group["bcos"]:
                    current_denom = denom.sqrt()

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
                full_step = full_step.mul(scale_factor_mask)

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

                param_fp32.add_(full_step, alpha=-lr)

                if abm_order > 1 and step % abm_k == 0:
                    storage_device = "cpu" if abm_cpu_storage else param_fp32.device
                    state["p_history"].appendleft(param_fp32.detach().to(storage_device))
                    history = list(state["p_history"])
                    current_k = len(history)
                    if current_k > 1:
                        history_compute = [hist.to(compute_device) for hist in history]
                        ab_c = self.ab_coeffs[current_k]
                        p_pred = torch.zeros_like(history_compute[0])
                        for hist, coeff in zip(history_compute, ab_c, strict=False):
                            p_pred.add_(hist, alpha=coeff)
                        am_c = self.am_coeffs[current_k]
                        corrector_hist = [p_pred, *history_compute[:-1]]
                        p_corrected = torch.zeros_like(history_compute[0])
                        for hist, coeff in zip(corrector_hist, am_c, strict=False):
                            p_corrected.add_(hist, alpha=coeff)
                        param_fp32.copy_(p_corrected)

                if device.type == "cpu":
                    if param.dtype == torch.bfloat16:
                        copy_stochastic_(param.data, param_fp32)
                    else:
                        param.data.copy_(param_fp32.to(device))
                else:
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

                if used_cuda and (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            if used_cuda:
                torch.cuda.synchronize()

        return loss
