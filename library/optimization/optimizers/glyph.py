import math
from enum import IntEnum

import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import copy_stochastic_
from library.optimization.optimizers.utils.shampoo import zero_power_via_newton_schulz_5


class LMONorm(IntEnum):
    r"""Normalization types."""

    NONE = 0
    AUTO = 1
    SPECTRAL = 2
    SPECTRALCONV = 3
    SIGN = 4
    BIAS = 5
    COL = 6
    ROW = 7


class Norm:
    r"""Base class for Glyph's LMO normalization helpers."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return grad


class Col(Norm):
    r"""col-wise normalization.

    :param normalized: bool. normalize by the input dimension. use for non-input layers.
    :param transpose: bool. transpose input before normalization. use for embedding layers which have a shape of
        (vocab_size, embedding_dim)
    """
    
    def __init__(self, normalized: bool = False, transpose: bool = False) -> None:
        self.normalized = normalized
        self.transpose = transpose

    def init(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self.transpose:
            x = x.transpose(0, 1)

        torch.nn.init.normal_(x)
        x.div_(x.norm(dim=0, keepdim=True)).mul_(math.sqrt(x.size(0)))
        if self.normalized:
            x.div_(x.size(1))

        x = x.to(dtype=dtype)
        if self.transpose:
            x = x.transpose(0, 1)
        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if self.transpose:
            grad = grad.transpose(0, 1)

        d_in, d_out = grad.size()
        rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=0, keepdim=True)) / math.sqrt(d_in)
        if self.normalized:
            rms_value.mul_(d_out)

        grad /= rms_value.add_(eps)

        if self.transpose:
            grad = grad.transpose(0, 1)
        return grad


class Row(Norm):
    r"""row-wise normalization.

    :param normalized: bool. normalize by the input dimension. use for non-input layers.
    :param transpose: bool. transpose input before normalization. use for embedding layers which have a shape of
        (vocab_size, embedding_dim)
    """

    def __init__(self, normalized: bool = True, transpose: bool = False) -> None:
        self.normalized = normalized
        self.transpose = transpose

    def init(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self.transpose:
            x = x.transpose(0, 1)

        torch.nn.init.normal_(x)
        x.div_(x.norm(dim=-1, keepdim=True))
        if self.normalized:
            x.div_(math.sqrt(x.size(-1)))

        x = x.to(dtype=dtype)
        if self.transpose:
            x = x.transpose(0, 1)
        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        if self.transpose:
            grad = grad.transpose(0, 1)

        rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=-1, keepdim=True))
        if self.normalized:
            rms_value.mul_(math.sqrt(grad.size(-1)))

        grad /= rms_value.add_(eps)

        if self.transpose:
            grad = grad.transpose(0, 1)
        return grad


class BiasRMS(Norm):
    r"""bias RMS."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.zeros_(x)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        rms_value = torch.sqrt(torch.sum(grad.pow(2), dim=0, keepdim=True))
        grad /= rms_value.add_(eps)
        return grad


class SpectralConv(Norm):
    r"""spectral-convolution normalization.

    :param num_steps: int. number of steps of zero-power Newton-Schulz 5.
    """

    def __init__(self, num_steps: int = 5) -> None:
        self.num_steps = num_steps

    def init(self, x: torch.Tensor) -> torch.Tensor:
        x_fp64 = x.double()
        d_out, d_in, kernel_size, *_ = x_fp64.size()

        for i in range(kernel_size):
            for j in range(kernel_size):
                torch.nn.init.orthogonal_(x_fp64[..., i, j])

        x_fp64.mul_(math.sqrt(d_out / d_in) / (kernel_size**2))
        return x_fp64.to(dtype=x.dtype)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        del eps
        grad = zero_power_via_newton_schulz_5(grad.view(len(grad), -1), self.num_steps).view(grad.shape)
        d_out, d_in, kernel_size, *_ = grad.size()
        grad *= math.sqrt(d_out / d_in) / (kernel_size**2)
        return grad


class Spectral(Norm):
    r"""spectral normalization.

    :param max_scale: bool. set upper bound (1.0) of the scale.
    :param normalize: bool. normalize by the input dimension. use for non-input layers.
    :param num_steps: int. number of steps of zero-power Newton-Schulz 5.
    """

    def __init__(self, max_scale: bool = False, normalize: bool = True, num_steps: int = 5) -> None:
        self.max_scale = max_scale
        self.normalize = normalize
        self.num_steps = num_steps

    def init(self, x: torch.Tensor) -> torch.Tensor:
        x_fp64 = x.double()
        torch.nn.init.orthogonal_(x_fp64)

        d_out, d_in = x_fp64.size()
        scale = math.sqrt(d_out / d_in) if self.normalize else math.sqrt(d_out)
        if self.max_scale:
            scale = max(1.0, scale)

        x_fp64.mul_(scale)
        return x_fp64.to(dtype=x.dtype)

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        del eps
        grad = zero_power_via_newton_schulz_5(grad.view(len(grad), -1), self.num_steps).view(grad.shape)

        d_out, d_in = grad.size()
        scale = math.sqrt(d_out / d_in) if self.normalize else math.sqrt(d_out)
        if self.max_scale:
            scale = max(1.0, scale)

        grad *= scale
        return grad


class Sign(Norm):
    r"""sign normalization.

    :param zero_init: bool. initialize with zero.
    :param normalize: bool. normalize by the input dimension. use for non-input layers.
    """

    def __init__(self, zero_init: bool = False, normalize: bool = True) -> None:
        self.zero_init = zero_init
        self.normalize = normalize

    def init(self, x: torch.Tensor) -> torch.Tensor:
        if self.zero_init:
            return torch.nn.init.zeros_(x)

        d_in = x.size(1)
        x = 2 * torch.randint(0, 2, x.shape, dtype=x.dtype, device=x.device) - 1
        if self.normalize:
            x.div_(d_in)
        return x

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        del eps
        d_in = grad.size(1)
        return torch.sign(grad).div_(d_in) if self.normalize else torch.sign(grad)


class Auto(Norm):
    r"""choose Norm type automatically."""

    def init(self, x: torch.Tensor) -> torch.Tensor:
        ndim = x.ndim
        if ndim in (0, 1):
            return BiasRMS().init(x)
        if ndim == 2:
            return Spectral().init(x)
        if ndim in (3, 4):
            return SpectralConv().init(x)
        raise NotImplementedError

    def lmo(self, grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        ndim = grad.ndim
        if ndim in (0, 1):
            return BiasRMS().lmo(grad, eps=eps)
        if ndim == 2:
            return Spectral().lmo(grad, eps=eps)
        if ndim in (3, 4):
            return SpectralConv().lmo(grad, eps=eps)
        raise NotImplementedError


def build_lmo_norm(norm_type: int, **kwargs) -> Norm:  # noqa: PLR0911
    r"""Build LMONorm by given norm_type."""

    if norm_type == LMONorm.AUTO:
        return Auto()
    if norm_type == LMONorm.SPECTRAL:
        return Spectral(**kwargs)
    if norm_type == LMONorm.SPECTRALCONV:
        return SpectralConv(**kwargs)
    if norm_type == LMONorm.SIGN:
        return Sign(**kwargs)
    if norm_type == LMONorm.BIAS:
        return BiasRMS()
    if norm_type == LMONorm.COL:
        return Col(**kwargs)
    if norm_type == LMONorm.ROW:
        return Row(**kwargs)
    return Norm()


class Glyph(Optimizer):
    r"""
    Glyph: Cutting through noise via adaptation, normalization, and scale-invariance. 
    
    For optimal use: Utilize a gradient accumulation size of 1, highest batch size you can handle, adjust LR as needed (If reducing your total batch size, reduce your LR). May be prone to excessive updates with a higher LR.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float):
            Coefficient used for computing the running average, and the running square of running average (default: 0.95, 0.999999)
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step (default: 0.998).
        amp (float):
            Beta-adjusted scaling parameter for adding the running nesterov average to the gradient, functionally acts as strength value for a low-pass filter. (default: 1.0).
        orthograd (bool):
            Modify the gradient to apply an orthogonal gradient update, - https://arxiv.org/abs/2501.04697 (default: False).
        adaptive_ema (bool):
            Scale the EMA using a modified cautious mask (default: True).
        atan2 (bool):
            Divide the gradient using .atan2 instead of .div for stability and scale-invariance, removes epsilon/eps - https://arxiv.org/abs/2407.05872 (default: True).
        cautious_min (bool):
            Use cautious mask on full step update, clamped to a minimum of cautious_min - https://arxiv.org/abs/2411.16085 (default: 1.0, thus disabling the mask. Use 0 to fully utilize the mask).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.95, 1.0 - 1e-6),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.998,
        amp: float = 1.0,
        orthograd: bool = False,
        adaptive_ema: bool = True,
        atan2: bool = True,
        cautious_min: float = 1.0,
        stochastic_fp: bool = True,
    ):
        self._init_lr = lr

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decay_rate": weight_decay_rate,
            "amp": amp,
            "orthograd": orthograd,
            "adaptive_ema": adaptive_ema,
            "atan2": atan2,
            "cautious_min": cautious_min,
            "stochastic_fp": stochastic_fp,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "Glyph"

    # Implementation from: https://github.com/LoganBooker/prodigy-plus-schedule-free/blob/1d2cfa2fe692a828d46a5a29b9667ec924961ac7/prodigyplus/core_optimiser.py#L169C5-L177C48
    @torch.no_grad()
    def orthograd(self, param, grad):
        w = param.view(-1)
        g = grad.view(-1)

        proj = torch.dot(w, g).div(torch.dot(w, w).add_(1e-30))
        g_orth = g.to(dtype=torch.float32, copy=True).sub(w, alpha=proj)
        g_orth_scaled = g_orth.mul(g.norm(2).div_(g_orth.norm(2).clamp_(min=1e-30)))
        grad.copy_(g_orth_scaled.view_as(grad))

    @torch.no_grad()
    def smoothen_oscillation(self, grad, prev_grad_or_ema):
        return grad.arctan().sin_().div_(prev_grad_or_ema.arctan().cos_())

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def init(self):
        for group in self.param_groups:
            norm = build_lmo_norm(LMONorm.AUTO)
            for param in group["params"]:
                norm.init(param)

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
            norm = build_lmo_norm(LMONorm.AUTO)

            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                grad = param.grad.data

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["ema"] = torch.zeros_like(param.data)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.ones_like(param.data)
                    state["prev_grad"] = torch.zeros_like(grad)

                param_fp32 = param.detach().clone()
                ema = state["ema"].detach().clone()
                ema_squared = state["ema_squared"].detach().clone()
                prev_grad = state["prev_grad"].detach().clone()

                if param.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    grad = grad.to(torch.float32)
                    ema = state["ema"].detach().clone().to(torch.float32)
                    ema_squared = state["ema_squared"].detach().clone().to(torch.float32)
                    prev_grad = state["prev_grad"].detach().clone().to(torch.float32)
                    param_fp32 = param.detach().clone().to(torch.float32)

                slow_beta = (betas[1] ** step - betas[1]) / (betas[1] ** step - 1.0)
                bias_correction = 1 - betas[0] ** step
                step_size = lr * bias_correction

                if group["orthograd"] and param_fp32.data.nelement() > 1:
                    self.orthograd(param_fp32, grad)

                # Stabilize gradient oscillations via weird but cool math that I don't have a name for
                grad = self.smoothen_oscillation(grad, prev_grad)
                
                # MARS
                correction = (((1.0 - betas[0]) / 2) * betas[0]) / (1 - betas[0]) * (grad - prev_grad)
                c_t = grad + correction
                
                # SCION spectral norm
                c_t = norm.lmo(c_t, eps=1e-30)
                
                # Parameter-based amplification
                c_t = self.smoothen_oscillation(c_t, param_fp32.data)

                # Update ema
                ema = ema.mul(betas[0]).add_(c_t)

                # Adaptive ema
                if group["adaptive_ema"]:
                    mask = (c_t * ema > 0).to(c_t.dtype)
                    mask.clamp_min_(betas[0])
                    mask.div_(mask.mean().clamp_(min=1e-3)) # Divide by mean (0.001-1.0)
                    ema = ema.mul(mask)

                # Compass amplification (functionally/practically a low-pass filter when used with a denom)
                update = c_t.add(ema, alpha=group["amp"] * betas[0])
                denom = ema_squared.sqrt() if group["atan2"] else torch.clamp(ema_squared.sqrt(), 1e-16)

                # AMSGrad with decay (to prevent little learning later on during training)
                ema_squared_new = ema_squared.mul(slow_beta).addcmul_(update, update, value=1 - slow_beta)
                ema_squared = torch.maximum(ema_squared.mul(slow_beta), ema_squared_new)

                # ADOPT update (update squared EMA after creation of denominator)
                if group["atan2"]:
                    full_step = update.atan2(denom).mul_(1.27323954474) # Multiply by reciprocal of atan2(1,1)
                else:
                    clip_lambda = step**0.25
                    full_step = update.div(denom).clamp_(-clip_lambda, clip_lambda)  # Ensure updates aren't obscenely large for the first few steps, there may be a better way...

                if weight_decay != 0:
                    # Perform weight decay
                    full_step = full_step.add(param_fp32.data, alpha=weight_decay * weight_decay_rate**step)

                # Apply caution as per 'Cautious Optimizers' with a modified minimum.
                if group["cautious_min"] != 1.0:
                    mask = (full_step * grad > 0).to(full_step.dtype)
                    mask.clamp_min_(group["cautious_min"])
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    full_step = full_step.mul(mask)

                param_fp32.data.add_(full_step, alpha=-step_size)

                if param.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    copy_stochastic_(state["ema"], ema)
                    copy_stochastic_(state["ema_squared"], ema_squared)
                    copy_stochastic_(state["prev_grad"], grad)
                    copy_stochastic_(param, param_fp32)
                else:
                    state["ema"].copy_(ema)
                    state["ema_squared"].copy_(ema_squared)
                    state["prev_grad"].copy_(grad)
                    param.copy_(param_fp32)

        return loss
