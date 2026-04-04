import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import (
    bias_rms,
    bias_rms_compile,
    copy_stochastic_,
    zero_power_via_newton_schulz_6,
    zero_power_via_newton_schulz_6_compile,
)


class Dehaze(Optimizer):
    r"""
    Dehaze: Cutting through noise via adaptation, normalization, and scale-invariance.

    For optimal use: Utilize a gradient accumulation size of 1, highest batch size you can handle, adjust LR as needed
    (If reducing your total batch size, reduce your LR). May be prone to excessive updates with a higher LR.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float, float):
            Coefficient used for computing the sign momentum, stage1 (short-term) squared running average, and the
            stage2 (long-term) squared running average (default: 0.95, 0.98, 0.9999999)
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step
            (default: 0.995).
        stage1_atan2 (bool):
            Divide the gradient using .atan2 instead of .div for stability and scale-invariance, removes epsilon/eps -
            https://arxiv.org/abs/2407.05872 (default: True).
        stage2_atan2 (bool):
            Divide the smooth gradient using .atan2 instead of .div for further stability and scale-invariance,
            removes epsilon/eps - https://arxiv.org/abs/2407.05872 (default: True).
        adaptive_muon (bool):
            Utilize six optimized Newton-Schulz iterations per step to compute the orthogonalization of the gradient,
            and adapt to the gradient norm - https://arxiv.org/abs/2410.21265 - https://github.com/leloykun/adaptive-muon
            (default: False).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.95, 0.9, 0.9999999),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.995,
        stage1_atan2: bool = True,
        stage2_atan2: bool = True,
        adaptive_muon: bool = False,
        stochastic_fp: bool = True,
        torch_compile: bool = True,
        **kwargs,
    ):
        self._init_lr = lr

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            weight_decay_rate=weight_decay_rate,
            stage1_atan2=stage1_atan2,
            stage2_atan2=stage2_atan2,
            adaptive_muon=adaptive_muon,
            stochastic_fp=stochastic_fp,
            torch_compile=torch_compile,
            **kwargs,
        )

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
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            step = group["step"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                grad = p.grad.data

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["stage1_emasq"] = torch.ones_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["stage2_emasq"] = torch.ones_like(p.data)
                    state["sign_momentum"] = torch.zeros_like(grad)

                # Detach
                p_fp32 = p.detach().clone()
                stage1_emasq = state["stage1_emasq"].detach().clone()
                stage2_emasq = state["stage2_emasq"].detach().clone()
                sign_momentum = state["sign_momentum"].detach().clone()

                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    grad = grad.to(torch.float32)
                    stage1_emasq = state["stage1_emasq"].detach().clone().to(torch.float32)
                    stage2_emasq = state["stage2_emasq"].detach().clone().to(torch.float32)
                    sign_momentum = state["sign_momentum"].detach().clone().to(torch.float32)
                    p_fp32 = p.detach().clone().to(torch.float32)

                # Create betas
                slow_beta1 = (betas[1] ** (step) - betas[1]) / (betas[1] ** (step) - 1.0)
                slow_beta2 = (betas[2] ** (step) - betas[2]) / (betas[2] ** (step) - 1.0)

                clip_lambda = step**0.25

                # Update sign momentum
                sign_momentum = sign_momentum.lerp(grad.sign(), weight=1.0 - betas[0])

                # Adaptive Muon / Newton Schulz iters
                if group["adaptive_muon"]:
                    grad_normed = grad
                    if grad.ndim > 0:
                        if group["torch_compile"]:
                            grad_normed = zero_power_via_newton_schulz_6_compile(grad_normed)
                        else:
                            grad_normed = zero_power_via_newton_schulz_6(grad_normed)
                    elif grad_normed.numel() > 1:
                        if group["torch_compile"]:
                            grad_normed = bias_rms_compile(grad_normed)
                        else:
                            grad_normed = bias_rms(grad_normed)
                    grad = grad_normed

                # Denom (Stage 1)
                if group["stage1_atan2"]:
                    c_t = grad.atan2(stage1_emasq.sqrt()).mul_(1.27323954474)
                else:
                    stage1_denom = torch.clamp(stage1_emasq.sqrt(), 1e-16)
                    c_t = grad.div(stage1_denom).clamp_(-clip_lambda, clip_lambda)

                # ADOPT-style update squared momentum (Stage 1)
                grad = torch.where(
                    grad.abs() > 255,
                    grad.mul(255 / grad.abs()),
                    grad,
                )
                stage1_emasq = stage1_emasq.mul(slow_beta1).addcmul_(grad, grad, value=1 - slow_beta1)

                # Denom (Stage 2)
                if group["stage2_atan2"]:
                    full_step = c_t.atan2(stage2_emasq.sqrt()).mul_(1.27323954474)
                else:
                    stage2_denom = torch.clamp(stage2_emasq.sqrt(), 1e-16)
                    full_step = c_t.div(stage2_denom).clamp_(-clip_lambda, clip_lambda)

                # ADOPT-style update squared momentum (Stage 2)
                stage2_emasq = stage2_emasq.mul(slow_beta2).addcmul_(c_t, c_t, value=1 - slow_beta2)

                # Apply sign momentum to the gradient
                full_step = full_step.abs().mul_(sign_momentum)

                # Perform weight decay
                if weight_decay != 0:
                    grad_weights = p_fp32.data

                    full_step = full_step.add(grad_weights, alpha=weight_decay * weight_decay_rate ** group["step"])

                p_fp32.data.add_(full_step, alpha=-lr / (1.0 - betas[0] ** step))
                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    copy_stochastic_(state["stage1_emasq"], stage1_emasq)
                    copy_stochastic_(state["stage2_emasq"], stage2_emasq)
                    copy_stochastic_(state["sign_momentum"], sign_momentum)
                    copy_stochastic_(p, p_fp32)
                else:
                    state["stage1_emasq"].copy_(stage1_emasq)
                    state["stage2_emasq"].copy_(stage2_emasq)
                    state["sign_momentum"].copy_(sign_momentum)
                    p.copy_(p_fp32)
        return loss
