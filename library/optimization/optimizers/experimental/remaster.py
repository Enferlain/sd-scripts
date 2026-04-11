# REMASTER from https://github.com/Clybius/Personalized-Optimizers by Clybius

import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import copy_stochastic_


class REMASTER(Optimizer):
    r"""
    REMASTER: Applying the idea of no gradient accumulation, as its been
    superseded by momentum. Faster training, smoother weights, Papa Johns.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float):
            Coefficient used for computing the running average, and the
            running square of running average (default: 0.95, 0.9999)
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied,
            weight_decay * weight_decay_rate**step (default: 0.998).
        amp (float):
            Beta-adjusted scaling parameter for adding the running average to
            the gradient. (default: 5.0).
        reset_interval (int):
            Resets the optimizers running averages after
            (reset_interval + reset_increment * times_reset) steps.
        reset_increment (int):
            Increments the reset_interval by this amount after every reset.
        orthograd (bool):
            Modify the gradient to apply an orthogonal gradient update.
        cautious_min (bool):
            Use cautious mask on full step update, clamped to a minimum of
            cautious_min.
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.95, 0.9999),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.998,
        amp: float = 5.0,
        reset_interval: int = 0,
        reset_increment: int = 0,
        orthograd: bool = True,
        cautious_min: float = 1.0,
        stochastic_fp: bool = True,
        **kwargs,
    ):
        del kwargs

        self._init_lr = lr
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decay_rate": weight_decay_rate,
            "amp": amp,
            "reset_interval": reset_interval,
            "reset_increment": reset_increment,
            "orthograd": orthograd,
            "cautious_min": cautious_min,
            "stochastic_fp": stochastic_fp,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "REMASTER"

    @torch.no_grad()
    def orthograd(self, parameter):
        # Donor atan2-style orthograd projection.
        weight = parameter.view(-1)
        grad = parameter.grad.view(-1)

        proj = torch.dot(weight, grad).atan2_(torch.dot(weight, weight)).mul_(1.27323954474)
        grad_orth = grad.to(dtype=torch.float32, copy=True).sub_(weight, alpha=proj)
        grad_orth_scaled = grad_orth.mul_(grad.norm(2).div_(grad_orth.norm(2).clamp_(min=1e-3)))
        parameter.grad.copy_(grad_orth_scaled.view_as(parameter.grad))

    @torch.no_grad()
    def reset_momentums(self, momentum, sq_momentum):
        momentum.copy_(torch.zeros_like(momentum))
        sq_momentum.copy_(torch.zeros_like(sq_momentum))

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
            orthograd = group["orthograd"]
            step = group["step"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]

                if orthograd and parameter.ndim >= 2:
                    self.orthograd(parameter)

                grad = parameter.grad.data

                # State initialization.
                if len(state) == 0:
                    state["ema"] = torch.zeros_like(parameter.data)
                    state["ema_squared"] = torch.zeros_like(parameter.data)
                    if group["reset_interval"] > 0:
                        state["times_zero"] = 0
                        state["steps_since_reset"] = 1

                parameter_fp32 = parameter.detach().clone()
                ema = state["ema"].detach().clone()
                ema_squared = state["ema_squared"].detach().clone()

                # Unpack.
                if parameter.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    grad = grad.to(torch.float32)
                    ema = state["ema"].detach().clone().to(torch.float32)
                    ema_squared = state["ema_squared"].detach().clone().to(torch.float32)
                    parameter_fp32 = parameter.detach().clone().to(torch.float32)

                if group["reset_interval"] > 0:
                    reset_period = group["reset_interval"] + (group["reset_increment"] * state["times_zero"])
                    if state["steps_since_reset"] // reset_period > 0:
                        self.reset_momentums(ema, ema_squared)
                        state["times_zero"] += 1
                        state["steps_since_reset"] = 1
                    step = state["steps_since_reset"]

                slow_beta = (betas[1] ** step - betas[1]) / (betas[1] ** step - 1.0)

                # Can apply to step_size, but this leads to significant initial
                # updates and could be too much without warmup.
                bias_correction = 1 - betas[0] ** step
                bias_correction_sqrt = (1 - slow_beta**step) ** 0.5
                atan2_mul = 1.27323954474
                step_size = lr * atan2_mul

                # RMS Norm.
                rms = grad.pow(2).mean().sqrt_().clamp_min_(1)
                grad.div_(rms)

                # Smooth EMA norm.
                grad_norm, ema_norm = grad.norm(2), ema.norm(2)
                normalization_val = grad_norm.atan2(ema_norm).mul_(atan2_mul)
                if normalization_val > 1e-6:
                    grad.div_(normalization_val)

                # Update ema.
                ema = ema.mul(betas[0]).add_(grad, alpha=1 - betas[0])

                # Adaptive ema.
                mask = (grad * ema > 0).to(grad.dtype)
                mask.clamp_min_(betas[0])
                mask.div_(mask.mean().clamp_(min=1e-3))
                ema = ema.mul(mask)

                # Compass amplification.
                corrected_grad = grad.add(ema.div(bias_correction), alpha=group["amp"])

                # AdamW debias.
                denom = ema_squared.sqrt().div_(bias_correction_sqrt)

                # ADOPT update.
                ema_squared = ema_squared.mul(slow_beta).addcmul_(corrected_grad, corrected_grad, value=1 - slow_beta)

                # Atan2-AdamW.
                full_step = corrected_grad.atan2(denom)

                if weight_decay != 0:
                    # Perform weight decay.
                    grad_weights = parameter_fp32.data
                    full_step.add_(grad_weights, alpha=weight_decay * weight_decay_rate ** group["step"])

                # Apply caution as per 'Cautious Optimizers' with a modified minimum.
                if group["cautious_min"] != 1.0:
                    mask = (full_step * grad > 0).to(full_step.dtype)
                    mask.clamp_min_(group["cautious_min"])
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    full_step.mul_(mask)

                parameter_fp32.data.add_(full_step, alpha=-step_size)
                if parameter.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    copy_stochastic_(state["ema"], ema)
                    copy_stochastic_(state["ema_squared"], ema_squared)
                    copy_stochastic_(parameter, parameter_fp32)
                else:
                    state["ema"].copy_(ema)
                    state["ema_squared"].copy_(ema_squared)
                    parameter.copy_(parameter_fp32)

                if group["reset_interval"] > 0:
                    state["steps_since_reset"] += 1

        return loss
