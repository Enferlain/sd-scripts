from typing import Literal

import torch
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup

from library.optimization.optimizers.utils import adaptive_eps, copy_stochastic_


MASK_GRADS = Literal["grad", "approx_grad_nat", "grad_nat"]


class FARMSCropV2(BaseOptimizer):
    r"""
    FARMSCropV2: Fisher-Accelerated RMSprop, with momentum-based Compass-style amplification, with ADOPT's AdamW changes. (https://arxiv.org/abs/2411.02853).
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float):
            coefficients used for computing running averages of
            gradient difference FIM and approx. natural grad FIM (default: 0.999, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.0).
        centralization (float):
            Center model grad (default: 0.0).
        diff_mult (float):
            Multiplier for difference amplification (default: 1.0).
        momentum_beta (float):
            Beta value for slow momentum / EMA (default: 0.9999) (Alternative recommendation: 0.99999).
        momentum_lambda (float):
            Amplification exponent for slow momentum / EMA (default: 0.25) (Alternative recommendation: 0.5).
        clip (float):
            Value to clip the grad's RMS at (default: 1.0)
        cautious (bool):
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: False)
        cautious_grad (str):
            Which form of grad to use for the cautious mask, valid options are 'grad', 'approx_grad_nat' 'grad_nat' (Default: grad)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-4,
        betas: Betas = (0.999, 0.9999),
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float | None = None,
        weight_decay: float = 0.0,
        centralization: float = 0.0,
        diff_mult: float = 1.0,
        momentum_beta: float = 0.9999,
        momentum_lambda: float = 0.25,
        clip: float = 1.0,
        cautious: bool = False,
        cautious_grad: MASK_GRADS = "grad",
        **kwargs,
    ):
        del kwargs

        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, "weight_decay")
        self.validate_non_negative(eps, "eps")
        self.validate_non_negative(eps2, "eps2")

        # Override zero to 1e-37, as zero and float32.tiny NaNs
        # Using 1e-37 as 1e-38 NaNs for Flux loras
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults: Defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "eps2": eps2,
            "eps_floor": eps_floor,
            "weight_decay": weight_decay,
            "centralization": centralization,
            "diff_mult": diff_mult,
            "momentum_beta": momentum_beta,
            "momentum_lambda": momentum_lambda,
            "clip": clip,
            "cautious": cautious,
            "cautious_grad": cautious_grad,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "FARMSCropV2"

    def init_group(self, group, **kwargs) -> None:
        del group, kwargs

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group["step"] = 0
            for param in group["params"]:
                state = self.state[param]
                state["fim"] = torch.ones_like(param.data)
                state["momentum"] = torch.zeros_like(param.data)
                if group["diff_mult"] > 0:
                    state["previous_grad"] = torch.zeros_like(param.data).detach()
                    state["grad_diff_fim"] = torch.ones_like(param.data)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            momentum_beta = group["momentum_beta"]
            momentum_lambda = group["momentum_lambda"]
            clip = group["clip"]
            step = group["step"]
            cautious_grad = group["cautious_grad"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                param_fp32 = param

                diff_mult = group["diff_mult"]
                if len(state) == 0:
                    state["fim"] = torch.ones_like(param.data)
                    state["momentum"] = torch.zeros_like(param.data)
                    if diff_mult > 0:
                        state["previous_grad"] = -grad.clone().to(param.dtype).detach()
                        state["grad_diff_fim"] = torch.ones_like(param.data)

                fim = state["fim"]
                momentum = state["momentum"]

                if param.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    fim = state["fim"].to(torch.float32)
                    momentum = state["momentum"].to(torch.float32)
                    param_fp32 = param.clone().to(torch.float32)

                clip_lambda = step**0.25
                fim_slow_beta = ((beta2**step - beta2) / (beta2**step - 1.0)) ** 0.5
                curr_eps = adaptive_eps(grad, group)

                if diff_mult > 0:
                    prev_grad = state["previous_grad"]
                    grad_diff_fim = state["grad_diff_fim"]
                    if param.dtype in {torch.float16, torch.bfloat16}:
                        prev_grad = state["previous_grad"].to(torch.float32)
                        grad_diff_fim = state["grad_diff_fim"].to(torch.float32)

                    grad_diff = prev_grad.add(grad) * diff_mult
                    rms = grad_diff.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_diff.div_(divisor)

                    diff_fim_base = grad_diff_fim.sqrt().add_(curr_eps)
                    grad_diff_fim.mul_(beta1).addcmul_(grad_diff, grad_diff, value=1 - beta1).clamp_(
                        -clip_lambda, clip_lambda
                    )

                    if param.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["grad_diff_fim"], grad_diff_fim)
                else:
                    diff_fim_base = 1.0

                approx_grad_nat = grad.div(diff_fim_base)
                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                approx_grad_nat.div_(divisor)

                fim_base = fim.sqrt().add_(curr_eps)
                grad_nat = grad.div(fim_base).div_(diff_fim_base)
                rms = grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                grad_nat.div_(divisor)

                full_step = grad_nat.add(momentum, alpha=step**momentum_lambda)

                if centralization != 0 and full_step.dim() > 1:
                    full_step.sub_(
                        full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(centralization)
                    )

                if weight_decay != 0:
                    grad_weights = param_fp32.data.div(fim_base).div_(diff_fim_base)
                    rms = grad_weights.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_weights.div_(divisor)
                    param_fp32.data.add_(grad_weights, alpha=-lr * weight_decay)

                if group["cautious"]:
                    if cautious_grad == "grad":
                        grad_for_mask = grad
                    elif cautious_grad == "approx_grad_nat":
                        grad_for_mask = approx_grad_nat
                    else:
                        grad_for_mask = grad_nat
                    mask = (full_step * grad_for_mask > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                else:
                    mask = 1.0

                param_fp32.data.add_(full_step * mask, alpha=-lr)

                fim.mul_(fim_slow_beta).addcmul_(approx_grad_nat, approx_grad_nat, value=1 - fim_slow_beta).clamp_(
                    -clip_lambda, clip_lambda
                )
                momentum.mul_(momentum_beta).add_(grad_nat, alpha=1 - momentum_beta)

                if param.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    if diff_mult > 0:
                        copy_stochastic_(state["previous_grad"], -grad)
                    copy_stochastic_(param, param_fp32)
                else:
                    if diff_mult > 0:
                        state["previous_grad"].copy_(-grad)

        return loss
