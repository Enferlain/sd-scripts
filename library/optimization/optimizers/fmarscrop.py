import math

import torch
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup

from library.optimization.optimizers.utils import NORM_TYPE, adaptive_eps, agc, copy_stochastic_


class FMARSCrop(BaseOptimizer):
    r"""
    Fisher-accelerated MARS with Compass-style amplification and ADOPT-style AdamW updates.
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 5e-4,
        betas: Betas = (0.999, 0.9999),
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float | None = None,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        centralization: float = 0.0,
        moment_centralization: float = 0.0,
        diff_mult: float = 1.0,
        momentum_lambda: float = 0.1,
        momentum_beta: float = 0.99,
        clip: float = 1.0,
        cautious: bool = True,
        gamma: float = 0.0005,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = "global",
        stable_weight_decay: bool = False,
        debias_beta2: bool = True,
        **kwargs,
    ):
        del kwargs

        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, "weight_decay")
        self.validate_non_negative(eps, "eps")

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
            "moment_centralization": moment_centralization,
            "diff_mult": diff_mult,
            "momentum_beta": momentum_beta,
            "momentum_lambda": momentum_lambda,
            "clip": clip,
            "cautious": cautious,
            "gamma": gamma,
            "adaptive_clip": adaptive_clip,
            "adaptive_clip_eps": adaptive_clip_eps,
            "adaptive_clip_type": adaptive_clip_type,
            "stable_weight_decay": stable_weight_decay,
            "debias_beta2": debias_beta2,
            "weight_decouple": weight_decouple,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "FMARSCrop"

    def init_group(self, group, **kwargs) -> None:
        del group, kwargs

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group["step"] = 0
            group["fim_mean_sqrt"] = None
            for parameter in group["params"]:
                state = self.state[parameter]
                state["fim"] = torch.ones_like(parameter.data)
                state["momentum"] = torch.zeros_like(parameter.data)
                state["prev_grad"] = torch.zeros_like(parameter.data).detach()
                if group["diff_mult"] > 0:
                    state["grad_diff_fim"] = torch.ones_like(parameter.data)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1
                group["fim_mean_sqrt"] = None

            param_size = 0
            fim_sum = 0.0

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            moment_centralization = group["moment_centralization"]
            diff_mult = group["diff_mult"]
            momentum_beta = group["momentum_beta"]
            momentum_lambda = group["momentum_lambda"]
            clip = group["clip"]
            step = group["step"]
            gamma = group["gamma"]
            adaptive_clip = group["adaptive_clip"]
            adaptive_clip_type = group["adaptive_clip_type"]
            adaptive_clip_eps = group["adaptive_clip_eps"]
            stable_weight_decay = group["stable_weight_decay"]
            weight_decouple = group["weight_decouple"]

            clip_lambda = (step - 1) ** 0.25

            if group["debias_beta2"]:
                current_beta2 = self.debias_beta(beta2, group["step"])
            else:
                current_beta2 = beta2

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]

                if stable_weight_decay:
                    param_size += parameter.numel()

                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(parameter)
                    state["fim"] = torch.ones_like(parameter)
                    state["prev_grad"] = -parameter.grad.to(dtype=parameter.dtype, copy=True).detach()
                    if diff_mult > 0:
                        state["grad_diff_fim"] = torch.ones_like(parameter)

                grad = parameter.grad
                parameter_fp32 = parameter
                prev_grad = state["prev_grad"]
                fim = state["fim"]
                momentum = state["momentum"]

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    fim = fim.to(torch.float32)
                    momentum = momentum.to(torch.float32)
                    prev_grad = prev_grad.to(torch.float32)
                    parameter_fp32 = parameter.to(dtype=torch.float32, copy=True)

                prev_grad = prev_grad.add(grad)
                correction = (gamma * (beta1 / (1.0 - beta1))) * prev_grad
                corrected_grad = grad + correction

                if adaptive_clip > 0.0:
                    corrected_grad = agc(
                        p=parameter_fp32,
                        grad=corrected_grad,
                        agc_clip_val=adaptive_clip,
                        agc_eps=adaptive_clip_eps,
                        norm_type=adaptive_clip_type,
                    )

                current_eps = adaptive_eps(grad, group)

                if diff_mult > 0:
                    grad_diff = prev_grad * diff_mult
                    rms = grad_diff.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_diff.div_(divisor)

                    grad_diff_fim = state["grad_diff_fim"]
                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff_fim = grad_diff_fim.to(torch.float32)

                    diff_fim_base = grad_diff_fim.sqrt().add_(current_eps)
                    grad_diff_fim.mul_(beta1).addcmul_(grad_diff, grad_diff, value=1.0 - beta1).clamp_(
                        -clip_lambda, clip_lambda
                    )

                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["grad_diff_fim"], grad_diff_fim)
                else:
                    diff_fim_base = 1.0

                approx_grad_nat = corrected_grad.div(diff_fim_base)
                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                approx_grad_nat.div_(divisor)

                if group["step"] == 1:
                    fim.addcmul_(approx_grad_nat, approx_grad_nat)
                else:
                    fim_base = fim.sqrt().add_(current_eps)
                    grad_nat = approx_grad_nat.div(fim_base).div_(diff_fim_base)
                    rms = grad_nat.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_nat.div_(divisor)

                    momentum.mul_(momentum_beta).add_(grad_nat, alpha=1.0 - momentum_beta)

                    if moment_centralization != 0:
                        momentum_cent = momentum.sub(torch.mean(momentum).mul_(moment_centralization))
                    else:
                        momentum_cent = momentum

                    if group["cautious"]:
                        mask = (momentum_cent * grad_nat < 0).to(momentum_cent.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        momentum_cent = momentum_cent * mask

                    full_step = grad_nat.add(momentum_cent, alpha=step**momentum_lambda)

                    if centralization != 0 and full_step.dim() > 1:
                        full_step.sub_(
                            full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(centralization)
                        )

                    if weight_decay != 0 and weight_decouple:
                        if stable_weight_decay and group["fim_mean_sqrt"] > 0:
                            swd_scaling = 1.0 / group["fim_mean_sqrt"]
                        else:
                            swd_scaling = 1.0
                        parameter_fp32.mul_(1.0 - weight_decay * lr * swd_scaling)
                    elif weight_decay != 0:
                        grad_weights = parameter_fp32.data.div(fim_base).div_(diff_fim_base)
                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(clip, rms) / clip
                        grad_weights.div_(divisor)

                        if stable_weight_decay and group["fim_mean_sqrt"] is not None:
                            scale = 1.0 / group["fim_mean_sqrt"]
                        else:
                            scale = 1.0
                        parameter_fp32.data.add_(grad_weights, alpha=-lr * weight_decay * scale)

                    if group["cautious"]:
                        mask = (full_step * grad_nat > 0).to(grad_nat.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                    else:
                        mask = 1.0

                    parameter_fp32.data.add_(full_step * mask, alpha=-lr)
                    fim.mul_(current_beta2).addcmul_(approx_grad_nat, approx_grad_nat, value=1.0 - current_beta2).clamp_(
                        -clip_lambda, clip_lambda
                    )

                if stable_weight_decay:
                    fim_sum += fim.sum()

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["prev_grad"], -grad)
                    copy_stochastic_(parameter, parameter_fp32)
                else:
                    state["prev_grad"].copy_(-grad)

            if stable_weight_decay:
                group["fim_mean_sqrt"] = math.sqrt(fim_sum / param_size)

        return loss
