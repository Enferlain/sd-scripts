import torch
from pytorch_optimizer.base.optimizer import BaseOptimizer

from library.optimization.optimizers.utils import adaptive_eps, agc, copy_stochastic_


class FMARSCropV2(BaseOptimizer):
    r"""
    Fisher-accelerated MARS with Compass-style amplification and cautious stepping.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.999, 0.9999),
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float | None = None,
        weight_decay: float = 0.01,
        centralization: float = 0.0,
        moment_centralization: float = 0.0,
        diff_mult: float = 0.0,
        momentum_beta: float = 0.99,
        momentum_lambda: float = 0.1,
        gamma: float = 0.001,
        clip: float = 1.0,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        cautious: bool = True,
        debias_beta2: bool = True,
    ):
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-36

        defaults = {
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
            "gamma": gamma,
            "clip": clip,
            "adaptive_clip": adaptive_clip,
            "adaptive_clip_eps": adaptive_clip_eps,
            "cautious": cautious,
            "debias_beta2": debias_beta2,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "FMARSCropV2"

    def init_group(self, group, **kwargs) -> None:
        del group, kwargs

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group["step"] = 0
            for parameter in group["params"]:
                state = self.state[parameter]
                state["fim"] = torch.ones_like(parameter.data)
                state["momentum"] = torch.zeros_like(parameter.data)
                state["prev_grad"] = torch.zeros_like(parameter.data).detach()
                if group["diff_mult"] > 0:
                    state["grad_diff_fim"] = torch.ones_like(parameter.data)

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

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            moment_centralization = group["moment_centralization"]
            diff_mult = group["diff_mult"]
            momentum_beta = group["momentum_beta"]
            momentum_lambda = group["momentum_lambda"]
            gamma = group["gamma"]
            clip = group["clip"]
            step = group["step"]
            debias_beta2 = group["debias_beta2"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]

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
                correction = (gamma * (beta1 / (1 - beta1))) * prev_grad
                corrected_grad = grad + correction

                if group["adaptive_clip"] > 0.0:
                    corrected_grad = agc(
                        p=parameter_fp32,
                        grad=corrected_grad,
                        agc_clip_val=group["adaptive_clip"],
                        agc_eps=group["adaptive_clip_eps"],
                        norm_type="layer",
                    )

                clip_lambda = step**0.25

                if debias_beta2:
                    fim_slow_beta = ((beta2**step - beta2) / (beta2**step - 1.0)) ** 0.5
                else:
                    fim_slow_beta = beta2

                current_eps = adaptive_eps(grad, group)

                if diff_mult > 0:
                    grad_diff = prev_grad * diff_mult
                    rms = grad_diff.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_diff.div_(divisor)

                    grad_diff_fim = state["grad_diff_fim"]
                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff_fim = state["grad_diff_fim"].to(torch.float32)

                    diff_fim_base = grad_diff_fim.sqrt().add(current_eps)
                    grad_diff_fim.mul_(beta1).addcmul_(grad_diff, grad_diff, value=1 - beta1).clamp_(
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

                fim_base = fim.sqrt().add(current_eps)
                grad_nat = approx_grad_nat.div(fim_base).div_(diff_fim_base)
                rms = grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                grad_nat.div_(divisor)

                momentum.mul_(momentum_beta).add_(grad_nat, alpha=1 - momentum_beta)

                if moment_centralization != 0:
                    momentum_cent = momentum - torch.mean(momentum) * moment_centralization
                else:
                    momentum_cent = momentum

                if group["cautious"]:
                    mask = (momentum_cent * grad_nat < 0).to(grad_nat.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    momentum_cent = momentum_cent * mask

                full_step = grad_nat.add(momentum_cent, alpha=step**momentum_lambda)

                if centralization != 0 and full_step.dim() > 1:
                    full_step.sub_(
                        full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(centralization)
                    )

                if weight_decay != 0:
                    grad_weights = parameter_fp32.data.div(fim_base).div_(diff_fim_base)
                    rms = grad_weights.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_weights.div_(divisor)
                    parameter_fp32.data.add_(grad_weights, alpha=-lr * weight_decay)

                if group["cautious"]:
                    mask = (full_step * grad_nat > 0).to(grad_nat.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    full_step = full_step * mask
                parameter_fp32.data.add_(full_step, alpha=-lr)

                fim.mul_(fim_slow_beta).addcmul_(approx_grad_nat, approx_grad_nat, value=1 - fim_slow_beta).clamp_(
                    -clip_lambda, clip_lambda
                )

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["prev_grad"], -grad)
                    copy_stochastic_(parameter, parameter_fp32)
                else:
                    state["prev_grad"].copy_(-grad)

        return loss
