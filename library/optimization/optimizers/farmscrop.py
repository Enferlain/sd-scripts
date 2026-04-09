import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import adaptive_eps, copy_stochastic_


class FARMSCrop(Optimizer):
    r"""
    FARMSCrop: Fisher-Accelerated RMSProp with momentum-style Compass amplification.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.999, 0.9999),
        eps: float = 1e-8,
        eps2: float = 0.01,
        eps_floor: float | None = 1e-16,
        weight_decay: float = 1e-6,
        centralization: float = 1.0,
        diff_mult: float = 1.0,
        momentum_beta: float = 0.9999,
        momentum_amp: float = 5.0,
        **kwargs,
    ):
        del kwargs

        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "eps2": eps2,
            "eps_floor": eps_floor,
            "weight_decay": weight_decay,
            "centralization": centralization,
            "diff_mult": diff_mult,
            "momentum_beta": momentum_beta,
            "momentum_amp": momentum_amp,
        }

        self.eps = eps
        self.eps2 = eps2
        self.eps_floor = eps_floor
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "FARMSCrop"

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            diff_mult = group["diff_mult"]
            momentum_beta = group["momentum_beta"]
            momentum_amp = group["momentum_amp"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state["fim"] = torch.ones_like(param.data)
                    state["momentum"] = torch.zeros_like(param.data)
                    state["previous_grad"] = torch.zeros_like(param.data)
                    state["grad_diff_fim"] = torch.ones_like(param.data)

                if param.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    fim = state["fim"].to(torch.float32)
                    momentum = state["momentum"].to(torch.float32)
                    prev_grad = state["previous_grad"].to(torch.float32)
                    grad_diff_fim = state["grad_diff_fim"].to(torch.float32)
                    param_fp32 = param.clone().to(torch.float32)
                else:
                    fim = state["fim"]
                    momentum = state["momentum"]
                    prev_grad = state["previous_grad"]
                    grad_diff_fim = state["grad_diff_fim"]
                    param_fp32 = param

                fim_slow_beta = ((beta2**group["step"] - beta2) / (beta2**group["step"] - 1.0)) ** 0.5
                grad_diff = prev_grad.add(grad) * diff_mult
                grad_diff_fim.mul_(beta1).addcmul_(grad_diff, grad_diff, value=1 - beta1)

                curr_eps = adaptive_eps(grad, group)
                diff_fim_base = grad_diff_fim.sqrt().add_(curr_eps)

                approx_grad_nat = grad.div(diff_fim_base)
                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(1, rms)
                approx_grad_nat.div_(divisor)

                fim.mul_(fim_slow_beta).addcmul_(approx_grad_nat, approx_grad_nat, value=1 - fim_slow_beta)
                fim_base = fim.sqrt().add_(curr_eps)

                grad_nat = grad.div(fim_base).mul_(diff_fim_base)
                rms = grad_nat.pow(2).mean().sqrt_()
                divisor = max(1, rms)
                grad_nat.div_(divisor)

                if centralization != 0 and grad_nat.dim() > 1:
                    grad_nat.sub_(
                        grad_nat.mean(dim=tuple(range(1, grad_nat.dim())), keepdim=True).mul_(centralization)
                    )

                momentum.mul_(momentum_beta).add_(grad_nat, alpha=1 - momentum_beta)
                full_step = grad_nat.add(momentum, alpha=momentum_amp)

                if weight_decay != 0:
                    grad_weights = param_fp32.data.div(fim_base).mul_(diff_fim_base)
                    rms = grad_weights.pow(2).mean().sqrt_()
                    divisor = max(1, rms)
                    grad_weights.div_(divisor)
                    full_step.add_(grad_weights, alpha=weight_decay)

                param_fp32.data.add_(full_step, alpha=-lr)

                if param.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["previous_grad"], -grad)
                    copy_stochastic_(state["grad_diff_fim"], grad_diff_fim)
                    copy_stochastic_(param, param_fp32)
                else:
                    state["previous_grad"].copy_(-grad)

        return loss
