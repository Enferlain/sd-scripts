import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils.stochastic import copy_stochastic_


class LPFAdamW(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.9, 0.999)).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.9, 0.999),
        amp_fac: float = 2.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        centralization: float = 0.0,
        **kwargs,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "amp_fac": amp_fac,
            "eps": eps,
            "weight_decay": weight_decay,
            "centralization": centralization,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "LPFAdamW"

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1

            beta1, beta2, beta3 = group["betas"]
            amplification_factor = group["amp_fac"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            bias_correction = 1 - beta2 ** group["step"]
            bias_correction_sqrt = (1 - beta3 ** group["step"]) ** 0.5
            step_size = lr / bias_correction

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("LPFAdamW does not support sparse gradients")

                state = self.state[parameter]
                
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["smoothing"] = torch.zeros_like(parameter.data)
                    state["ema"] = torch.zeros_like(parameter.data)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(parameter.data)

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    parameter_fp32 = parameter.clone().to(torch.float32)
                    smoothing = state["smoothing"].to(torch.float32)
                    ema = state["ema"].to(torch.float32)
                    ema_squared = state["ema_squared"].to(torch.float32)
                else:
                    parameter_fp32 = parameter
                    smoothing = state["smoothing"]
                    ema = state["ema"]
                    ema_squared = state["ema_squared"]

                # center the gradient vector
                if centralization != 0 and grad.dim() > 1:
                    grad.sub_(grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization))

                smoothing.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad.add_(smoothing, alpha=amplification_factor)

                ema.mul_(beta2).add_(grad, alpha=1 - beta2)
                ema_squared.mul_(beta3).addcmul_(grad, grad, value=1 - beta3)
                denominator = (ema_squared.sqrt() / bias_correction_sqrt).add_(group["eps"])

                if weight_decay != 0:
                    parameter_fp32.data.mul_(1 - step_size * weight_decay)

                parameter_fp32.data.addcdiv_(grad, denominator, value=-step_size)

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["ema"], ema)
                    copy_stochastic_(state["ema_squared"], ema_squared)
                    copy_stochastic_(state["smoothing"], smoothing)
                    copy_stochastic_(parameter, parameter_fp32)

        return loss
