import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient

from library.optimization.optimizers.utils.stochastic import copy_stochastic_


class Adai(BaseOptimizer):
    """Repo-owned Adai optimizer adapted for the optimization layer."""

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-3,
        betas: Betas = (0.1, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        stable_weight_decay: bool = False,
        dampening: float = 1.0,
        use_gc: bool = False,
        eps: float = 1e-3,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, "weight_decay")
        self.validate_non_negative(eps, "eps")

        self.use_gc = use_gc
        defaults: Defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "stable_weight_decay": stable_weight_decay,
            "dampening": dampening,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "Adai"

    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group["step"] = 0
            for parameter in group["params"]:
                state = self.state[parameter]
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["beta1_prod"] = torch.ones_like(parameter)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size = 0
        exp_avg_sq_hat_sum = 0.0

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
            _, beta2 = group["betas"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                param_size += parameter.numel()
                state = self.state[parameter]
                parameter_fp32 = parameter

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    parameter_fp32 = parameter.clone().to(torch.float32)

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["beta1_prod"] = torch.ones_like(parameter)

                if self.use_gc:
                    centralize_gradient(grad, gc_conv_only=False)

                bias_correction2 = self.debias(beta2, group["step"])

                if not group["stable_weight_decay"] and group["weight_decay"] > 0.0:
                    self.apply_weight_decay(
                        p=parameter_fp32,
                        grad=grad,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        weight_decouple=group["weight_decouple"],
                        fixed_decay=group["fixed_decay"],
                    )

                exp_avg_sq = state["exp_avg_sq"]
                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    exp_avg_sq = exp_avg_sq.to(torch.float32)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                exp_avg_sq_hat_sum += (exp_avg_sq.sum() / bias_correction2).item()

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)

        if param_size == 0:
            raise ZeroParameterSizeError()

        exp_avg_sq_hat_mean = exp_avg_sq_hat_sum / param_size

        for group in self.param_groups:
            beta0, beta2 = group["betas"]
            beta0_dp = math.pow(beta0, 1.0 - group["dampening"])

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                state = self.state[parameter]
                parameter_fp32 = parameter

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    parameter_fp32 = parameter.clone().to(torch.float32)

                if group["stable_weight_decay"] and group["weight_decay"] > 0.0:
                    self.apply_weight_decay(
                        p=parameter_fp32,
                        grad=grad,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        weight_decouple=group["weight_decouple"],
                        fixed_decay=group["fixed_decay"],
                    )

                bias_correction2 = self.debias(beta2, group["step"])
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1_prod = state["beta1_prod"]

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    exp_avg = exp_avg.to(torch.float32)
                    exp_avg_sq = exp_avg_sq.to(torch.float32)
                    beta1_prod = beta1_prod.to(torch.float32)

                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                beta1 = (
                    1.0
                    - (exp_avg_sq_hat / exp_avg_sq_hat_mean).pow_(1.0 / (3.0 - 2.0 * group["dampening"])).mul_(beta0)
                ).clamp_(0.0, 1.0 - group["eps"])
                beta3 = (1.0 - beta1).pow_(group["dampening"])

                beta1_prod.mul_(beta1)
                exp_avg.mul_(beta1).addcmul_(beta3, grad)
                exp_avg_hat = exp_avg.div(1.0 - beta1_prod).mul_(beta0_dp)
                parameter_fp32.add_(exp_avg_hat, alpha=-group["lr"])

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["beta1_prod"], beta1_prod)
                    copy_stochastic_(parameter, parameter_fp32)

        return loss
