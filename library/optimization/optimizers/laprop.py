import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup

from library.optimization.optimizers.utils.stochastic import copy_stochastic_


class LaProp(BaseOptimizer):
    """Repo-owned LaProp optimizer adapted for the optimization layer."""

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 4e-4,
        betas: Betas = (0.9, 0.999),
        centered: bool = False,
        steps_before_using_centered: int = 10,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ams_bound: bool = False,
        cautious: bool = False,
        eps: float = 1e-15,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, "weight_decay")
        self.validate_non_negative(eps, "eps")

        self.cautious = cautious
        self.steps_before_using_centered = steps_before_using_centered

        defaults: Defaults = {
            "lr": lr,
            "betas": betas,
            "centered": centered,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "ams_bound": ams_bound,
            "eps": eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "LaProp"

    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group["step"] = 0
            group["exp_avg_lr_1"] = 0.0
            group["exp_avg_lr_2"] = 0.0
            for parameter in group["params"]:
                state = self.state[parameter]
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)

                if group["centered"]:
                    state["exp_mean_avg_beta2"] = torch.zeros_like(parameter)
                if group["ams_bound"]:
                    state["max_exp_avg_sq"] = torch.zeros_like(parameter)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
            group["exp_avg_lr_1"] = group.get("exp_avg_lr_1", 0.0)
            group["exp_avg_lr_2"] = group.get("exp_avg_lr_2", 0.0)

            beta1, beta2 = group["betas"]
            group["exp_avg_lr_1"] = group["exp_avg_lr_1"] * beta1 + (1.0 - beta1) * group["lr"]
            group["exp_avg_lr_2"] = group["exp_avg_lr_2"] * beta2 + (1.0 - beta2)

            bias_correction1 = group["exp_avg_lr_1"] / group["lr"] if group["lr"] != 0.0 else 1.0
            bias_correction2 = group["exp_avg_lr_2"]
            step_size = 1.0 / bias_correction1

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[parameter]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    if group["centered"]:
                        state["exp_mean_avg_beta2"] = torch.zeros_like(parameter)
                    if group["ams_bound"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(parameter)

                parameter_fp32 = parameter
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    parameter_fp32 = parameter.clone().to(torch.float32)
                    exp_avg = exp_avg.to(torch.float32)
                    exp_avg_sq = exp_avg_sq.to(torch.float32)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denominator = exp_avg_sq
                if group["centered"]:
                    exp_mean_avg_beta2 = state["exp_mean_avg_beta2"]
                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        exp_mean_avg_beta2 = exp_mean_avg_beta2.to(torch.float32)

                    exp_mean_avg_beta2.mul_(beta2).add_(grad, alpha=1.0 - beta2)
                    if group["step"] > self.steps_before_using_centered:
                        denominator = denominator - exp_mean_avg_beta2.pow(2)

                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["exp_mean_avg_beta2"], exp_mean_avg_beta2)

                if group["ams_bound"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        max_exp_avg_sq = max_exp_avg_sq.to(torch.float32)

                    if not (group["centered"] and group["step"] <= self.steps_before_using_centered):
                        torch.max(max_exp_avg_sq, denominator, out=max_exp_avg_sq)
                        denominator = max_exp_avg_sq

                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["max_exp_avg_sq"], max_exp_avg_sq)

                denominator = denominator.div(bias_correction2).sqrt_().add_(group["eps"])
                exp_avg.mul_(beta1).addcdiv_(grad, denominator, value=(1.0 - beta1) * group["lr"])

                if self.cautious:
                    mask = (exp_avg * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                else:
                    mask = 1.0

                parameter_fp32.add_(exp_avg * mask, alpha=-step_size)
                self.apply_weight_decay(
                    p=parameter_fp32,
                    grad=grad,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    weight_decouple=group["weight_decouple"],
                    fixed_decay=group["fixed_decay"],
                )

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(parameter, parameter_fp32)

        return loss
