import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup

from library.optimization.optimizers.utils.stochastic import copy_stochastic_


class SGDSaI(BaseOptimizer):
    r"""No More Adam: Learning Rate Scaling at Initialization is All You Need.

    :param params: ParamGroup. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float.  coefficients used for computing running averages of gradient.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        eps: float = 1e-8,
        cautious: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, "beta", 0.0, 1.0)
        self.validate_non_negative(weight_decay, "weight_decay")
        self.validate_non_negative(eps, "eps")

        self.has_warmup = False
        defaults: Defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "cautious": cautious,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "SGDSaI"

    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group["step"] = 0
            for parameter in group["params"]:
                state = self.state[parameter]
                if group["momentum"] > 0.0:
                    state["momentum_buffer"] = torch.zeros_like(parameter)

    @torch.no_grad()
    def warmup_step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)

                sigma = grad.std().nan_to_num_()
                grad_norm = grad.norm()
                gsnr = grad_norm.div_(sigma.add_(group["eps"])) if sigma != 0.0 else grad_norm
                self.state[parameter]["gsnr"] = gsnr

        self.has_warmup = True
        return loss

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if not self.has_warmup:
            self.warmup_step(closure)

        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad
                parameter_fp32 = parameter
                state = self.state[parameter]

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    parameter_fp32 = parameter.to(dtype=torch.float32, copy=True)

                if momentum > 0.0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = grad.clone()
                    momentum_buffer = state["momentum_buffer"]
                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        momentum_buffer = momentum_buffer.to(torch.float32)

                    momentum_buffer.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["momentum_buffer"], momentum_buffer)
                else:
                    momentum_buffer = grad

                self.apply_weight_decay(
                    parameter_fp32,
                    grad,
                    group["lr"],
                    group["weight_decay"],
                    group["weight_decouple"],
                    False,
                )

                if group["cautious"] and momentum > 0.0:
                    mask = (momentum_buffer * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                else:
                    mask = 1.0

                parameter_fp32.add_(momentum_buffer * mask, alpha=-group["lr"] * state["gsnr"])

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(parameter, parameter_fp32)

        return loss
