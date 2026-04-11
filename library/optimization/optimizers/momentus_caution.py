import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import copy_stochastic_


def unit_norm_func(x: torch.Tensor, norm: float = 2.0) -> torch.Tensor:
    r"""Get norm of unit."""
    keep_dim = True
    dim = None

    x_len = len(x.shape)
    if x_len <= 1:
        keep_dim = False
    elif x_len in (2, 3):
        dim = 1
    elif x_len == 4:
        dim = (1, 2, 3)
    else:
        dim = tuple(range(1, x_len))

    return x.norm(p=norm, dim=dim, keepdim=keep_dim)


def agc_global_norm(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    agc_eps: float,
    agc_clip_val: float,
    eps: float = 1e-6,
    unit_norm: bool = True,
) -> torch.Tensor:
    r"""Clip gradient values based on the global norm."""
    func = unit_norm_func if unit_norm else torch.linalg.norm
    parameter_norm = func(parameter).clamp_(min=agc_eps)
    grad_norm = func(grad)
    max_norm = parameter_norm * agc_clip_val
    clipped_grad = grad * (max_norm / grad_norm.clamp_min_(eps))
    return torch.where(grad_norm > max_norm, clipped_grad, grad)


class MomentusCaution(Optimizer):
    r"""
    MomentusCaution: MomentusCaution
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta: float = 0.9,
        momentum_beta: float = 0.0,
        weight_decay: float = 0.0,
        gamma_ratio: float = 0.5,
        adaptive_clip: float = 0.0,
        cautious: bool = True,
        nesterov: bool = False,
        **kwargs,
    ):
        del kwargs

        defaults = {
            "lr": lr,
            "beta": beta,
            "momentum_beta": momentum_beta,
            "weight_decay": weight_decay,
            "gamma_ratio": gamma_ratio,
            "adaptive_clip": adaptive_clip,
            "cautious": cautious,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "MomentusCaution"

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group["step"] = 0
            for parameter in group["params"]:
                state = self.state[parameter]
                state["momentum"] = torch.zeros_like(parameter)
                if group["gamma_ratio"] != 0:
                    state["prev_grad"] = torch.zeros_like(parameter)
                    copy_stochastic_(state["prev_grad"], -parameter)
                if group["momentum_beta"] > 0:
                    state["grad_momentum"] = torch.zeros_like(parameter)

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
            beta = group["beta"]
            momentum_beta = group["momentum_beta"]
            weight_decay = group["weight_decay"]
            gamma_ratio = group["gamma_ratio"]
            adaptive_clip = group["adaptive_clip"]
            step = group["step"]

            curr_beta = beta * (1 - beta ** (step - 1)) / (1 - beta**step)
            curr_gamma = (1.0 - curr_beta) * gamma_ratio

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                grad = parameter.grad.data
                parameter_fp32 = parameter

                # State initialization.
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(parameter)
                    if gamma_ratio != 0:
                        state["prev_grad"] = torch.zeros_like(parameter)
                        copy_stochastic_(state["prev_grad"], -grad)
                    if momentum_beta > 0:
                        state["grad_momentum"] = torch.zeros_like(parameter)

                momentum = state["momentum"]

                # Unpack.
                if parameter.dtype == torch.bfloat16:
                    grad = grad.to(torch.float32)
                    momentum = momentum.to(torch.float32)
                    parameter_fp32 = parameter.to(dtype=torch.float32, copy=True)

                if gamma_ratio != 0:
                    state["prev_grad"].add_(grad)
                    # Calculate c_t (gradient with correction term).
                    correction = curr_gamma * curr_beta / (1 - curr_beta) * state["prev_grad"]
                    corrected_grad = grad + correction
                else:
                    corrected_grad = grad

                # Gradient clipping (if necessary).
                if adaptive_clip > 0.0:
                    corrected_grad = agc_global_norm(parameter, corrected_grad, 1e-3, adaptive_clip, unit_norm=True)
                else:
                    grad_norm = torch.norm(corrected_grad)
                    if grad_norm > 1.0:
                        corrected_grad = corrected_grad / grad_norm

                var_reduced_grad = corrected_grad
                if momentum_beta > 0:
                    grad_momentum = state["grad_momentum"]
                    if parameter.dtype == torch.bfloat16:
                        grad_momentum = grad_momentum.to(torch.float32)

                    if group["nesterov"]:
                        grad_momentum.mul_(momentum_beta).add_(corrected_grad)
                        var_reduced_grad = corrected_grad.add(grad_momentum, alpha=momentum_beta).mul_(1.0 - momentum_beta)
                    else:
                        grad_momentum.mul_(momentum_beta).add_(corrected_grad, alpha=1 - momentum_beta)
                        var_reduced_grad = grad_momentum

                    if parameter.dtype == torch.bfloat16:
                        copy_stochastic_(state["grad_momentum"], grad_momentum)

                full_step = var_reduced_grad.div(momentum.sqrt().clamp_min_(1e-6))
                momentum.mul_(curr_beta).addcmul_(corrected_grad, corrected_grad, value=1 - curr_beta)

                if weight_decay != 0:
                    # Perform weight decay.
                    full_step = full_step.add(parameter_fp32, alpha=weight_decay)

                # Apply caution as per 'Cautious Optimizers' + 'Grams'.
                if group["cautious"]:
                    mask = (full_step * grad > 0).to(full_step.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    full_step = torch.sign(corrected_grad) * full_step.abs() * mask.clamp_min_(1)

                parameter_fp32.add_(full_step, alpha=-lr)

                # Pack.
                if parameter.dtype == torch.bfloat16:
                    copy_stochastic_(state["momentum"], momentum)
                    if gamma_ratio != 0:
                        copy_stochastic_(state["prev_grad"], -grad)
                    copy_stochastic_(parameter, parameter_fp32)
                elif gamma_ratio != 0:
                    state["prev_grad"].copy_(-grad)

        return loss
