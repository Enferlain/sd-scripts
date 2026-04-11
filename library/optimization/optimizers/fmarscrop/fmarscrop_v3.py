import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import adaptive_eps, copy_stochastic_


def agc_global_norm(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    agc_eps: float,
    agc_clip_val: float,
    eps: float = 1e-6,
    unit_norm: bool = True,
) -> torch.Tensor:
    r"""Clip gradient values based on the global norm."""
    del parameter

    if unit_norm:
        grad_norm = torch.linalg.norm(grad)
    else:
        grad_norm = torch.linalg.norm(grad)

    max_norm = grad_norm.clamp_(min=agc_eps) * agc_clip_val
    clipped_grad = grad * (max_norm / grad_norm.clamp_min_(eps))
    return torch.where(grad_norm > max_norm, clipped_grad, grad)


class FMARSCropV3(Optimizer):
    r"""
    FMARSCropV3: Fisher-accelerated MARS (https://arxiv.org/abs/2411.10438),
    with momentum-based Compass-style amplification, with customized ADOPT
    AdamW changes (https://arxiv.org/abs/2411.02853), and cautious stepping.
    Un-official MARS implementation is credited to Less Wright (lessw2020).
    Intended to arrive at the minima faster and in a more stable manner than
    FMARSCrop_ExMachina and V1.
    Thanks to Machina for introducing the usage of stochastic rounding,
    adaptive_eps, and further testing!

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float):
            coefficients used for computing running average of momentum and
            the FIM running average (default: 0.99, 0.95).
        eps (float):
            Term the denominator is minimally clamped to, to improve
            numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps.
            (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None,
            disabling adaptive eps. If 0, round to 1e-30).
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.01).
        centralization (float):
            Center model grad (default: 0.0).
        moment_centralization (float):
            Center the slow momentum / EMA (default: 0.0).
        diff_mult (float):
            Multiplier for difference amplification, adds another memory
            state (slightly increased VRAM usage) (default: 0.0).
        momentum_lambda (float):
            Amplification factor for slow momentum / EMA (default: 2.0).
        gamma (float):
            Scaling parameter for gradient correction for MARS
            (default: 0.05).
        clip_lambda (float):
            Value to clip the grad's RMS at (default: 1.0).
        adaptive_clip (float):
            Adaptive clip value to apply to the corrected gradient, before
            further use by the optimizer. (default: 0.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to
            avoid parameters not getting updates due to very small gradients
            being clipped excessively. (default: 1e-3).
        adaptive_clip_norm_type (bool):
            Whether or not to use the unit norm (default: 1) or the norm of
            the whole grad (0) for adaptive clipping.
        cautious (bool):
            Use cautious mask on parameter update
            https://arxiv.org/abs/2411.16085 (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.99, 0.95),
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float | None = None,
        weight_decay: float = 0.01,
        centralization: float = 0.0,
        moment_centralization: float = 0.0,
        diff_mult: float = 0.0,
        momentum_lambda: float = 2.0,
        gamma: float = 0.05,
        clip_lambda: float = 1.0,
        adaptive_clip: float = 0.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_norm_type: bool = True,
        cautious: bool = True,
    ):
        # Override zero to 1e-30, as zero and float32.tiny NaNs.
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-30

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
            "momentum_lambda": momentum_lambda,
            "gamma": gamma,
            "clip_lambda": clip_lambda,
            "adaptive_clip": adaptive_clip,
            "adaptive_clip_eps": adaptive_clip_eps,
            "adaptive_clip_norm_type": adaptive_clip_norm_type,
            "cautious": cautious,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "FMARSCropV3"

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group["step"] = 0
            for parameter in group["params"]:
                state = self.state[parameter]
                state["fim"] = torch.ones_like(parameter.data)
                # Fisher information matrix.
                state["momentum"] = torch.zeros_like(parameter.data)
                # Prev grad.
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
            momentum_lambda = group["momentum_lambda"]
            gamma = group["gamma"]
            clip_lambda = group["clip_lambda"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]

                # State initialization.
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(parameter)
                    state["fim"] = torch.ones_like(parameter)
                    state["prev_grad"] = -parameter.grad.clone().to(parameter.dtype).detach()
                    if diff_mult > 0:
                        state["grad_diff_fim"] = torch.ones_like(parameter)

                grad = parameter.grad
                parameter_fp32 = parameter
                prev_grad = state["prev_grad"]
                fim = state["fim"]
                momentum = state["momentum"]

                # Unpack.
                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    fim = state["fim"].to(torch.float32)
                    momentum = state["momentum"].to(torch.float32)
                    prev_grad = state["prev_grad"].to(torch.float32)
                    parameter_fp32 = parameter.clone().to(torch.float32)

                prev_grad = prev_grad.add(grad)

                # Calculate c_t (gradient with correction term).
                correction = gamma * beta2 / (1 - beta2) * prev_grad
                corrected_grad = grad + correction

                # Gradient clipping (if necessary).
                if group["adaptive_clip"] > 0.0:
                    corrected_grad = agc_global_norm(
                        parameter_fp32,
                        corrected_grad,
                        group["adaptive_clip_eps"],
                        group["adaptive_clip"],
                        unit_norm=group["adaptive_clip_norm_type"],
                    )
                grad_norm = torch.linalg.norm(corrected_grad)
                if grad_norm > clip_lambda:
                    corrected_grad = corrected_grad * clip_lambda / grad_norm

                current_eps = adaptive_eps(grad, group)

                if diff_mult > 0:
                    # grad_diff contains the difference between prev grad and current grad.
                    grad_diff = prev_grad * diff_mult
                    rms = grad_diff.pow(2).mean().sqrt_()
                    divisor = max(clip_lambda, rms) / clip_lambda
                    grad_diff.div_(divisor)

                    grad_diff_fim = state["grad_diff_fim"]

                    # Unpack.
                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff_fim = state["grad_diff_fim"].to(torch.float32)

                    # Get natural gradient (squared ema, obtained sqrt of ema).
                    diff_fim_base = torch.clamp(grad_diff_fim.sqrt(), current_eps)
                    grad_diff_fim.mul_(beta2).addcmul_(grad_diff, grad_diff, value=1 - beta2)

                    # Pack.
                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["grad_diff_fim"], grad_diff_fim)
                else:
                    diff_fim_base = 1.0

                approx_grad_nat = corrected_grad.div(diff_fim_base)
                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip_lambda, rms) / clip_lambda
                approx_grad_nat.div_(divisor)

                fim_base = torch.clamp(fim.sqrt(), current_eps)
                grad_nat = corrected_grad.div(fim_base)
                rms = grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip_lambda, rms) / clip_lambda
                grad_nat.div_(divisor)

                momentum.mul_(beta1).add_(grad_nat, alpha=1 - beta1)

                # Compass-style amplification.
                if moment_centralization != 0:
                    momentum_cent = momentum - torch.mean(momentum) * moment_centralization
                else:
                    momentum_cent = momentum
                full_step = grad_nat.add(momentum_cent, alpha=momentum_lambda)

                # Center the gradient vector.
                if centralization != 0 and full_step.dim() > 1:
                    full_step.sub_(
                        full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(centralization)
                    )

                if weight_decay != 0:
                    # Perform weight decay.
                    grad_weights = parameter_fp32.data.div(fim_base)
                    rms = grad_weights.pow(2).mean().sqrt_()
                    divisor = max(clip_lambda, rms) / clip_lambda
                    grad_weights.div_(divisor)
                    full_step.add_(grad_weights, alpha=weight_decay)

                # Apply full step.
                if group["cautious"]:
                    # Apply caution as per 'Cautious Optimizers'.
                    mask = (full_step * corrected_grad > 0).to(full_step.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    full_step = full_step * mask
                parameter_fp32.data.add_(full_step, alpha=-lr)

                fim.mul_(beta2).addcmul_(approx_grad_nat, approx_grad_nat, value=1 - beta2)

                # Pack.
                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["prev_grad"], -grad)
                    copy_stochastic_(parameter, parameter_fp32)
                else:
                    # Copy the negative of the current grad.
                    state["prev_grad"].copy_(-grad)

        return loss
