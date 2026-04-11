import math

import torch
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup

from library.optimization.optimizers.utils import NORM_TYPE, UPDATE_STRATEGY, adaptive_eps, agc, copy_stochastic_


class FMARSCropV2ExMachina(BaseOptimizer):
    r"""
    FMARSCrop: Fisher-accelerated MARS (https://arxiv.org/abs/2411.10438),
    with momentum-based Compass-style amplification, with ADOPT's AdamW
    changes (https://arxiv.org/abs/2411.02853).

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float, float):
            coefficients used for computing running averages of momentum,
            approx. natural grad FIM, and gradient difference FIM
            (default: 0.99, 0.9999, 0.999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps.
            (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs.
            (default: None, disabling adaptive eps).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.0).
        centralization (float):
            Center model grad (default: 0.0).
        moment_centralization (float):
            Center the slow momentum / EMA (default: 0.0).
        diff_mult (float):
            Multiplier for difference amplification (default: 1.0).
        momentum_lambda (float):
            The lambda value for slow momentum / EMA, controlling how much
            the momentum is amplified while being added to the update.
            (default: 2.0).
        clip (float):
            Value to clip the grad's RMS at (default: 1.0)
        cautious (bool) (deprecated, use update strategy):
            Use cautious mask on parameter update
            https://arxiv.org/abs/2411.16085 (default: False)
        update_strategy (str):
            Determine the update strategy to use, valid values are
            `unmodified`, `cautious` (https://arxiv.org/abs/2411.16085),
            and `grams` (https://arxiv.org/abs/2412.17107)
            (default: cautious)
        adaptive_clip (float):
            Adaptive clip value to applied to the MARS corrected gradient.
            (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to
            avoid parameters not getting updates due to very small gradients
            being clipped excessively. (default: 1e-3).
        adaptive_clip_type (string):
            The type of clipping, can be unit or layer. If done at the unit
            level can change the direction of the gradient, while layer only
            scales down the magnitude of the entire gradient proportionally.
            Traditional adaptive clipping uses unit-wise, while this
            implementation also supports layer. Valid values: layer, unit
            (default: layer).
        gamma (float):
            Scaling value for the MARS style correction of the gradient,
            0.025 or 0.05 are recommended by the paper, larger values apply
            more correction, and will require higher LRs to offset.
            (default: 0.0005)
        debias_beta1 (bool):
            Apply bias correction to step size (LR). (Default: False)
        debias_beta2 (bool):
            Apply bias correction to fim. (Default: True)
        debias_beta3 (bool):
            Apply bias correction to diff fim. (Default: False)
    """

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 5e-4,
        betas: Betas = (0.99, 0.9999, 0.999),
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float | None = None,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        centralization: float = 0.0,
        moment_centralization: float = 0.0,
        diff_mult: float = 1.0,
        momentum_lambda: float = 0.1,
        clip: float = 1.0,
        cautious: bool = False,
        gamma: float = 0.005,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = "global",
        stable_weight_decay: bool = False,
        debias_beta1: bool = False,
        debias_beta2: bool = True,
        debias_beta3: bool = False,
        update_strategy: UPDATE_STRATEGY = "cautious",
        **kwargs,
    ):
        del kwargs

        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, "weight_decay")
        self.validate_non_negative(eps, "eps")

        # Override zero to 1e-37, as zero and float32.tiny NaNs.
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        if update_strategy not in {"unmodified", "cautious", "grams"}:
            raise ValueError(f"Invalid update strategy: {update_strategy}")

        # Keep backwards compatibility with the donor cautious flag.
        if cautious:
            update_strategy = "cautious"

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
            "momentum_lambda": momentum_lambda,
            "clip": clip,
            "cautious": cautious,
            "gamma": gamma,
            "adaptive_clip": adaptive_clip,
            "adaptive_clip_eps": adaptive_clip_eps,
            "adaptive_clip_type": adaptive_clip_type,
            "stable_weight_decay": stable_weight_decay,
            "debias_beta1": debias_beta1,
            "debias_beta2": debias_beta2,
            "debias_beta3": debias_beta3,
            "weight_decouple": weight_decouple,
            "update_strategy": update_strategy,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "FMARSCropV2ExMachina"

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
                # Fisher information matrix.
                state["momentum"] = torch.zeros_like(parameter.data)
                # Prev grad.
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

            beta1, beta2, beta3 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            moment_centralization = group["moment_centralization"]
            diff_mult = group["diff_mult"]
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

            bias_correction1 = self.debias(beta1, group["step"])
            step_size = self.apply_adam_debias(
                adam_debias=not group["debias_beta1"],
                step_size=lr,
                bias_correction1=bias_correction1,
            )

            if group["debias_beta2"]:
                current_beta2 = self.debias_beta(beta2, group["step"])
            else:
                current_beta2 = beta2

            if group["debias_beta3"]:
                current_beta3 = self.debias_beta(beta3, group["step"])
            else:
                current_beta3 = beta3

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]

                if stable_weight_decay:
                    param_size += parameter.numel()

                # State initialization.
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

                # Unpack reduced-precision parameter state to float32.
                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    fim = fim.to(torch.float32)
                    momentum = momentum.to(torch.float32)
                    prev_grad = prev_grad.to(torch.float32)
                    parameter_fp32 = parameter.to(dtype=torch.float32, copy=True)

                prev_grad = prev_grad.add(grad)

                # Calculate c_t (gradient with correction term).
                corrected_grad = prev_grad.mul(gamma * (beta1 / (1.0 - beta1))).add_(grad)

                if adaptive_clip > 0.0:
                    # Apply Adaptive Gradient Clipping (AGC).
                    corrected_grad = agc(
                        parameter_fp32,
                        corrected_grad,
                        adaptive_clip,
                        adaptive_clip_eps,
                        norm_type=adaptive_clip_type,
                    )

                current_eps = adaptive_eps(grad, group)

                if diff_mult > 0:
                    # grad_diff contains the difference between prev grad and current grad.
                    grad_diff = prev_grad * diff_mult
                    rms = grad_diff.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_diff.div_(divisor)

                    grad_diff_fim = state["grad_diff_fim"]

                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff_fim = grad_diff_fim.to(torch.float32)

                    if group["step"] == 1:
                        grad_diff_fim.addcmul_(grad_diff, grad_diff).clamp_(-clip_lambda, clip_lambda)
                        diff_fim_base = 1.0
                    else:
                        # Get natural gradient denominator from the diff FIM.
                        diff_fim_base = grad_diff_fim.sqrt().add_(current_eps)
                        grad_diff_fim.mul_(current_beta3).addcmul_(grad_diff, grad_diff, value=1.0 - current_beta3).clamp_(
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

                    momentum.mul_(beta1).add_(grad_nat, alpha=1.0 - beta1)

                    if moment_centralization != 0:
                        momentum_cent = momentum.sub(torch.mean(momentum).mul_(moment_centralization))
                    else:
                        momentum_cent = momentum

                    if group["update_strategy"] in {"cautious", "grams"}:
                        if group["update_strategy"] == "cautious":
                            mask = (momentum_cent * grad_nat > 0).to(grad_nat.dtype)
                            mask.div_(mask.mean().clamp_(min=1e-3))
                            momentum_cent = momentum_cent * mask
                        elif group["update_strategy"] == "grams":
                            momentum_cent = torch.sign(grad_nat) * momentum_cent.abs()

                    # Compass-style amplification.
                    full_step = grad_nat.add(momentum_cent, alpha=step**momentum_lambda)

                    # Center the update vector if requested.
                    if centralization != 0 and full_step.dim() > 1:
                        full_step.sub_(
                            full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(centralization)
                        )

                    if stable_weight_decay and group["fim_mean_sqrt"] is not None:
                        swd_scaling = 1.0 / group["fim_mean_sqrt"]
                    else:
                        swd_scaling = 1.0

                    # Perform weight decay.
                    if weight_decay != 0 and weight_decouple:
                        parameter_fp32.mul_(1.0 - weight_decay * lr * swd_scaling)
                    elif weight_decay != 0:
                        grad_weights = parameter_fp32.data.div(fim_base).div_(diff_fim_base)
                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(clip, rms) / clip
                        grad_weights.div_(divisor)
                        parameter_fp32.data.add_(grad_weights, alpha=-lr * weight_decay * swd_scaling)

                    if group["update_strategy"] in {"cautious", "grams"}:
                        if group["update_strategy"] == "cautious":
                            mask = (full_step * grad_nat > 0).to(grad_nat.dtype)
                            mask.div_(mask.mean().clamp_(min=1e-3))
                            full_step = full_step * mask
                        elif group["update_strategy"] == "grams":
                            full_step.copy_(torch.sign(grad_nat) * full_step.abs())

                    # Apply the update after the strategy-specific shaping.
                    parameter_fp32.data.add_(full_step, alpha=-step_size)
                    fim.mul_(current_beta2).addcmul_(approx_grad_nat, approx_grad_nat, value=1.0 - current_beta2).clamp_(
                        -clip_lambda, clip_lambda
                    )

                if stable_weight_decay:
                    fim_sum += fim.sum()

                # Pack reduced-precision state back down after the float32 update.
                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["prev_grad"], -grad)
                    copy_stochastic_(parameter, parameter_fp32)
                else:
                    # Next step diff is -prev_grad + grad, or equivalently grad - prev_grad.
                    state["prev_grad"].copy_(-grad)

            if stable_weight_decay:
                group["fim_mean_sqrt"] = math.sqrt(fim_sum / param_size)

        return loss
