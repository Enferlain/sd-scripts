import logging
import math

import torch
from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup

from library.optimization.optimizers.utils import NORM_TYPE, UPDATE_STRATEGY, adaptive_eps, agc, copy_stochastic_, paper_orthograd
from library.optimization.optimizers.utils.orthograd import bias_rms, bias_rms_compile, zero_power_via_newton_schulz_6, zero_power_via_newton_schulz_6_compile
from library.optimization.optimizers.utils.stable_spam import stable_spam_clipping_compile_wrapper, stable_spam_clipping_impl
from library.optimization.optimizers.utils.update import apply_update_strategies


logger = logging.getLogger(__name__)


class AdEMAMix(BaseOptimizer):
    r"""Better, Faster, Older."""

    def __init__(
        self,
        params: ParamGroup,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        clip: float = 0.0,
        alpha: float = 5.0,
        t_alpha_beta3: float | None = None,
        eps: float = 1e-8,
        centralization: float = 0.0,
        cautious: bool = False,
        update_strategy: UPDATE_STRATEGY = "unmodified",
        adopt: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, "alpha")
        self.validate_non_negative(t_alpha_beta3, "t_alpha_beta3")
        self.validate_non_negative(weight_decay, "weight_decay")
        self.validate_non_negative(eps, "eps")
        self.validate_non_negative(clip, "clip")
        self.validate_non_negative(centralization, "centralization")

        if update_strategy not in {"unmodified", "cautious", "grams"}:
            raise ValueError(f"Invalid update strategy: {update_strategy}")
        if cautious:
            update_strategy = "cautious"

        defaults: Defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "clip": clip,
            "fixed_decay": fixed_decay,
            "alpha": alpha,
            "t_alpha_beta3": t_alpha_beta3,
            "eps": eps,
            "centralization": centralization,
            "cautious": cautious,
            "update_strategy": update_strategy,
            "adopt": adopt,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "AdEMAMix"

    def init_group(self, group, **kwargs) -> None:
        pass

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group["step"] = 0
            beta1, _beta2, _beta3 = group["betas"]

            for p in group["params"]:
                state = self.state[p]
                state["exp_avg"] = torch.zeros_like(p) if beta1 > 0.0 else None
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["exp_avg_slow"] = torch.zeros_like(p)

    @staticmethod
    def schedule_alpha(t_alpha_beta3: float | None, step: int, alpha: float) -> float:
        if t_alpha_beta3 is None:
            return alpha
        return min(step * alpha / t_alpha_beta3, alpha)

    @staticmethod
    def schedule_beta3(t_alpha_beta3: float | None, step: int, beta1: float, beta3: float, eps: float) -> float:
        if t_alpha_beta3 is None:
            return beta3

        log_beta1, log_beta3 = math.log(beta1 + eps), math.log(beta3)
        return min(
            math.exp(log_beta1 * log_beta3 / ((1.0 - step / t_alpha_beta3) * log_beta3 + (step / t_alpha_beta3) * log_beta1)),
            beta3,
        )

    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        return x.norm(2) / math.sqrt(x.numel())

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
            beta1, beta2, beta3 = group["betas"]
            step = group["step"]

            bias_correction1: float = self.debias(beta1, step)
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, step))
            eps = group["eps"]
            clip = group["clip"]
            centralization = group["centralization"]
            adopt = group["adopt"]
            alpha_t: float = self.schedule_alpha(group["t_alpha_beta3"], step, group["alpha"])
            beta3_t: float = self.schedule_beta3(group["t_alpha_beta3"], step, beta1, beta3, eps)

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                p_fp32 = p
                grad = p.grad
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    p_fp32 = p.to(torch.float32)

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p) if beta1 > 0.0 else None
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["exp_avg_slow"] = torch.zeros_like(p)

                if centralization > 0.0 and grad.dim() > 1:
                    grad.sub_(grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization))
                if clip > 0.0:
                    grad.div_(((self.get_rms(grad) + eps) / clip).clamp_(min=1.0))

                exp_avg, exp_avg_sq, exp_avg_slow = state["exp_avg"], state["exp_avg_sq"], state["exp_avg_slow"]
                if p.dtype in {torch.float16, torch.bfloat16}:
                    if beta1 > 0.0:
                        exp_avg = exp_avg.to(torch.float32)
                    exp_avg_sq, exp_avg_slow = exp_avg_sq.to(torch.float32), exp_avg_slow.to(torch.float32)

                if adopt and step == 0:
                    exp_avg_sq.add_(grad)
                else:
                    if not adopt:
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                        denominator = (exp_avg_sq.sqrt() / bias_correction2_sq).add_(eps)
                    else:
                        denominator = exp_avg_sq.sqrt().add_(eps)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                        adopt_clip: float = (step - 1) ** 0.25
                        scaled_adopt_clip = adopt_clip * denominator
                        grad = grad.clamp(-scaled_adopt_clip, scaled_adopt_clip)

                    if beta1 > 0.0:
                        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    else:
                        exp_avg = grad

                    exp_avg_slow.mul_(beta3_t).add_(grad, alpha=1.0 - beta3_t)
                    update = exp_avg.div(bias_correction1) + alpha_t * exp_avg_slow

                    if group["update_strategy"] == "cautious":
                        mask = (update * grad > 0).to(grad.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                        update = update * mask
                    elif group["update_strategy"] == "grams":
                        update.copy_(torch.sign(grad) * update.abs())

                    update = update / denominator
                    self.apply_weight_decay(
                        p=p_fp32,
                        grad=update,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        weight_decouple=group["weight_decouple"],
                        fixed_decay=group["fixed_decay"],
                    )
                    p_fp32.add_(-group["lr"] * update)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    if beta1 > 0.0:
                        copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    copy_stochastic_(state["exp_avg_slow"], exp_avg_slow)
                    copy_stochastic_(p, p_fp32)

        return loss


class SimplifiedAdEMAMix(BaseOptimizer):
    r"""Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants."""

    def __init__(
        self,
        params: ParamGroup,
        lr: float | torch.Tensor = 1e-4,
        betas: Betas = (0.99, 0.95),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        alpha: float = 1.0,
        beta1_warmup: int | None = None,
        min_beta1: float = 0.9,
        eps: float = 1e-8,
        eps2: float = 1e-2,
        eps_floor: float | None = None,
        use_orthograd: bool = False,
        adaptive_clip: float | None = None,
        adaptive_clip_eps: float = 1e-3,
        adaptive_clip_type: NORM_TYPE = "layer",
        update_strategy: UPDATE_STRATEGY = "unmodified",
        bias_correction1: bool = False,
        bias_correction2: bool = True,
        use_stable_spam_clipping: bool = False,
        use_adopt: bool = False,
        torch_compile: bool = False,
        sync_chunk_size: int = 128,
        state_storage_dtype: str | torch.dtype = torch.bfloat16,
        state_storage_device: str | torch.device = "cpu",
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, "alpha")
        self.validate_non_negative(min_beta1, "min_beta1")
        self.validate_non_negative(weight_decay, "weight_decay")
        self.validate_non_negative(eps, "eps")

        for key in kwargs:
            logger.warning("Unrecognized optimizer argument '%s'. It will be ignored.", key)

        if isinstance(state_storage_dtype, str):
            normalized = state_storage_dtype.strip().lower()
            if normalized == "float32":
                final_dtype = torch.float32
            elif normalized == "float16":
                final_dtype = torch.float16
            else:
                final_dtype = torch.bfloat16
        else:
            final_dtype = state_storage_dtype

        self.sync_chunk_size = sync_chunk_size
        self.state_storage_dtype = final_dtype
        self.state_storage_device = state_storage_device

        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = torch.finfo(torch.float32).tiny
        if update_strategy not in {"unmodified", "cautious", "grams", "both"}:
            raise ValueError(f"Invalid update strategy: {update_strategy}")

        defaults: Defaults = {
            "lr": lr,
            "betas": betas,
            "alpha": alpha,
            "beta1_warmup": beta1_warmup,
            "min_beta1": min_beta1,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "fixed_decay": fixed_decay,
            "eps": eps,
            "eps2": eps2,
            "eps_floor": eps_floor,
            "use_orthograd": use_orthograd,
            "adaptive_clip": adaptive_clip,
            "adaptive_clip_eps": adaptive_clip_eps,
            "adaptive_clip_type": adaptive_clip_type,
            "update_strategy": update_strategy,
            "bias_correction1": bias_correction1,
            "bias_correction2": bias_correction2,
            "use_stable_spam_clipping": use_stable_spam_clipping,
            "use_adopt": use_adopt,
            "torch_compile": torch_compile,
            "sync_chunk_size": sync_chunk_size,
            "state_storage_dtype": final_dtype,
            "state_storage_device": state_storage_device,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "SimplifiedAdEMAMix"

    def init_group(self, group, **kwargs) -> None:
        pass

    def reset(self):
        pass

    @staticmethod
    def linear_hl_warmup_scheduler(step: int, beta_end: float, beta_start: float = 0.0, warmup: int = 1) -> float:
        def f(beta: float, eps: float = 1e-8) -> float:
            return math.log(0.5) / math.log(beta + eps) - 1.0

        def f_inv(t: float) -> float:
            return math.pow(0.5, 1.0 / (t + 1))

        if step < warmup:
            a: float = step / float(warmup)
            return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))
        return beta_end

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
            adopt_clip: float = (group["step"] - 1) ** 0.25
            beta1, beta2 = group["betas"]
            use_stable_spam_clipping = group["use_stable_spam_clipping"]
            apply_ortho_to_group = group.get("is_ortho_group", False)

            if group["beta1_warmup"]:
                beta1 = self.linear_hl_warmup_scheduler(group["step"], beta_end=beta1, beta_start=group["min_beta1"], warmup=group["beta1_warmup"])

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                device = p.device
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p.data, dtype=self.state_storage_dtype, device=self.state_storage_device)
                    state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=self.state_storage_dtype, device=self.state_storage_device)
                    if self.state_storage_device == "cpu":
                        state["exp_avg"] = state["exp_avg"].pin_memory()
                        state["exp_avg_sq"] = state["exp_avg_sq"].pin_memory()
                    state["num_sum"] = 0.0
                    state["den_sum"] = 0.0

                compute_device = torch.cuda.current_device() if device.type == "cpu" else device
                exp_avg = state["exp_avg"].to(compute_device, non_blocking=True, dtype=torch.float32)
                exp_avg_sq = state["exp_avg_sq"].to(compute_device, non_blocking=True, dtype=torch.float32)
                grad = grad.to(torch.float32).to(compute_device, non_blocking=True)
                p_fp32 = p.to(compute_device, dtype=torch.float32, non_blocking=True)

                if apply_ortho_to_group and group["use_orthograd"]:
                    paper_orthograd(param=p_fp32, grad=grad)
                if group["adaptive_clip"] is not None and group["adaptive_clip"] > 0:
                    grad = agc(p=p_fp32, grad=grad, agc_clip_val=group["adaptive_clip"], agc_eps=group["adaptive_clip_eps"], norm_type=group["adaptive_clip_type"])
                if use_stable_spam_clipping:
                    if group["torch_compile"]:
                        grad = stable_spam_clipping_compile_wrapper(state, grad, step=group["step"])
                    else:
                        grad = stable_spam_clipping_impl(state, grad, step=group["step"])

                curr_eps = adaptive_eps(grad, group)
                if group["use_adopt"] and group["step"] == 1:
                    exp_avg_sq.addcmul_(grad, grad)
                else:
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    state["num_sum"] = beta1 * state["num_sum"] + 1.0
                    state["den_sum"] = beta2 * state["den_sum"] + (1.0 - beta2)

                    if group["use_adopt"]:
                        denominator = exp_avg_sq.sqrt().add_(math.sqrt(state["den_sum"]) * curr_eps)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    else:
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                        denominator = exp_avg_sq.sqrt().add_(math.sqrt(state["den_sum"]) * curr_eps)

                    update = group["alpha"] * grad + exp_avg
                    if group["update_strategy"] in {"cautious", "grams", "both"}:
                        if group["update_strategy"] in {"cautious", "both"}:
                            mask = (update * grad > 0).to(grad.dtype)
                            mask.div_(mask.mean().clamp_(min=1e-3))
                            update = update * mask
                        if group["update_strategy"] in {"grams", "both"}:
                            update.copy_(torch.sign(grad) * update.abs())

                    update.div_(denominator)
                    if group["bias_correction1"]:
                        update.div_(state["num_sum"])
                    if group["bias_correction2"]:
                        update.mul_(math.sqrt(state["den_sum"]))
                    if group["use_adopt"]:
                        update.clamp_(-adopt_clip, adopt_clip)

                    self.apply_weight_decay(
                        p=p_fp32,
                        grad=grad,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        weight_decouple=group["weight_decouple"],
                        fixed_decay=group["fixed_decay"],
                    )
                    p_fp32.add_(update, alpha=-group["lr"])

                    if device.type == "cpu":
                        if p.dtype == torch.bfloat16:
                            copy_stochastic_(p.data, p_fp32)
                        else:
                            p.data.copy_(p_fp32)
                    else:
                        if p.dtype == torch.bfloat16:
                            copy_stochastic_(p, p_fp32)
                        else:
                            p.data.copy_(p_fp32, non_blocking=True)
                    if self.state_storage_dtype == torch.bfloat16:
                        copy_stochastic_(state["exp_avg"], exp_avg)
                        copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                    else:
                        state["exp_avg"].copy_(exp_avg, non_blocking=True)
                        state["exp_avg_sq"].copy_(exp_avg_sq, non_blocking=True)

                if (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            torch.cuda.synchronize()

        return loss


class SimplifiedAdEMAMixExM(BaseOptimizer):
    r"""Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants."""

    def __init__(
        self,
        params: ParamGroup,
        lr: float | torch.Tensor = 2e-4,
        betas: Betas = (0.95, 0.997),
        min_beta1: float = 0.95,
        beta1_warmup: int | None = None,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        alpha: float = 1.0,
        eps: float = 1e-8,
        eps_floor: float | None = 1e-12,
        use_orthograd: bool = True,
        update_strategy: UPDATE_STRATEGY = "unmodified",
        update_strategy_scale: float = 1.0,
        use_stable_spam_clipping: bool = True,
        use_compass: bool = False,
        use_adabelief: bool = True,
        use_newton_schulz: bool = True,
        amsgrad_min_decay_rate: float = 0.98,
        amsgrad_max_decay_rate: float = 0.98,
        torch_compile: bool = True,
        sync_chunk_size: int = 128,
        state_storage_dtype: str | torch.dtype = torch.bfloat16,
        state_storage_device: str | torch.device = "cpu",
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(alpha, "alpha")
        self.validate_non_negative(min_beta1, "min_beta1")
        self.validate_non_negative(weight_decay, "weight_decay")
        self.validate_non_negative(eps, "eps")

        if isinstance(state_storage_dtype, str):
            normalized = state_storage_dtype.strip().lower()
            if normalized == "float32":
                final_dtype = torch.float32
            elif normalized == "float16":
                final_dtype = torch.float16
            else:
                final_dtype = torch.bfloat16
        else:
            final_dtype = state_storage_dtype

        self.sync_chunk_size = sync_chunk_size
        self.state_storage_dtype = final_dtype
        self.state_storage_device = state_storage_device

        if not (0.0 <= update_strategy_scale <= 1.0):
            raise ValueError(f"update_strategy_scale ({update_strategy_scale}) must lie in [0.0, 1.0].")
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = torch.finfo(torch.float32).tiny
        if update_strategy not in {"unmodified", "cautious", "grams", "both"}:
            raise ValueError(f"Invalid update strategy: {update_strategy}")

        defaults: Defaults = {
            "lr": lr,
            "betas": betas,
            "alpha": alpha,
            "beta1_warmup": beta1_warmup,
            "min_beta1": min_beta1,
            "weight_decay": weight_decay,
            "weight_decouple": weight_decouple,
            "eps": eps,
            "eps2": 1e-2,
            "eps_floor": eps_floor,
            "use_orthograd": use_orthograd,
            "update_strategy": update_strategy,
            "update_strategy_scale": update_strategy_scale,
            "use_stable_spam_clipping": use_stable_spam_clipping,
            "use_compass": use_compass,
            "use_adabelief": use_adabelief,
            "torch_compile": torch_compile,
            "amsgrad_max_decay_rate": amsgrad_max_decay_rate,
            "amsgrad_min_decay_rate": amsgrad_min_decay_rate,
            "use_newton_schulz": use_newton_schulz,
            "sync_chunk_size": sync_chunk_size,
            "state_storage_dtype": final_dtype,
            "state_storage_device": state_storage_device,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "SimplifiedAdEMAMixExM"

    def init_group(self, group, **kwargs) -> None:
        pass

    def reset(self):
        pass

    @staticmethod
    def linear_hl_warmup_scheduler(step: int, beta_end: float, beta_start: float = 0.0, warmup: int = 1) -> float:
        def f(beta: float, eps: float = 1e-8) -> float:
            return math.log(0.5) / math.log(beta + eps) - 1.0

        def f_inv(t: float) -> float:
            return math.pow(0.5, 1.0 / (t + 1))

        if step < warmup:
            a: float = step / float(warmup)
            return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))
        return beta_end

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
            step = group["step"]
            adopt_clip: float = (step - 1) ** 0.25
            beta1, beta2 = group["betas"]
            use_stable_spam_clipping = group["use_stable_spam_clipping"]
            apply_ortho_to_group = group.get("is_ortho_group", False)
            eps_floor = group["eps_floor"]

            if group["beta1_warmup"]:
                beta1 = self.linear_hl_warmup_scheduler(step, beta_end=beta1, beta_start=group["min_beta1"], warmup=group["beta1_warmup"])

            beta2 = (beta2**step - beta2) / (beta2**step - 1.0)
            bias_correction1 = 1 - beta1**step
            bias_correction2_sqrt = (1 - beta2**step) ** 0.5

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad
                device = p.device
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    if self.state_storage_device == "cpu":
                        state["exp_avg"] = torch.zeros_like(p.data, dtype=self.state_storage_dtype, device=self.state_storage_device).pin_memory()
                        state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=self.state_storage_dtype, device=self.state_storage_device).pin_memory()
                    else:
                        state["exp_avg"] = torch.zeros_like(p.data, dtype=self.state_storage_dtype, device=self.state_storage_device)
                        state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=self.state_storage_dtype, device=self.state_storage_device)

                compute_device = torch.cuda.current_device() if device.type == "cpu" else device
                exp_avg = state["exp_avg"].to(compute_device, non_blocking=True, dtype=torch.float32)
                exp_avg_sq = state["exp_avg_sq"].to(compute_device, non_blocking=True, dtype=torch.float32)
                grad = grad.to(torch.float32).to(compute_device, non_blocking=True)
                p_fp32 = p.to(compute_device, dtype=torch.float32, non_blocking=True)

                if apply_ortho_to_group and group["use_orthograd"]:
                    paper_orthograd(param=p_fp32, grad=grad)
                if use_stable_spam_clipping:
                    if group["torch_compile"]:
                        grad = stable_spam_clipping_compile_wrapper(state, grad, step=step, eps=eps_floor)
                    else:
                        grad = stable_spam_clipping_impl(state, grad, step=step, eps=eps_floor)

                rms_grad = torch.sqrt(torch.mean(grad.pow(2)))
                curr_eps = adaptive_eps(grad, group, rms_grad=rms_grad)
                grad_normed = grad.div(rms_grad.clamp_min_(1))

                if group["use_newton_schulz"]:
                    if grad_normed.ndim > 0:
                        grad_normed = zero_power_via_newton_schulz_6_compile(grad_normed) if group["torch_compile"] else zero_power_via_newton_schulz_6(grad_normed)
                    elif grad_normed.numel() > 1:
                        grad_normed = bias_rms_compile(grad_normed) if group["torch_compile"] else bias_rms(grad_normed)

                mask = (grad_normed * exp_avg > 0).to(grad_normed.dtype)
                mask.clamp_min_(beta1)
                mask.div_(mask.mean().clamp_(min=1e-3))
                exp_avg.mul_(mask)
                exp_avg.mul_(beta1).add_(grad_normed, alpha=1.0 - beta1)

                if group["use_compass"]:
                    bias_corrected_exp_avg = exp_avg.div(bias_correction1)
                    c_t = grad_normed.add(bias_corrected_exp_avg, alpha=group["alpha"])
                else:
                    c_t = grad_normed

                if step == 1:
                    if group["use_compass"]:
                        grad_residual = c_t.add(grad_normed.add(bias_corrected_exp_avg, alpha=-1))
                    else:
                        grad_residual = grad_normed - exp_avg
                    exp_avg_sq.addcmul_(grad_residual, grad_residual)
                else:
                    denominator = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(curr_eps)
                    if group["use_adabelief"]:
                        if group["use_compass"]:
                            grad_residual = c_t.add(grad_normed.add(bias_corrected_exp_avg, alpha=-1))
                        else:
                            grad_residual = grad_normed - exp_avg
                        new_exp_avg_sq = exp_avg_sq.mul(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2)
                    else:
                        new_exp_avg_sq = exp_avg_sq.mul(beta2).addcmul_(c_t, c_t, value=1.0 - beta2)

                    torch.maximum(exp_avg_sq.mul(max(min(beta2, group["amsgrad_max_decay_rate"]), group["amsgrad_min_decay_rate"])), new_exp_avg_sq, out=exp_avg_sq)
                    update = c_t if group["use_compass"] else (group["alpha"] * grad_normed + exp_avg)
                    update = apply_update_strategies(update, grad_normed, group["update_strategy"], group["update_strategy_scale"])
                    update.div_(denominator)

                    if not group["use_compass"]:
                        update.div_(bias_correction1)
                    update.clamp_(-adopt_clip, adopt_clip)

                    self.apply_weight_decay(
                        p=p_fp32,
                        grad=grad_normed,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        weight_decouple=group["weight_decouple"],
                        fixed_decay=False,
                    )
                    p_fp32.add_(update, alpha=-group["lr"])

                if device.type == "cpu":
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p.data, p_fp32)
                    else:
                        p.data.copy_(p_fp32)
                else:
                    if p.dtype == torch.bfloat16:
                        copy_stochastic_(p, p_fp32)
                    else:
                        p.data.copy_(p_fp32, non_blocking=True)

                if self.state_storage_dtype == torch.bfloat16:
                    copy_stochastic_(state["exp_avg"], exp_avg)
                    copy_stochastic_(state["exp_avg_sq"], exp_avg_sq)
                else:
                    state["exp_avg"].copy_(exp_avg, non_blocking=True)
                    state["exp_avg_sq"].copy_(exp_avg_sq, non_blocking=True)

                if (i + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize()

            torch.cuda.synchronize()

        return loss
