import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import adaptive_eps, copy_stochastic_

try:
    from bitsandbytes.functional import dequantize_blockwise, quantize_blockwise
except ImportError:  # pragma: no cover - exercised through loader/registry import handling
    dequantize_blockwise = None
    quantize_blockwise = None


class FishMonger(Optimizer):
    r"""
    FishMonger: Screw it, fisher everything.

    Tracks a fast momentum/FIM path plus a slower natural-gradient path, with optional differential amplification
    between successive gradients.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.99, 0.999),
        eps: float = 1e-8,
        eps2: float = 0.01,
        eps_floor: float | None = 1e-16,
        weight_decay: float = 0.0,
        clip: float = 1.0,
        centralization: float = 1.0,
        diff_amp: float = 1.0,
        diff_amp_beta: float = 0.999,
        **kwargs,
    ):
        del kwargs

        # The donor treats non-positive floors as numerically dangerous and lifts them.
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "eps2": eps2,
            "eps_floor": eps_floor,
            "weight_decay": weight_decay,
            "clip": clip,
            "centralization": centralization,
            "diff_amp": diff_amp,
            "diff_amp_beta": diff_amp_beta,
        }

        self.eps = eps
        self.eps2 = eps2
        self.eps_floor = eps_floor

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "FishMonger"

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1

            beta1, beta2, beta3 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            clip = 1.0 if group["clip"] <= 0 else group["clip"]
            centralization = group["centralization"]
            diff_amp = group["diff_amp"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("FishMonger does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(param.data)
                    state["momentum_slow"] = torch.zeros_like(param.data)
                    state["momentum_slow_squared"] = torch.zeros_like(param.data)
                    state["fim"] = torch.ones_like(param.data)
                    if diff_amp:
                        state["ema_diff"] = torch.zeros_like(param.data)
                        state["previous_grad"] = grad.data.clone().mul_(-1.0)

                if param.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32).data
                    momentum = state["momentum"].to(torch.float32)
                    momentum_slow = state["momentum_slow"].to(torch.float32)
                    momentum_slow_squared = state["momentum_slow_squared"].to(torch.float32)
                    fim = state["fim"].to(torch.float32)
                    param_fp32 = param.detach().clone().to(torch.float32)
                    ema_diff = state["ema_diff"].to(torch.float32) if diff_amp else 0
                else:
                    grad = grad.data
                    momentum = state["momentum"]
                    momentum_slow = state["momentum_slow"]
                    momentum_slow_squared = state["momentum_slow_squared"]
                    fim = state["fim"]
                    ema_diff = state["ema_diff"] if diff_amp else 0

                if diff_amp:
                    if param.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff = state["previous_grad"].to(torch.float32)
                    else:
                        grad_diff = state["previous_grad"]
                    grad_diff.add_(grad)

                    ema_diff.mul_(group["diff_amp_beta"]).add_(grad_diff, alpha=1 - group["diff_amp_beta"])

                    if param.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["previous_grad"], -grad)
                    else:
                        state["previous_grad"].copy_(-grad)

                momentum.mul_(beta1).add_(grad, alpha=1 - beta1)

                fim_beta = (beta2**group["step"] - beta2) / (beta2**group["step"] - 1.0)
                bias_correction = 1 - beta1**group["step"]

                fim.mul_(fim_beta).addcmul_(momentum, momentum, value=1 - fim_beta)

                curr_eps = adaptive_eps(grad, group)
                fim_base = fim.sqrt() + curr_eps

                grad_nat = momentum / fim_base
                rms = grad_nat.pow(2).mean().sqrt_().add_(curr_eps)
                divisor = max(1, rms) / clip
                grad_nat.div_(divisor)

                momentum_slow.mul_(beta1).add_(grad_nat, alpha=1 - beta1)

                squared_fim_beta = (beta3**group["step"] - beta3) / (beta3**group["step"] - 1.0)
                momentum_slow_squared.mul_(squared_fim_beta).addcmul_(momentum_slow, momentum_slow, value=1 - squared_fim_beta)

                fim_slow_base = momentum_slow_squared.sqrt() + curr_eps
                grad_nat_2 = grad / fim_base / fim_slow_base

                rms = grad_nat_2.pow(2).mean().sqrt_().add_(curr_eps)
                divisor = max(1, rms) / clip
                grad_nat_2.div_(divisor)

                if param.dtype in {torch.float16, torch.bfloat16}:
                    grad_weights = param_fp32.data / fim_base / fim_slow_base
                else:
                    grad_weights = param.data / fim_base / fim_slow_base

                rms = grad_weights.pow(2).mean().sqrt_().add_(curr_eps)
                divisor = max(1, rms) / clip
                grad_weights.div_(divisor)

                diff_weights = ema_diff / fim_base / fim_slow_base if diff_amp else 0
                if diff_amp:
                    rms = diff_weights.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / clip
                    diff_weights.div_(divisor)

                full_step = grad_nat_2 + (weight_decay * grad_weights) - (diff_amp * diff_weights)

                if centralization != 0 and full_step.dim() > 1:
                    full_step.sub_(full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(centralization))

                if param.dtype in {torch.float16, torch.bfloat16}:
                    param_fp32.data.add_(full_step, alpha=-lr / bias_correction)
                else:
                    param.data.add_(full_step, alpha=-lr / bias_correction)

                if param.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["momentum_slow"], momentum_slow)
                    copy_stochastic_(state["momentum_slow_squared"], momentum_slow_squared)
                    copy_stochastic_(state["fim"], fim)
                    if diff_amp:
                        copy_stochastic_(state["ema_diff"], ema_diff)
                    copy_stochastic_(param, param_fp32)

        return loss


class FishMonger8BitBNB(Optimizer):
    r"""
    FishMonger8BitBNB: Screw it, fisher everything.

    Quantizes optimizer state blockwise with bitsandbytes while preserving the donor FishMonger update shape.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.99, 0.999),
        eps: float = 1e-8,
        eps2: float = 0.01,
        eps_floor: float | None = 1e-16,
        weight_decay: float = 0.0,
        clip: float = 1.0,
        centralization: float = 1.0,
        diff_amp: float = 1.0,
        diff_amp_beta: float = 0.999,
        quantization_group_size: int = 64,
        **kwargs,
    ):
        del kwargs

        if quantize_blockwise is None or dequantize_blockwise is None:
            raise ImportError("No bitsandbytes")
        if quantization_group_size not in {64, 128, 256, 512, 1024, 2048, 4096}:
            raise ValueError("FishMonger8BitBNB quantization_group_size must be one of 64, 128, 256, 512, 1024, 2048, 4096")

        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-37

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "eps2": eps2,
            "eps_floor": eps_floor,
            "weight_decay": weight_decay,
            "clip": clip,
            "centralization": centralization,
            "diff_amp": diff_amp,
            "diff_amp_beta": diff_amp_beta,
            "group_size": quantization_group_size,
        }

        self.eps = eps
        self.eps2 = eps2
        self.eps_floor = eps_floor

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return "FishMonger8BitBNB"

    @staticmethod
    def _ensure_cuda_params(param_groups) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("FishMonger8BitBNB requires CUDA-enabled bitsandbytes blockwise quantization")
        if any(param.device.type != "cuda" for group in param_groups for param in group["params"]):
            raise RuntimeError("FishMonger8BitBNB only supports CUDA parameters")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._ensure_cuda_params(self.param_groups)

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1

            beta1, beta2, beta3 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            clip = 1.0 if group["clip"] <= 0 else group["clip"]
            centralization = group["centralization"]
            diff_amp = group["diff_amp"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("FishMonger8BitBNB does not support sparse gradients")

                state = self.state[param]

                if len(state) == 0:
                    state["momentum"] = quantize_blockwise(torch.zeros_like(param.data), blocksize=group["group_size"])
                    state["momentum_slow"] = quantize_blockwise(torch.zeros_like(param.data), blocksize=group["group_size"])
                    state["momentum_slow_squared"] = quantize_blockwise(
                        torch.zeros_like(param.data),
                        blocksize=group["group_size"],
                    )
                    state["fim"] = quantize_blockwise(torch.zeros_like(param.data), blocksize=group["group_size"])
                    if diff_amp:
                        state["ema_diff"] = quantize_blockwise(torch.zeros_like(param.data), blocksize=group["group_size"])
                        state["previous_grad"] = quantize_blockwise(
                            grad.data.clone().mul_(-1.0),
                            blocksize=group["group_size"],
                        )

                if param.dtype in {torch.float16, torch.bfloat16}:
                    param_fp32 = param.to(torch.float32)
                    grad = grad.to(torch.float32).data
                else:
                    grad = grad.data

                momentum = dequantize_blockwise(*state["momentum"])
                momentum_slow = dequantize_blockwise(*state["momentum_slow"])
                momentum_slow_squared = dequantize_blockwise(*state["momentum_slow_squared"])
                fim = dequantize_blockwise(*state["fim"])
                ema_diff = dequantize_blockwise(*state["ema_diff"]) if diff_amp else 0

                if diff_amp:
                    grad_diff = dequantize_blockwise(*state["previous_grad"])
                    grad_diff.add_(grad)

                    ema_diff.mul_(group["diff_amp_beta"]).add_(grad_diff, alpha=1 - group["diff_amp_beta"])

                    state["previous_grad"] = quantize_blockwise(-grad, blocksize=group["group_size"])

                momentum.mul_(beta1).add_(grad, alpha=1 - beta1)

                fim_beta = (beta2**group["step"] - beta2) / (beta2**group["step"] - 1.0)
                bias_correction = 1 - beta1**group["step"]

                fim.mul_(fim_beta).addcmul_(momentum, momentum, value=1 - fim_beta)

                curr_eps = adaptive_eps(grad, group)
                fim_base = fim.sqrt() + curr_eps

                grad_nat = momentum / fim_base
                rms = grad_nat.pow(2).mean().sqrt_().add_(curr_eps)
                divisor = max(1, rms) / clip
                grad_nat.div_(divisor)

                momentum_slow.mul_(beta1).add_(grad_nat, alpha=1 - beta1)

                squared_fim_beta = (beta3**group["step"] - beta3) / (beta3**group["step"] - 1.0)
                momentum_slow_squared.mul_(squared_fim_beta).addcmul_(momentum_slow, momentum_slow, value=1 - squared_fim_beta)

                fim_slow_base = momentum_slow_squared.sqrt() + curr_eps
                grad_nat_2 = grad / fim_base / fim_slow_base

                rms = grad_nat_2.pow(2).mean().sqrt_().add_(curr_eps)
                divisor = max(1, rms) / clip
                grad_nat_2.div_(divisor)

                if param.dtype in {torch.float16, torch.bfloat16}:
                    grad_weights = param_fp32.data / fim_base / fim_slow_base
                else:
                    grad_weights = param.data / fim_base / fim_slow_base

                rms = grad_weights.pow(2).mean().sqrt_().add_(curr_eps)
                divisor = max(1, rms) / clip
                grad_weights.div_(divisor)

                diff_weights = ema_diff / fim_base / fim_slow_base if diff_amp else 0
                if diff_amp:
                    rms = diff_weights.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / clip
                    diff_weights.div_(divisor)

                full_step = grad_nat_2 + (weight_decay * grad_weights) - (diff_amp * diff_weights)

                if centralization > 0.0 and full_step.dim() > 1:
                    full_step.sub_(full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(centralization))

                if param.dtype in {torch.float16, torch.bfloat16}:
                    param_fp32.data.add_(full_step, alpha=-lr / bias_correction)
                else:
                    param.data.add_(full_step, alpha=-lr / bias_correction)

                state["momentum"] = quantize_blockwise(momentum, blocksize=group["group_size"])
                state["momentum_slow"] = quantize_blockwise(momentum_slow, blocksize=group["group_size"])
                state["momentum_slow_squared"] = quantize_blockwise(momentum_slow_squared, blocksize=group["group_size"])
                state["fim"] = quantize_blockwise(fim, blocksize=group["group_size"])
                if diff_amp:
                    state["ema_diff"] = quantize_blockwise(ema_diff, blocksize=group["group_size"])

                if param.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(param, param_fp32)

        return loss
