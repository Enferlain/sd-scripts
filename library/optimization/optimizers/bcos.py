# Source: https://github.com/facebookresearch/bcos

import torch
from torch.optim import Optimizer

from library.optimization.optimizers.utils import copy_stochastic_, resolve_state_storage_dtype


class BCOS(Optimizer):
    r"""Repo-owned BCOS optimizer adapted for the optimization layer.

    BCOS supports three denominator-estimation modes:
    - ``g``: gradient EMA only
    - ``m``: momentum plus EMA variance
    - ``c``: conditional variance estimate using the momentum state
    """

    def __init__(
        self,
        params,
        lr,
        beta=0.9,
        beta2=None,
        eps=1e-6,
        weight_decay=0.1,
        mode="c",
        decouple_wd=True,
        simple_cond=False,
        sync_chunk_size: int = 128,
        state_storage_dtype: str | torch.dtype = torch.bfloat16,
        state_storage_device: str | torch.device = "cpu",
        **kwargs,
    ):
        del kwargs

        if mode not in {"g", "m", "c"}:
            raise ValueError(f"BCOS mode {mode} not supported")

        final_dtype = resolve_state_storage_dtype(state_storage_dtype)
        defaults = {
            "lr": lr,
            "beta": beta,
            "beta2": beta2,
            "eps": eps,
            "wd": weight_decay,
            "sync_chunk_size": sync_chunk_size,
            "state_storage_dtype": final_dtype,
            "state_storage_device": state_storage_device,
        }
        super().__init__(params, defaults)

        self.mode = mode
        self.decouple_wd = decouple_wd  # True for BCOSW
        self.simple_cond = simple_cond  # True for simple alternative v estimator in 'c' mode
        self.sync_chunk_size = sync_chunk_size
        self.state_storage_dtype = final_dtype
        self.state_storage_device = state_storage_device

    def __str__(self) -> str:
        return "BCOS"

    def _initialize_state(self, grad: torch.Tensor) -> torch.Tensor:
        state_tensor = grad.detach().to(dtype=self.state_storage_dtype, device=self.state_storage_device)
        if self.state_storage_device == "cpu" and torch.cuda.is_available():
            state_tensor = state_tensor.pin_memory()
        return state_tensor

    @staticmethod
    def _get_compute_device(parameter_device: torch.device) -> torch.device:
        if parameter_device.type == "cpu" and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return parameter_device

    @staticmethod
    def _copy_parameter_back(parameter: torch.Tensor, parameter_fp32: torch.Tensor) -> None:
        destination = parameter_fp32.to(parameter.device, non_blocking=parameter.device.type == "cuda")
        if parameter.dtype == torch.bfloat16:
            copy_stochastic_(parameter.data, destination)
        else:
            parameter.data.copy_(destination, non_blocking=parameter.device.type == "cuda")

    def _copy_state_back(self, state_tensor: torch.Tensor, compute_tensor: torch.Tensor) -> None:
        destination = compute_tensor.to(state_tensor.device, non_blocking=state_tensor.device.type == "cuda")
        if self.state_storage_dtype == torch.bfloat16:
            copy_stochastic_(state_tensor, destination)
        else:
            state_tensor.copy_(destination, non_blocking=state_tensor.device.type == "cuda")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
            lr = group["lr"]
            beta = group["beta"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["wd"]

            for index, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    continue

                grad = parameter.grad.data
                state = self.state[parameter]

                if self.mode in {"m", "c"} and "m" not in state:
                    state["m"] = self._initialize_state(grad)
                if self.mode in {"g", "m"} and "v" not in state:
                    state["v"] = self._initialize_state(grad)

                compute_device = self._get_compute_device(parameter.device)
                non_blocking = compute_device.type == "cuda"

                momentum = None
                variance = None
                if self.mode in {"m", "c"}:
                    momentum = state["m"].to(compute_device, non_blocking=non_blocking, dtype=torch.float32)
                if self.mode in {"g", "m"}:
                    variance = state["v"].to(compute_device, non_blocking=non_blocking, dtype=torch.float32)

                grad_fp32 = grad.to(compute_device, non_blocking=non_blocking, dtype=torch.float32)
                parameter_fp32 = parameter.detach().to(compute_device, non_blocking=non_blocking, dtype=torch.float32)

                # decoupled weight decay or absorb in gradient
                if self.decouple_wd:  # p := (1 - lr * wd) * p
                    parameter_fp32.mul_(1 - lr * weight_decay)
                else:  # g := g + wd * p
                    grad_fp32.add_(parameter_fp32, alpha=weight_decay)

                if self.mode in {"m", "c"}:
                    if self.mode == "c":  # conditional estimator
                        if self.simple_cond:
                            beta_variance = 1 - (1 - beta) ** 2 if beta2 is None else beta2
                            variance = beta_variance * momentum.square() + (1 - beta_variance) * grad_fp32.square()
                        else:
                            variance = (
                                (3 * beta**2 - 2 * beta**3) * momentum.square()
                                + (1 - beta) ** 2 * grad_fp32.square()
                                + 2 * beta * (1 - beta) ** 2 * momentum * grad_fp32
                            )

                    # update momentum
                    momentum.mul_(beta).add_(grad_fp32, alpha=1 - beta)
                    direction = momentum
                else:
                    direction = grad_fp32

                if self.mode in {"g", "m"}:  # EMA estimator
                    beta_variance = beta if beta2 is None else beta2
                    variance.mul_(beta_variance).add_(direction.square(), alpha=1 - beta_variance)

                # BCOS update: p := p - lr * (d / (sqrt(v) + eps))
                parameter_fp32.add_(direction.div(variance.sqrt().add_(eps)), alpha=-lr)

                self._copy_parameter_back(parameter, parameter_fp32)
                if self.mode in {"m", "c"}:
                    self._copy_state_back(state["m"], momentum)
                if self.mode in {"g", "m"}:
                    self._copy_state_back(state["v"], variance)

                if compute_device.type == "cuda" and (index + 1) % self.sync_chunk_size == 0:
                    torch.cuda.synchronize(compute_device)

            if any(parameter.device.type == "cuda" for parameter in group["params"]) or (
                torch.cuda.is_available() and any(parameter.device.type == "cpu" for parameter in group["params"])
            ):
                compute_device = self._get_compute_device(group["params"][0].device)
                if compute_device.type == "cuda":
                    torch.cuda.synchronize(compute_device)

        return loss
