import torch
from torch.optim import Optimizer

from library.optimization.stochastic import copy_stochastic_


class ScheduleFreeWrapper(Optimizer):
    r"""
    Wrap any optimizer to make it Schedule-Free.

    This version uses a memory-efficient swap operation but may be slower than
    the reference version. In most cases the performance difference is
    negligible. For the best possible performance and memory usage,
    Schedule-Free needs to be directly integrated with the base optimizer.

    When using this version, you can disable the base optimizer's momentum, as
    it's no longer necessary when using our wrapper's momentum (although you
    can use both types of momentum if you want).

    If you set weight decay on the base optimizer, it computes weight decay at
    ``z``. We offer the option to compute weight decay at ``y`` via the
    ``weight_decay_at_y`` parameter, which seems to give better results in our
    experiments. This approach to decay only works correctly if the base
    optimizer uses ``group["lr"]`` as the current learning rate.
    """

    @torch.no_grad()
    def __init__(
        self,
        optimizer: Optimizer,
        momentum: float = 0.9,
        weight_decay_at_y: float = 0.0,
        weight_lr_power: float = 2.0,
        r: float = 0.0,
    ) -> None:
        if weight_decay_at_y < 0.0:
            raise ValueError("weight_decay_at_y must be non-negative")
        if momentum < 0.0:
            raise ValueError("momentum must be non-negative")
        if weight_lr_power < 0.0:
            raise ValueError("weight_lr_power must be non-negative")
        if r < 0.0:
            raise ValueError("r must be non-negative")

        self.base_optimizer = optimizer
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay_at_y = weight_decay_at_y
        self.weight_lr_power = weight_lr_power
        self.r = r
        self.train_mode = False

        defaults = {
            "momentum": momentum,
            "weight_decay_at_y": weight_decay_at_y,
            "weight_lr_power": weight_lr_power,
            "r": r,
        }
        self._initializing = True
        super().__init__(optimizer.param_groups, defaults)
        self._initializing = False
        self.param_groups = self.base_optimizer.param_groups

    def __str__(self) -> str:
        return "ScheduleFreeWrapper"

    def add_param_group(self, param_group):
        if getattr(self, "_initializing", False):
            Optimizer.add_param_group(self, param_group)
            return

        self.base_optimizer.add_param_group(param_group)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group["sf_step"] = 0

            for parameter in group["params"]:
                state = self.state[parameter]
                state["z"] = torch.clone(parameter, memory_format=torch.preserve_format)

        reset_fn = getattr(self.base_optimizer, "reset", None)
        if reset_fn is not None:
            reset_fn()

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if self.train_mode:
                for parameter in group["params"]:
                    state = self.state[parameter]
                    if "z" not in state:
                        continue

                    parameter_fp32 = parameter
                    z = state["z"]

                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        z = z.to(torch.float32)
                        parameter_fp32 = parameter.to(dtype=torch.float32, copy=True)

                    parameter_fp32.data.lerp_(end=z, weight=1.0 - 1.0 / self.momentum)

                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(parameter, parameter_fp32)

                self.train_mode = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if not self.train_mode:
                for parameter in group["params"]:
                    state = self.state[parameter]
                    if "z" not in state:
                        continue

                    parameter_fp32 = parameter
                    z = state["z"]

                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        z = z.to(torch.float32)
                        parameter_fp32 = parameter.to(dtype=torch.float32, copy=True)

                    parameter_fp32.data.lerp_(end=z, weight=1.0 - self.momentum)

                    if parameter.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(parameter, parameter_fp32)

                self.train_mode = True

    @staticmethod
    def swap(x: torch.Tensor, y: torch.Tensor):
        # Convert to uint8 while preserving dimensions by viewing as bytes.
        x_bytes = x.view(-1).view(torch.uint8)
        y_bytes = y.view(-1).view(torch.uint8)

        # Perform bitwise XOR operations.
        x_bytes.bitwise_xor_(y_bytes)
        y_bytes.bitwise_xor_(x_bytes)
        x_bytes.bitwise_xor_(y_bytes)

    @torch.no_grad()
    def step(self, closure=None):
        if not self.train_mode:
            raise RuntimeError(
                "Optimizer was not in train mode when step is called. "
                "Please insert .train() and .eval() calls on the optimizer."
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            group["sf_step"] = group.get("sf_step", 0) + 1

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                if "z" not in state:
                    state["z"] = torch.clone(parameter, memory_format=torch.preserve_format)

                z = state["z"]
                parameter_fp32 = parameter

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    parameter_fp32 = parameter.to(dtype=torch.float32, copy=True)
                    z = z.to(torch.float32)

                if self.weight_decay_at_y != 0.0:
                    z.sub_(parameter_fp32, alpha=lr * self.weight_decay_at_y)
                    parameter_fp32.sub_(parameter_fp32, alpha=lr * self.weight_decay_at_y * (1.0 - self.momentum))

                parameter_fp32.lerp_(end=z, weight=1.0 - 1.0 / self.momentum)

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["z"], z)
                    copy_stochastic_(parameter, parameter_fp32)

                z = state["z"]
                self.swap(z, parameter)

        self.base_optimizer.step()

        for group in self.param_groups:
            lr = max(group["lr"] * 1.0, 1e-8)
            lr_max = group["lr_max"] = max(lr, group.get("lr_max", 0.0))

            weight = (group["sf_step"] ** self.r) * (lr_max**self.weight_lr_power)
            weight_sum = group["sf_weight_sum"] = group.get("sf_weight_sum", 0.0) + weight
            ckp1 = weight / weight_sum

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                z = state["z"]
                self.swap(z, parameter)

                parameter_fp32 = parameter
                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    parameter_fp32 = parameter.to(dtype=torch.float32, copy=True)

                parameter_fp32.lerp_(end=z.to(torch.float32), weight=ckp1)
                parameter_fp32.lerp_(end=state["z"].to(torch.float32), weight=1.0 - self.momentum)

                if parameter.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(parameter, parameter_fp32)

        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "wrapper_state": super().state_dict(),
            "train_mode": self.train_mode,
        }

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.param_groups = self.base_optimizer.param_groups
        super().load_state_dict(state_dict["wrapper_state"])
        self.base_optimizer.param_groups = self.param_groups
        self.train_mode = state_dict.get("train_mode", False)
