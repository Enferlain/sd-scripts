import torch


class WrappedOptimizerProxy(torch.optim.Optimizer):
    """Expose wrapper objects through a stable Optimizer-shaped surface."""

    def __init__(self, wrapped, base_optimizer):
        self._wrapped = wrapped
        self.base_optimizer = base_optimizer

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    @property
    def state(self):
        return self._wrapped.state

    @state.setter
    def state(self, state):
        self._wrapped.state = state

    @property
    def param_groups(self):
        return self._wrapped.param_groups

    @param_groups.setter
    def param_groups(self, param_groups):
        self._wrapped.param_groups = param_groups

    @property
    def defaults(self):
        return self._wrapped.defaults

    @defaults.setter
    def defaults(self, defaults):
        self._wrapped.defaults = defaults

    def add_param_group(self, param_group):
        self._wrapped.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self._wrapped.load_state_dict(state_dict)

    def state_dict(self):
        return self._wrapped.state_dict()

    def zero_grad(self, set_to_none: bool = False):
        return self._wrapped.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        return self._wrapped.step(closure)

    def train(self):
        train_fn = getattr(self._wrapped, "train", None)
        if train_fn is not None:
            return train_fn()
        return None

    def eval(self):
        eval_fn = getattr(self._wrapped, "eval", None)
        if eval_fn is not None:
            return eval_fn()
        return None


class SNOOASGD(torch.optim.Optimizer):
    """Averaged-SGD wrapper adapted into the repo-owned optimization layer."""

    @torch.no_grad()
    def __init__(
        self,
        optimizer,
        lr: float = 1.0,
        alpha: float = 0.75,
        lambd: float | None = None,
        t0: int = 0,
        accelerate_k: int = 20,
        accelerate_lr: float = 0.5,
        accelerate_momentum: float = 0.5,
        accelerate_nesterov: bool = True,
    ) -> None:
        self.base_optimizer = optimizer
        self.optimizer = optimizer
        self.lr = lr
        self.alpha = alpha
        self.lambd = lambd
        self.t0 = t0
        self.accelerate_k = accelerate_k
        self.accelerate_lr = accelerate_lr
        self.accelerate_momentum = accelerate_momentum
        self.accelerate_nesterov = accelerate_nesterov
        self.current_step = 0
        self.n_averaged = 0
        self.model_params = None
        self.averaged_params_cpu = None
        self.non_averaged_params_cpu = None
        self.nesterov_params_cpu = None
        self.nesterov_buffer_cpu = None
        self.is_swapped = False

    @property
    def state(self):
        return self.base_optimizer.state

    @state.setter
    def state(self, state):
        self.base_optimizer.state = state

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups):
        self.base_optimizer.param_groups = param_groups

    @property
    def defaults(self):
        return self.base_optimizer.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.base_optimizer.defaults = defaults

    def add_param_group(self, param_group):
        self.base_optimizer.add_param_group(param_group)

    @torch.no_grad()
    def _initialize_state(self):
        params = [p for pg in self.param_groups for p in pg["params"] if isinstance(p, torch.Tensor)]
        if not params:
            return

        self.model_params = list(params)
        self.averaged_params_cpu = [p.clone().to("cpu") for p in self.model_params]
        self.non_averaged_params_cpu = [p.clone().to("cpu") for p in self.model_params]
        self.nesterov_params_cpu = [p.clone().to("cpu") for p in self.model_params] if self.accelerate_k > 0 else None
        self.nesterov_buffer_cpu = [torch.zeros_like(p).to("cpu") for p in self.model_params] if self.accelerate_k > 0 else None

    @torch.no_grad()
    def step(self, closure=None):
        if self.averaged_params_cpu is None:
            if self.param_groups:
                self._initialize_state()
            if self.averaged_params_cpu is None:
                return self.base_optimizer.step(closure)

        if self.is_swapped:
            self._swap_parameters()

        loss = self.base_optimizer.step(closure)
        self.current_step += 1

        if self.current_step >= self.t0:
            self.n_averaged += 1
            decay = self.lambd if self.lambd is not None else self.lr / (self.n_averaged**self.alpha)

            for p_gpu, p_avg_cpu in zip(self.model_params, self.averaged_params_cpu, strict=True):
                delta = p_gpu.data.to("cpu", non_blocking=True) - p_avg_cpu.data
                p_avg_cpu.data.add_(delta, alpha=decay)
        else:
            for p_gpu, p_avg_cpu in zip(self.model_params, self.averaged_params_cpu, strict=True):
                p_avg_cpu.data.copy_(p_gpu.data, non_blocking=True)

        if self.accelerate_k > 0 and self.current_step % self.accelerate_k == 0:
            for p_nesterov_cpu, buffer_nesterov_cpu, p_gpu in zip(
                self.nesterov_params_cpu,
                self.nesterov_buffer_cpu,
                self.model_params,
                strict=True,
            ):
                delta = p_gpu.data.to("cpu", non_blocking=True) - p_nesterov_cpu.data
                if self.accelerate_nesterov:
                    buffer_nesterov_cpu.data.mul_(self.accelerate_momentum).add_(delta)
                    grad = (
                        buffer_nesterov_cpu.data.mul(self.accelerate_momentum).add_(delta).mul_(1.0 - self.accelerate_momentum).to(
                            p_gpu.data.device,
                            non_blocking=True,
                        )
                    )
                else:
                    buffer_nesterov_cpu.data.lerp_(delta, weight=1.0 - self.accelerate_momentum)
                    grad = delta.lerp(buffer_nesterov_cpu.data, weight=self.accelerate_momentum).to(
                        p_gpu.data.device,
                        non_blocking=True,
                    )

                p_nesterov_cpu.data.copy_(p_gpu.data, non_blocking=True)
                p_gpu.data.add_(grad, alpha=self.accelerate_lr)

        if not self.is_swapped:
            self._swap_parameters()

        return loss

    def zero_grad(self, set_to_none: bool = False):
        if self.is_swapped:
            self._swap_parameters()
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _swap_parameters(self):
        if self.averaged_params_cpu is None:
            return

        if not self.is_swapped:
            for p_gpu, p_non_avg_cpu in zip(self.model_params, self.non_averaged_params_cpu, strict=True):
                p_non_avg_cpu.copy_(p_gpu.data, non_blocking=True)
            for p_gpu, p_avg_cpu in zip(self.model_params, self.averaged_params_cpu, strict=True):
                p_gpu.copy_(p_avg_cpu.data, non_blocking=True)
            self.is_swapped = True
            return

        for p_gpu, p_non_avg_cpu in zip(self.model_params, self.non_averaged_params_cpu, strict=True):
            p_gpu.copy_(p_non_avg_cpu.data, non_blocking=True)
        self.is_swapped = False

    def state_dict(self):
        return {
            "inner_optimizer": self.base_optimizer.state_dict(),
            "asgd_wrapper": {
                "t0": self.t0,
                "lr": self.lr,
                "alpha": self.alpha,
                "lambd": self.lambd,
                "accelerate_k": self.accelerate_k,
                "accelerate_lr": self.accelerate_lr,
                "accelerate_momentum": self.accelerate_momentum,
                "accelerate_nesterov": self.accelerate_nesterov,
                "current_step": self.current_step,
                "n_averaged": self.n_averaged,
                "averaged_params_cpu": self.averaged_params_cpu,
                "non_averaged_params_cpu": self.non_averaged_params_cpu,
                "nesterov_params_cpu": self.nesterov_params_cpu if self.accelerate_k > 0 else None,
                "nesterov_buffer_cpu": self.nesterov_buffer_cpu if self.accelerate_k > 0 else None,
                "is_swapped": self.is_swapped,
            },
        }

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict["inner_optimizer"])
        wrapper_state = state_dict["asgd_wrapper"]

        if self.averaged_params_cpu is None and self.param_groups:
            self._initialize_state()

        self.t0 = wrapper_state.get("t0", 0)
        self.lr = wrapper_state.get("lr", 1.0)
        self.alpha = wrapper_state.get("alpha", 0.75)
        self.lambd = wrapper_state.get("lambd")
        self.accelerate_k = wrapper_state.get("accelerate_k", 20)
        self.accelerate_lr = wrapper_state.get("accelerate_lr", 0.5)
        self.accelerate_momentum = wrapper_state.get("accelerate_momentum", 0.5)
        self.accelerate_nesterov = wrapper_state.get("accelerate_nesterov", True)
        self.current_step = wrapper_state["current_step"]
        self.n_averaged = wrapper_state["n_averaged"]
        self.averaged_params_cpu = wrapper_state["averaged_params_cpu"]
        self.non_averaged_params_cpu = wrapper_state["non_averaged_params_cpu"]
        self.nesterov_params_cpu = wrapper_state["nesterov_params_cpu"]
        self.nesterov_buffer_cpu = wrapper_state["nesterov_buffer_cpu"]
        self.is_swapped = wrapper_state["is_swapped"]

        if self.is_swapped:
            for p_gpu, p_avg_cpu in zip(self.model_params, self.averaged_params_cpu, strict=True):
                p_gpu.copy_(p_avg_cpu.data, non_blocking=True)
