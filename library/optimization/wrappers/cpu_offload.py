from __future__ import annotations

from collections.abc import Mapping

import torch
from torchao.optim import CPUOffloadOptimizer as TorchAOCPUOffloadOptimizer
from torchao.utils import get_available_devices


def _clone_param_groups(param_groups):
    cloned_groups = []
    for group in param_groups:
        cloned_group = {key: value for key, value in group.items() if key != "params"}
        cloned_group["params"] = list(group["params"])
        cloned_groups.append(cloned_group)
    return cloned_groups


class CPUOffloadOptimizerWrapper(torch.optim.Optimizer):
    """Expose upstream TorchAO CPU offload through the repo's wrapper surface."""

    def __init__(
        self,
        optimizer,
        *,
        base_optimizer_kwargs: Mapping[str, object] | None = None,
        offload_gradients: bool = False,
        minimal_size: int = 4096,
    ) -> None:
        available_devices = get_available_devices()
        device_type = str(available_devices[-1]) if available_devices else "cpu"
        if device_type not in {"cuda", "xpu"}:
            raise RuntimeError("CPUOffloadOptimizer requires a CUDA or XPU runtime")

        optimizer_class = optimizer.__class__
        param_groups = _clone_param_groups(optimizer.param_groups)
        optimizer_kwargs = dict(base_optimizer_kwargs or {})

        self._wrapped = TorchAOCPUOffloadOptimizer(
            param_groups,
            optimizer_class=optimizer_class,
            offload_gradients=offload_gradients,
            minimal_size=minimal_size,
            **optimizer_kwargs,
        )
        self.base_optimizer = self._wrapped
        self.source_optimizer_class = optimizer_class
        self.offload_gradients = offload_gradients
        self.minimal_size = minimal_size
        self.base_optimizer_kwargs = optimizer_kwargs

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
        return getattr(self._wrapped, "defaults", {})

    @defaults.setter
    def defaults(self, defaults):
        self._wrapped.defaults = defaults

    def add_param_group(self, param_group):
        self._wrapped.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self._wrapped.load_state_dict(state_dict)

    def state_dict(self):
        return self._wrapped.state_dict()

    def zero_grad(self, set_to_none: bool = True):
        return self._wrapped.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        return self._wrapped.step(closure)

    def __str__(self) -> str:
        return "CPUOffloadOptimizer"
