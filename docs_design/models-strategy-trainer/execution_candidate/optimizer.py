"""Eager physical optimizer for the CPU candidate, not a production API.

The installed torch.optim.Optimizer wrapper imports Dynamo/Triton even for
eager SGD. Reuse PyTorch's functional implementation without that wrapper;
this adapter offers only the plain SGD configuration the candidate already
uses. The private functional API is coupled to the repo's tested Torch stack.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import _functional


class EagerSGD:
    def __init__(self, parameters: Iterable[nn.Parameter], *, lr: float):
        if lr < 0:
            raise ValueError("Learning rate must be non-negative")
        self.parameters = tuple(parameters)
        if not self.parameters:
            raise ValueError("Optimizer needs parameters")
        self.learning_rate = lr

    @torch.no_grad()
    def step(self) -> None:
        parameters: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameters.append(parameter)
                gradients.append(parameter.grad)
        _functional.sgd(
            parameters,
            gradients,
            [],
            foreach=False,
            fused=False,
            weight_decay=0.0,
            momentum=0.0,
            lr=self.learning_rate,
            dampening=0.0,
            nesterov=False,
            maximize=False,
        )

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                if gradient.grad_fn is not None:
                    gradient.detach_()
                else:
                    gradient.requires_grad_(False)
                gradient.zero_()

    def state_dict(self) -> dict[str, object]:
        # No momentum/other numerical state exists in this offered profile.
        # Return detached configuration containers, never live parameters.
        return {"state": {}, "param_groups": [{"lr": self.learning_rate, "params": list(range(len(self.parameters)))}]}
