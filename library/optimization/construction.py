"""Building an optimizer by its registered name, loading only the optimizer selected (D21).

A strategy names its unit's optimizer and settings, such as
``@adamw8bit(lr 1e-05)``; Trainer optimization builds it here on the unit's
resolved parameters. Each registration names its class by import path, which
is imported only when that optimizer is built: some optimizer libraries turn
the GIL on when they load, as bitsandbytes does through Triton, and only a
run that selects one of them pays for that (D15).
"""

import importlib
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from library.optimization.registry import get_optimizer_registration


def optimizer_class(name: str) -> type[torch.optim.Optimizer]:
    """The class registered under the name, imported now; raises ValueError for a name that registers no plain optimizer."""
    registration = get_optimizer_registration(name)
    if registration is None or registration.target is None:
        raise ValueError(f"no optimizer is registered as {name!r}")
    if registration.kind != "optimizer":
        raise ValueError(f"{name!r} is registered as a {registration.kind}, and only a plain optimizer is built by name so far")
    module, _, attribute = registration.target.rpartition(".")
    found = getattr(importlib.import_module(module), attribute)
    if not (isinstance(found, type) and issubclass(found, torch.optim.Optimizer)):
        raise ValueError(f"{registration.target} is not an optimizer class")
    return found


def build(name: str, parameters: Iterable[torch.nn.Parameter], settings: Mapping[str, Any]) -> torch.optim.Optimizer:
    """The named optimizer over the parameters, in their order, with the settings as its keyword arguments."""
    built: Any = optimizer_class(name)
    return built(list(parameters), **settings)


__all__ = ["build", "optimizer_class"]
