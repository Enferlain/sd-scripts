from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from torch import nn


@dataclass(slots=True)
class ParameterGroup:
    """Typed optimizer parameter-group description.

    Phase 1 keeps the runtime payload compatible with the existing optimizer
    factories while giving the optimization layer one shared group shape.
    """

    params: list[Any]
    lr: float | None = None
    label: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def to_optimizer_dict(self) -> dict[str, Any]:
        group: dict[str, Any] = {"params": self.params}
        if self.lr is not None:
            group["lr"] = self.lr
        group.update(self.options)
        return group


def build_parameter_group(
    params: Iterable[Any],
    *,
    lr: float | None = None,
    label: str | None = None,
    **options: Any,
) -> ParameterGroup:
    """Build a typed parameter group from any parameter iterable."""
    return ParameterGroup(params=list(params), lr=lr, label=label, options=options)


def build_module_parameter_group(
    module: nn.Module,
    *,
    lr: float | None = None,
    label: str | None = None,
    **options: Any,
) -> ParameterGroup:
    """Build a typed parameter group directly from a module.

    The materialized optimizer dict keeps `named_params` alongside `params`
    so optimizers that need parameter-name context can still build through
    the shared factory path.
    """
    named_params = list(module.named_parameters())
    return build_parameter_group((param for _, param in named_params), lr=lr, label=label, named_params=named_params, **options)


def materialize_parameter_groups(trainable_params: Any) -> Any:
    """Convert typed parameter groups into the legacy optimizer dict payload."""
    if not isinstance(trainable_params, list):
        return trainable_params

    materialized: list[Any] = []
    for group in trainable_params:
        if isinstance(group, ParameterGroup):
            materialized.append(group.to_optimizer_dict())
        else:
            materialized.append(group)
    return materialized
