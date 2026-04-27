from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from torch import nn

from library.optimization.targets import OptimizationTargetRef


@dataclass(slots=True)
class AdapterTrainableParameterRef:
    """Repo-owned trainable-parameter description for adapter optimization handoff."""

    param: nn.Parameter
    name: str
    algorithm: str
    component: str
    component_key: str
    target_path: str
    source_target_ref: OptimizationTargetRef | None = None


@runtime_checkable
class AdapterTrainableParameterProvider(Protocol):
    """Adapter runtime surface for exposing trainable refs to optimization."""

    def describe_trainable_parameter_refs(self) -> list[AdapterTrainableParameterRef]: ...


def attach_trainable_parameter_provider(
    adapter: Any,
    describe_fn: Callable[[], list[AdapterTrainableParameterRef]],
) -> Any:
    """Attach a repo-owned trainable-ref describer to a compatibility adapter."""

    adapter.describe_trainable_parameter_refs = describe_fn
    return adapter


def get_trainable_parameter_refs(adapter: Any) -> list[AdapterTrainableParameterRef]:
    """Read adapter trainable refs through the repo-owned runtime contract."""

    describe_fn = getattr(adapter, "describe_trainable_parameter_refs", None)
    if describe_fn is None:
        raise TypeError("Adapter runtime does not expose describe_trainable_parameter_refs()")

    refs = describe_fn()
    if not isinstance(refs, list):
        raise TypeError("describe_trainable_parameter_refs() must return a list of AdapterTrainableParameterRef")

    for ref in refs:
        if not isinstance(ref, AdapterTrainableParameterRef):
            raise TypeError("describe_trainable_parameter_refs() returned a non-AdapterTrainableParameterRef item")
        if not isinstance(ref.param, nn.Parameter):
            raise TypeError(f"AdapterTrainableParameterRef '{ref.name}' has a param field that is not an nn.Parameter")

    return refs
