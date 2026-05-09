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
    adapter_module_path: str | None = None
    source_target_ref: OptimizationTargetRef | None = None


@runtime_checkable
class AdapterTrainableParameterProvider(Protocol):
    """Adapter runtime surface for exposing trainable refs to optimization."""

    def describe_trainable_parameter_refs(self) -> list[AdapterTrainableParameterRef]: ...


def build_adapter_module_path(module_name: str, param_name: str) -> str:
    """Return the adapter-module owner path for a named parameter."""

    owner_path, _, _ = param_name.rpartition(".")
    if not owner_path:
        return module_name
    return f"{module_name}.{owner_path}"


def build_named_parameter_refs(
    *,
    module_name: str,
    module: nn.Module,
    algorithm: str,
    component: str,
    component_key: str,
    target_path: str,
    source_target_ref: OptimizationTargetRef | None = None,
) -> list[AdapterTrainableParameterRef]:
    """Build repo-owned trainable refs for one adapter module's named parameters."""

    refs: list[AdapterTrainableParameterRef] = []
    for param_name, param in module.named_parameters():
        refs.append(
            AdapterTrainableParameterRef(
                param=param,
                name=f"{module_name}.{param_name}",
                algorithm=algorithm,
                component=component,
                component_key=component_key,
                target_path=target_path,
                adapter_module_path=build_adapter_module_path(module_name, param_name),
                source_target_ref=source_target_ref,
            )
        )
    return refs


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
