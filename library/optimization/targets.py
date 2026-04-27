from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from torch import nn

from library.models.parameter_dump import build_selector_name

OptimizationTargetKind = Literal["component", "module", "parameter"]


@dataclass(slots=True)
class OptimizationTargetRef:
    """Optimization-owned reference to one selected targetable object."""

    kind: OptimizationTargetKind
    component: str
    component_key: str
    path: str
    selector: str
    obj: Any
    module_type: str | None = None
    owner_module_path: str | None = None
    owner_module_type: str | None = None
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)


def _copy_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return {} if metadata is None else dict(metadata)


def _module_type_name(module: Any) -> str | None:
    return module.__class__.__name__ if module is not None else None


def build_component_target_ref(
    *,
    component: str,
    component_key: str,
    obj: Any,
    tags: frozenset[str] = frozenset(),
    metadata: dict[str, Any] | None = None,
) -> OptimizationTargetRef:
    """Build a component-level target ref with the public selector surface."""

    return OptimizationTargetRef(
        kind="component",
        component=component,
        component_key=component_key,
        path="",
        selector=build_selector_name(component, ""),
        obj=obj,
        tags=tags,
        metadata=_copy_metadata(metadata),
    )


def build_module_target_ref(
    *,
    component: str,
    component_key: str,
    path: str,
    module: Any,
    tags: frozenset[str] = frozenset(),
    metadata: dict[str, Any] | None = None,
) -> OptimizationTargetRef:
    """Build a module-level target ref from a component-local module path."""

    return OptimizationTargetRef(
        kind="module",
        component=component,
        component_key=component_key,
        path=path,
        selector=build_selector_name(component, path),
        obj=module,
        module_type=_module_type_name(module),
        tags=tags,
        metadata=_copy_metadata(metadata),
    )


def build_parameter_target_ref(
    *,
    component: str,
    component_key: str,
    path: str,
    parameter: nn.Parameter,
    owner_module_path: str | None = None,
    owner_module_type: str | None = None,
    tags: frozenset[str] = frozenset(),
    metadata: dict[str, Any] | None = None,
) -> OptimizationTargetRef:
    """Build a parameter-level target ref from a component-local parameter path."""

    return OptimizationTargetRef(
        kind="parameter",
        component=component,
        component_key=component_key,
        path=path,
        selector=build_selector_name(component, path),
        obj=parameter,
        owner_module_path=owner_module_path,
        owner_module_type=owner_module_type,
        tags=tags,
        metadata=_copy_metadata(metadata),
    )


def resolve_parameter_owner_modules(root_module: nn.Module) -> dict[int, tuple[str, str | None]]:
    """Resolve the deepest direct owner-module path/type for each parameter."""

    owner_modules: dict[int, tuple[str, str | None]] = {}
    for module_path, module in root_module.named_modules():
        module_type = _module_type_name(module)
        for parameter in module.parameters(recurse=False):
            owner_modules[id(parameter)] = (module_path, module_type)
    return owner_modules
