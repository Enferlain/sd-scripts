"""Repo-owned adapter runtime scaffolding."""

from .registry import (
    AdapterMethodRegistration,
    get_adapter_method,
    get_adapter_method_for_legacy_module,
    list_adapter_methods,
)
from .runtime import (
    AdapterBuildContext,
    AdapterModelContext,
    AdapterResolvedTarget,
    AdapterResolvedTargets,
    build_adapter,
    build_adapter_for_legacy_module,
    build_adapter_from_weights,
    build_adapter_from_weights_for_legacy_module,
)

__all__ = [
    "AdapterBuildContext",
    "AdapterMethodRegistration",
    "AdapterModelContext",
    "AdapterResolvedTarget",
    "AdapterResolvedTargets",
    "build_adapter",
    "build_adapter_for_legacy_module",
    "build_adapter_from_weights",
    "build_adapter_from_weights_for_legacy_module",
    "get_adapter_method",
    "get_adapter_method_for_legacy_module",
    "list_adapter_methods",
]
