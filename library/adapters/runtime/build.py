from __future__ import annotations

import importlib

from library.adapters.registry import get_adapter_method, get_adapter_method_for_legacy_module
from library.adapters.runtime.context import AdapterBuildRequest


def build_adapter(request: AdapterBuildRequest):
    """Build an adapter runtime from a repo-owned adapter type name.

    The explicit request object is the first repo-owned runtime boundary for
    adapter construction during the migration away from raw positional inputs.
    """

    registration = get_adapter_method(request.adapter.adapter_type)
    runtime_module = importlib.import_module(registration.runtime_module_path)
    return runtime_module.create_adapter(request)


def build_adapter_from_weights(request: AdapterBuildRequest, weights_path: str):
    """Build an adapter runtime from weights using a repo-owned adapter type."""

    registration = get_adapter_method(request.adapter.adapter_type)
    runtime_module = importlib.import_module(registration.runtime_module_path)
    return runtime_module.create_adapter_from_weights(request, weights_path)


def build_adapter_for_legacy_module(module_path: str, request: AdapterBuildRequest):
    """Build an adapter runtime from the current compatibility-era module path."""

    registration = get_adapter_method_for_legacy_module(module_path)
    if request.adapter.adapter_type != registration.name:
        raise ValueError(
            f"Adapter build request type '{request.adapter.adapter_type}' does not match legacy module "
            f"'{module_path}' resolved as '{registration.name}'"
        )
    runtime_module = importlib.import_module(registration.runtime_module_path)
    return runtime_module.create_adapter(request)


def build_adapter_from_weights_for_legacy_module(module_path: str, request: AdapterBuildRequest, weights_path: str):
    """Build an adapter runtime from weights via the current legacy module path."""

    registration = get_adapter_method_for_legacy_module(module_path)
    if request.adapter.adapter_type != registration.name:
        raise ValueError(
            f"Adapter build request type '{request.adapter.adapter_type}' does not match legacy module "
            f"'{module_path}' resolved as '{registration.name}'"
        )
    runtime_module = importlib.import_module(registration.runtime_module_path)
    return runtime_module.create_adapter_from_weights(request, weights_path)
