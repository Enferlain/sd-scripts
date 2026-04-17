from __future__ import annotations

import importlib
from typing import Any

from ..registry import get_adapter_method, get_adapter_method_for_legacy_module


def build_adapter(adapter_type: str, *args: Any, **kwargs: Any) -> Any:
    """Build an adapter runtime from a repo-owned adapter type name.

    ``Any`` is intentional for this first migration slice while the concrete
    repo-owned adapter runtime surface is still being established.
    """

    registration = get_adapter_method(adapter_type)
    runtime_module = importlib.import_module(registration.runtime_module_path)
    return runtime_module.create_adapter(*args, **kwargs)


def build_adapter_from_weights(adapter_type: str, *args: Any, **kwargs: Any) -> Any:
    """Build an adapter runtime from weights using a repo-owned adapter type."""

    registration = get_adapter_method(adapter_type)
    runtime_module = importlib.import_module(registration.runtime_module_path)
    return runtime_module.create_adapter_from_weights(*args, **kwargs)


def build_adapter_for_legacy_module(module_path: str, *args: Any, **kwargs: Any) -> Any:
    """Build an adapter runtime from the current compatibility-era module path."""

    registration = get_adapter_method_for_legacy_module(module_path)
    runtime_module = importlib.import_module(registration.runtime_module_path)
    return runtime_module.create_adapter(*args, **kwargs)


def build_adapter_from_weights_for_legacy_module(module_path: str, *args: Any, **kwargs: Any) -> Any:
    """Build an adapter runtime from weights via the current legacy module path."""

    registration = get_adapter_method_for_legacy_module(module_path)
    runtime_module = importlib.import_module(registration.runtime_module_path)
    return runtime_module.create_adapter_from_weights(*args, **kwargs)
