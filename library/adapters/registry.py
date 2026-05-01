from __future__ import annotations

from .methods.peft import PEFT_METHOD_REGISTRATIONS
from .types import AdapterMethodRegistration


_BUILTIN_ADAPTER_METHODS = (
    *PEFT_METHOD_REGISTRATIONS,
)

_ADAPTER_METHODS_BY_NAME = {method.name: method for method in _BUILTIN_ADAPTER_METHODS}
_ADAPTER_METHODS_BY_LEGACY_MODULE = {method.legacy_module_path: method for method in _BUILTIN_ADAPTER_METHODS}


def list_adapter_methods() -> tuple[AdapterMethodRegistration, ...]:
    """Return the built-in adapter types available in the repo-owned runtime."""

    return _BUILTIN_ADAPTER_METHODS


def get_adapter_method(name: str) -> AdapterMethodRegistration:
    """Resolve an adapter type by repo-owned name."""

    try:
        return _ADAPTER_METHODS_BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTER_METHODS_BY_NAME))
        raise KeyError(f"Unknown adapter type '{name}'. Available adapter types: {available}") from exc


def get_adapter_method_for_legacy_module(module_path: str) -> AdapterMethodRegistration:
    """Resolve an adapter type from the current compatibility-era module path."""

    try:
        return _ADAPTER_METHODS_BY_LEGACY_MODULE[module_path]
    except KeyError as exc:
        available = ", ".join(sorted(_ADAPTER_METHODS_BY_LEGACY_MODULE))
        raise KeyError(f"Unknown adapter module '{module_path}'. Available adapter modules: {available}") from exc
