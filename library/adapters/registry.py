from __future__ import annotations

from .methods.peft.loha.config import CONFIG_BINDING as LOHA_CONFIG_BINDING
from .methods.peft.lora.config import CONFIG_BINDING as LORA_CONFIG_BINDING
from .types import AdapterMethodRegistration


_BUILTIN_ADAPTER_METHODS = (
    AdapterMethodRegistration(
        name="loha",
        # This remains the compatibility-era selection string for now even
        # though the repo-owned runtime lives under adapters.methods.
        legacy_module_path="library.adapters.loha",
        runtime_module_path="library.adapters.methods.peft.loha.runtime",
        config_binding=LOHA_CONFIG_BINDING,
    ),
    AdapterMethodRegistration(
        name="lora",
        legacy_module_path="library.adapters.lora",
        runtime_module_path="library.adapters.methods.peft.lora.runtime",
        config_binding=LORA_CONFIG_BINDING,
    ),
    # Left over on purpose as legacy bridges. The current adapter-system task
    # is centered on the LoRA-shaped path; Dylora/OFT will be revisited from
    # the vendor LyCORIS direction rather than by extending the already-present
    # local implementations here.
    AdapterMethodRegistration(
        name="dylora_deprecated",
        legacy_module_path="library.adapters.dylora",
        runtime_module_path="library.adapters.methods.peft.dylora_deprecated.runtime",
    ),
    AdapterMethodRegistration(
        name="oft_deprecated",
        legacy_module_path="library.adapters.oft",
        runtime_module_path="library.adapters.methods.peft.oft_deprecated.runtime",
    ),
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
