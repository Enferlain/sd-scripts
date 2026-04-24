from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LegacyAdapterModule(Protocol):
    """Compatibility-era adapter module surface used during migration."""

    def create_adapter(self, *args: Any, **kwargs: Any) -> Any: ...

    def create_adapter_from_weights(self, *args: Any, **kwargs: Any) -> Any: ...


AdapterRuntimeSettingsBuilder = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class AdapterMethodConfigBinding:
    """Method-local config binding used by the adapter registry."""

    config_key: str
    config_type: type
    runtime_settings_builder: AdapterRuntimeSettingsBuilder


@dataclass(frozen=True, slots=True)
class AdapterMethodRegistration:
    """Registry entry for one repo-owned adapter type."""

    name: str
    legacy_module_path: str
    runtime_module_path: str
    config_binding: AdapterMethodConfigBinding | None = None
