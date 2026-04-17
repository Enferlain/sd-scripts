from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LegacyAdapterModule(Protocol):
    """Compatibility-era adapter module surface used during migration."""

    def create_adapter(self, *args: Any, **kwargs: Any) -> Any: ...

    def create_adapter_from_weights(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class AdapterMethodRegistration:
    """Registry entry for one repo-owned adapter type."""

    name: str
    legacy_module_path: str
    runtime_module_path: str
