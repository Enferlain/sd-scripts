from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AdapterStateIO(Protocol):
    """Persistence surface currently used by the adapter training path."""

    def load_weights(self, file: str) -> Any: ...

    def save_weights(self, file: str, dtype: Any, metadata: dict[str, str] | None) -> None: ...
