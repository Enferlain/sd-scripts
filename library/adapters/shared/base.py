from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AdapterRuntime(Protocol):
    """Training-facing adapter runtime surface used by the shared adapter path."""

    def apply_to(self, *args: Any, **kwargs: Any) -> None: ...

    def prepare_grad_etc(self, *args: Any, **kwargs: Any) -> None: ...

    def enable_gradient_checkpointing(self) -> None: ...

    def get_trainable_params(self) -> list[Any]: ...
