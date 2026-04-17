from __future__ import annotations

from typing import Any

from ... import lora as legacy_lora


def create_adapter(*args: Any, **kwargs: Any) -> Any:
    """Repo-owned wrapper for the built-in LoRA adapter runtime."""

    return legacy_lora.create_adapter(*args, **kwargs)


def create_adapter_from_weights(*args: Any, **kwargs: Any) -> Any:
    """Repo-owned wrapper for loading the built-in LoRA adapter from weights."""

    return legacy_lora.create_adapter_from_weights(*args, **kwargs)
