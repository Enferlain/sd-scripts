from __future__ import annotations

from typing import Any

from ... import dylora as legacy_dylora


def create_adapter(*args: Any, **kwargs: Any) -> Any:
    """Repo-owned wrapper for the built-in DyLoRA adapter runtime."""

    return legacy_dylora.create_adapter(*args, **kwargs)


def create_adapter_from_weights(*args: Any, **kwargs: Any) -> Any:
    """Repo-owned wrapper for loading the built-in DyLoRA adapter from weights."""

    return legacy_dylora.create_adapter_from_weights(*args, **kwargs)
