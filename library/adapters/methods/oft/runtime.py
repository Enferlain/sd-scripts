from __future__ import annotations

from typing import Any

from ... import oft as legacy_oft


def create_adapter(*args: Any, **kwargs: Any) -> Any:
    """Repo-owned wrapper for the built-in OFT adapter runtime."""

    return legacy_oft.create_adapter(*args, **kwargs)


def create_adapter_from_weights(*args: Any, **kwargs: Any) -> Any:
    """Repo-owned wrapper for loading the built-in OFT adapter from weights."""

    return legacy_oft.create_adapter_from_weights(*args, **kwargs)
