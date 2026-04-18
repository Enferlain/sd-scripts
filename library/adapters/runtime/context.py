from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .targets import AdapterResolvedTargets


@dataclass(slots=True)
class AdapterModelContext:
    """Model objects needed to construct an adapter runtime."""

    vae: Any
    text_encoder: Any | list[Any]
    denoiser: Any


@dataclass(slots=True)
class AdapterBuildContext:
    """Common construction context for the repo-owned adapter runtime."""

    model: AdapterModelContext
    multiplier: float = 1.0
    for_inference: bool = False


@dataclass(slots=True)
class AdapterRuntimeSpec:
    """Adapter-type selection plus adapter-local runtime settings."""

    adapter_type: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AdapterBuildRequest:
    """Explicit request payload for building an adapter runtime."""

    adapter: AdapterRuntimeSpec
    context: AdapterBuildContext
    resolved_targets: AdapterResolvedTargets
