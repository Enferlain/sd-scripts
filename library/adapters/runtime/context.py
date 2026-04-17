from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AdapterModelContext:
    """Model objects needed to construct an adapter runtime."""

    vae: Any
    text_encoder: Any
    denoiser: Any


@dataclass(slots=True)
class AdapterBuildContext:
    """Common construction context for the repo-owned adapter runtime."""

    model: AdapterModelContext
    multiplier: float = 1.0
    for_inference: bool = False
