from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AdapterResolvedTarget:
    """Adapter-facing view of one resolved original-model target."""

    component: str
    path: str
    module: Any
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AdapterResolvedTargets:
    """Adapter-facing bundle of resolved targets for one training run."""

    family: str
    targets: list[AdapterResolvedTarget] = field(default_factory=list)
