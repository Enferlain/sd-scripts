"""Typed optimizer and scheduler metadata facts.

These are orchestration/runtime facts, not export records. The optimizer
stack uses them as a central, typed snapshot of behavior that other layers can
inspect or alias without forcing serialization helpers onto the classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class SchedulerRuntimeFacts:
    """Explicit scheduler/runtime ownership facts for optimizer orchestration."""

    mode: Literal["external", "embedded", "none"] = "external"
    target: Literal["optimizer", "base_optimizer"] = "optimizer"


@dataclass(slots=True)
class OptimizerRuntimeFacts:
    """Explicit optimizer runtime behavior facts for orchestration."""

    supports_train_eval_toggle: bool = False
