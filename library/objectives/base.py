from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from library.losses.loss_modifiers import LossModifier
from library.timesteps.runtime import TimestepRuntime


@dataclass
class ObjectiveRuntime:
    """Trainer-owned runtime bundle for the active objective/formulation."""

    noise_scheduler: Any
    timestep_runtime: TimestepRuntime
    loss_modifier: LossModifier


class ObjectiveDefinition(ABC):
    """Configuration-driven objective owner for runtime assembly and metadata."""

    name: str

    @abstractmethod
    def build_runtime(self, cfg: Any, accelerator: Any) -> ObjectiveRuntime:
        """Build the objective-owned runtime helpers used by the trainer."""
        raise NotImplementedError
