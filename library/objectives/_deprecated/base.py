from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from library.losses.loss_modifiers import LossModifier
from library.timesteps.runtime import TimestepRuntime


@dataclass
class ObjectiveRuntime:
    """Trainer-owned runtime bundle for the active objective/formulation."""

    name: str
    num_train_timesteps: int
    timestep_runtime: TimestepRuntime | None
    loss_modifier: LossModifier

    def advance_to_step(self, global_step: int) -> tuple[int, int] | None:
        """Advance any objective-owned timestep scheduling state to the given step."""
        if self.timestep_runtime is None:
            return None
        return self.timestep_runtime.advance_to_step(global_step)

    def update_from_batch(self, timesteps: torch.Tensor, per_sample_loss: torch.Tensor) -> None:
        """Feed batch timestep/loss observations back into runtime-owned samplers."""
        if self.timestep_runtime is None:
            return
        self.timestep_runtime.update_from_batch(timesteps, per_sample_loss)


class ObjectiveDefinition(ABC):
    """Configuration-driven objective owner for runtime assembly and metadata."""

    name: str

    @abstractmethod
    def build_runtime(self, cfg: Any, accelerator: Any) -> ObjectiveRuntime:
        """Build the objective-owned runtime helpers used by the trainer."""
        raise NotImplementedError
