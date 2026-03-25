from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from library.losses.loss_modifiers import LossModifierOutput


@dataclass
class EDM2LossModifier:
    """Concrete trainer-owned loss modifier for EDM2 adaptive weighting."""

    config: Any = None
    training_config: Any = None
    model: Any = None
    optimizer: Any = None
    lr_scheduler: Any = None
    loss_recorder: Any = None
    current_global_step_loss_scaled: float | None = None

    @property
    def name(self) -> str:
        return "edm2"

    @property
    def is_enabled(self) -> bool:
        """Return whether EDM2 loss weighting is active for the current run."""
        return self.model is not None

    @property
    def sidecar_suffix(self) -> str | None:
        return "_edm2_loss_weights"

    @property
    def accumulation_model(self) -> Any | None:
        """Return the EDM2 sidecar model for shared gradient accumulation."""
        return self.model

    def apply(
        self,
        *,
        per_sample_loss: torch.Tensor,
        timesteps: torch.Tensor,
        batch: dict[str, Any] | None = None,
        global_step: int = 0,
    ) -> LossModifierOutput:
        """Apply EDM2 weighting to a per-sample loss tensor when enabled."""
        del batch, global_step
        if not self.is_enabled:
            return LossModifierOutput(loss=per_sample_loss.mean())

        weighted_loss, scaled_loss = self.model(per_sample_loss, timesteps)
        return LossModifierOutput(
            loss=weighted_loss.mean(),
            metrics={"loss/current_scaled": float(scaled_loss.mean().detach().item())},
        )

    def optimizer_step(self) -> None:
        """Advance the EDM2 optimizer and scheduler when active."""
        if self.optimizer is not None:
            self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def zero_grad(self) -> None:
        """Clear EDM2 optimizer gradients when active."""
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def get_lr(self) -> float | None:
        """Return the active EDM2 learning rate when available."""
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, "get_last_lr"):
            last_lr = self.lr_scheduler.get_last_lr()
            if last_lr:
                return float(last_lr[0])
        if self.optimizer is not None and getattr(self.optimizer, "param_groups", None):
            return float(self.optimizer.param_groups[0]["lr"])
        return None

    def save_sidecar(self, ckpt_path: str, metadata: dict[str, str]) -> None:
        """Persist EDM2 weights via the sidecar model when enabled."""
        if self.model is None or not hasattr(self.model, "save_weights"):
            return
        self.model.save_weights(ckpt_path, dtype=torch.float32, metadata=metadata)

    def load_sidecar(self, ckpt_path: str) -> None:
        """Load EDM2 sidecar weights when the sidecar model supports it."""
        if self.model is None or not hasattr(self.model, "load_weights"):
            return
        self.model.load_weights(ckpt_path)

    def should_plot(self, global_step: int) -> bool:
        """Return whether EDM2 wants to emit a loss-weighting plot now."""
        if self.config is None or self.training_config is None:
            return False
        from library.losses.edm2.plotting import plot_edm2_loss_weighting_check

        return plot_edm2_loss_weighting_check(self.config, self.training_config, global_step)

    def plot(self, output_name: str, step: int, device: Any) -> None:
        """Emit the EDM2 loss-weighting plot when the modifier is enabled."""
        if self.model is None or self.config is None:
            return
        from library.losses.edm2.plotting import plot_edm2_loss_weighting

        plot_edm2_loss_weighting(
            self.config,
            output_name,
            step,
            self.model,
            1000,
            device,
        )
