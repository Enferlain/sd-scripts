from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


def normalize_modifier_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Normalize modifier metric values and enforce namespaced metric keys."""
    normalized: dict[str, float] = {}
    for name, value in metrics.items():
        if "/" not in name:
            raise ValueError(
                "Loss modifier metrics must be namespaced, e.g. 'loss/current_scaled' or 'modifier/example_metric'. "
                f"Got '{name}'."
            )
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Loss modifier metric '{name}' must be a scalar tensor, got shape {tuple(value.shape)}")
            value = value.detach().item()
        normalized[name] = float(value)
    return normalized


@dataclass
class BatchLossOutput:
    """Base diffusion loss state returned by model-family strategies."""

    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    timesteps: torch.Tensor
    sampling_loss: torch.Tensor | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metrics = normalize_modifier_metrics(self.metrics)


@dataclass
class LossModifierOutput:
    """Result from an optional trainer-owned loss modifier."""

    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metrics = normalize_modifier_metrics(self.metrics)


class LossModifier(Protocol):
    """Minimal trainer-facing protocol for optional post-loss modifiers."""

    @property
    def name(self) -> str:
        """Return the display name used for modifier-specific logs/artifacts."""
        ...

    @property
    def is_enabled(self) -> bool:
        """Return whether the modifier is active for the current run."""
        ...

    @property
    def sidecar_suffix(self) -> str | None:
        """Return the checkpoint suffix used for sidecar artifacts, if any."""
        ...

    @property
    def accumulation_model(self) -> Any | None:
        """Return an auxiliary model to include in gradient accumulation, if any."""
        ...

    def apply(
        self,
        *,
        per_sample_loss: torch.Tensor,
        timesteps: torch.Tensor,
        batch: dict[str, Any] | None = None,
        global_step: int = 0,
    ) -> LossModifierOutput:
        """Transform a per-sample loss tensor into the final scalar loss."""
        ...

    def optimizer_step(self) -> None:
        """Advance any optimizer/scheduler owned by the modifier runtime."""
        ...

    def zero_grad(self) -> None:
        """Clear gradients for any optimizer owned by the modifier runtime."""
        ...

    def get_lr(self) -> float | None:
        """Return the current modifier learning rate when one exists."""
        ...

    def save_sidecar(self, ckpt_path: str, metadata: dict[str, str]) -> None:
        """Persist sidecar state for the modifier."""
        ...

    def load_sidecar(self, ckpt_path: str) -> None:
        """Load previously saved sidecar state for the modifier."""
        ...

    def should_plot(self, global_step: int) -> bool:
        """Return whether the modifier wants to emit a plot at the current step."""
        ...

    def plot(self, output_name: str, step: int, device: Any) -> None:
        """Emit optional modifier-owned plots or artifacts."""
        ...


@dataclass
class NoOpLossModifier:
    """Disabled loss modifier used when no adaptive feature is active."""

    model: Any = None
    optimizer: Any = None
    lr_scheduler: Any = None
    loss_recorder: Any = None
    current_global_step_loss_scaled: float | None = None

    @property
    def name(self) -> str:
        return "noop"

    @property
    def is_enabled(self) -> bool:
        return False

    @property
    def sidecar_suffix(self) -> str | None:
        return None

    @property
    def accumulation_model(self) -> Any | None:
        return None

    def apply(
        self,
        *,
        per_sample_loss: torch.Tensor,
        timesteps: torch.Tensor,
        batch: dict[str, Any] | None = None,
        global_step: int = 0,
    ) -> LossModifierOutput:
        del timesteps, batch, global_step
        return LossModifierOutput(loss=per_sample_loss.mean())

    def optimizer_step(self) -> None:
        return None

    def zero_grad(self) -> None:
        return None

    def get_lr(self) -> float | None:
        return None

    def save_sidecar(self, ckpt_path: str, metadata: dict[str, str]) -> None:
        del ckpt_path, metadata
        return None

    def load_sidecar(self, ckpt_path: str) -> None:
        del ckpt_path
        return None

    def should_plot(self, global_step: int) -> bool:
        del global_step
        return False

    def plot(self, output_name: str, step: int, device: Any) -> None:
        del output_name, step, device
        return None


def build_loss_modifier(loss_config, training_config, noise_scheduler, accelerator) -> LossModifier:
    """Build the active trainer-owned loss modifier for the run."""
    if getattr(loss_config.edm2, "enabled", False):
        from library.losses.edm2.factory import create_edm2_modifier

        return create_edm2_modifier(loss_config.edm2, training_config, noise_scheduler, accelerator)
    return NoOpLossModifier()
