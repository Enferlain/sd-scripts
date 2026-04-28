"""Factory for selecting the active training mode implementation."""

from __future__ import annotations

from typing import Any

from library.training.modes.base import TrainingMode
from library.training.modes.finetune_mode import FineTuneMode
from library.training.modes.adapter_mode import AdapterMode


_ACTIVE_MODE_REGISTRY: dict[str, type[TrainingMode]] = {
    "finetune": FineTuneMode,
    "adapter": AdapterMode,
}


def build_training_mode(cfg: Any) -> TrainingMode:
    """Construct the active training mode for the current config."""
    mode_name = cfg.mode

    if mode_name == "textual_inversion":
        raise NotImplementedError(
            "mode=textual_inversion is not implemented in the active Trainer/TrainingMode launcher yet. "
            "Use the dedicated textual inversion entrypoint until that mode is migrated."
        )

    try:
        mode_cls = _ACTIVE_MODE_REGISTRY[mode_name]
    except KeyError as exc:
        known_modes = ", ".join(sorted(_ACTIVE_MODE_REGISTRY | {"textual_inversion"}))
        raise ValueError(f"Unsupported mode={mode_name!r}. Known modes: {known_modes}.") from exc

    return mode_cls()
