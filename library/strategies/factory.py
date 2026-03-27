"""Factory for selecting the active model-family training strategy."""

from __future__ import annotations

from typing import Any

from library.strategies.base.contracts import TrainingStrategy
from library.strategies.sd.training import SdTrainingStrategy
from library.strategies.sdxl.training import SdxlTrainingStrategy


_STRATEGY_REGISTRY: dict[str, type[TrainingStrategy]] = {
    "sd1": SdTrainingStrategy,
    "sd15": SdTrainingStrategy,
    "sd2": SdTrainingStrategy,
    "sdxl": SdxlTrainingStrategy,
}


def build_training_strategy(cfg: Any) -> TrainingStrategy:
    """Construct the training strategy for the requested model family."""
    model_type = cfg.model.model_type

    try:
        strategy_cls = _STRATEGY_REGISTRY[model_type]
    except KeyError as exc:
        known_types = ", ".join(sorted(_STRATEGY_REGISTRY))
        raise ValueError(
            f"Unsupported model.model_type={model_type!r} for the active strategy path. Known active model types: {known_types}."
        ) from exc

    return strategy_cls(cfg)
