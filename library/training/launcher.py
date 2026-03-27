"""Shared launcher for the active config-driven training entrypoints."""

from __future__ import annotations

from library.config.config_validation import prepare_config, validate_config
from library.config.dataclasses.run import RunConfig
from library.strategies.factory import build_training_strategy
from library.training.modes.factory import build_training_mode
from library.training.runners.trainer import Trainer


def run_training(cfg: RunConfig) -> None:
    """Prepare config, construct mode/strategy objects, and run training."""
    prepare_config(cfg)
    validate_config(cfg)

    strategies = build_training_strategy(cfg)
    mode = build_training_mode(cfg)

    trainer = Trainer(cfg, strategies, mode)
    trainer.train()
