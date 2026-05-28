"""Canonical root entrypoint for the active config-driven training launcher."""

import time

import hydra

from library.config.config_validation import prepare_config, validate_config
from library.config.dataclasses.run import RunConfig
from library.strategies.factory import build_training_strategy
from library.training.interrupts import install_double_ctrl_c_guard
from library.training.modes.factory import build_training_mode
from library.training.runners.trainer import Trainer
from library.utils.common_utils import setup_logging

PROCESS_LAUNCHED_PERF = time.perf_counter()


def train(cfg: RunConfig) -> None:
    """Run training through the shared config-driven launcher."""
    setup_logging(cfg.output.logging, reset=True)
    prepare_config(cfg)
    validate_config(cfg)

    strategies = build_training_strategy(cfg)
    mode = build_training_mode(cfg)

    trainer = Trainer(cfg, strategies, mode, process_launched_perf=PROCESS_LAUNCHED_PERF)
    with install_double_ctrl_c_guard(trainer):
        trainer.train()


@hydra.main(version_base=None, config_path="configs", config_name=None)
def main(cfg: RunConfig) -> None:
    """Main entrypoint for the active training launcher."""
    train(cfg)

if __name__ == "__main__":
    from library.config.schemas import register_run

    register_run()
    main()
