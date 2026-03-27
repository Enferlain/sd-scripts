"""Compatibility wrapper for launching the SDXL fine-tune preset."""

import logging

import hydra

from library.config.dataclasses.run import RunConfig
from library.training.launcher import run_training


logger = logging.getLogger(__name__)


def train(cfg: RunConfig) -> None:
    """Run SDXL fine-tuning through the shared launcher."""
    run_training(cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="presets/sdxl_finetune")
def main(cfg: RunConfig) -> None:
    """Main entry point for the SDXL fine-tune compatibility wrapper."""
    train(cfg)


if __name__ == "__main__":
    from library.config.schemas import register_run

    register_run()
    main()
