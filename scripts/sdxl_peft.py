"""
SDXL PEFT (LoRA/LyCORIS) Training Script

This script is the entry point for SDXL PEFT training.
All training logic is now in Trainer and its phase functions.

Model-specific operations are delegated to:
- library/strategies/sdxl/training.py (strategy pattern)
- library/training/runners/trainer.py (trainer orchestration)
- library/training/phases/*.py (phase-specific logic)
"""

import logging
import hydra

from library.config.config_validation import prepare_config, validate_config
from library.config.dataclasses.sdxl_peft import SDXLPeftConfig
from library.strategies.sdxl.training import SdxlTrainingStrategy
from library.training.runners.trainer import Trainer
from library.training.modes import PeftMode

from library.utils.device_utils import init_ipex

init_ipex()


logger = logging.getLogger(__name__)


def train(cfg: SDXLPeftConfig, strategies: SdxlTrainingStrategy) -> None:
    """Run SDXL PEFT training.

    Args:
        cfg: Training configuration
        strategies: SDXL-specific training strategies
    """
    mode = PeftMode()
    trainer = Trainer(cfg, strategies, mode)
    trainer.train()


@hydra.main(version_base=None, config_path="../configs", config_name="sdxl_peft")
def main(cfg: SDXLPeftConfig):
    """Main entry point for SDXL PEFT training."""
    prepare_config(cfg)
    validate_config(cfg)

    strategies = SdxlTrainingStrategy()
    train(cfg, strategies)


if __name__ == "__main__":
    # Register Hydra schema only when running as script
    from library.config.schemas import register_sdxl_peft

    register_sdxl_peft()
    main()
