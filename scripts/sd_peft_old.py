"""
SD1.5/2 PEFT (LoRA/LyCORIS) Training Script

This is a thin orchestrator that:
1. Loads Hydra config
2. Instantiates SD strategy
3. Calls the generic train() function

The actual training logic is in:
- library/training/peft_trainer.py (train function)
- library/strategies/peft_strategy_sd.py (model-specific methods)
- library/training/peft_common.py (shared utilities)
"""

import logging
import hydra

from hydra.core.config_store import ConfigStore

from library.config.dataclasses.sd_peft import SDPeftConfig
from library.config.config_validation import prepare_config, validate_config
from library.utils.common_utils import setup_logging
from library.utils.device_utils import init_ipex
from library.strategies.peft_strategy_sd import SdPeftStrategy
from library.training.peft_trainer import train

init_ipex()

setup_logging()
logger = logging.getLogger(__name__)


# Register the structure config with Hydra
cs = ConfigStore.instance()
cs.store(name="sd_peft", node=SDPeftConfig)


@hydra.main(version_base=None, config_path="../configs", config_name="sd_peft")
def main(cfg: SDPeftConfig):
    """Main entry point for SD PEFT training."""
    prepare_config(cfg)
    validate_config(cfg)

    strategies = SdPeftStrategy()
    train(cfg, strategies)


if __name__ == "__main__":
    main()
