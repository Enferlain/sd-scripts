"""
Hydra ConfigStore schema registration.

This module provides explicit registration functions for each config schema.
Scripts should call only the specific register function they need.
Tests can call register_all() to register all schemas at once.

Pattern follows Hydra's documented "library registers its configs" approach.
"""

from hydra.core.config_store import ConfigStore


def register_sd_peft():
    """Register SD PEFT config schema."""
    from library.config.dataclasses.sd_peft import SDPeftConfig

    cs = ConfigStore.instance()
    cs.store(name="sd_peft_schema", node=SDPeftConfig)


def register_sdxl_peft():
    """Register SDXL PEFT config schema."""
    from library.config.dataclasses.sdxl_peft import SDXLPeftConfig

    cs = ConfigStore.instance()
    cs.store(name="sdxl_peft_schema", node=SDXLPeftConfig)


def register_sd_finetune():
    """Register SD FineTune config schema."""
    from library.config.dataclasses.sd_finetune import SDFineTuneConfig

    cs = ConfigStore.instance()
    cs.store(name="sd_finetune_schema", node=SDFineTuneConfig)


def register_sdxl_finetune():
    """Register SDXL FineTune config schema."""
    from library.config.dataclasses.sdxl_finetune import SDXLFineTuneConfig

    cs = ConfigStore.instance()
    cs.store(name="sdxl_finetune_schema", node=SDXLFineTuneConfig)


def register_sd_textual_inversion():
    """Register SD Textual Inversion config schema."""
    from library.config.dataclasses.sd_textual_inversion import TextualInversionConfig

    cs = ConfigStore.instance()
    cs.store(name="sd_textual_inversion_schema", node=TextualInversionConfig)


def register_sdxl_textual_inversion():
    """Register SDXL Textual Inversion config schema."""
    from library.config.dataclasses.sdxl_textual_inversion import SDXLTextualInversionConfig

    cs = ConfigStore.instance()
    cs.store(name="sdxl_textual_inversion_schema", node=SDXLTextualInversionConfig)


def register_all():
    """Register all config schemas. Use in tests or tools that need all configs."""
    register_sd_peft()
    register_sdxl_peft()
    register_sd_finetune()
    register_sdxl_finetune()
    register_sd_textual_inversion()
    register_sdxl_textual_inversion()
