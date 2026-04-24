from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from library.adapters.types import AdapterMethodConfigBinding


@dataclass
class PeftLoraConfig:
    """Method-specific LoRA settings under the PEFT shell."""

    rank: int | None = field(default=None, metadata={"help": "LoRA rank/dimensions (higher = more capacity, more VRAM)"})
    alpha: float = field(default=1.0, metadata={"help": "LoRA alpha scaling"})
    dropout: float | None = field(default=None, metadata={"help": "Dropout rate for LoRA neurons during training"})
    conv_rank: int | None = field(default=None, metadata={"help": "LoRA rank for Conv2d layers"})
    conv_alpha: float | None = field(default=None, metadata={"help": "LoRA alpha for Conv2d layers"})
    rank_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to the LoRA rank dimension"})
    module_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to entire LoRA modules"})
    # Left over on purpose for the current adapter-system slice: these legacy
    # optimizer-policy knobs are kept as explicit reference points while the
    # repo-owned grouping boundary rejects them in the active PEFT path.
    block_ranks: list[int] | None = field(default=None, metadata={"help": "Per-block LoRA rank values"})
    block_alphas: list[float] | None = field(default=None, metadata={"help": "Per-block LoRA alpha values"})
    conv_block_ranks: list[int] | None = field(default=None, metadata={"help": "Per-block Conv2d LoRA rank values"})
    conv_block_alphas: list[float] | None = field(default=None, metadata={"help": "Per-block Conv2d LoRA alpha values"})
    down_lr_weight: Any | None = field(default=None, metadata={"help": "Legacy block LR weight preset or values for down blocks"})
    mid_lr_weight: Any | None = field(default=None, metadata={"help": "Legacy block LR weight preset or value for the mid block"})
    up_lr_weight: Any | None = field(default=None, metadata={"help": "Legacy block LR weight preset or values for up blocks"})
    block_lr_zero_threshold: float | None = field(default=None, metadata={"help": "Threshold below which block LR is treated as zero"})
    loraplus_lr_ratio: float | None = field(default=None, metadata={"help": "Legacy LoRA+ learning rate ratio"})
    loraplus_unet_lr_ratio: float | None = field(default=None, metadata={"help": "Legacy LoRA+ learning rate ratio for the denoiser"})
    loraplus_text_encoder_lr_ratio: float | None = field(
        default=None, metadata={"help": "Legacy LoRA+ learning rate ratio for text encoders"}
    )


def build_runtime_settings(config: PeftLoraConfig) -> dict[str, object]:
    """Translate forward LoRA config into normalized runtime settings."""

    settings: dict[str, object] = {
        "adapter_rank": config.rank,
        "adapter_alpha": config.alpha,
        "neuron_dropout": config.dropout,
    }
    optional_settings = {
        "conv_dim": config.conv_rank,
        "conv_alpha": config.conv_alpha,
        "rank_dropout": config.rank_dropout,
        "module_dropout": config.module_dropout,
        "block_dims": config.block_ranks,
        "block_alphas": config.block_alphas,
        "conv_block_dims": config.conv_block_ranks,
        "conv_block_alphas": config.conv_block_alphas,
    }
    for key, value in optional_settings.items():
        if value is not None:
            settings[key] = value
    return settings


CONFIG_BINDING = AdapterMethodConfigBinding(
    config_key="lora",
    config_type=PeftLoraConfig,
    runtime_settings_builder=build_runtime_settings,
)
