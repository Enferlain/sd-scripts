from __future__ import annotations

from dataclasses import dataclass, field

from library.adapters.types import AdapterMethodConfigBinding


def _validate_probability(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"adapter.peft.lora.{field_name} must be between 0.0 and 1.0 inclusive.")


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


def build_runtime_settings(config: PeftLoraConfig) -> dict[str, object]:
    """Translate forward LoRA config into normalized runtime settings."""

    if config.rank is not None and config.rank <= 0:
        raise ValueError("adapter.peft.lora.rank must be a positive integer when set.")
    if config.conv_rank is not None and config.conv_rank <= 0:
        raise ValueError("adapter.peft.lora.conv_rank must be a positive integer when set.")
    if config.conv_alpha is not None and config.conv_rank is None:
        raise ValueError("adapter.peft.lora.conv_alpha requires adapter.peft.lora.conv_rank to also be set.")

    _validate_probability(config.dropout, field_name="dropout")
    _validate_probability(config.rank_dropout, field_name="rank_dropout")
    _validate_probability(config.module_dropout, field_name="module_dropout")

    settings: dict[str, object] = {
        "adapter_alpha": config.alpha,
    }
    if config.rank is not None:
        settings["adapter_rank"] = config.rank
    if config.dropout is not None:
        # Keep both keys during the transition because some existing adapter
        # tests and call paths still look for the older generic `dropout` name.
        settings["neuron_dropout"] = config.dropout
        settings["dropout"] = config.dropout

    optional_settings = {
        "conv_dim": config.conv_rank,
        "conv_alpha": config.conv_alpha,
        "rank_dropout": config.rank_dropout,
        "module_dropout": config.module_dropout,
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
