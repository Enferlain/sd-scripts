from __future__ import annotations

from dataclasses import dataclass, field

from library.adapters.types import AdapterMethodConfigBinding


def _validate_probability(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"adapter.peft.dylora.{field_name} must be between 0.0 and 1.0 inclusive.")


@dataclass
class PeftDyloraConfig:
    """Method-specific DyLoRA settings under the PEFT shell."""

    rank: int | None = field(default=None, metadata={"help": "DyLoRA rank/dimensions (higher = more capacity, more VRAM)"})
    alpha: float = field(default=1.0, metadata={"help": "DyLoRA alpha scaling"})
    block_size: int = field(
        default=1,
        metadata={
            "help": "DyLoRA update unit. Each training step activates a prefix whose rank is a multiple of this block size."
        },
    )
    module_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to entire DyLoRA modules"})
    bypass_mode: bool | None = field(default=None, metadata={"help": "Optional DyLoRA bypass-mode toggle"})


def build_runtime_settings(config: PeftDyloraConfig) -> dict[str, object]:
    """Translate forward DyLoRA config into normalized runtime settings."""

    if config.rank is None:
        raise ValueError("adapter.peft.dylora.rank must be set to a positive integer.")
    if config.rank <= 0:
        raise ValueError("adapter.peft.dylora.rank must be a positive integer when set.")
    if config.block_size <= 0:
        raise ValueError("adapter.peft.dylora.block_size must be a positive integer.")
    if config.rank % config.block_size != 0:
        raise ValueError("adapter.peft.dylora.block_size must divide adapter.peft.dylora.rank exactly.")
    _validate_probability(config.module_dropout, field_name="module_dropout")

    settings: dict[str, object] = {
        "adapter_rank": config.rank,
        "adapter_alpha": config.alpha,
        "block_size": config.block_size,
    }
    if config.module_dropout is not None:
        settings["module_dropout"] = config.module_dropout
    if config.bypass_mode is not None:
        settings["bypass_mode"] = config.bypass_mode
    return settings


CONFIG_BINDING = AdapterMethodConfigBinding(
    config_key="dylora",
    config_type=PeftDyloraConfig,
    runtime_settings_builder=build_runtime_settings,
)
