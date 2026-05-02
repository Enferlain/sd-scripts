from __future__ import annotations

from dataclasses import dataclass, field

from library.adapters.types import AdapterMethodConfigBinding


def _validate_probability(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"adapter.peft.glora.{field_name} must be between 0.0 and 1.0 inclusive.")


@dataclass
class PeftGloraConfig:
    """Method-specific GLoRA settings under the PEFT shell."""

    rank: int | None = field(default=None, metadata={"help": "GLoRA rank/dimensions (higher = more capacity, more VRAM)"})
    alpha: float = field(default=1.0, metadata={"help": "GLoRA alpha scaling"})
    dropout: float | None = field(
        default=None,
        metadata={
            "help": "Plain GLoRA dropout probability. In rebuilt-weight mode this drops the forward input before the GLoRA branches are applied, rather than dropping reconstructed branch weights."
        },
    )
    rank_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to the GLoRA rank dimension"})
    module_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to entire GLoRA modules"})
    use_tucker: bool = field(default=False, metadata={"help": "Use Tucker factorization for the GLoRA B branch on supported convolution targets"})
    use_scalar: bool = field(default=False, metadata={"help": "Enable scalar scaling in the GLoRA module"})
    rank_dropout_scale: bool = field(default=False, metadata={"help": "Scale GLoRA updates when rank dropout is active"})
    bypass_mode: bool | None = field(default=None, metadata={"help": "Optional GLoRA bypass-mode toggle"})
    rs_lora: bool = field(default=False, metadata={"help": "Enable RS-LoRA scaling behavior for GLoRA"})
    orthogonalize: bool = field(default=False, metadata={"help": "Orthogonalize GLoRA factors during training; implies scalar mode"})


def build_runtime_settings(config: PeftGloraConfig) -> dict[str, object]:
    """Translate forward GLoRA config into normalized runtime settings."""

    if config.rank is None:
        raise ValueError("adapter.peft.glora.rank must be set to a positive integer.")
    if config.rank <= 0:
        raise ValueError("adapter.peft.glora.rank must be a positive integer when set.")
    _validate_probability(config.dropout, field_name="dropout")
    _validate_probability(config.rank_dropout, field_name="rank_dropout")
    _validate_probability(config.module_dropout, field_name="module_dropout")

    settings: dict[str, object] = {
        "adapter_rank": config.rank,
        "adapter_alpha": config.alpha,
    }
    if config.dropout is not None:
        settings["dropout"] = config.dropout
    if config.rank_dropout is not None:
        settings["rank_dropout"] = config.rank_dropout
    if config.module_dropout is not None:
        settings["module_dropout"] = config.module_dropout
    if config.use_tucker:
        settings["use_tucker"] = True
    if config.use_scalar:
        settings["use_scalar"] = True
    if config.rank_dropout_scale:
        settings["rank_dropout_scale"] = True
    if config.bypass_mode is not None:
        settings["bypass_mode"] = config.bypass_mode
    if config.rs_lora:
        settings["rs_lora"] = True
    if config.orthogonalize:
        settings["orthogonalize"] = True
    return settings


CONFIG_BINDING = AdapterMethodConfigBinding(
    config_key="glora",
    config_type=PeftGloraConfig,
    runtime_settings_builder=build_runtime_settings,
)
