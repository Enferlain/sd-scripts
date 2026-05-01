from __future__ import annotations

from dataclasses import dataclass, field

from library.adapters.types import AdapterMethodConfigBinding


def _validate_probability(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"adapter.peft.boft.{field_name} must be between 0.0 and 1.0 inclusive.")


@dataclass
class PeftBoftConfig:
    """Method-specific BOFT settings under the PEFT shell."""

    factor: int | None = field(
        default=None,
        metadata={
            "help": "Preferred BOFT factorization hint; this is passed into the LyCORIS-style power-of-two search rather than "
            "acting as a literal stage count or fixed block size"
        },
    )
    constraint: float = field(
        default=0.0,
        metadata={"help": "Constraint strength for the BOFT skew-symmetric blocks; 0 disables clamping"},
    )
    num_stages: int | None = field(
        default=None,
        metadata={
            "help": "Optional partial butterfly depth. When omitted, BOFT uses the full stage count implied by the factorized block layout."
        },
    )
    rescaled: bool = field(
        default=False,
        metadata={"help": "Learn an additional per-output rescale parameter, following the absorbed LyCORIS BOFT design"},
    )
    dropout: float | None = field(
        default=None,
        metadata={"help": "BOFT multiplicative dropout on butterfly rotation blocks/components"},
    )
    module_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to entire BOFT modules"})
    bypass_mode: bool | None = field(default=None, metadata={"help": "Optional BOFT bypass-mode toggle"})


def build_runtime_settings(config: PeftBoftConfig) -> dict[str, object]:
    """Translate forward BOFT config into normalized runtime settings."""

    if config.factor is None:
        raise ValueError("adapter.peft.boft.factor must be set to a positive integer.")
    if config.factor <= 0:
        raise ValueError("adapter.peft.boft.factor must be a positive integer when set.")
    if config.constraint < 0.0:
        raise ValueError("adapter.peft.boft.constraint must be non-negative.")
    if config.num_stages is not None and config.num_stages <= 0:
        raise ValueError("adapter.peft.boft.num_stages must be a positive integer when set.")
    _validate_probability(config.dropout, field_name="dropout")
    _validate_probability(config.module_dropout, field_name="module_dropout")

    settings: dict[str, object] = {
        "factor": config.factor,
        "constraint": config.constraint,
    }
    if config.num_stages is not None:
        settings["num_stages"] = config.num_stages
    if config.rescaled:
        settings["rescaled"] = True
    if config.dropout is not None:
        settings["dropout"] = config.dropout
    if config.module_dropout is not None:
        settings["module_dropout"] = config.module_dropout
    if config.bypass_mode is not None:
        settings["bypass_mode"] = config.bypass_mode
    return settings


CONFIG_BINDING = AdapterMethodConfigBinding(
    config_key="boft",
    config_type=PeftBoftConfig,
    runtime_settings_builder=build_runtime_settings,
)
