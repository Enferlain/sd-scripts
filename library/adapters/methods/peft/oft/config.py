from __future__ import annotations

from dataclasses import dataclass, field

from library.adapters.types import AdapterMethodConfigBinding


def _validate_probability(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"adapter.peft.oft.{field_name} must be between 0.0 and 1.0 inclusive.")


@dataclass
class PeftOftConfig:
    """Method-specific OFT settings under the PEFT shell."""

    factor: int | None = field(
        default=None,
        metadata={
            "help": "Preferred OFT factorization hint; this is passed into the LyCORIS-style factorization search rather than "
            "acting as a literal rank or fixed block size"
        },
    )
    constraint: float = field(
        default=0.0,
        metadata={"help": "Constraint strength for the skew-symmetric OFT blocks; 0 disables clamping"},
    )
    rescaled: bool = field(
        default=False,
        metadata={"help": "Learn an additional per-output rescale parameter, following the absorbed LyCORIS OFT design"},
    )
    dropout: float | None = field(
        default=None,
        metadata={"help": "OFT plain dropout; preserved as a method-local semantic from the absorbed LyCORIS behavior"},
    )
    rank_dropout: float | None = field(
        default=None,
        metadata={"help": "Dropout applied to the rebuilt OFT transform in the non-bypass execution path"},
    )
    module_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to entire OFT modules"})
    bypass_mode: bool | None = field(default=None, metadata={"help": "Optional OFT bypass-mode toggle"})


def build_runtime_settings(config: PeftOftConfig) -> dict[str, object]:
    """Translate forward OFT config into normalized runtime settings."""

    if config.factor is None:
        raise ValueError("adapter.peft.oft.factor must be set to a positive integer.")
    if config.factor <= 0:
        raise ValueError("adapter.peft.oft.factor must be a positive integer when set.")
    if config.constraint < 0.0:
        raise ValueError("adapter.peft.oft.constraint must be non-negative.")
    _validate_probability(config.dropout, field_name="dropout")
    _validate_probability(config.rank_dropout, field_name="rank_dropout")
    _validate_probability(config.module_dropout, field_name="module_dropout")

    settings: dict[str, object] = {
        "factor": config.factor,
        "constraint": config.constraint,
    }
    if config.rescaled:
        settings["rescaled"] = True
    if config.dropout is not None:
        settings["dropout"] = config.dropout
    if config.rank_dropout is not None:
        settings["rank_dropout"] = config.rank_dropout
    if config.module_dropout is not None:
        settings["module_dropout"] = config.module_dropout
    if config.bypass_mode is not None:
        settings["bypass_mode"] = config.bypass_mode
    return settings


CONFIG_BINDING = AdapterMethodConfigBinding(
    config_key="oft",
    config_type=PeftOftConfig,
    runtime_settings_builder=build_runtime_settings,
)
