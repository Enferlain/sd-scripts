from __future__ import annotations

from dataclasses import dataclass, field

from library.adapters.types import AdapterMethodConfigBinding


def _validate_probability(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"adapter.peft.vera.{field_name} must be between 0.0 and 1.0 inclusive.")


@dataclass
class PeftVeraConfig:
    """Method-specific VeRA settings under the PEFT shell."""

    rank: int | None = field(default=None, metadata={"help": "Shared VeRA projection rank"})
    dropout: float | None = field(default=None, metadata={"help": "Dropout applied before the shared VeRA projection"})
    d_initial: float = field(default=0.1, metadata={"help": "Initial value for the learned VeRA lambda_d vector"})
    projection_prng_key: int = field(
        default=0,
        metadata={"help": "Deterministic seed used to initialize shared VeRA projection tensors"},
    )
    save_projection: bool = field(
        default=True,
        metadata={"help": "Persist shared VeRA projection tensors in the adapter artifact"},
    )
    init_weights: bool = field(
        default=True,
        metadata={"help": "Whether to reset VeRA lambda vectors to the HF-style zero-delta initialization"},
    )


def build_runtime_settings(config: PeftVeraConfig) -> dict[str, object]:
    """Translate forward VeRA config into normalized runtime settings."""

    if config.rank is None:
        raise ValueError("adapter.peft.vera.rank must be set to a positive integer.")
    if config.rank <= 0:
        raise ValueError("adapter.peft.vera.rank must be a positive integer when set.")
    _validate_probability(config.dropout, field_name="dropout")

    settings: dict[str, object] = {
        "adapter_rank": config.rank,
        "d_initial": config.d_initial,
        "projection_prng_key": config.projection_prng_key,
        "save_projection": config.save_projection,
        "init_weights": config.init_weights,
    }
    if config.dropout is not None:
        settings["dropout"] = config.dropout
    return settings


CONFIG_BINDING = AdapterMethodConfigBinding(
    config_key="vera",
    config_type=PeftVeraConfig,
    runtime_settings_builder=build_runtime_settings,
)
