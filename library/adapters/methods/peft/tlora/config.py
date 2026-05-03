from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from library.adapters.types import AdapterMethodConfigBinding


TloraSigType = Literal["principal", "last", "middle"]
_VALID_SIG_TYPES = {"principal", "last", "middle"}


def _validate_probability(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"adapter.peft.tlora.{field_name} must be between 0.0 and 1.0 inclusive.")


@dataclass
class PeftTloraConfig:
    """Method-specific TLora settings under the PEFT shell."""

    rank: int | None = field(default=None, metadata={"help": "TLora rank"})
    alpha: float = field(
        default=1.0,
        metadata={"help": "TLora alpha scaling. Set this equal to rank to match the original paper's unit scaling."},
    )
    dropout: float | None = field(
        default=None,
        metadata={
            "help": "Plain TLora dropout probability. In the active timestep-aware training path this applies on the bypass delta output."
        },
    )
    module_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to entire TLora modules"})
    use_scalar: bool = field(default=False, metadata={"help": "Enable scalar scaling in the TLora module"})
    bypass_mode: bool | None = field(default=None, metadata={"help": "Optional TLora bypass-mode toggle"})
    sig_type: TloraSigType = field(default="principal", metadata={"help": "Singular-vector selection policy"})
    use_data_init: bool = field(
        default=True,
        metadata={"help": "Initialize TLora factors from the target weight SVD instead of a random matrix SVD"},
    )
    min_rank: int = field(
        default=1,
        metadata={"help": "Minimum active rank at the highest-noise timesteps"},
    )
    mask_alpha: float = field(
        default=1.0,
        metadata={"help": "Exponent controlling how aggressively active rank grows as timesteps decrease"},
    )
    max_timestep: int = field(default=1000, metadata={"help": "Maximum timestep used when building TLora rank masks"})


def build_runtime_settings(config: PeftTloraConfig) -> dict[str, object]:
    """Translate forward TLora config into normalized runtime settings."""

    if config.rank is None:
        raise ValueError("adapter.peft.tlora.rank must be set to a positive integer.")
    if config.rank <= 0:
        raise ValueError("adapter.peft.tlora.rank must be a positive integer when set.")
    _validate_probability(config.dropout, field_name="dropout")
    _validate_probability(config.module_dropout, field_name="module_dropout")
    if config.sig_type not in _VALID_SIG_TYPES:
        valid = ", ".join(sorted(_VALID_SIG_TYPES))
        raise ValueError(f"adapter.peft.tlora.sig_type must be one of: {valid}.")
    if config.min_rank <= 0:
        raise ValueError("adapter.peft.tlora.min_rank must be a positive integer.")
    if config.min_rank > config.rank:
        raise ValueError("adapter.peft.tlora.min_rank cannot exceed adapter.peft.tlora.rank.")
    if config.mask_alpha <= 0.0:
        raise ValueError("adapter.peft.tlora.mask_alpha must be greater than 0.0.")
    if config.max_timestep <= 0:
        raise ValueError("adapter.peft.tlora.max_timestep must be a positive integer.")

    settings: dict[str, object] = {
        "adapter_rank": config.rank,
        "adapter_alpha": config.alpha,
        "sig_type": config.sig_type,
        "use_data_init": config.use_data_init,
        "mask_min_rank": config.min_rank,
        "mask_alpha": config.mask_alpha,
        "mask_max_timestep": config.max_timestep,
    }
    if config.dropout is not None:
        settings["dropout"] = config.dropout
    if config.module_dropout is not None:
        settings["module_dropout"] = config.module_dropout
    if config.use_scalar:
        settings["use_scalar"] = True
    if config.bypass_mode is not None:
        settings["bypass_mode"] = config.bypass_mode
    return settings


CONFIG_BINDING = AdapterMethodConfigBinding(
    config_key="tlora",
    config_type=PeftTloraConfig,
    runtime_settings_builder=build_runtime_settings,
)
