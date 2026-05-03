from __future__ import annotations

from dataclasses import dataclass, field

from library.adapters.types import AdapterMethodConfigBinding


def _validate_probability(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"adapter.peft.abba.{field_name} must be between 0.0 and 1.0 inclusive.")


@dataclass
class PeftAbbaConfig:
    """Method-specific ABBA settings under the PEFT shell."""

    rank: int | None = field(
        default=None,
        metadata={"help": "ABBA rank/dimensions before the method-local split into two factor pairs"},
    )
    alpha: float = field(default=1.0, metadata={"help": "ABBA alpha scaling"})
    dropout: float | None = field(
        default=None,
        metadata={
            "help": "Plain ABBA dropout probability. In weight-decompose mode this drops the forward input before the merged ABBA weight is applied."
        },
    )
    rank_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to the ABBA output dimension"})
    module_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to entire ABBA modules"})
    use_scalar: bool = field(default=False, metadata={"help": "Enable scalar scaling in the ABBA module"})
    rank_dropout_scale: bool = field(default=False, metadata={"help": "Scale ABBA updates when rank dropout is active"})
    weight_decompose: bool = field(default=False, metadata={"help": "Enable weight decomposition mode"})
    wd_on_output: bool = field(default=True, metadata={"help": "Apply weight decomposition on the output side"})
    bypass_mode: bool | None = field(default=None, metadata={"help": "Optional ABBA bypass-mode toggle"})


def build_runtime_settings(config: PeftAbbaConfig) -> dict[str, object]:
    """Translate forward ABBA config into normalized runtime settings."""

    if config.rank is None:
        raise ValueError("adapter.peft.abba.rank must be set to an integer greater than or equal to 2.")
    if config.rank < 2:
        raise ValueError("adapter.peft.abba.rank must be greater than or equal to 2 when set.")
    if config.weight_decompose and config.bypass_mode:
        raise ValueError("adapter.peft.abba.bypass_mode cannot be enabled when adapter.peft.abba.weight_decompose is true.")
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
    if config.use_scalar:
        settings["use_scalar"] = True
    if config.rank_dropout_scale:
        settings["rank_dropout_scale"] = True
    if config.weight_decompose:
        settings["weight_decompose"] = True
    if config.wd_on_output is not True:
        settings["wd_on_output"] = config.wd_on_output
    if config.bypass_mode is not None:
        settings["bypass_mode"] = config.bypass_mode
    return settings


CONFIG_BINDING = AdapterMethodConfigBinding(
    config_key="abba",
    config_type=PeftAbbaConfig,
    runtime_settings_builder=build_runtime_settings,
)
