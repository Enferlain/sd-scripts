from __future__ import annotations

from dataclasses import dataclass, field

from library.adapters.types import AdapterMethodConfigBinding


def _validate_probability(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"adapter.peft.ia3.{field_name} must be between 0.0 and 1.0 inclusive.")


@dataclass
class PeftIa3Config:
    """Method-specific IA3 settings under the PEFT shell."""

    train_on_input: bool | None = field(
        default=None,
        metadata={"help": "Optional global IA3 axis override. When unset, IA3 auto-selects input-vs-output scaling per target using the known IA3 target-name patterns."},
    )
    module_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to entire IA3 modules"})
    bypass_mode: bool | None = field(default=None, metadata={"help": "Optional IA3 bypass-mode toggle"})


def build_runtime_settings(config: PeftIa3Config) -> dict[str, object]:
    """Translate forward IA3 config into normalized runtime settings."""

    _validate_probability(config.module_dropout, field_name="module_dropout")

    settings: dict[str, object] = {}
    if config.train_on_input is not None:
        settings["train_on_input"] = config.train_on_input
    if config.module_dropout is not None:
        settings["module_dropout"] = config.module_dropout
    if config.bypass_mode is not None:
        settings["bypass_mode"] = config.bypass_mode
    return settings


CONFIG_BINDING = AdapterMethodConfigBinding(
    config_key="ia3",
    config_type=PeftIa3Config,
    runtime_settings_builder=build_runtime_settings,
)
