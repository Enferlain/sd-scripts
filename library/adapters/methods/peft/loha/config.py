from __future__ import annotations

from dataclasses import dataclass, field

from library.adapters.types import AdapterMethodConfigBinding


@dataclass
class PeftLohaConfig:
    """Method-specific LoHa settings under the PEFT shell."""

    rank: int | None = field(default=None, metadata={"help": "LoHa rank/dimensions (higher = more capacity, more VRAM)"})
    alpha: float = field(default=1.0, metadata={"help": "LoHa alpha scaling"})
    dropout: float | None = field(default=None, metadata={"help": "Dropout rate for LoHa neurons during training"})
    rank_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to the LoHa rank dimension"})
    module_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to entire LoHa modules"})
    use_tucker: bool = field(default=False, metadata={"help": "Use Tucker factorization for supported LoHa targets"})
    use_scalar: bool = field(default=False, metadata={"help": "Enable scalar scaling in the LoHa module"})
    rank_dropout_scale: bool = field(default=False, metadata={"help": "Scale LoHa updates when rank dropout is active"})
    weight_decompose: bool = field(default=False, metadata={"help": "Enable weight decomposition mode"})
    wd_on_output: bool = field(default=True, metadata={"help": "Apply weight decomposition on the output side"})
    bypass_mode: bool | None = field(default=None, metadata={"help": "Optional LoHa bypass-mode toggle"})
    rs_lora: bool = field(default=False, metadata={"help": "Enable RS-LoRA scaling behavior for LoHa"})


def build_runtime_settings(config: PeftLohaConfig) -> dict[str, object]:
    """Translate forward LoHa config into normalized runtime settings."""

    settings: dict[str, object] = {
        "adapter_rank": config.rank,
        "adapter_alpha": config.alpha,
        "neuron_dropout": config.dropout,
    }
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
    if config.weight_decompose:
        settings["weight_decompose"] = True
    if config.wd_on_output is not True:
        settings["wd_on_output"] = config.wd_on_output
    if config.bypass_mode is not None:
        settings["bypass_mode"] = config.bypass_mode
    if config.rs_lora:
        settings["rs_lora"] = True
    return settings


CONFIG_BINDING = AdapterMethodConfigBinding(
    config_key="loha",
    config_type=PeftLohaConfig,
    runtime_settings_builder=build_runtime_settings,
)
