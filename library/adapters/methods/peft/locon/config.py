from __future__ import annotations

from dataclasses import dataclass, field

from library.adapters.types import AdapterMethodConfigBinding


LOCON_INIT_MODES = ("lycoris_legacy", "zero_delta_he", "random_nonzero")


@dataclass
class PeftLoconConfig:
    """Method-specific LoCon settings under the PEFT shell."""

    rank: int | None = field(default=None, metadata={"help": "LoCon rank/dimensions (higher = more capacity, more VRAM)"})
    alpha: float = field(default=1.0, metadata={"help": "LoCon alpha scaling"})
    dropout: float | None = field(
        default=None,
        metadata={"help": "LoCon plain dropout; mirrors the absorbed LyCORIS semantics for bypass and DoRA paths"},
    )
    rank_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to the LoCon rank dimension"})
    module_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to entire LoCon modules"})
    use_tucker: bool = field(default=False, metadata={"help": "Use Tucker factorization for supported LoCon convolution targets"})
    use_scalar: bool = field(default=False, metadata={"help": "Enable scalar scaling in the LoCon module"})
    init_mode: str = field(
        default="lycoris_legacy",
        metadata={"help": "LoCon initialization policy: lycoris_legacy, zero_delta_he, or random_nonzero"},
    )
    rank_dropout_scale: bool = field(default=False, metadata={"help": "Scale LoCon updates when rank dropout is active"})
    weight_decompose: bool = field(default=False, metadata={"help": "Enable weight decomposition mode"})
    wd_on_output: bool = field(default=True, metadata={"help": "Apply weight decomposition on the output side"})
    bypass_mode: bool | None = field(default=None, metadata={"help": "Optional LoCon bypass-mode toggle"})
    rs_lora: bool = field(default=False, metadata={"help": "Enable RS-LoRA scaling behavior for LoCon"})
    orthogonalize: bool = field(default=False, metadata={"help": "Orthogonalize LoCon factors during training; implies scalar mode"})


def build_runtime_settings(config: PeftLoconConfig) -> dict[str, object]:
    """Translate forward LoCon config into normalized runtime settings."""

    if config.weight_decompose and config.bypass_mode:
        raise ValueError(
            "adapter.peft.locon.bypass_mode cannot be enabled when adapter.peft.locon.weight_decompose is true."
        )
    if config.init_mode not in LOCON_INIT_MODES:
        raise ValueError(
            "adapter.peft.locon.init_mode must be one of "
            f"{', '.join(LOCON_INIT_MODES)}, got {config.init_mode!r}."
        )
    if config.rank is None:
        raise ValueError("adapter.peft.locon.rank must be set to a positive integer.")
    if config.rank <= 0:
        raise ValueError("adapter.peft.locon.rank must be a positive integer when set.")

    settings: dict[str, object] = {
        "adapter_rank": config.rank,
        "adapter_alpha": config.alpha,
    }
    if config.dropout is not None:
        settings["dropout"] = config.dropout
    if config.init_mode != "lycoris_legacy":
        settings["init_mode"] = config.init_mode
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
    if config.orthogonalize:
        settings["orthogonalize"] = True
    return settings


CONFIG_BINDING = AdapterMethodConfigBinding(
    config_key="locon",
    config_type=PeftLoconConfig,
    runtime_settings_builder=build_runtime_settings,
)
