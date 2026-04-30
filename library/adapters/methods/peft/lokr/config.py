from __future__ import annotations

from dataclasses import dataclass, field

from library.adapters.types import AdapterMethodConfigBinding


LOKR_INIT_MODES = ("lycoris_legacy", "zero_delta_he", "random_nonzero")


@dataclass
class PeftLokrConfig:
    """Method-specific LoKr settings under the PEFT shell."""

    rank: int | None = field(default=None, metadata={"help": "LoKr rank/dimensions (higher = more capacity, more VRAM)"})
    alpha: float = field(default=1.0, metadata={"help": "LoKr alpha scaling"})
    dropout: float | None = field(
        default=None,
        metadata={"help": "Deprecated/unsupported for repo-owned LoKr; use rank_dropout or module_dropout instead"},
    )
    rank_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to the rebuilt LoKr diff weight"})
    module_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to entire LoKr modules"})
    use_tucker: bool = field(default=False, metadata={"help": "Use Tucker factorization for supported LoKr convolution targets"})
    use_scalar: bool = field(default=False, metadata={"help": "Enable scalar scaling in the LoKr module"})
    init_mode: str = field(
        default="lycoris_legacy",
        metadata={"help": "LoKr initialization policy: lycoris_legacy, zero_delta_he, or random_nonzero"},
    )
    decompose_both: bool = field(default=False, metadata={"help": "Low-rank decompose both Kronecker factors when rank allows"})
    factor: int = field(default=-1, metadata={"help": "Preferred LoKr factorization divisor/search bound; -1 searches automatically"})
    full_matrix: bool = field(
        default=False,
        metadata={
            "help": "Force full-matrix Kronecker factors instead of rank decomposition; full-matrix mode uses unit scaling "
            "by overriding alpha to rank"
        },
    )
    rank_dropout_scale: bool = field(default=False, metadata={"help": "Scale LoKr updates when rank dropout is active"})
    weight_decompose: bool = field(default=False, metadata={"help": "Enable weight decomposition mode"})
    wd_on_output: bool = field(default=True, metadata={"help": "Apply weight decomposition on the output side"})
    bypass_mode: bool | None = field(default=None, metadata={"help": "Optional LoKr bypass-mode toggle"})
    rs_lora: bool = field(default=False, metadata={"help": "Enable RS-LoRA scaling behavior for LoKr"})
    unbalanced_factorization: bool = field(default=False, metadata={"help": "Swap LoKr output factors after factorization"})
    orthogonalize: bool = field(default=False, metadata={"help": "Orthogonalize LoKr factors during training; implies scalar mode"})


def _validate_probability(value: float | None, *, field_name: str) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"adapter.peft.lokr.{field_name} must be between 0.0 and 1.0 inclusive.")


def build_runtime_settings(config: PeftLokrConfig) -> dict[str, object]:
    """Translate forward LoKr config into normalized runtime settings."""

    if config.weight_decompose and config.bypass_mode:
        raise ValueError("adapter.peft.lokr.bypass_mode cannot be enabled when adapter.peft.lokr.weight_decompose is true.")
    if config.dropout is not None:
        raise ValueError("adapter.peft.lokr.dropout is not supported; use rank_dropout or module_dropout instead.")
    if config.init_mode not in LOKR_INIT_MODES:
        raise ValueError(f"adapter.peft.lokr.init_mode must be one of {', '.join(LOKR_INIT_MODES)}, got {config.init_mode!r}.")
    if config.rank is None:
        raise ValueError("adapter.peft.lokr.rank must be set to a positive integer.")
    if config.rank <= 0:
        raise ValueError("adapter.peft.lokr.rank must be a positive integer when set.")
    if config.factor == 0:
        raise ValueError("adapter.peft.lokr.factor must be -1 or a positive integer.")
    _validate_probability(config.rank_dropout, field_name="rank_dropout")
    _validate_probability(config.module_dropout, field_name="module_dropout")

    settings: dict[str, object] = {
        "adapter_rank": config.rank,
        "adapter_alpha": config.alpha,
    }
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
    if config.decompose_both:
        settings["decompose_both"] = True
    if config.factor != -1:
        settings["factor"] = config.factor
    if config.full_matrix:
        settings["full_matrix"] = True
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
    if config.unbalanced_factorization:
        settings["unbalanced_factorization"] = True
    if config.orthogonalize:
        settings["orthogonalize"] = True
    return settings


CONFIG_BINDING = AdapterMethodConfigBinding(
    config_key="lokr",
    config_type=PeftLokrConfig,
    runtime_settings_builder=build_runtime_settings,
)
