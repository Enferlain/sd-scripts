from dataclasses import dataclass, field
from typing import Any


_DEFAULT_LEGACY_ORTHOGRAD_TARGETS = (
    "lora_down.weight",
    "lora_up.weight",
    "lora_down1.weight",
    "lora_up1.weight",
    "lora_down2.weight",
    "lora_up2.weight",
    "a1.weight",
    "a2.weight",
    "b1.weight",
    "b2.weight",
    "c1.weight",
)


@dataclass
class PeftLoraConfig:
    """Method-specific LoRA settings under the PEFT compatibility shell."""

    rank: int | None = field(default=None, metadata={"help": "LoRA rank/dimensions (higher = more capacity, more VRAM)"})
    alpha: float = field(default=1.0, metadata={"help": "LoRA alpha scaling"})
    dropout: float | None = field(default=None, metadata={"help": "Dropout rate for LoRA neurons during training"})
    conv_rank: int | None = field(default=None, metadata={"help": "LoRA rank for Conv2d layers"})
    conv_alpha: float | None = field(default=None, metadata={"help": "LoRA alpha for Conv2d layers"})
    rank_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to the LoRA rank dimension"})
    module_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to entire LoRA modules"})
    # Left over on purpose for the current adapter-system slice: these legacy
    # optimizer-policy knobs are kept as explicit reference points while the
    # repo-owned grouping boundary rejects them in the active PEFT path.
    block_ranks: list[int] | None = field(default=None, metadata={"help": "Per-block LoRA rank values"})
    block_alphas: list[float] | None = field(default=None, metadata={"help": "Per-block LoRA alpha values"})
    conv_block_ranks: list[int] | None = field(default=None, metadata={"help": "Per-block Conv2d LoRA rank values"})
    conv_block_alphas: list[float] | None = field(default=None, metadata={"help": "Per-block Conv2d LoRA alpha values"})
    down_lr_weight: Any | None = field(default=None, metadata={"help": "Legacy block LR weight preset or values for down blocks"})
    mid_lr_weight: Any | None = field(default=None, metadata={"help": "Legacy block LR weight preset or value for the mid block"})
    up_lr_weight: Any | None = field(default=None, metadata={"help": "Legacy block LR weight preset or values for up blocks"})
    block_lr_zero_threshold: float | None = field(default=None, metadata={"help": "Threshold below which block LR is treated as zero"})
    loraplus_lr_ratio: float | None = field(default=None, metadata={"help": "Legacy LoRA+ learning rate ratio"})
    loraplus_unet_lr_ratio: float | None = field(default=None, metadata={"help": "Legacy LoRA+ learning rate ratio for the denoiser"})
    loraplus_text_encoder_lr_ratio: float | None = field(
        default=None, metadata={"help": "Legacy LoRA+ learning rate ratio for text encoders"}
    )


@dataclass
class PeftConfig:
    """PEFT shell/orchestration config plus method-specific adapter config."""

    lora: PeftLoraConfig = field(default_factory=PeftLoraConfig)

    adapter_weights: str | None = field(default=None, metadata={"help": "Path to pretrained adapter weights to continue training"})
    adapter_module: str | None = field(default=None, metadata={"help": "Python module path for adapter (e.g., library.adapters.lora)"})
    adapter_args: list[str] | None = field(default=None, metadata={"help": "Additional adapter arguments as key=value pairs"})
    adapter_rank_from_weights: bool = field(default=False, metadata={"help": "Automatically determine rank from loaded weights"})
    scale_weight_norms: float | None = field(
        default=None, metadata={"help": "Scale weight norms to prevent exploding gradients (1.0 recommended)"}
    )
    base_weights: list[str] | None = field(default=None, metadata={"help": "Adapter weights to merge into model before training"})
    base_weights_multiplier: list[float] | None = field(default=None, metadata={"help": "Multipliers for base_weights when merging"})
    training_comment: str | None = field(default=None, metadata={"help": "Arbitrary comment stored in model metadata"})
    # Left over on purpose as a legacy reference surface for later orthograd
    # follow-up. The active repo-owned PEFT path does not currently consume it.
    orthograd_targets: list[str] | None = field(
        default_factory=lambda: list(_DEFAULT_LEGACY_ORTHOGRAD_TARGETS),
        metadata={"help": "Parameter names to apply orthogonal gradient to"},
    )
