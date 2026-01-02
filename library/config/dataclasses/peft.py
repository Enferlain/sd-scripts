from dataclasses import dataclass, field
from typing import Any

@dataclass
class PeftConfig:
    """PEFT/LoRA adapter configuration."""
    adapter_rank: int | None = field(default=None, metadata={"help": "Adapter rank/dimensions (higher = more capacity, more VRAM)"})
    adapter_alpha: float = field(default=1.0, metadata={"help": "Alpha for LoRA weight scaling (1 = same as rank)"})
    neuron_dropout: float | None = field(default=None, metadata={"help": "Dropout rate for neurons during training (0-1, None to disable)"})
    adapter_weights: str | None = field(default=None, metadata={"help": "Path to pretrained adapter weights to continue training"})
    adapter_module: str | None = field(default=None, metadata={"help": "Python module path for adapter (e.g., library.adapters.lora)"})
    adapter_args: list[str] | None = field(default=None, metadata={"help": "Additional adapter arguments as key=value pairs"})
    adapter_rank_from_weights: bool = field(default=False, metadata={"help": "Automatically determine rank from loaded weights"})
    scale_weight_norms: float | None = field(default=None, metadata={"help": "Scale weight norms to prevent exploding gradients (1.0 recommended)"})
    base_weights: list[str] | None = field(default=None, metadata={"help": "Adapter weights to merge into model before training"})
    base_weights_multiplier: list[float] | None = field(default=None, metadata={"help": "Multipliers for base_weights when merging"})
    training_comment: str | None = field(default=None, metadata={"help": "Arbitrary comment stored in model metadata"})
    orthograd_targets: list[str] | None = field(default=None, metadata={"help": "Parameter names to apply orthogonal gradient to"})

    # LoRA-specific fields
    conv_dim: int | None = field(default=None, metadata={"help": "Rank for Conv2d layers (default: same as adapter_rank)"})
    conv_alpha: float | None = field(default=None, metadata={"help": "Alpha for Conv2d layers (default: same as adapter_alpha)"})
    rank_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to LoRA rank dimension"})
    module_dropout: float | None = field(default=None, metadata={"help": "Dropout applied to entire LoRA modules"})
    block_dims: list[int] | None = field(default=None, metadata={"help": "Per-block rank values for UNet blocks"})
    block_alphas: list[float] | None = field(default=None, metadata={"help": "Per-block alpha values for UNet blocks"})
    conv_block_dims: list[int] | None = field(default=None, metadata={"help": "Per-block Conv2d rank values"})
    conv_block_alphas: list[float] | None = field(default=None, metadata={"help": "Per-block Conv2d alpha values"})
    # LR Weights can be list of floats or str (preset name)
    down_lr_weight: Any | None = field(default=None, metadata={"help": "LR weight for down blocks (list or preset name)"})
    mid_lr_weight: Any | None = field(default=None, metadata={"help": "LR weight for mid block (float or preset name)"})
    up_lr_weight: Any | None = field(default=None, metadata={"help": "LR weight for up blocks (list or preset name)"})
    block_lr_zero_threshold: float | None = field(default=None, metadata={"help": "Threshold below which block LR is treated as zero"})
    loraplus_lr_ratio: float | None = field(default=None, metadata={"help": "LoRA+ learning rate ratio (B matrix LR multiplier)"})
    loraplus_unet_lr_ratio: float | None = field(default=None, metadata={"help": "LoRA+ LR ratio for UNet specifically"})
    loraplus_text_encoder_lr_ratio: float | None = field(default=None, metadata={"help": "LoRA+ LR ratio for text encoder specifically"})
