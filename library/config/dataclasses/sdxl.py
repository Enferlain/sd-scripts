from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SDXLConfig:
    cache_text_encoder_outputs: bool = field(default=False, metadata={"help": "cache text encoder outputs"})
    cache_text_encoder_outputs_to_disk: bool = field(default=False, metadata={"help": "cache text encoder outputs to disk"})
    disable_mmap_load_safetensors: bool = field(default=False, metadata={"help": "disable mmap load for safetensors"})
    train_text_encoder: bool = field(default=False, metadata={"help": "train text encoder"})
    fused_optimizer_groups: Optional[int] = field(default=None, metadata={"help": "number of optimizers for fused backward pass and optimizer step"})

