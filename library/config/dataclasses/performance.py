from dataclasses import dataclass, field
from typing import Optional
from .deepspeed import DeepSpeedConfig


@dataclass
class PerformanceConfig:
    mem_eff_attn: bool = False
    torch_compile: bool = False
    dynamo_backend: str = "inductor"
    xformers: bool = False
    sdpa: bool = False
    gradient_checkpointing: bool = False
    cpu_offload_checkpointing: bool = False
    mixed_precision: str = "no"
    full_fp16: bool = False
    full_bf16: bool = False
    fp8_base: bool = False
    fp8_base_unet: bool = False
    no_half_vae: bool = False
    ddp_timeout: Optional[int] = None
    ddp_gradient_as_bucket_view: bool = False
    ddp_static_graph: bool = False
    lowram: bool = False
    highvram: bool = False
    disable_cuda_reduced_precision_operations: bool = False
    enable_cuda_reduced_precision_operations: bool = False
    use_ramtorch: bool = field(default=False, metadata={"help": "Use RamTorch to reduce GPU memory usage by keeping model weights on CPU."})
    direct_ramtorch: bool = field(default=False, metadata={"help": "Train orig weights in lyco full module and save diff instead of keep both."})
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
