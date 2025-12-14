from dataclasses import dataclass, field
from typing import Optional
from .deepspeed import DeepspeedConfig


@dataclass
class PerformanceConfig:
    mem_eff_attn: bool = False
    torch_compile: bool = False
    dynamo_backend: str = "inductor"
    xformers: bool = False
    sdpa: bool = False
    gradient_checkpointing: bool = False
    mixed_precision: str = "no"
    full_fp16: bool = False
    full_bf16: bool = False
    fp8_base: bool = False
    ddp_timeout: Optional[int] = None
    ddp_gradient_as_bucket_view: bool = False
    ddp_static_graph: bool = False
    lowram: bool = False
    highvram: bool = False
    disable_cuda_reduced_precision_operations: bool = False
    enable_cuda_reduced_precision_operations: bool = False
    deepspeed: DeepspeedConfig = field(default_factory=DeepspeedConfig)
