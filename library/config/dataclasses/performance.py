from dataclasses import dataclass, field
from typing import Optional
from .deepspeed import DeepSpeedConfig


@dataclass
class PrecisionConfig:
    """Numeric precision settings."""
    mixed_precision: str = "no"
    full_fp16: bool = False
    full_bf16: bool = False
    fp8_base: bool = False
    fp8_base_unet: bool = False
    no_half_vae: bool = False
    disable_cuda_reduced_precision_operations: bool = False
    enable_cuda_reduced_precision_operations: bool = False


@dataclass
class MemoryConfig:
    """Memory management settings."""
    gradient_checkpointing: bool = False
    cpu_offload_checkpointing: bool = False
    lowram: bool = False
    highvram: bool = False
    use_ramtorch: bool = field(default=False, metadata={"help": "Use RamTorch to reduce GPU memory usage by keeping model weights on CPU."})
    direct_ramtorch: bool = field(default=False, metadata={"help": "Train orig weights in lyco full module and save diff instead of keep both."})


@dataclass
class AttentionConfig:
    """Attention implementation settings."""
    mem_eff_attn: bool = False
    xformers: bool = False
    sdpa: bool = False
    diffusers_xformers: bool = field(default=False, metadata={"help": "use xformers by diffusers"})


@dataclass
class CompilationConfig:
    """Torch compilation settings."""
    torch_compile: bool = False
    dynamo_backend: str = "inductor"


@dataclass
class DistributedConfig:
    """DDP/distributed training settings."""
    ddp_timeout: Optional[int] = None
    ddp_gradient_as_bucket_view: bool = False
    ddp_static_graph: bool = False


@dataclass
class CachingConfig:
    """Text encoder and model caching settings."""
    cache_text_encoder_outputs: bool = field(default=False, metadata={"help": "cache text encoder outputs"})
    cache_text_encoder_outputs_to_disk: bool = field(default=False, metadata={"help": "cache text encoder outputs to disk"})
    disable_mmap_load_safetensors: bool = field(default=False, metadata={"help": "disable mmap load for safetensors"})


@dataclass
class PerformanceConfig:
    """Performance configuration with organized subcategories."""
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    compilation: CompilationConfig = field(default_factory=CompilationConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
