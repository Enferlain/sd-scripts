from dataclasses import dataclass, field


@dataclass
class PrecisionConfig:
    """Numeric precision settings."""

    mixed_precision: str = field(default="no", metadata={"help": "Mixed precision mode: no, fp16, or bf16"})
    full_fp16: bool = field(default=False, metadata={"help": "Enable full fp16 training (requires mixed_precision=fp16)"})
    full_bf16: bool = field(default=False, metadata={"help": "Enable full bf16 training (requires mixed_precision=bf16)"})
    fp8_base: bool = field(default=False, metadata={"help": "Use fp8 for base model to reduce VRAM (requires torch>=2.1)"})
    fp8_base_unet: bool = field(default=False, metadata={"help": "Use fp8 for UNet only (text encoder stays in weight_dtype)"})
    no_half_vae: bool = field(default=False, metadata={"help": "Keep VAE in float32 precision"})
    # NOTE: These two flags control TF32 and reduced precision matmul. They are mutually exclusive.
    # If neither is set, PyTorch defaults are used. Consider consolidating into a single enum field.
    disable_cuda_reduced_precision_operations: bool = field(
        default=False, metadata={"help": "Disable TF32 and reduced precision for maximum accuracy"}
    )
    enable_cuda_reduced_precision_operations: bool = field(
        default=False, metadata={"help": "Enable TF32 and reduced precision for faster computation"}
    )


@dataclass
class MemoryConfig:
    """Memory management settings."""

    gradient_checkpointing: bool = field(default=False, metadata={"help": "Enable gradient checkpointing to reduce VRAM usage"})
    cpu_offload_checkpointing: bool = field(default=False, metadata={"help": "Offload gradient checkpoints to CPU for more VRAM savings"})
    lowram: bool = field(default=False, metadata={"help": "Load models directly to GPU (for systems with limited RAM but sufficient VRAM)"})
    # TODO: highvram is currently unused - the HIGH_VRAM constant in constants.py is never set from config.
    # This should either be wired up to set the constant, or removed entirely.
    highvram: bool = field(default=False, metadata={"help": "Skip memory cleanup after operations (currently unused)"})
    use_ramtorch: bool = field(default=False, metadata={"help": "Use RamTorch to reduce GPU memory usage by keeping model weights on CPU"})
    direct_ramtorch: bool = field(
        default=False, metadata={"help": "Train original weights in LyCORIS full module and save diff instead of keeping both"}
    )


@dataclass
class AttentionConfig:
    """Attention implementation settings."""

    mem_eff_attn: bool = field(default=False, metadata={"help": "Use memory-efficient attention implementation"})
    xformers: bool = field(default=False, metadata={"help": "Use xformers for attention (requires xformers package)"})
    sdpa: bool = field(default=False, metadata={"help": "Use PyTorch SDPA (scaled dot product attention)"})
    diffusers_xformers: bool = field(default=False, metadata={"help": "Use xformers via diffusers library"})


@dataclass
class CompilationConfig:
    """Torch compilation settings."""

    torch_compile: bool = field(default=False, metadata={"help": "Enable torch.compile for model optimization"})
    dynamo_backend: str = field(default="inductor", metadata={"help": "Torch dynamo backend: inductor, eager, aot_eager, etc."})


@dataclass
class DistributedConfig:
    """DDP/distributed training settings."""

    ddp_timeout: int | None = field(default=None, metadata={"help": "Distributed training timeout in seconds"})
    ddp_gradient_as_bucket_view: bool = field(default=False, metadata={"help": "Use gradient as bucket view for DDP memory optimization"})
    ddp_static_graph: bool = field(default=False, metadata={"help": "Enable DDP static graph optimization"})


@dataclass
class CachingConfig:
    """Text encoder and model caching settings."""

    cache_text_encoder_outputs: bool = field(default=False, metadata={"help": "Cache text encoder outputs to reduce VRAM usage"})
    cache_text_encoder_outputs_to_disk: bool = field(default=False, metadata={"help": "Cache text encoder outputs to disk"})
    disable_mmap_load_safetensors: bool = field(default=False, metadata={"help": "Disable memory-mapped loading for safetensors files"})


@dataclass
class DeepSpeedConfig:
    """DeepSpeed training settings."""

    deepspeed: bool = field(default=False, metadata={"help": "Enable DeepSpeed training"})
    zero_stage: int = field(default=2, metadata={"help": "ZeRO optimization stage: 0, 1, 2, or 3"})
    offload_optimizer_device: str | None = field(default=None, metadata={"help": "Offload optimizer state to: none, cpu, or nvme"})
    offload_optimizer_nvme_path: str | None = field(default=None, metadata={"help": "NVMe path for optimizer offload (e.g., /nvme)"})
    offload_param_device: str | None = field(default=None, metadata={"help": "Offload parameters to: none, cpu, or nvme"})
    offload_param_nvme_path: str | None = field(default=None, metadata={"help": "NVMe path for parameter offload (e.g., /nvme)"})
    zero3_init_flag: bool = field(default=False, metadata={"help": "Enable deepspeed.zero.Init for constructing massive models"})
    zero3_save_16bit_model: bool = field(default=False, metadata={"help": "Save 16-bit model weights with ZeRO-3"})
    fp16_master_weights_and_gradients: bool = field(
        default=False, metadata={"help": "Keep fp16 master weights and gradients (requires compatible optimizer)"}
    )


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
