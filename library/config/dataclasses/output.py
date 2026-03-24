from dataclasses import dataclass, field


@dataclass
class SavingConfig:
    """Checkpoint and model saving settings."""

    output_dir: str = field(default="outputs", metadata={"help": "Directory for saving checkpoints and models"})
    output_name: str | None = field(default=None, metadata={"help": "Base name for output files (default: epoch/step/last)"})
    save_precision: str | None = field(default=None, metadata={"help": "Model save precision: fp16, bf16, or float"})
    save_every_n_epochs: int = field(default=1, metadata={"help": "Save checkpoint every N epochs"})
    save_every_n_steps: int | None = field(default=None, metadata={"help": "Save checkpoint every N training steps (disabled by default)"})
    save_n_epoch_ratio: int | None = field(default=None, metadata={"help": "Compute save_every_n_epochs as total_epochs / ratio"})
    save_last_n_epochs: int | None = field(default=None, metadata={"help": "Keep only the last N epoch checkpoints (delete older)"})
    save_last_n_epochs_state: int | None = field(default=None, metadata={"help": "Keep only the last N epoch states (delete older)"})
    save_last_n_steps: int | None = field(default=None, metadata={"help": "Keep only the last N step checkpoints (delete older)"})
    save_last_n_steps_state: int | None = field(default=None, metadata={"help": "Keep only the last N step states (delete older)"})
    save_state: bool = field(default=False, metadata={"help": "Save training state for resuming (optimizer, scheduler, etc.)"})
    save_state_on_train_end: bool = field(default=False, metadata={"help": "Save training state when training completes"})
    resume: str | None = field(default=None, metadata={"help": "Path to training state to resume from"})
    save_model_as: str | None = field(default=None, metadata={"help": "Model format: safetensors, ckpt, diffusers, diffusers_safetensors"})
    use_safetensors: bool = field(default=False, metadata={"help": "Use safetensors format for saving (deprecated, use save_model_as)"})
    no_metadata: bool = field(default=False, metadata={"help": "Don't save training metadata in model file"})
    hash_algorithm: str = field(
        default="sha256", metadata={"help": "Hash algorithm for model checksums: md5, sha1, sha256, sha512, blake3"}
    )


@dataclass
class ResourceMonitorConfig:
    """Resource monitor settings."""

    enabled: bool = field(default=False, metadata={"help": "Enable resource monitor hooks"})
    mode: str = field(default="off", metadata={"help": "Resource monitor mode: off, basic, sampled, deep"})
    log_every_n_steps: int = field(default=0, metadata={"help": "Emit resource step logs every N steps (0 disables)"})
    sample_interval_sec: float = field(default=1.0, metadata={"help": "Background sampler interval in seconds (sampled/deep modes)"})
    rank_scope: str = field(default="main", metadata={"help": "Which ranks emit resource logs: main or all"})
    device_scope: str = field(default="local", metadata={"help": "Device scope for resource collection: local or all_visible"})
    output_jsonl: str | None = field(default=None, metadata={"help": "Optional JSONL output path for resource events"})
    jsonl_flush_mode: str = field(default="auto", metadata={"help": "JSONL flush mode: auto, line, batch"})
    jsonl_flush_every_n_events: int = field(default=50, metadata={"help": "When batch flush mode is used, flush every N events"})
    queue_maxsize: int = field(default=1024, metadata={"help": "Max queued sampled events before drop policy applies"})
    drop_policy: str = field(default="drop_oldest", metadata={"help": "Queue policy: drop_oldest, drop_newest, block"})
    max_collection_ms: float = field(default=0.0, metadata={"help": "Optional per-collection budget in ms (0 disables enforcement)"})
    phase_summary: bool = field(default=True, metadata={"help": "Emit phase start/end resource summaries"})
    component_breakdown: bool = field(default=True, metadata={"help": "Emit startup component memory estimates"})
    deep_window_steps: int = field(default=0, metadata={"help": "Deep mode active window in steps (0 disables)"})
    deep_window_seconds: float = field(default=0.0, metadata={"help": "Deep mode active window in seconds (0 disables)"})


@dataclass
class LoggingConfig:
    """Logging and tracking settings."""

    logging_dir: str | None = field(default=None, metadata={"help": "Directory for TensorBoard/W&B logs"})
    log_with: str | None = field(default=None, metadata={"help": "Tracker to use: tensorboard, wandb, or all"})
    log_prefix: str | None = field(default=None, metadata={"help": "Prefix for log directory name"})
    log_tracker_name: str | None = field(default=None, metadata={"help": "Project/experiment name for tracker"})
    wandb_run_name: str | None = field(default=None, metadata={"help": "Run name for Weights & Biases"})
    log_tracker_config: dict | None = field(default_factory=dict, metadata={"help": "Additional config dict passed to tracker.init()"})
    wandb_api_key: str | None = field(default=None, metadata={"help": "Weights & Biases API key for authentication"})
    log_config: bool = field(default=False, metadata={"help": "Log the training configuration"})
    console_log_level: str | None = field(default=None, metadata={"help": "Console log level: DEBUG, INFO, WARNING, ERROR"})
    console_log_file: str | None = field(default=None, metadata={"help": "Path to save console output to file"})
    console_log_simple: bool = field(default=False, metadata={"help": "Use simplified console log format"})
    log_every_n_steps: int = field(default=1, metadata={"help": "Emit metrics to tracker every N steps (1 = every step)"})
    log_timestep_distribution_every_n_steps: int | None = field(
        default=None, metadata={"help": "Save timesteps distribution chart every N steps"}
    )
    live_plot_port: int | None = field(default=None, metadata={"help": "Launch live interactive dashboard server on this port"})
    resource_monitor: ResourceMonitorConfig = field(default_factory=ResourceMonitorConfig)


@dataclass
class HuggingFaceConfig:
    """HuggingFace Hub upload settings."""

    huggingface_repo_id: str | None = field(default=None, metadata={"help": "HuggingFace repository ID (user/repo)"})
    huggingface_repo_type: str | None = field(default=None, metadata={"help": "Repository type: model, dataset, or space"})
    huggingface_path_in_repo: str | None = field(default=None, metadata={"help": "Path within the repository for uploads"})
    huggingface_token: str | None = field(default=None, metadata={"help": "HuggingFace API token for authentication"})
    huggingface_repo_visibility: str | None = field(default=None, metadata={"help": "Repository visibility: public or private"})
    save_state_to_huggingface: bool = field(default=False, metadata={"help": "Upload training states to HuggingFace"})
    resume_from_huggingface: bool = field(default=False, metadata={"help": "Resume training from HuggingFace repository"})
    async_upload: bool = field(default=False, metadata={"help": "Upload to HuggingFace asynchronously"})


@dataclass
class SamplingConfig:
    """Sample image generation settings."""

    sample_every_n_steps: int | None = field(default=None, metadata={"help": "Generate sample images every N steps"})
    sample_at_first: bool = field(default=False, metadata={"help": "Generate sample images before training starts (step 0)"})
    sample_every_n_epochs: int | None = field(default=None, metadata={"help": "Generate sample images every N epochs"})
    sample_prompt: str | None = field(
        default=None, metadata={"help": "Inline sample prompt used when sample_prompt_file is not set"}
    )
    sample_prompt_file: str | None = field(default=None, metadata={"help": "Path to prompts file (.txt, .toml, or .json)"})
    sample_negative_prompt: str | None = field(default=None, metadata={"help": "Default negative prompt for sample generation"})
    sample_width: int | None = field(default=None, metadata={"help": "Default sample width when not specified per prompt"})
    sample_height: int | None = field(default=None, metadata={"help": "Default sample height when not specified per prompt"})
    sample_steps: int | None = field(default=None, metadata={"help": "Default inference steps when not specified per prompt"})
    sample_cfg_scale: float | None = field(default=None, metadata={"help": "Default CFG/guidance scale when not specified per prompt"})
    sample_seed: int | None = field(default=None, metadata={"help": "Default seed for sample generation when not specified per prompt"})
    sample_sampler: str = field(default="ddim", metadata={"help": "Default sampler: ddim, euler, euler_a, dpmsolver, etc."})
    sample_vae_dtype: str | None = field(
        default=None, metadata={"help": "VAE dtype for sampling: fp16, bf16, or fp32. If None, uses training VAE dtype."}
    )


@dataclass
class MetadataConfig:
    """Model metadata settings."""

    metadata_title: str | None = field(default=None, metadata={"help": "Title for model metadata (default is output_name)"})
    metadata_author: str | None = field(default=None, metadata={"help": "Author name for model metadata"})
    metadata_description: str | None = field(default=None, metadata={"help": "Description for model metadata"})
    metadata_license: str | None = field(default=None, metadata={"help": "License for model metadata"})
    metadata_tags: str | None = field(default=None, metadata={"help": "Tags for model metadata, separated by comma"})
    metadata_usage_hint: str | None = field(default=None, metadata={"help": "Usage hint for model metadata"})
    metadata_thumbnail: str | None = field(default=None, metadata={"help": "Thumbnail image as data URL or file path for model metadata"})
    metadata_merged_from: str | None = field(default=None, metadata={"help": "Source models for merged model metadata"})
    metadata_trigger_phrase: str | None = field(default=None, metadata={"help": "Trigger phrase for model metadata"})
    metadata_preprocessor: str | None = field(default=None, metadata={"help": "Preprocessor used for model metadata"})
    metadata_is_negative_embedding: str | None = field(
        default=None, metadata={"help": "Whether this is a negative embedding for model metadata"}
    )


@dataclass
class OutputConfig:
    """Output configuration with organized subcategories for saving, logging, and publishing."""

    saving: SavingConfig = field(default_factory=SavingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
