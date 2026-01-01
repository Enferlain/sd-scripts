from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class SavingConfig:
    """Checkpoint and model saving settings."""
    output_dir: Optional[str] = field(default=None, metadata={"help": "Directory for saving checkpoints and models"})
    output_name: Optional[str] = field(default=None, metadata={"help": "Base name for output files (default: epoch/step/last)"})
    save_precision: Optional[str] = field(default=None, metadata={"help": "Model save precision: fp16, bf16, or float"})
    save_every_n_epochs: Optional[int] = field(default=None, metadata={"help": "Save checkpoint every N epochs"})
    save_every_n_steps: Optional[int] = field(default=None, metadata={"help": "Save checkpoint every N training steps"})
    save_n_epoch_ratio: Optional[int] = field(default=None, metadata={"help": "Compute save_every_n_epochs as total_epochs / ratio"})
    save_last_n_epochs: Optional[int] = field(default=None, metadata={"help": "Keep only the last N epoch checkpoints (delete older)"})
    save_last_n_epochs_state: Optional[int] = field(default=None, metadata={"help": "Keep only the last N epoch states (delete older)"})
    save_last_n_steps: Optional[int] = field(default=None, metadata={"help": "Keep only the last N step checkpoints (delete older)"})
    save_last_n_steps_state: Optional[int] = field(default=None, metadata={"help": "Keep only the last N step states (delete older)"})
    save_state: bool = field(default=False, metadata={"help": "Save training state for resuming (optimizer, scheduler, etc.)"})
    save_state_on_train_end: bool = field(default=False, metadata={"help": "Save training state when training completes"})
    resume: Optional[str] = field(default=None, metadata={"help": "Path to training state to resume from"})
    save_model_as: Optional[str] = field(default=None, metadata={"help": "Model format: safetensors, ckpt, diffusers, diffusers_safetensors"})
    use_safetensors: bool = field(default=False, metadata={"help": "Use safetensors format for saving (deprecated, use save_model_as)"})
    no_metadata: bool = field(default=False, metadata={"help": "Don't save training metadata in model file"})
    hash_algorithm: str = field(default='sha256', metadata={"help": "Hash algorithm for model checksums: md5, sha1, sha256, sha512, blake3"})


@dataclass
class LoggingConfig:
    """Logging and tracking settings."""
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Directory for TensorBoard/W&B logs"})
    log_with: Optional[str] = field(default=None, metadata={"help": "Tracker to use: tensorboard, wandb, or all"})
    log_prefix: Optional[str] = field(default=None, metadata={"help": "Prefix for log directory name"})
    log_tracker_name: Optional[str] = field(default=None, metadata={"help": "Project/experiment name for tracker"})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "Run name for Weights & Biases"})
    log_tracker_config: Optional[Dict] = field(default_factory=dict, metadata={"help": "Additional config dict passed to tracker.init()"})
    wandb_api_key: Optional[str] = field(default=None, metadata={"help": "Weights & Biases API key for authentication"})
    log_config: bool = field(default=False, metadata={"help": "Log the training configuration"})
    console_log_level: Optional[str] = field(default=None, metadata={"help": "Console log level: DEBUG, INFO, WARNING, ERROR"})
    console_log_file: Optional[str] = field(default=None, metadata={"help": "Path to save console output to file"})
    console_log_simple: bool = field(default=False, metadata={"help": "Use simplified console log format"})
    log_timestep_distribution_every_n_steps: Optional[int] = field(default=None, metadata={"help": "Save timesteps distribution chart every N steps"})
    live_plot_port: Optional[int] = field(default=None, metadata={"help": "Launch live interactive dashboard server on this port"})


@dataclass
class HuggingFaceConfig:
    """HuggingFace Hub upload settings."""
    huggingface_repo_id: Optional[str] = field(default=None, metadata={"help": "HuggingFace repository ID (user/repo)"})
    huggingface_repo_type: Optional[str] = field(default=None, metadata={"help": "Repository type: model, dataset, or space"})
    huggingface_path_in_repo: Optional[str] = field(default=None, metadata={"help": "Path within the repository for uploads"})
    huggingface_token: Optional[str] = field(default=None, metadata={"help": "HuggingFace API token for authentication"})
    huggingface_repo_visibility: Optional[str] = field(default=None, metadata={"help": "Repository visibility: public or private"})
    save_state_to_huggingface: bool = field(default=False, metadata={"help": "Upload training states to HuggingFace"})
    resume_from_huggingface: bool = field(default=False, metadata={"help": "Resume training from HuggingFace repository"})
    async_upload: bool = field(default=False, metadata={"help": "Upload to HuggingFace asynchronously"})


@dataclass
class SamplingConfig:
    """Sample image generation settings."""
    sample_every_n_steps: Optional[int] = field(default=None, metadata={"help": "Generate sample images every N steps"})
    sample_at_first: bool = field(default=False, metadata={"help": "Generate sample images before training starts (step 0)"})
    sample_every_n_epochs: Optional[int] = field(default=None, metadata={"help": "Generate sample images every N epochs"})
    sample_prompts: Optional[str] = field(default=None, metadata={"help": "Path to prompts file (.txt, .toml, or .json)"})
    sample_sampler: str = field(default="ddim", metadata={"help": "Default sampler: ddim, euler, euler_a, dpmsolver, etc."})


@dataclass
class MetadataConfig:
    """Model metadata settings."""
    metadata_title: Optional[str] = field(default=None, metadata={"help": "Title for model metadata (default is output_name)"})
    metadata_author: Optional[str] = field(default=None, metadata={"help": "Author name for model metadata"})
    metadata_description: Optional[str] = field(default=None, metadata={"help": "Description for model metadata"})
    metadata_license: Optional[str] = field(default=None, metadata={"help": "License for model metadata"})
    metadata_tags: Optional[str] = field(default=None, metadata={"help": "Tags for model metadata, separated by comma"})
    metadata_usage_hint: Optional[str] = field(default=None, metadata={"help": "Usage hint for model metadata"})
    metadata_thumbnail: Optional[str] = field(default=None, metadata={"help": "Thumbnail image as data URL or file path for model metadata"})
    metadata_merged_from: Optional[str] = field(default=None, metadata={"help": "Source models for merged model metadata"})
    metadata_trigger_phrase: Optional[str] = field(default=None, metadata={"help": "Trigger phrase for model metadata"})
    metadata_preprocessor: Optional[str] = field(default=None, metadata={"help": "Preprocessor used for model metadata"})
    metadata_is_negative_embedding: Optional[str] = field(default=None, metadata={"help": "Whether this is a negative embedding for model metadata"})


@dataclass
class OutputConfig:
    """Output configuration with organized subcategories for saving, logging, and publishing."""
    saving: SavingConfig = field(default_factory=SavingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
