from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class SavingConfig:
    """Checkpoint and model saving settings."""
    output_dir: Optional[str] = None
    output_name: Optional[str] = None
    save_precision: Optional[str] = None
    save_every_n_epochs: Optional[int] = None
    save_every_n_steps: Optional[int] = None
    save_n_epoch_ratio: Optional[int] = None
    save_last_n_epochs: Optional[int] = None
    save_last_n_epochs_state: Optional[int] = None
    save_last_n_steps: Optional[int] = None
    save_last_n_steps_state: Optional[int] = None
    save_state: bool = False
    save_state_on_train_end: bool = False
    resume: Optional[str] = None
    save_model_as: Optional[str] = None
    use_safetensors: bool = False
    no_metadata: bool = False


@dataclass
class LoggingConfig:
    """Logging and tracking settings."""
    logging_dir: Optional[str] = None
    log_with: Optional[str] = None
    log_prefix: Optional[str] = None
    log_tracker_name: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_tracker_config: Optional[Dict] = field(default_factory=dict)
    wandb_api_key: Optional[str] = None
    log_config: bool = False
    console_log_level: Optional[str] = None
    console_log_file: Optional[str] = None
    console_log_simple: bool = False
    log_timestep_distribution_every_n_steps: Optional[int] = field(default=None, metadata={"help": "Saves a snapshot of the timestep distribution chart every N steps."})
    live_plot_port: Optional[int] = field(default=None, metadata={"help": "Launches the live interactive dashboard server on this port."})


@dataclass
class HuggingFaceConfig:
    """HuggingFace Hub upload settings."""
    huggingface_repo_id: Optional[str] = None
    huggingface_repo_type: Optional[str] = None
    huggingface_path_in_repo: Optional[str] = None
    huggingface_token: Optional[str] = None
    huggingface_repo_visibility: Optional[str] = None
    save_state_to_huggingface: bool = False
    resume_from_huggingface: bool = False
    async_upload: bool = False


@dataclass
class SamplingConfig:
    """Sample image generation settings."""
    sample_every_n_steps: Optional[int] = None
    sample_at_first: bool = False
    sample_every_n_epochs: Optional[int] = None
    sample_prompts: Optional[str] = None
    sample_sampler: str = "ddim"


@dataclass
class MetadataConfig:
    """Model metadata settings."""
    metadata_title: Optional[str] = field(default=None, metadata={"help": "title for model metadata (default is output_name)"})
    metadata_author: Optional[str] = field(default=None, metadata={"help": "author name for model metadata"})
    metadata_description: Optional[str] = field(default=None, metadata={"help": "description for model metadata"})
    metadata_license: Optional[str] = field(default=None, metadata={"help": "license for model metadata"})
    metadata_tags: Optional[str] = field(default=None, metadata={"help": "tags for model metadata, separated by comma"})
    metadata_usage_hint: Optional[str] = field(default=None, metadata={"help": "usage hint for model metadata"})
    metadata_thumbnail: Optional[str] = field(default=None, metadata={"help": "thumbnail image as data URL or file path (will be converted to data URL) for model metadata"})
    metadata_merged_from: Optional[str] = field(default=None, metadata={"help": "source models for merged model metadata"})
    metadata_trigger_phrase: Optional[str] = field(default=None, metadata={"help": "trigger phrase for model metadata"})
    metadata_preprocessor: Optional[str] = field(default=None, metadata={"help": "preprocessor used for model metadata"})
    metadata_is_negative_embedding: Optional[str] = field(default=None, metadata={"help": "whether this is a negative embedding for model metadata"})


@dataclass
class OutputConfig:
    """Output configuration with organized subcategories for saving, logging, and publishing."""
    saving: SavingConfig = field(default_factory=SavingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
