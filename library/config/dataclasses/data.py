from dataclasses import dataclass, field


@dataclass
class SourceConfig:
    """Data source paths and dataset definition."""

    train_data_dir: str | None = field(default=None, metadata={"help": "directory for train images"})
    reg_data_dir: str | None = field(default=None, metadata={"help": "directory for regularization images"})
    dataset_config: str | None = field(default=None, metadata={"help": "Load dataset config from a file"})
    in_json: str | None = field(default=None, metadata={"help": "json metadata for dataset"})
    dataset_class: str | None = field(default=None, metadata={"help": "dataset class for arbitrary dataset"})
    dataset_repeats: int = field(default=1, metadata={"help": "repeat dataset when training with captions"})
    # Placeholder for subsets to be populated by hydra or manually
    subsets: list[dict] = field(
        default_factory=list, metadata={"help": "List of dataset subset configurations (populated via YAML or dataset_config)"}
    )


@dataclass
class PreprocessingConfig:
    """Image preprocessing, augmentation, and debug settings."""

    resolution: str | None = field(default=None, metadata={"help": "resolution in training"})
    resize_interpolation: str | None = field(default=None, metadata={"help": "Resize interpolation when required"})
    alpha_mask: bool = field(default=False, metadata={"help": "use alpha channel as mask for training"})
    flip_aug: bool = field(default=False, metadata={"help": "enable horizontal flip augmentation"})
    color_aug: bool = field(default=False, metadata={"help": "enable weak color augmentation"})
    random_crop: bool = field(default=False, metadata={"help": "enable random crop"})
    face_crop_aug_range: str | None = field(default=None, metadata={"help": "enable face-centered crop augmentation and its range"})
    debug_dataset: bool = field(default=False, metadata={"help": "show images for debugging"})
    cache_info: bool = field(default=False, metadata={"help": "cache meta information for faster dataset loading"})


@dataclass
class CaptionConfig:
    """Caption handling, dropout, and warmup settings."""

    caption_extension: str = field(default=".caption", metadata={"help": "extension of caption files"})
    caption_extention: str | None = field(default=None, metadata={"help": "extension of caption files (backward compatibility)"})
    caption_separator: str = field(default=",", metadata={"help": "separator for caption"})
    shuffle_caption: bool = field(default=False, metadata={"help": "shuffle separated caption"})
    keep_tokens: int = field(default=0, metadata={"help": "keep heading N tokens when shuffling caption tokens"})
    keep_tokens_separator: str = field(
        default="", metadata={"help": "A custom separator to divide the caption into fixed and flexible parts"}
    )
    secondary_separator: str | None = field(default=None, metadata={"help": "a secondary separator for caption"})
    enable_wildcard: bool = field(default=False, metadata={"help": "enable wildcard for caption"})
    caption_prefix: str | None = field(default=None, metadata={"help": "prefix for caption text"})
    caption_suffix: str | None = field(default=None, metadata={"help": "suffix for caption text"})
    caption_dropout_rate: float = field(default=0.0, metadata={"help": "Rate out dropout caption"})
    caption_dropout_every_n_epochs: int = field(default=0, metadata={"help": "Dropout all captions every N epochs"})
    caption_tag_dropout_rate: float = field(default=0.0, metadata={"help": "Rate out dropout comma separated tokens"})
    weighted_captions: bool = field(default=False, metadata={"help": "enable weighted captions"})
    token_warmup_min: int = field(default=1, metadata={"help": "start learning at N tags"})
    token_warmup_step: float = field(default=0.0, metadata={"help": "tag length reaches maximum on N steps"})


@dataclass
class BucketingConfig:
    """Aspect ratio bucketing settings."""

    enable_bucket: bool = field(default=False, metadata={"help": "enable buckets for multi aspect ratio training"})
    min_bucket_reso: int = field(default=256, metadata={"help": "minimum resolution for buckets"})
    max_bucket_reso: int = field(default=1024, metadata={"help": "maximum resolution for buckets"})
    bucket_reso_steps: int = field(default=64, metadata={"help": "steps of resolution for buckets"})
    bucket_no_upscale: bool = field(default=False, metadata={"help": "make bucket for each image without upscaling"})


@dataclass
class CachingConfig:
    """Latent caching settings."""

    cache_latents: bool = field(default=False, metadata={"help": "cache latents to main memory to reduce VRAM usage"})
    cache_latents_to_disk: bool = field(default=False, metadata={"help": "cache latents to disk to reduce VRAM usage"})
    vae_batch_size: int = field(default=1, metadata={"help": "batch size for caching latents"})
    skip_cache_check: bool = field(default=False, metadata={"help": "skip the content validation of cache"})


@dataclass
class LoaderConfig:
    """DataLoader settings."""

    max_workers: int = field(default=8, metadata={"help": "max number of data loader workers (0 to disable multiprocessing)"})
    persistent_workers: bool = field(default=False, metadata={"help": "keep data loader workers alive between epochs"})


@dataclass
class DataConfig:
    """Data configuration with organized subcategories."""

    source: SourceConfig = field(default_factory=SourceConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    bucketing: BucketingConfig = field(default_factory=BucketingConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
