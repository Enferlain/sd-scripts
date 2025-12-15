from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class DatasetConfig:
    train_data_dir: Optional[str] = field(default=None, metadata={"help": "directory for train images"})
    dataset_config: Optional[str] = field(default=None, metadata={"help": "Load dataset config from a file"})
    cache_info: bool = field(default=False, metadata={"help": "cache meta information for faster dataset loading"})
    shuffle_caption: bool = field(default=False, metadata={"help": "shuffle separated caption"})
    caption_separator: str = field(default=",", metadata={"help": "separator for caption"})
    caption_extension: str = field(default=".caption", metadata={"help": "extension of caption files"})
    caption_extention: Optional[str] = field(default=None, metadata={"help": "extension of caption files (backward compatibility)"})
    keep_tokens: int = field(default=0, metadata={"help": "keep heading N tokens when shuffling caption tokens"})
    keep_tokens_separator: str = field(default="", metadata={"help": "A custom separator to divide the caption into fixed and flexible parts"})
    secondary_separator: Optional[str] = field(default=None, metadata={"help": "a secondary separator for caption"})
    enable_wildcard: bool = field(default=False, metadata={"help": "enable wildcard for caption"})
    caption_prefix: Optional[str] = field(default=None, metadata={"help": "prefix for caption text"})
    caption_suffix: Optional[str] = field(default=None, metadata={"help": "suffix for caption text"})
    color_aug: bool = field(default=False, metadata={"help": "enable weak color augmentation"})
    flip_aug: bool = field(default=False, metadata={"help": "enable horizontal flip augmentation"})
    face_crop_aug_range: Optional[str] = field(default=None, metadata={"help": "enable face-centered crop augmentation and its range"})
    random_crop: bool = field(default=False, metadata={"help": "enable random crop"})
    debug_dataset: bool = field(default=False, metadata={"help": "show images for debugging"})
    resolution: Optional[str] = field(default=None, metadata={"help": "resolution in training"})
    cache_latents: bool = field(default=False, metadata={"help": "cache latents to main memory to reduce VRAM usage"})
    vae_batch_size: int = field(default=1, metadata={"help": "batch size for caching latents"})
    cache_latents_to_disk: bool = field(default=False, metadata={"help": "cache latents to disk to reduce VRAM usage"})
    skip_cache_check: bool = field(default=False, metadata={"help": "skip the content validation of cache"})
    resize_interpolation: Optional[str] = field(default=None, metadata={"help": "Resize interpolation when required"})
    token_warmup_min: int = field(default=1, metadata={"help": "start learning at N tags"})
    token_warmup_step: float = field(default=0.0, metadata={"help": "tag length reaches maximum on N steps"})
    alpha_mask: bool = field(default=False, metadata={"help": "use alpha channel as mask for training"})
    dataset_class: Optional[str] = field(default=None, metadata={"help": "dataset class for arbitrary dataset"})
    caption_dropout_rate: float = field(default=0.0, metadata={"help": "Rate out dropout caption"})
    caption_dropout_every_n_epochs: int = field(default=0, metadata={"help": "Dropout all captions every N epochs"})
    caption_tag_dropout_rate: float = field(default=0.0, metadata={"help": "Rate out dropout comma separated tokens"})
    reg_data_dir: Optional[str] = field(default=None, metadata={"help": "directory for regularization images"})
    in_json: Optional[str] = field(default=None, metadata={"help": "json metadata for dataset"})
    dataset_repeats: int = field(default=1, metadata={"help": "repeat dataset when training with captions"})
    weighted_captions: bool = field(default=False, metadata={"help": "enable weighted captions"})
    validation_split: float = field(default=0.0, metadata={"help": "Split for validation images out of the training dataset"})
    validation_seed: Optional[int] = field(default=None, metadata={"help": "Validation seed for shuffling validation dataset, training `--seed` used otherwise"})

    # Placeholder for subsets to be populated by hydra or manually
    subsets: List[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.cache_latents_to_disk and not self.cache_latents:
            self.cache_latents = True
        if self.caption_extention is not None:
            self.caption_extension = self.caption_extention
