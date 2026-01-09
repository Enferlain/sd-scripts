# Data Pipeline Package
#
# This package provides the high-performance data loading pipeline.
#
# Structure:
# - structures.py: Core data structures (CacheEntry, Bucket, EpochManifest, DatasetManifest)
# - image_utils.py: Image dimension reading, alpha detection, ID generation
# - caption_processor.py: Caption processing, augmentation, tag parsing
# - bucketing.py: Aspect ratio bucket generation and selection
# - scanners.py: Directory and metadata file scanning
# - manifest.py: Manifest creation, loading, saving, and config hashing
# - epoch_preparation.py: Per-epoch batch organization and tokenization
# - dataloader.py: TrainingDataset and DataLoader creation
# - caching_engine.py: Latent and TE output caching with multi-GPU coordination


from library.data.structures import (
    CacheEntry,
    Bucket,
    BatchInfo,
    EpochManifest,
    DatasetManifest,
)
from library.data.manifest import (
    save_dataset_manifest,
    load_dataset_manifest,
    save_epoch_manifest,
    load_epoch_manifest,
    get_or_create_manifest,
    create_manifest,
    create_manifest_from_config,
)
from library.data.caching_engine import CachingStrategy, CachingEngine
from library.data.dataloader import TrainingDataset, create_training_dataloader
from library.data.epoch_preparation import (
    prepare_epoch,
    prepare_validation_epoch,
    tokenize_epoch_manifest,
    load_epoch_tokens,
)
from library.data.scanners import ScannedImage, scan_directory, scan_metadata_file
from library.data.bucketing import make_bucket_resolutions, select_bucket
from library.data.caption_processor import (
    CaptionConfig,
    process_caption,
    compute_tag_frequency,
    read_caption,
)

__all__ = [
    # Dataclasses
    "CacheEntry",
    "Bucket",
    "BatchInfo",
    "EpochManifest",
    "DatasetManifest",
    # Caption processing
    "CaptionConfig",
    "process_caption",
    # Manifest I/O
    "save_dataset_manifest",
    "load_dataset_manifest",
    "save_epoch_manifest",
    "load_epoch_manifest",
    "get_or_create_manifest",
    # Caching
    "CachingStrategy",
    "CachingEngine",
    # DataLoader
    "TrainingDataset",
    "create_training_dataloader",
    # Epoch preparation
    "prepare_epoch",
    "prepare_validation_epoch",
    "tokenize_epoch_manifest",
    "load_epoch_tokens",
    # Dataset scanner (Phase 1)
    "ScannedImage",
    "scan_directory",
    "scan_metadata_file",
    "read_caption",
    "make_bucket_resolutions",
    "select_bucket",
    "create_manifest",
    "create_manifest_from_config",
    "compute_tag_frequency",
]
