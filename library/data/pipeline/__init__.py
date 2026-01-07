# Data Pipeline Package
#
# This package provides the new high-performance data loading pipeline.
# See DATA_PIPELINE_PLAN.md for architecture details.
#
# Structure:
# - dataclasses.py: Core data structures (CacheEntry, EpochManifest, etc.)
# - manifest.py: Manifest reading/writing/validation
# - caching_engine.py: Fast batch caching with multi-GPU coordination
# - dataloader.py: TrainingDataLoader for epoch iteration
# - epoch_preparation.py: Phase 3 batch organization
# - dataset_scanner.py: Phase 1 directory scanning and manifest creation

from library.data.pipeline.dataclasses import (
    CacheEntry,
    Bucket,
    BatchInfo,
    EpochManifest,
    DatasetManifest,
)
from library.data.pipeline.manifest import (
    save_dataset_manifest,
    load_dataset_manifest,
    save_epoch_manifest,
    load_epoch_manifest,
    get_or_create_manifest,
)
from library.data.pipeline.caching_engine import CachingStrategy, CachingEngine
from library.data.pipeline.dataloader import TrainingDataset, create_training_dataloader
from library.data.pipeline.epoch_preparation import (
    prepare_epoch,
    prepare_validation_epoch,
    tokenize_epoch_manifest,
    load_epoch_tokens,
)
from library.data.pipeline.dataset_scanner import (
    ScannedImage,
    scan_directory,
    scan_metadata_file,
    read_caption,
    make_bucket_resolutions,
    select_bucket,
    create_manifest,
    create_manifest_from_config,
    compute_tag_frequency,
)
from library.data.pipeline.caption_processor import (
    CaptionConfig,
    process_caption,
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
