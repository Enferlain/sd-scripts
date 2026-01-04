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

from library.data.pipeline.dataclasses import (
    CacheEntry,
    Bucket,
    EpochManifest,
    DatasetManifest,
)
from library.data.pipeline.manifest import (
    save_dataset_manifest,
    load_dataset_manifest,
    save_epoch_manifest,
    load_epoch_manifest,
)
from library.data.pipeline.caching_engine import CachingStrategy, CachingEngine
from library.data.pipeline.dataloader import TrainingDataset, create_training_dataloader
from library.data.pipeline.epoch_preparation import prepare_epoch, prepare_validation_epoch

__all__ = [
    # Dataclasses
    "CacheEntry",
    "Bucket",
    "EpochManifest",
    "DatasetManifest",
    # Manifest I/O
    "save_dataset_manifest",
    "load_dataset_manifest",
    "save_epoch_manifest",
    "load_epoch_manifest",
    # Caching
    "CachingStrategy",
    "CachingEngine",
    # DataLoader
    "TrainingDataset",
    "create_training_dataloader",
    # Epoch preparation
    "prepare_epoch",
    "prepare_validation_epoch",
]
