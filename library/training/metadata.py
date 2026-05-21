"""Compatibility wrappers for training metadata assembly."""

from library.metadata.emitters.checkpoint import build_checkpoint_metadata
from library.metadata.emitters.run import TrainingMetadataBuildContext, TrainingMetadataBundle, TrainingMetadataState
from library.metadata.emitters.run import build_objective_ss_metadata, build_training_metadata_bundle, select_minimum_training_metadata

__all__ = [
    "TrainingMetadataBuildContext",
    "TrainingMetadataBundle",
    "TrainingMetadataState",
    "build_checkpoint_metadata",
    "build_objective_ss_metadata",
    "build_training_metadata_bundle",
    "select_minimum_training_metadata",
]
