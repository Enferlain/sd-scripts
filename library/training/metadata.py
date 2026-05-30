"""Compatibility wrappers for training metadata assembly."""

from library.metadata.emitters.checkpoint import build_checkpoint_metadata
from library.metadata.emitters.run import TrainingMetadataBuildContext, TrainingMetadataBundle, TrainingMetadataState
from library.metadata.emitters.run import build_objective_run_metadata, build_training_metadata_bundle

__all__ = [
    "TrainingMetadataBuildContext",
    "TrainingMetadataBundle",
    "TrainingMetadataState",
    "build_checkpoint_metadata",
    "build_objective_run_metadata",
    "build_training_metadata_bundle",
]
