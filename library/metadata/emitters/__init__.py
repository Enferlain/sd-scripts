"""Central metadata emitters and compatibility builders."""

from library.metadata.emitters.checkpoint import build_checkpoint_artifact_metadata, build_checkpoint_metadata
from library.metadata.emitters.observability import (
    build_analytics_snapshot_metadata,
    build_logged_artifact_metadata,
    build_resource_monitor_metadata,
    build_run_lifecycle_metadata,
    build_run_report_metadata,
)
from library.metadata.emitters.resource import (
    build_resource_accounting_gap_metadata,
    build_resource_accounting_metadata,
    build_resource_observation_metadata,
    build_resource_profile_metadata,
    build_structural_resource_metadata,
)
from library.metadata.emitters.run import (
    TrainingMetadataBuildContext,
    TrainingMetadataBundle,
    TrainingMetadataState,
    build_model_spec_metadata,
    build_objective_run_metadata,
    build_training_metadata_bundle,
    build_training_run_metadata,
)

__all__ = [
    "TrainingMetadataBuildContext",
    "TrainingMetadataBundle",
    "TrainingMetadataState",
    "build_checkpoint_artifact_metadata",
    "build_checkpoint_metadata",
    "build_analytics_snapshot_metadata",
    "build_logged_artifact_metadata",
    "build_model_spec_metadata",
    "build_objective_run_metadata",
    "build_resource_monitor_metadata",
    "build_resource_observation_metadata",
    "build_resource_profile_metadata",
    "build_run_lifecycle_metadata",
    "build_run_report_metadata",
    "build_structural_resource_metadata",
    "build_training_metadata_bundle",
    "build_training_run_metadata",
    "build_resource_accounting_metadata",
    "build_resource_accounting_gap_metadata",
]
