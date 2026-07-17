"""Typed shared metadata fact dataclasses.

These are the canonical shared fact shapes that domain code hands to the
central metadata backbone. They stay intentionally small and explicit, while
the backend/projection layers decide how to persist or export them.
"""

from library.metadata.dataclasses.artifact import CheckpointArtifactFacts
from library.metadata.dataclasses.model import (
    build_model_component_identifier,
    build_model_family_contribution_identifier,
    build_model_realization_identifier,
    ModelArtifactFacts,
    ModelArtifactPresentation,
    ModelArtifactResolutionContext,
    ModelFamilyDeclarationReference,
    ModelFamilyMetadataContribution,
    ModelFamilyMetadataField,
    ModelRealizationFacts,
    ModelSpecFacts,
    RealizedModelComponentFacts,
)
from library.metadata.dataclasses.optimization import (
    OptimizerRuntimeFacts,
    SchedulerRuntimeFacts,
)
from library.metadata.dataclasses.run import RunMetadataFacts

from library.metadata.dataclasses.observability import (
    AnalyticsSnapshotFacts,
    LoggedArtifactFacts,
    ResourceMonitorFacts,
    RunLifecycleFacts,
    RunReportFacts,
)

from library.metadata.dataclasses.resource import (
    ResourceAccountingFacts,
    ResourceAccountingGapFacts,
    ResourceCollectorStatusFacts,
    ResourceFactReference,
    ResourceObservationFrameFacts,
    ResourceObservationFacts,
    ResourceObservationMeasurementFacts,
    ResourceProfileFacts,
    StructuralResourceFacts,
)


__all__ = [
    "CheckpointArtifactFacts",
    "LoggedArtifactFacts",
    "RunLifecycleFacts",
    "ResourceMonitorFacts",
    "RunReportFacts",
    "AnalyticsSnapshotFacts",
    "OptimizerRuntimeFacts",
    "ResourceAccountingFacts",
    "ResourceAccountingGapFacts",
    "ResourceCollectorStatusFacts",
    "ResourceFactReference",
    "ResourceObservationFrameFacts",
    "ResourceObservationFacts",
    "ResourceObservationMeasurementFacts",
    "ResourceProfileFacts",
    "ModelSpecFacts",
    "ModelArtifactFacts",
    "ModelArtifactPresentation",
    "ModelArtifactResolutionContext",
    "ModelFamilyDeclarationReference",
    "ModelFamilyMetadataContribution",
    "ModelFamilyMetadataField",
    "ModelRealizationFacts",
    "RealizedModelComponentFacts",
    "RunMetadataFacts",
    "SchedulerRuntimeFacts",
    "StructuralResourceFacts",
    "build_model_component_identifier",
    "build_model_family_contribution_identifier",
    "build_model_realization_identifier",
]
