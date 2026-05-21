"""Runtime-facing metadata filing API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from library.metadata.backends import InMemoryMetadataBackend, MetadataBackend, MetadataSnapshot
from library.metadata.dataclasses.observability import (
    AnalyticsSnapshotFacts,
    LoggedArtifactFacts,
    ResourceMonitorFacts,
    RunLifecycleFacts,
    RunReportFacts,
)
from library.metadata.emitters.observability import (
    build_analytics_snapshot_metadata,
    build_logged_artifact_metadata,
    build_resource_monitor_metadata,
    build_run_lifecycle_metadata,
    build_run_report_metadata,
)
from library.metadata.providers import MetadataProviderResult
from library.metadata.validation import MetadataItemValidationError, validate_metadata_item

MetadataRuntimeItem: TypeAlias = (
    LoggedArtifactFacts
    | RunLifecycleFacts
    | ResourceMonitorFacts
    | RunReportFacts
    | AnalyticsSnapshotFacts
)

_SUPPORTED_METADATA_ITEM_TYPES = (
    LoggedArtifactFacts,
    RunLifecycleFacts,
    ResourceMonitorFacts,
    RunReportFacts,
    AnalyticsSnapshotFacts,
)


def build_metadata_result(item: MetadataRuntimeItem) -> MetadataProviderResult:
    """Build one provider result from one accepted typed metadata item."""
    validate_metadata_item(item)

    if isinstance(item, LoggedArtifactFacts):
        return build_logged_artifact_metadata(item)
    if isinstance(item, RunLifecycleFacts):
        return build_run_lifecycle_metadata(item)
    if isinstance(item, ResourceMonitorFacts):
        return build_resource_monitor_metadata(item)
    if isinstance(item, RunReportFacts):
        return build_run_report_metadata(item)
    if isinstance(item, AnalyticsSnapshotFacts):
        return build_analytics_snapshot_metadata(item)

    supported = ", ".join(item_type.__name__ for item_type in _SUPPORTED_METADATA_ITEM_TYPES)
    raise MetadataItemValidationError(
        f"Unsupported metadata item type {type(item).__name__}. Supported types: {supported}."
    )


@dataclass(slots=True)
class MetadataRuntime:
    """Small runtime-facing API for filing typed metadata items."""

    backend: MetadataBackend = field(default_factory=InMemoryMetadataBackend)

    def file(self, item: MetadataRuntimeItem) -> MetadataProviderResult:
        """File one accepted typed metadata item into the metadata backend."""
        result = build_metadata_result(item)
        self.backend.ingest(result)
        return result

    def snapshot(self) -> MetadataSnapshot:
        """Return the current collected metadata snapshot."""
        return self.backend.snapshot()

    def validate(self) -> None:
        """Validate the currently collected backend state."""
        self.backend.validate()
