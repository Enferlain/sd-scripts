"""Declarative registry for accepted runtime metadata items and routes."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

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
    ResourceObservationFacts,
    ResourceObservationFrameFacts,
    ResourceProfileFacts,
    StructuralResourceFacts,
)


type MetadataRuntimeItem = (
    LoggedArtifactFacts
    | RunLifecycleFacts
    | ResourceMonitorFacts
    | RunReportFacts
    | AnalyticsSnapshotFacts
    | ResourceObservationFrameFacts
    | ResourceObservationFacts
    | StructuralResourceFacts
    | ResourceProfileFacts
    | ResourceAccountingFacts
    | ResourceAccountingGapFacts
    | ResourceCollectorStatusFacts
)

METADATA_ITEM_ROUTES: Mapping[type[object], str] = MappingProxyType(
    {
        LoggedArtifactFacts: "observability.logged_artifact",
        RunLifecycleFacts: "observability.run_lifecycle",
        ResourceMonitorFacts: "observability.resource_monitor_compatibility",
        RunReportFacts: "observability.run_report",
        AnalyticsSnapshotFacts: "observability.analytics_snapshot",
        ResourceObservationFrameFacts: "resource.observation_frame",
        ResourceObservationFacts: "resource.observation",
        StructuralResourceFacts: "resource.structural",
        ResourceProfileFacts: "resource.profile",
        ResourceAccountingFacts: "resource.accounting",
        ResourceAccountingGapFacts: "resource.accounting_gap",
        ResourceCollectorStatusFacts: "resource.collector_status",
    }
)

METADATA_ITEM_TYPES = tuple(METADATA_ITEM_ROUTES)


def metadata_item_route(item: object) -> str | None:
    """Return the registered emitter route for one exact accepted item type."""
    return METADATA_ITEM_ROUTES.get(type(item))


def supported_metadata_item_names() -> str:
    """Return stable user-facing names for accepted runtime item types."""
    return ", ".join(item_type.__name__ for item_type in METADATA_ITEM_TYPES)
