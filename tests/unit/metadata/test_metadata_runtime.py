"""Unit tests for the runtime-facing metadata filing API."""

from __future__ import annotations

import pytest

from library.metadata import (
    AnalyticsSnapshotFacts,
    LoggedArtifactFacts,
    METADATA_PAYLOAD_VERSION,
    MetadataRuntime,
    RunLifecycleFacts,
)
from library.metadata.validation import MetadataItemValidationError
from library.metadata.registry import METADATA_ITEM_ROUTES, METADATA_ITEM_TYPES
from library.metadata.runtime import _METADATA_EMITTERS_BY_ROUTE


@pytest.mark.unit
def test_metadata_runtime_files_items_into_backend_snapshot() -> None:
    runtime = MetadataRuntime()

    runtime.file(
        RunLifecycleFacts(
            run_identifier="run-1",
            event_type="run_started",
            run_name="training",
            status="running",
        )
    )
    runtime.file(
        LoggedArtifactFacts(
            path="/tmp/report.md",
            kind="benchmark_report",
            metadata={"format": "markdown"},
        )
    )
    runtime.file(
        AnalyticsSnapshotFacts(
            snapshot_identifier="/tmp/report.json#payload",
            snapshot_kind="benchmark_report_payload",
            source="unit_test",
            payload={"status": "succeeded", "resource_monitor": {"event_count": 3}},
            run_identifier="run-1",
        )
    )

    snapshot = runtime.snapshot()

    assert [event.event_type for event in snapshot.events] == [
        "run_started",
        "artifact_registered",
    ]
    assert len(snapshot.records) == 1
    assert snapshot.records[0].facts["kind"] == "benchmark_report_payload"
    assert snapshot.records[0].facts["payload"]["resource_monitor"]["event_count"] == 3
    assert snapshot.records[0].schema_version == METADATA_PAYLOAD_VERSION
    assert snapshot.records[0].identity.schema_version == METADATA_PAYLOAD_VERSION
    assert snapshot.events[0].schema_version == METADATA_PAYLOAD_VERSION
    assert snapshot.events[0].identity.schema_version == METADATA_PAYLOAD_VERSION
    assert snapshot.events[1].facts["metadata"] == {"format": "markdown"}


@pytest.mark.unit
def test_metadata_runtime_rejects_unknown_items() -> None:
    runtime = MetadataRuntime()

    with pytest.raises(MetadataItemValidationError):
        runtime.file(object())  # type: ignore[arg-type]


@pytest.mark.unit
def test_metadata_item_registry_is_the_shared_validation_and_routing_catalog() -> None:
    assert tuple(METADATA_ITEM_ROUTES) == METADATA_ITEM_TYPES
    assert all(route.strip() for route in METADATA_ITEM_ROUTES.values())
    assert set(METADATA_ITEM_ROUTES.values()) == set(_METADATA_EMITTERS_BY_ROUTE)
