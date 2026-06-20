"""Unit tests for observability metadata emitters."""

from __future__ import annotations

import pytest

from library.metadata import (
    AnalyticsSnapshotFacts,
    InMemoryMetadataBackend,
    LoggedArtifactFacts,
    ResourceMonitorFacts,
    RunLifecycleFacts,
    RunReportFacts,
)

from library.metadata.emitters import (
    build_analytics_snapshot_metadata,
    build_logged_artifact_metadata,
    build_resource_monitor_metadata,
    build_run_lifecycle_metadata,
    build_run_report_metadata,
)


@pytest.mark.unit
def test_observability_providers_collect_events_and_records() -> None:
    backend = InMemoryMetadataBackend()
    backend.ingest(
        build_run_lifecycle_metadata(
            RunLifecycleFacts(
                run_identifier="run-1",
                event_type="run_started",
                run_name="training",
                status="running",
            )
        )
    )
    backend.ingest(
        build_logged_artifact_metadata(
            LoggedArtifactFacts(
                path="/tmp/report.md",
                kind="benchmark_report",
                metadata={"format": "markdown"},
                run_identifier="run-1",
            )
        )
    )
    backend.ingest(
        build_resource_monitor_metadata(
            ResourceMonitorFacts(
                run_identifier="run-1",
                event_name="session_start",
                gpu_used_mb=128.0,
            )
        )
    )
    backend.ingest(
        build_run_report_metadata(
            RunReportFacts(
                report_identifier="/tmp/report.md",
                run_identifier="run-1",
                status="succeeded",
                generated_at=123.0,
                output_name="report",
                resource_event_count=1,
            )
        )
    )
    backend.ingest(
        build_analytics_snapshot_metadata(
            AnalyticsSnapshotFacts(
                snapshot_identifier="snapshot-1",
                snapshot_kind="debug_snapshot",
                source="tests",
                payload={"rows": 4},
                run_identifier="run-1",
            )
        )
    )

    snapshot = backend.snapshot()

    assert len(snapshot.events) == 3
    assert len(snapshot.records) == 2
    assert snapshot.events[0].event_type == "run_started"
    assert snapshot.events[1].event_type == "artifact_registered"
    assert snapshot.records[0].facts["kind"] == "benchmark_report"
    assert snapshot.records[1].facts["kind"] == "debug_snapshot"
    assert len(snapshot.edges) == 3
    assert {edge.relationship for edge in snapshot.edges} == {"derived_from"}
    assert {edge.target.identifier for edge in snapshot.edges} == {"run-1"}
    assert {edge.source.identifier for edge in snapshot.edges} == {"/tmp/report.md", "snapshot-1"}
