"""Unit tests for the runtime-facing metadata filing API."""

from __future__ import annotations

import pytest

from library.metadata import LoggedArtifactFacts, METADATA_PAYLOAD_VERSION, MetadataRuntime, RunLifecycleFacts
from library.metadata.validation import MetadataItemValidationError


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

    snapshot = runtime.snapshot()

    assert [event.event_type for event in snapshot.events] == [
        "run_started",
        "artifact_registered",
    ]
    assert snapshot.events[0].schema_version == METADATA_PAYLOAD_VERSION
    assert snapshot.events[0].identity.schema_version == METADATA_PAYLOAD_VERSION
    assert snapshot.events[1].facts["metadata"] == {"format": "markdown"}


@pytest.mark.unit
def test_metadata_runtime_rejects_unknown_items() -> None:
    runtime = MetadataRuntime()

    with pytest.raises(MetadataItemValidationError):
        runtime.file(object())  # type: ignore[arg-type]
