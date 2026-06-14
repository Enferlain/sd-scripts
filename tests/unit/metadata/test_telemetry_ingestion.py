"""Tests for bounded high-frequency metadata ingestion."""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from typing import Literal

import pytest

from library.metadata import (
    InMemoryMetadataBackend,
    MetadataRelationship,
    MetadataBufferPolicy,
    MetadataRuntime,
    ResourceObservationFrameFacts,
    ResourceObservationMeasurementFacts,
    SQLiteMetadataStore,
)


def _frame(index: int) -> ResourceObservationFrameFacts:
    return ResourceObservationFrameFacts(
        frame_identifier=f"frame-{index}",
        run_identifier="run-1",
        event_name="sample",
        ts=float(index),
        collector_id="sampled",
        rank=0,
        measurements=(
            ResourceObservationMeasurementFacts(
                measurement_identifier=f"frame-{index}:gpu-used",
                resource_kind="gpu_memory",
                measurement_kind="used",
                value=1000.0 + index,
                unit="MiB",
                source="nvml",
                device_identifier="cuda:0",
            ),
            ResourceObservationMeasurementFacts(
                measurement_identifier=f"frame-{index}:cpu-rss",
                resource_kind="cpu_memory",
                measurement_kind="rss",
                value=500.0 + index,
                unit="MiB",
                source="psutil",
            ),
        ),
    )


@pytest.mark.unit
def test_telemetry_buffer_flushes_bounded_batches() -> None:
    runtime = MetadataRuntime(
        telemetry_buffer_policy=MetadataBufferPolicy(capacity=10, flush_max_items=2),
    )
    for index in range(5):
        assert runtime.buffer(_frame(index)) is True

    first_report = runtime.flush_buffer()
    second_report = runtime.flush_buffer()

    assert first_report.flushed_items == 2
    assert first_report.pending_items == 3
    assert second_report.flushed_items == 4
    assert second_report.pending_items == 1
    assert len(runtime.snapshot().records_for(entity_type="resource_observation_frame")) == 4


@pytest.mark.unit
@pytest.mark.parametrize(
    ("drop_policy", "expected_frames"),
    [
        ("drop_oldest", {"frame-1", "frame-2"}),
        ("drop_newest", {"frame-0", "frame-1"}),
    ],
)
def test_telemetry_buffer_applies_retention_policy_and_records_degradation(
    drop_policy: Literal["drop_oldest", "drop_newest"],
    expected_frames: set[str],
) -> None:
    runtime = MetadataRuntime(
        telemetry_buffer_policy=MetadataBufferPolicy(
            capacity=2,
            drop_policy=drop_policy,
            flush_max_items=10,
        ),
    )

    assert runtime.buffer(_frame(0)) is True
    assert runtime.buffer(_frame(1)) is True
    assert runtime.buffer(_frame(2)) is (drop_policy == "drop_oldest")
    report = runtime.flush_buffer()
    snapshot = runtime.snapshot()

    assert report.dropped_items == 1
    assert report.capacity_dropped_items == 1
    assert report.ingestion_failed_items == 0
    assert report.degraded is True
    assert {
        record.identity.identifier
        for record in snapshot.records_for(entity_type="resource_observation_frame")
    } == expected_frames
    degradation_events = [event for event in snapshot.events if event.event_type == "metadata_ingestion_degraded"]
    assert len(degradation_events) == 1
    assert degradation_events[0].facts["dropped_items"] == 1
    assert degradation_events[0].facts["capacity_dropped_items"] == 1
    assert degradation_events[0].facts["ingestion_failed_items"] == 0
    assert degradation_events[0].facts["drop_policy"] == drop_policy


@pytest.mark.unit
def test_telemetry_buffer_degrades_without_raising_on_backend_failure() -> None:
    class FailingBackend(InMemoryMetadataBackend):
        def ingest_many(self, results) -> None:
            raise RuntimeError("storage unavailable")

    runtime = MetadataRuntime(backend=FailingBackend())
    runtime.buffer(_frame(0))

    report = runtime.flush_buffer(max_items=1)

    assert report.flushed_items == 0
    assert report.dropped_items == 1
    assert report.capacity_dropped_items == 0
    assert report.ingestion_failed_items == 1
    assert report.pending_items == 0
    assert report.degraded is True
    assert report.error_message == "storage unavailable"


@pytest.mark.unit
def test_telemetry_buffer_records_prior_backend_failure_after_recovery() -> None:
    class RecoveringBackend(InMemoryMetadataBackend):
        fail_next = True

        def ingest_many(self, results) -> None:
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("storage unavailable")
            super().ingest_many(results)

    runtime = MetadataRuntime(backend=RecoveringBackend())
    runtime.buffer(_frame(0))
    runtime.flush_buffer(max_items=1)

    recovered_report = runtime.flush_buffer(max_items=1)
    degradation_events = [
        event
        for event in runtime.snapshot().events
        if event.event_type == "metadata_ingestion_degraded"
    ]

    assert recovered_report.error_message is None
    assert len(degradation_events) == 1
    assert degradation_events[0].facts["dropped_items"] == 1
    assert degradation_events[0].facts["capacity_dropped_items"] == 0
    assert degradation_events[0].facts["ingestion_failed_items"] == 1
    assert degradation_events[0].facts["error_message"] == "storage unavailable"


@pytest.mark.unit
def test_telemetry_buffer_serializes_concurrent_flushes_without_blocking_producers() -> None:
    class TrackingBackend(InMemoryMetadataBackend):
        def __init__(self) -> None:
            super().__init__()
            self._tracking_lock = Lock()
            self.entered_ingestion = Event()
            self.allow_ingestion = Event()
            self.active_ingestions = 0
            self.max_active_ingestions = 0

        def ingest_many(self, results) -> None:
            with self._tracking_lock:
                self.active_ingestions += 1
                self.max_active_ingestions = max(self.max_active_ingestions, self.active_ingestions)
            self.entered_ingestion.set()
            assert self.allow_ingestion.wait(timeout=1)
            try:
                time.sleep(0.02)
                super().ingest_many(results)
            finally:
                with self._tracking_lock:
                    self.active_ingestions -= 1

    backend = TrackingBackend()
    runtime = MetadataRuntime(
        backend=backend,
        telemetry_buffer_policy=MetadataBufferPolicy(capacity=10, flush_max_items=1),
    )
    runtime.buffer(_frame(0))
    runtime.buffer(_frame(1))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_flush = executor.submit(runtime.flush_buffer)
        assert backend.entered_ingestion.wait(timeout=1)
        second_flush = executor.submit(runtime.flush_buffer)
        assert runtime.buffer(_frame(2)) is True
        backend.allow_ingestion.set()
        flushes = [first_flush, second_flush]
        reports = [flush.result() for flush in flushes]

    assert backend.max_active_ingestions == 1
    assert max(report.flushed_items for report in reports) == 2
    assert runtime.buffer_report().pending_items == 1


@pytest.mark.unit
def test_resource_observation_frame_volume_round_trips_through_memory_and_sqlite() -> None:
    frame_count = 250
    frames = tuple(_frame(index) for index in range(frame_count))
    runtimes = (
        MetadataRuntime(),
        MetadataRuntime(
            backend=InMemoryMetadataBackend(
                store=SQLiteMetadataStore(sqlite3.connect(":memory:")),
            )
        ),
    )

    for runtime in runtimes:
        runtime.file_many(frames)
        snapshot = runtime.snapshot()

        assert len(snapshot.records_for(entity_type="resource_observation_frame")) == frame_count
        assert len(snapshot.records_for(entity_type="resource_observation")) == frame_count * 2
        assert len(snapshot.edges) == frame_count * 7
        assert {edge.relationship for edge in snapshot.edges} == {
            MetadataRelationship.CONTAINED_IN,
            MetadataRelationship.OBSERVED_DURING,
            MetadataRelationship.OBSERVED_IN,
            MetadataRelationship.OBSERVED_ON,
            MetadataRelationship.PRODUCED_BY,
        }
