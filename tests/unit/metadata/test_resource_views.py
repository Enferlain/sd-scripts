"""Tests for resource metadata graph views."""

from __future__ import annotations

import pytest

from library.metadata import (
    MetadataEntityType,
    MetadataRelationship,
    MetadataRuntime,
    ResourceAccountingFacts,
    ResourceAccountingGapFacts,
    ResourceFactReference,
    ResourceObservationFacts,
    ResourceObservationFrameFacts,
    ResourceObservationMeasurementFacts,
    ResourceProfileFacts,
    ResourceRunView,
    StructuralResourceFacts,
    edges_from,
    metadata_identity,
    records_by_identity,
    source_identities,
    target_identities,
)


@pytest.mark.unit
def test_metadata_graph_helpers_traverse_snapshot_edges_and_records() -> None:
    runtime = MetadataRuntime()
    runtime.file(_resource_frame())
    snapshot = runtime.snapshot()
    frame = snapshot.record_for(entity_type="resource_observation_frame", identifier="frame-1")

    assert frame is not None

    run_targets = target_identities(
        snapshot,
        frame.identity,
        relationship=MetadataRelationship.OBSERVED_DURING,
        target_type=MetadataEntityType.RUN,
    )
    device_sources = source_identities(
        snapshot,
        metadata_identity(entity_type=MetadataEntityType.DEVICE, identifier="cuda:0"),
        relationship=MetadataRelationship.OBSERVED_ON,
        source_type="resource_observation",
    )
    resolved_records = records_by_identity(snapshot, device_sources)

    assert [identity.identifier for identity in run_targets] == ["run-1"]
    assert [record.identity.identifier for record in resolved_records] == ["frame-1:gpu-used"]
    assert edges_from(
        snapshot,
        frame.identity,
        relationship=MetadataRelationship.OBSERVED_DURING,
        target_type=MetadataEntityType.PHASE,
        target_identifier="run-1:phase:training.epoch.0",
    )


@pytest.mark.unit
def test_resource_run_view_queries_run_scoped_resource_facts_and_evidence() -> None:
    runtime = MetadataRuntime()
    runtime.file_many(
        (
            _resource_frame(),
            _direct_observation(),
            _structural_fact(),
            _profile(),
            _accounting(),
            _accounting_gap(),
            _other_run_frame(),
        )
    )
    snapshot = runtime.snapshot()
    view = ResourceRunView.from_snapshot(snapshot, run_identifier="run-1")

    assert [record.identity.identifier for record in view.observation_frames()] == ["frame-1"]
    assert [record.identity.identifier for record in view.observations()] == [
        "frame-1:gpu-used",
        "frame-1:cpu-rss",
        "obs-direct",
    ]
    assert [record.identity.identifier for record in view.observation_frames_for_phase("training.epoch.0")] == [
        "frame-1"
    ]
    assert [record.identity.identifier for record in view.observation_frames_for_step(12)] == ["frame-1"]
    assert [record.identity.identifier for record in view.observations_for_device("cuda:0")] == [
        "frame-1:gpu-used"
    ]
    assert [record.identity.identifier for record in view.structural_facts()] == ["struct-1"]
    assert [record.identity.identifier for record in view.profiles()] == ["profile-1"]
    assert [record.identity.identifier for record in view.accounting_statements()] == ["acct-1"]
    assert [record.identity.identifier for record in view.accounting_gaps()] == ["gap-1"]

    frame = view.observation_frames()[0]
    profile = view.profiles()[0]
    accounting = view.accounting_statements()[0]
    gap = view.accounting_gaps()[0]

    assert [record.identity.identifier for record in view.measurements_for_frame(frame)] == [
        "frame-1:gpu-used",
        "frame-1:cpu-rss",
    ]
    assert [record.identity.identifier for record in view.source_records_for(profile)] == ["frame-1:gpu-used"]
    assert [record.identity.identifier for record in view.source_records_for(accounting)] == ["struct-1"]
    assert [record.identity.identifier for record in view.source_records_for(gap)] == ["frame-1:gpu-used"]
    assert view.profiles()[0] is snapshot.record_for(entity_type="resource_profile", identifier="profile-1")


def _resource_frame() -> ResourceObservationFrameFacts:
    return ResourceObservationFrameFacts(
        frame_identifier="frame-1",
        run_identifier="run-1",
        event_name="step_sample",
        ts=123.0,
        collector_id="sampled",
        phase="training.epoch.0",
        global_step=12,
        epoch=3,
        rank=0,
        world_size=2,
        host_identifier="host-a",
        process_identifier="pid-123",
        measurements=(
            ResourceObservationMeasurementFacts(
                measurement_identifier="frame-1:gpu-used",
                resource_kind="gpu_memory",
                measurement_kind="used",
                value=2048.0,
                unit="MiB",
                source="nvml",
                scope_type="device",
                device_identifier="cuda:0",
            ),
            ResourceObservationMeasurementFacts(
                measurement_identifier="frame-1:cpu-rss",
                resource_kind="cpu_memory",
                measurement_kind="rss",
                value=512.0,
                unit="MiB",
                source="psutil",
                scope_type="process",
            ),
        ),
    )


def _direct_observation() -> ResourceObservationFacts:
    return ResourceObservationFacts(
        observation_identifier="obs-direct",
        run_identifier="run-1",
        resource_kind="cpu_memory",
        measurement_kind="rss",
        value=768.0,
        unit="MiB",
        source="psutil",
        phase="training.epoch.0",
    )


def _structural_fact() -> StructuralResourceFacts:
    return StructuralResourceFacts(
        structural_identifier="struct-1",
        run_identifier="run-1",
        owner_type="model_component",
        owner_identifier="denoiser",
        resource_kind="parameter_memory",
        quantity=1024.0,
        unit="MiB",
        basis="parameter_bytes",
        source="startup_component_memory",
        component_key="unet",
    )


def _profile() -> ResourceProfileFacts:
    return ResourceProfileFacts(
        profile_identifier="profile-1",
        run_identifier="run-1",
        profile_kind="session_peaks",
        derivation_version="v1",
        values={"gpu_used_peak_mib": 2048.0},
        source_fact_references=(
            ResourceFactReference(
                entity_type="resource_observation",
                identifier="frame-1:gpu-used",
            ),
        ),
    )


def _accounting() -> ResourceAccountingFacts:
    return ResourceAccountingFacts(
        accounting_identifier="acct-1",
        run_identifier="run-1",
        resource_kind="parameter_memory",
        quantity=1024.0,
        unit="MiB",
        owner_type="model_component",
        owner_identifier="denoiser",
        basis="structural",
        derivation_method="parameter_bytes",
        derivation_version="v1",
        source_fact_references=(
            ResourceFactReference(
                entity_type="resource_structural_fact",
                identifier="struct-1",
                relationship="supported_by",
            ),
        ),
    )


def _accounting_gap() -> ResourceAccountingGapFacts:
    return ResourceAccountingGapFacts(
        gap_identifier="gap-1",
        run_identifier="run-1",
        resource_kind="gpu_memory",
        quantity=256.0,
        unit="MiB",
        basis="observed_minus_accounted",
        derivation_method="session_peak",
        derivation_version="v1",
        source_fact_references=(
            ResourceFactReference(
                entity_type="resource_observation",
                identifier="frame-1:gpu-used",
            ),
        ),
    )


def _other_run_frame() -> ResourceObservationFrameFacts:
    return ResourceObservationFrameFacts(
        frame_identifier="other-frame",
        run_identifier="run-other",
        event_name="step_sample",
        measurements=(
            ResourceObservationMeasurementFacts(
                measurement_identifier="other-frame:gpu-used",
                resource_kind="gpu_memory",
                measurement_kind="used",
                value=4096.0,
                unit="MiB",
                source="nvml",
                device_identifier="cuda:0",
            ),
        ),
    )
