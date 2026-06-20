"""Tests for resource metadata graph views."""

from __future__ import annotations

import sqlite3

import pytest

from library.metadata import (
    edges_from,
    InMemoryMetadataBackend,
    metadata_identity,
    MetadataEntityType,
    MetadataGraphIndex,
    MetadataRecord,
    MetadataRelationship,
    MetadataRuntime,
    MetadataSnapshot,
    project_resource_accounting_export,
    project_resource_monitor_compatibility_event,
    project_resource_profile_export,
    project_resource_report,
    project_resource_run_compatibility_events,
    records_by_identity,
    RESOURCE_ACCOUNTING_EXPORT_SCHEMA,
    RESOURCE_EXPORT_SCHEMA_VERSION,
    RESOURCE_PROFILE_EXPORT_SCHEMA,
    RESOURCE_REPORT_EXPORT_SCHEMA,
    ResourceAccountingFacts,
    ResourceAccountingGapFacts,
    ResourceFactReference,
    ResourceObservationFacts,
    ResourceObservationFrameFacts,
    ResourceObservationMeasurementFacts,
    ResourceProfileFacts,
    ResourceRunView,
    RunReportFacts,
    source_identities,
    SQLiteMetadataStore,
    StructuralResourceFacts,
    target_identities,
)


@pytest.fixture(params=("memory", "sqlite"))
def resource_runtime(request: pytest.FixtureRequest) -> MetadataRuntime:
    if request.param == "memory":
        return MetadataRuntime()

    connection = sqlite3.connect(":memory:")
    request.addfinalizer(connection.close)
    return MetadataRuntime(
        backend=InMemoryMetadataBackend(
            store=SQLiteMetadataStore(connection),
        )
    )


@pytest.mark.unit
def test_metadata_graph_helpers_traverse_snapshot_edges_and_records() -> None:
    runtime = MetadataRuntime()
    runtime.file(_resource_frame())
    snapshot = runtime.snapshot()
    graph = MetadataGraphIndex.from_snapshot(snapshot)
    frame = snapshot.record_for(entity_type="resource_observation_frame", identifier="frame-1")

    assert frame is not None

    run_targets = target_identities(
        graph,
        frame.identity,
        relationship=MetadataRelationship.OBSERVED_DURING,
        target_type=MetadataEntityType.RUN,
    )
    device_sources = source_identities(
        graph,
        metadata_identity(entity_type=MetadataEntityType.DEVICE, identifier="cuda:0"),
        relationship=MetadataRelationship.OBSERVED_ON,
        source_type="resource_observation",
    )
    resolved_records = records_by_identity(graph, device_sources)

    assert [identity.identifier for identity in run_targets] == ["run-1"]
    assert [record.identity.identifier for record in resolved_records] == ["frame-1:gpu-used"]
    assert edges_from(
        graph,
        frame.identity,
        relationship=MetadataRelationship.OBSERVED_DURING,
        target_type=MetadataEntityType.PHASE,
        target_identifier="run-1:phase:training.epoch.0",
    )
    assert MetadataGraphIndex.from_snapshot(graph) is graph


@pytest.mark.unit
def test_metadata_graph_index_matches_snapshot_namespace_lookup_semantics() -> None:
    first = MetadataRecord(
        identity=metadata_identity(entity_type="resource_profile", identifier="profile-1", namespace="first"),
        producer="test",
    )
    second = MetadataRecord(
        identity=metadata_identity(entity_type="resource_profile", identifier="profile-1", namespace="second"),
        producer="test",
    )
    snapshot = MetadataSnapshot(records=(first, second))
    graph = MetadataGraphIndex.from_snapshot(snapshot)

    assert graph.record_for(entity_type="resource_profile", identifier="profile-1") is second
    assert graph.record_for(entity_type="resource_profile", identifier="profile-1", namespace="first") is first
    assert graph.record_for(entity_type="resource_profile", identifier="profile-1", namespace="second") is second


@pytest.mark.unit
def test_resource_compatibility_projection_is_identical_before_and_after_filing() -> None:
    frame = _resource_frame()
    runtime = MetadataRuntime()
    runtime.file(frame)
    view = ResourceRunView.from_snapshot(runtime.snapshot(), run_identifier="run-1")

    assert project_resource_run_compatibility_events(view) == (
        project_resource_monitor_compatibility_event(frame),
    )


@pytest.mark.unit
def test_resource_exports_are_versioned_and_preserve_profile_accounting_evidence() -> None:
    runtime = MetadataRuntime()
    runtime.file_many(
        (
            _resource_frame(),
            _structural_fact(),
            _profile(),
            _accounting(),
            _accounting_gap(),
        )
    )
    view = ResourceRunView.from_snapshot(runtime.snapshot(), run_identifier="run-1")

    report = project_resource_report(view)
    profiles = project_resource_profile_export(view)
    accounting = project_resource_accounting_export(view)

    assert report.schema_name == RESOURCE_REPORT_EXPORT_SCHEMA
    assert report.schema_version == RESOURCE_EXPORT_SCHEMA_VERSION
    assert report.payload["input_source"] == "metadata_view"
    assert report.payload["event_count"] == 1
    assert [identity.identifier for identity in report.source_identities] == [
        "frame-1",
        "frame-1:gpu-used",
        "frame-1:cpu-rss",
    ]

    profile_document = profiles.document()
    assert profile_document["schema_name"] == RESOURCE_PROFILE_EXPORT_SCHEMA
    assert profile_document["run_identifier"] == "run-1"
    assert profile_document["profiles"][0]["facts"]["semantic_class"] == "profile"
    assert profile_document["profiles"][0]["resolved_source_identities"][0]["identifier"] == "frame-1:gpu-used"

    accounting_document = accounting.document()
    assert accounting_document["schema_name"] == RESOURCE_ACCOUNTING_EXPORT_SCHEMA
    assert accounting_document["statements"][0]["facts"]["owner_identifier"] == "denoiser"
    assert accounting_document["gaps"][0]["facts"]["semantic_class"] == "accounting_gap"
    assert "owner_identifier" not in accounting_document["gaps"][0]["facts"]
    assert {identity.identifier for identity in accounting.source_identities} == {
        "acct-1",
        "gap-1",
        "struct-1",
        "frame-1:gpu-used",
    }


@pytest.mark.unit
def test_resource_run_view_preserves_semantic_fact_classes_and_evidence(
    resource_runtime: MetadataRuntime,
) -> None:
    resource_runtime.file_many(
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
    snapshot = resource_runtime.snapshot()
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

    semantic_classes = {
        record.identity.identifier: record.facts["semantic_class"]
        for record in view.resource_records()
    }
    assert semantic_classes == {
        "frame-1": "observation",
        "frame-1:gpu-used": "observation",
        "frame-1:cpu-rss": "observation",
        "obs-direct": "observation",
        "struct-1": "structural",
        "profile-1": "profile",
        "acct-1": "accounting",
        "gap-1": "accounting_gap",
    }

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
    assert profile.facts["values"] == {"gpu_used_peak_mib": 2048.0}


@pytest.mark.unit
def test_resource_run_view_queries_multi_scope_and_artifact_linked_facts() -> None:
    runtime = MetadataRuntime()
    runtime.file_many(
        (
            RunReportFacts(
                report_identifier="report-12.json",
                run_identifier="run-1",
                status="succeeded",
                generated_at=123.0,
                output_name="report-12",
                global_step=12,
            ),
            ResourceObservationFacts(
                observation_identifier="source-run:gpu-used",
                run_identifier="source-run",
                resource_kind="gpu_memory",
                measurement_kind="used",
                value=1536.0,
                unit="MiB",
                source="nvml",
                device_identifier="cuda:7",
            ),
            _scoped_frame(
                frame_identifier="frame-rank-0",
                rank=0,
                device_identifier="cuda:0",
                phase="training.epoch.0",
                global_step=11,
            ),
            _scoped_frame(
                frame_identifier="frame-rank-1",
                rank=1,
                device_identifier="cuda:1",
                phase="training.epoch.1",
                global_step=12,
            ),
            _component_structural_fact(
                structural_identifier="struct-unet",
                component_identifier="unet",
            ),
            _component_structural_fact(
                structural_identifier="struct-text-encoder",
                component_identifier="text_encoder",
            ),
            ResourceProfileFacts(
                profile_identifier="profile-linked",
                run_identifier="run-1",
                profile_kind="checkpoint_context",
                derivation_version="v1",
                values={"gpu_used_peak_mib": 2048.0},
                source_fact_references=(
                    ResourceFactReference(
                        entity_type="artifact",
                        identifier="report-12.json",
                        relationship="supported_by",
                    ),
                    ResourceFactReference(
                        entity_type="resource_observation",
                        identifier="source-run:gpu-used",
                    ),
                    ResourceFactReference(
                        entity_type="resource_observation",
                        identifier="missing-observation",
                    ),
                ),
            ),
        )
    )
    view = ResourceRunView.from_snapshot(runtime.snapshot(), run_identifier="run-1")

    assert [record.identity.identifier for record in view.observation_frames_for_rank(0)] == ["frame-rank-0"]
    assert [record.identity.identifier for record in view.observation_frames_for_rank(1)] == ["frame-rank-1"]
    assert [record.identity.identifier for record in view.observations_for_device("cuda:0")] == [
        "frame-rank-0:gpu-used"
    ]
    assert [record.identity.identifier for record in view.observations_for_device("cuda:1")] == [
        "frame-rank-1:gpu-used"
    ]
    assert [record.identity.identifier for record in view.observation_frames_for_phase("training.epoch.1")] == [
        "frame-rank-1"
    ]
    assert [record.identity.identifier for record in view.structural_facts_for_component("unet")] == [
        "struct-unet"
    ]
    assert [record.identity.identifier for record in view.structural_facts_for_component("text_encoder")] == [
        "struct-text-encoder"
    ]

    source_records = view.source_records_for(view.profiles()[0])
    assert [(record.identity.entity_type, record.identity.identifier) for record in source_records] == [
        ("artifact", "report-12.json"),
        ("resource_observation", "source-run:gpu-used"),
    ]


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


def _scoped_frame(
    *,
    frame_identifier: str,
    rank: int,
    device_identifier: str,
    phase: str,
    global_step: int,
) -> ResourceObservationFrameFacts:
    return ResourceObservationFrameFacts(
        frame_identifier=frame_identifier,
        run_identifier="run-1",
        event_name="step_sample",
        phase=phase,
        global_step=global_step,
        rank=rank,
        world_size=2,
        measurements=(
            ResourceObservationMeasurementFacts(
                measurement_identifier=f"{frame_identifier}:gpu-used",
                resource_kind="gpu_memory",
                measurement_kind="used",
                value=2048.0 + rank,
                unit="MiB",
                source="nvml",
                scope_type="device",
                device_identifier=device_identifier,
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


def _component_structural_fact(
    *,
    structural_identifier: str,
    component_identifier: str,
) -> StructuralResourceFacts:
    return StructuralResourceFacts(
        structural_identifier=structural_identifier,
        run_identifier="run-1",
        owner_type="model_component",
        owner_identifier=component_identifier,
        resource_kind="parameter_memory",
        quantity=512.0,
        unit="MiB",
        basis="parameter_bytes",
        source="startup_component_memory",
        component_key=component_identifier,
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
