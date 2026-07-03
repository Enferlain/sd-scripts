"""Tests for resource metadata graph views."""

from __future__ import annotations

import sqlite3

from unittest.mock import patch

import pytest

from library.metadata import (
    compare_resource_profiles,
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
    project_resource_report_compatibility_events,
    project_resource_run_compatibility_events,
    records_by_identity,
    RESOURCE_ACCOUNTING_EXPORT_SCHEMA,
    RESOURCE_EXPORT_SCHEMA_VERSION,
    RESOURCE_PROFILE_EXPORT_SCHEMA,
    RESOURCE_REPORT_EXPORT_SCHEMA,
    ResourceAccountingFacts,
    ResourceAccountingGapFacts,
    ResourceCollectorStatusFacts,
    ResourceFactReference,
    ResourceObservationFacts,
    ResourceObservationFrameFacts,
    ResourceObservationMeasurementFacts,
    ResourceProfileFacts,
    ResourceProfileValueComparison,
    ResourceRunView,
    RunReportFacts,
    source_identities,
    SQLiteMetadataStore,
    StructuralResourceFacts,
    target_identities,
)


_RESOURCE_REPORT_COMPATIBILITY_EXACT_KEYS = (
    "event_count",
    "total_resource_event_count",
    "total_jsonl_event_count",
    "session_start",
    "session_end",
    "gpu_used_peak_session_mb",
    "gpu_used_peak_session_by_device_mb",
    "phases",
    "debug",
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

    assert project_resource_run_compatibility_events(view) == (project_resource_monitor_compatibility_event(frame),)


@pytest.mark.unit
def test_resource_report_projection_matches_compatibility_output_for_declared_fields() -> None:
    runtime = MetadataRuntime()
    runtime.file_many(
        (
            _resource_frame(),
            _scoped_frame(
                frame_identifier="frame-phase-start",
                rank=0,
                device_identifier="cuda:0",
                phase="training.epoch.0",
                global_step=12,
            ),
            _scoped_frame(
                frame_identifier="frame-phase-end",
                rank=0,
                device_identifier="cuda:0",
                phase="training.epoch.0",
                global_step=13,
            ),
        )
    )
    view = ResourceRunView.from_snapshot(runtime.snapshot(), run_identifier="run-1")
    compatibility_events = project_resource_run_compatibility_events(view)

    metadata_report = project_resource_report(
        view,
        jsonl_path="resource.jsonl",
        total_jsonl_event_count=len(compatibility_events),
    )
    compatibility_report = project_resource_report_compatibility_events(
        compatibility_events,
        run_identifier="run-1",
        jsonl_path="resource.jsonl",
        input_source="jsonl_compatibility",
        total_jsonl_event_count=len(compatibility_events),
    )

    assert metadata_report.payload["input_source"] == "metadata_view"
    assert compatibility_report.payload["input_source"] == "jsonl_compatibility"
    assert metadata_report.schema_name == compatibility_report.schema_name
    assert metadata_report.schema_version == compatibility_report.schema_version
    for key in _RESOURCE_REPORT_COMPATIBILITY_EXACT_KEYS:
        assert metadata_report.payload[key] == compatibility_report.payload[key]


@pytest.mark.unit
def test_frame_measurements_preserve_declared_order_and_run_boundary(
    resource_runtime: MetadataRuntime,
) -> None:
    resource_runtime.file_many((_resource_frame(), _other_run_frame()))
    snapshot = resource_runtime.snapshot()
    reordered_snapshot = MetadataSnapshot(
        records=snapshot.records,
        events=snapshot.events,
        edges=tuple(reversed(snapshot.edges)),
        required_facts=snapshot.required_facts,
    )
    view = ResourceRunView.from_snapshot(reordered_snapshot, run_identifier="run-1")
    frame = snapshot.record_for(entity_type="resource_observation_frame", identifier="frame-1")
    other_frame = snapshot.record_for(entity_type="resource_observation_frame", identifier="other-frame")

    assert frame is not None
    assert other_frame is not None
    assert [record.identity.identifier for record in view.measurements_for_frame(frame)] == [
        "frame-1:gpu-used",
        "frame-1:cpu-rss",
    ]
    assert view.measurements_for_frame(other_frame) == ()


@pytest.mark.unit
def test_resource_report_projection_traverses_run_observations_once(
    resource_runtime: MetadataRuntime,
) -> None:
    resource_runtime.file_many(
        tuple(
            _scoped_frame(
                frame_identifier=f"frame-{index}",
                rank=index % 2,
                device_identifier=f"cuda:{index % 2}",
                phase="training.epoch.0",
                global_step=index,
            )
            for index in range(25)
        )
    )
    view = ResourceRunView.from_snapshot(resource_runtime.snapshot(), run_identifier="run-1")
    original_observations = ResourceRunView.observations
    observation_calls = 0

    def counted_observations(current_view: ResourceRunView) -> tuple[MetadataRecord, ...]:
        nonlocal observation_calls
        observation_calls += 1
        return original_observations(current_view)

    with patch.object(ResourceRunView, "observations", counted_observations):
        report = project_resource_report(view)

    assert report.payload["event_count"] == 25
    assert observation_calls == 1


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
        "profile-1",
        "acct-1",
        "struct-1",
        "gap-1",
    ]
    assert report.payload["profile_views"][0]["facts"]["semantic_class"] == "profile"
    assert report.payload["profile_views"][0]["facts"]["values"] == {"gpu_used_peak_mib": 2048.0}
    assert report.payload["profile_views"][0]["resolved_source_identities"][0]["identifier"] == "frame-1:gpu-used"
    report_accounting = report.payload["accounting_views"]
    assert report_accounting["statements"][0]["facts"]["semantic_class"] == "accounting"
    assert report_accounting["statements"][0]["facts"]["owner_identifier"] == "denoiser"
    assert report_accounting["gaps"][0]["facts"]["semantic_class"] == "accounting_gap"
    assert "owner_identifier" not in report_accounting["gaps"][0]["facts"]

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
    assert [record.identity.identifier for record in view.observation_frames_for_phase("training.epoch.0")] == ["frame-1"]
    assert [record.identity.identifier for record in view.observation_frames_for_step(12)] == ["frame-1"]
    assert [record.identity.identifier for record in view.observations_for_device("cuda:0")] == ["frame-1:gpu-used"]
    assert [record.identity.identifier for record in view.structural_facts()] == ["struct-1"]
    assert [record.identity.identifier for record in view.profiles()] == ["profile-1"]
    assert [record.identity.identifier for record in view.accounting_statements()] == ["acct-1"]
    assert [record.identity.identifier for record in view.accounting_gaps()] == ["gap-1"]

    semantic_classes = {record.identity.identifier: record.facts["semantic_class"] for record in view.resource_records()}
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
def test_resource_run_view_compares_stored_profiles_for_regression_queries(
    resource_runtime: MetadataRuntime,
) -> None:
    resource_runtime.file_many(
        (
            _profile_for_run(
                profile_identifier="baseline-profile",
                run_identifier="run-baseline",
                generated_at=1.0,
                values={
                    "gpu_used_peak_mib": 100.0,
                    "session_duration_s": 10.0,
                    "phase_names": ["training.epoch.0"],
                    "missing_from_candidate": 7.0,
                },
            ),
            _profile_for_run(
                profile_identifier="candidate-profile-old",
                run_identifier="run-candidate",
                generated_at=1.0,
                values={"gpu_used_peak_mib": 999.0},
            ),
            _profile_for_run(
                profile_identifier="candidate-profile-new",
                run_identifier="run-candidate",
                generated_at=2.0,
                values={
                    "gpu_used_peak_mib": 120.0,
                    "session_duration_s": 9.0,
                    "phase_names": ["training.epoch.0"],
                    "new_candidate_value": 3.0,
                },
            ),
            _profile_for_run(
                profile_identifier="candidate-profile-newer-same-timestamp",
                run_identifier="run-candidate",
                generated_at=2.0,
                values={
                    "gpu_used_peak_mib": 120.0,
                    "session_duration_s": 9.0,
                    "phase_names": ["training.epoch.0"],
                    "new_candidate_value": 3.0,
                },
            ),
            _profile_for_run(
                profile_identifier="candidate-profile-v2",
                run_identifier="run-candidate",
                derivation_version="v2",
                generated_at=3.0,
                values={"gpu_used_peak_mib": 130.0},
            ),
        )
    )
    snapshot = resource_runtime.snapshot()
    baseline_view = ResourceRunView.from_snapshot(snapshot, run_identifier="run-baseline")
    candidate_view = ResourceRunView.from_snapshot(snapshot, run_identifier="run-candidate")

    assert [profile.identity.identifier for profile in candidate_view.profiles_for(profile_kind="session_peaks")] == [
        "candidate-profile-old",
        "candidate-profile-new",
        "candidate-profile-newer-same-timestamp",
        "candidate-profile-v2",
    ]
    assert (
        candidate_view.latest_profile(
            profile_kind="session_peaks",
            derivation_version="v1",
        ).identity.identifier
        == "candidate-profile-newer-same-timestamp"
    )

    comparison = candidate_view.compare_profile_to(
        baseline_view,
        profile_kind="session_peaks",
        derivation_version="v1",
        regression_thresholds={
            "gpu_used_peak_mib": 16.0,
            "session_duration_s": 0.0,
        },
    )

    assert comparison is not None
    assert comparison.compatible is True
    assert comparison.profile_kind == "session_peaks"
    assert comparison.derivation_version == "v1"
    by_key = {value.key: value for value in comparison.values}
    assert by_key["gpu_used_peak_mib"] == ResourceProfileValueComparison(
        key="gpu_used_peak_mib",
        baseline_value=100.0,
        candidate_value=120.0,
        status="numeric",
        delta=20.0,
        ratio=1.2,
        regression_threshold=16.0,
        regression=True,
    )
    assert by_key["session_duration_s"].delta == -1.0
    assert by_key["session_duration_s"].regression is False
    assert by_key["phase_names"].status == "non_numeric"
    assert by_key["missing_from_candidate"].status == "missing_candidate"
    assert by_key["new_candidate_value"].status == "missing_baseline"
    assert [value.key for value in comparison.regressions()] == ["gpu_used_peak_mib"]


@pytest.mark.unit
def test_resource_profile_comparison_rejects_incompatible_profile_shapes() -> None:
    baseline = _profile_for_run(
        profile_identifier="baseline-profile",
        run_identifier="run-baseline",
        profile_kind="session_peaks",
        derivation_version="v1",
        values={"gpu_used_peak_mib": 100.0},
    )
    different_version = _profile_for_run(
        profile_identifier="candidate-profile-v2",
        run_identifier="run-candidate",
        profile_kind="session_peaks",
        derivation_version="v2",
        values={"gpu_used_peak_mib": 120.0},
    )
    different_kind = _profile_for_run(
        profile_identifier="candidate-structural",
        run_identifier="run-candidate",
        profile_kind="structural_summary",
        derivation_version="v1",
        values={"gpu_used_peak_mib": 120.0},
    )
    runtime = MetadataRuntime()
    runtime.file_many((baseline, different_version, different_kind))
    snapshot = runtime.snapshot()

    baseline_record = snapshot.record_for(entity_type="resource_profile", identifier="baseline-profile")
    version_record = snapshot.record_for(entity_type="resource_profile", identifier="candidate-profile-v2")
    kind_record = snapshot.record_for(entity_type="resource_profile", identifier="candidate-structural")

    assert baseline_record is not None
    assert version_record is not None
    assert kind_record is not None
    version_comparison = compare_resource_profiles(baseline_record, version_record)
    kind_comparison = compare_resource_profiles(baseline_record, kind_record)

    assert version_comparison.compatible is False
    assert version_comparison.reason == "derivation_version_mismatch"
    assert version_comparison.values == ()
    assert kind_comparison.compatible is False
    assert kind_comparison.reason == "profile_kind_mismatch"
    assert kind_comparison.values == ()


@pytest.mark.unit
def test_resource_run_view_exposes_collector_status_as_operational_context() -> None:
    runtime = MetadataRuntime()
    runtime.file_many(
        (
            _resource_frame(),
            _collector_status(
                status_identifier="status-nvml-fallback",
                collector_id="resource_monitor.nvml_gpu_used",
                status="unavailable",
                degraded=True,
                reason="collector_unavailable",
                fallback_collector_id="resource_monitor.torch_gpu_used",
                rank=0,
            ),
            _collector_status(
                status_identifier="status-sampler-ok",
                collector_id="resource_monitor.sampler",
                status="ok",
                degraded=False,
                reason="sample_completed",
                rank=1,
            ),
            _collector_status(
                status_identifier="other-status",
                run_identifier="run-other",
                collector_id="resource_monitor.nvml_gpu_used",
                status="failed",
                degraded=True,
                reason="collector_failed",
                rank=0,
            ),
        )
    )
    view = ResourceRunView.from_snapshot(runtime.snapshot(), run_identifier="run-1")

    assert [record.identity.identifier for record in view.collector_statuses()] == [
        "status-nvml-fallback",
        "status-sampler-ok",
    ]
    assert [record.identity.identifier for record in view.degraded_collector_statuses()] == ["status-nvml-fallback"]
    assert [record.identity.identifier for record in view.collector_statuses_for_collector("resource_monitor.nvml_gpu_used")] == [
        "status-nvml-fallback"
    ]
    assert [record.identity.identifier for record in view.collector_statuses_for_rank(0)] == ["status-nvml-fallback"]
    assert {record.identity.identifier for record in view.observations()} == {
        "frame-1:gpu-used",
        "frame-1:cpu-rss",
    }
    assert "status-nvml-fallback" not in {record.identity.identifier for record in view.resource_records()}
    assert view.degraded_collector_statuses()[0].facts["fallback_collector_id"] == ("resource_monitor.torch_gpu_used")


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
    assert [record.identity.identifier for record in view.observations_for_device("cuda:0")] == ["frame-rank-0:gpu-used"]
    assert [record.identity.identifier for record in view.observations_for_device("cuda:1")] == ["frame-rank-1:gpu-used"]
    assert [record.identity.identifier for record in view.observation_frames_for_phase("training.epoch.1")] == ["frame-rank-1"]
    assert [record.identity.identifier for record in view.structural_facts_for_component("unet")] == ["struct-unet"]
    assert [record.identity.identifier for record in view.structural_facts_for_component("text_encoder")] == ["struct-text-encoder"]

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


def _collector_status(
    *,
    status_identifier: str,
    collector_id: str,
    status: str,
    degraded: bool,
    reason: str,
    run_identifier: str = "run-1",
    fallback_collector_id: str | None = None,
    rank: int | None = None,
) -> ResourceCollectorStatusFacts:
    return ResourceCollectorStatusFacts(
        status_identifier=status_identifier,
        run_identifier=run_identifier,
        collector_id=collector_id,
        status=status,
        degraded=degraded,
        reason=reason,
        rank=rank,
        fallback_collector_id=fallback_collector_id,
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


def _profile_for_run(
    *,
    profile_identifier: str,
    run_identifier: str,
    values: dict[str, object],
    profile_kind: str = "session_peaks",
    derivation_version: str = "v1",
    generated_at: float | None = None,
) -> ResourceProfileFacts:
    return ResourceProfileFacts(
        profile_identifier=profile_identifier,
        run_identifier=run_identifier,
        profile_kind=profile_kind,
        derivation_version=derivation_version,
        values=values,
        source_fact_references=(
            ResourceFactReference(
                entity_type="resource_observation",
                identifier="frame-1:gpu-used",
            ),
        ),
        generated_at=generated_at,
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
