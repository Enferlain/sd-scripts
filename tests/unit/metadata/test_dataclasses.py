"""Unit tests for typed metadata fact dataclasses."""

import pytest

from library.constants import SS_METADATA_KEY_ADAPTER_MODULE as LEGACY_ADAPTER_MODULE_KEY
from library.metadata import CheckpointArtifactFacts as TopLevelCheckpointArtifactFacts
from library.metadata import LoggedArtifactFacts as TopLevelLoggedArtifactFacts
from library.metadata import METADATA_PAYLOAD_VERSION
from library.metadata import ModelSpecFacts as TopLevelModelSpecFacts
from library.metadata import OptimizerRuntimeFacts as TopLevelOptimizerRuntimeFacts
from library.metadata import ResourceMonitorFacts as TopLevelResourceMonitorFacts
from library.metadata import RunLifecycleFacts as TopLevelRunLifecycleFacts
from library.metadata import RunMetadataFacts as TopLevelRunMetadataFacts
from library.metadata import RunReportFacts as TopLevelRunReportFacts
from library.metadata import SchedulerRuntimeFacts as TopLevelSchedulerRuntimeFacts
from library.metadata.dataclasses import (
    AnalyticsSnapshotFacts,
    CheckpointArtifactFacts,
    LoggedArtifactFacts,
    ModelSpecFacts,
    OptimizerRuntimeFacts,
    ResourceMonitorFacts,
    RunLifecycleFacts,
    RunMetadataFacts,
    RunReportFacts,
    SchedulerRuntimeFacts,
)
from library.metadata.emitters import (
    build_analytics_snapshot_metadata,
    build_checkpoint_artifact_metadata,
    build_logged_artifact_metadata,
    build_model_spec_metadata,
    build_resource_monitor_metadata,
    build_run_lifecycle_metadata,
    build_run_report_metadata,
    build_training_run_metadata,
)
from library.metadata.keys import (
    SS_METADATA_KEY_ADAPTER_MODULE,
)


@pytest.mark.unit
def test_metadata_keys_are_owned_by_metadata_package_and_reexported_from_constants() -> None:
    assert SS_METADATA_KEY_ADAPTER_MODULE == "ss_adapter_module"
    assert SS_METADATA_KEY_ADAPTER_MODULE == LEGACY_ADAPTER_MODULE_KEY


@pytest.mark.unit
def test_metadata_package_reexports_shared_fact_dataclasses() -> None:
    assert TopLevelCheckpointArtifactFacts is CheckpointArtifactFacts
    assert TopLevelLoggedArtifactFacts is LoggedArtifactFacts
    assert TopLevelModelSpecFacts is ModelSpecFacts
    assert TopLevelOptimizerRuntimeFacts is OptimizerRuntimeFacts
    assert TopLevelResourceMonitorFacts is ResourceMonitorFacts
    assert TopLevelRunLifecycleFacts is RunLifecycleFacts
    assert TopLevelRunMetadataFacts is RunMetadataFacts
    assert TopLevelRunReportFacts is RunReportFacts
    assert TopLevelSchedulerRuntimeFacts is SchedulerRuntimeFacts


@pytest.mark.unit
def test_run_facts_feed_training_run_metadata_emitter() -> None:
    facts = RunMetadataFacts.from_metadata(
        {"session_id": "123", "seed": "42"},
        run_identifier="123",
    )

    record = build_training_run_metadata(facts).records[0]

    assert record.identity.entity_type == "run"
    assert record.identity.identifier == "123"
    assert record.producer == "training.run"
    assert record.facts["seed"] == "42"


@pytest.mark.unit
def test_modelspec_facts_feed_model_spec_metadata_emitter() -> None:
    facts = ModelSpecFacts.from_modelspec_metadata(
        {
            "modelspec.title": "LoRA",
            "modelspec.architecture": "stable-diffusion-xl-v1-base/lora",
            "modelspec.implementation": "sgm",
            "modelspec.prediction_type": "epsilon",
        }
    )

    record = build_model_spec_metadata(facts).records[0]

    assert record.facts["modelspec.title"] == "LoRA"
    assert record.facts["architecture"] == "stable-diffusion-xl-v1-base/lora"
    assert record.facts["implementation"] == "sgm"
    assert record.facts["prediction_type"] == "epsilon"


@pytest.mark.unit
def test_checkpoint_artifact_facts_feed_checkpoint_metadata_emitter() -> None:
    facts = CheckpointArtifactFacts.for_checkpoint(
        artifact_identifier="adapter.safetensors",
        no_metadata=False,
        step=12,
        epoch=3,
    )

    record = build_checkpoint_artifact_metadata(facts).records[0]

    assert record.identity.entity_type == "artifact"
    assert record.identity.identifier == "adapter.safetensors"
    assert record.facts["kind"] == "checkpoint"
    assert record.facts["format"] == "safetensors"
    assert record.facts["metadata_policy"] == "full"
    assert record.facts["step"] == 12
    assert record.facts["epoch"] == 3


@pytest.mark.unit
def test_observability_facts_feed_metadata_emitters() -> None:
    artifact = LoggedArtifactFacts(
        path="/tmp/report.md",
        kind="benchmark_report",
        metadata={"format": "markdown"},
    )
    lifecycle = RunLifecycleFacts(
        run_identifier="run-1",
        event_type="run_started",
        run_name="training",
        status="running",
        global_step=1,
    )
    resource = ResourceMonitorFacts(
        run_identifier="run-1",
        event_name="session_start",
        gpu_used_mb=128.0,
        gpu_used_by_device_mb={"0": 80.0, "1": 48.0},
        cpu_rss_mb=256.0,
        cpu_vms_mb=512.0,
    )
    report = RunReportFacts(
        report_identifier="/tmp/report.md",
        run_identifier="run-1",
        status="succeeded",
        generated_at=123.0,
        output_name="report",
        resource_event_count=2,
    )
    snapshot = AnalyticsSnapshotFacts(
        snapshot_identifier="snapshot-1",
        snapshot_kind="debug_snapshot",
        source="tests",
        payload={"rows": 4},
    )

    artifact_event = build_logged_artifact_metadata(
        artifact
    ).events[0]
    lifecycle_event = build_run_lifecycle_metadata(lifecycle).events[0]
    resource_event = build_resource_monitor_metadata(resource).events[0]
    report_record = build_run_report_metadata(report).records[0]
    snapshot_record = build_analytics_snapshot_metadata(snapshot).records[0]

    assert artifact_event.identity.entity_type == "artifact"
    assert artifact_event.facts["kind"] == "benchmark_report"
    assert lifecycle_event.identity.entity_type == "run"
    assert lifecycle_event.facts["status"] == "running"
    assert resource_event.facts["gpu_used_mb"] == 128.0
    assert resource_event.facts["gpu_used_by_device_mb"] == {"0": 80.0, "1": 48.0}
    assert resource_event.facts["cpu_vms_mb"] == 512.0
    assert report_record.identity.entity_type == "artifact"
    assert report_record.facts["kind"] == "benchmark_report"
    assert snapshot_record.facts["payload"] == {"rows": 4}
    assert artifact_event.schema_version == METADATA_PAYLOAD_VERSION
    assert artifact_event.identity.schema_version == METADATA_PAYLOAD_VERSION
    assert lifecycle_event.schema_version == METADATA_PAYLOAD_VERSION
    assert report_record.schema_version == METADATA_PAYLOAD_VERSION

    artifact_only_result = build_logged_artifact_metadata(artifact)

    assert len(artifact_only_result.events) == 1
    assert artifact_only_result.events[0].event_type == "artifact_registered"


@pytest.mark.unit
def test_optimizer_runtime_facts_are_central_catalog_entries() -> None:
    scheduler = SchedulerRuntimeFacts(mode="embedded", target="optimizer")
    optimizer = OptimizerRuntimeFacts(supports_train_eval_toggle=True)

    assert scheduler.mode == "embedded"
    assert scheduler.target == "optimizer"
    assert optimizer.supports_train_eval_toggle is True
