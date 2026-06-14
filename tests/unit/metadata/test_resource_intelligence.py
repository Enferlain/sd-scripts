"""Unit tests for resource-intelligence metadata facts."""

from __future__ import annotations

import pytest

from library.metadata import (
    METADATA_PAYLOAD_VERSION,
    MetadataEntityType,
    MetadataItemValidationError,
    MetadataRelationship,
    MetadataRuntime,
    ResourceAccountingFacts,
    ResourceAccountingGapFacts,
    ResourceFactReference,
    ResourceObservationFrameFacts,
    ResourceObservationFacts,
    ResourceObservationMeasurementFacts,
    ResourceProfileFacts,
    StructuralResourceFacts,
)
from library.metadata.emitters import (
    build_resource_accounting_gap_metadata,
    build_resource_accounting_metadata,
    build_resource_observation_frame_metadata,
    build_resource_observation_metadata,
    build_resource_profile_metadata,
    build_structural_resource_metadata,
)


@pytest.mark.unit
def test_resource_intelligence_facts_feed_metadata_emitters() -> None:
    observation = ResourceObservationFacts(
        observation_identifier="obs-1",
        run_identifier="run-1",
        resource_kind="gpu_memory",
        measurement_kind="allocated",
        value=128.0,
        unit="mb",
        source="torch.cuda.memory_allocated",
        phase="training.epoch.0",
        device_identifier="cuda:0",
        metadata={"device_scope": "local"},
    )
    structural = StructuralResourceFacts(
        structural_identifier="struct-1",
        run_identifier="run-1",
        owner_type="model_component",
        owner_identifier="denoiser",
        resource_kind="parameter_memory",
        quantity=256.0,
        unit="mb",
        basis="parameter_bytes",
        source="startup_component_memory",
    )
    source_ref = ResourceFactReference(
        entity_type="resource_observation",
        identifier="obs-1",
    )
    profile = ResourceProfileFacts(
        profile_identifier="profile-1",
        run_identifier="run-1",
        profile_kind="session_peaks",
        derivation_version="v1",
        values={"gpu_memory_peak_mb": 128.0},
        source_fact_references=(source_ref,),
    )
    accounting = ResourceAccountingFacts(
        accounting_identifier="acct-1",
        run_identifier="run-1",
        resource_kind="parameter_memory",
        quantity=256.0,
        unit="mb",
        owner_type="model_component",
        owner_identifier="denoiser",
        basis="structural",
        derivation_method="parameter_bytes",
        derivation_version="v1",
        source_fact_references=(
            ResourceFactReference(
                entity_type="resource_structural_fact",
                identifier="struct-1",
            ),
        ),
    )
    gap = ResourceAccountingGapFacts(
        gap_identifier="gap-1",
        run_identifier="run-1",
        resource_kind="gpu_memory",
        quantity=64.0,
        unit="mb",
        basis="observed_minus_accounted",
        derivation_method="phase_summary",
        derivation_version="v1",
        source_fact_references=(source_ref,),
        reason="no structural owner matched observed peak",
    )

    observation_record = build_resource_observation_metadata(observation).records[0]
    observation_edges = build_resource_observation_metadata(observation).edges
    structural_record = build_structural_resource_metadata(structural).records[0]
    profile_result = build_resource_profile_metadata(profile)
    accounting_result = build_resource_accounting_metadata(accounting)
    gap_result = build_resource_accounting_gap_metadata(gap)

    assert observation_record.identity.entity_type == "resource_observation"
    assert observation_record.facts["semantic_class"] == "observation"
    assert observation_record.facts["measurement_kind"] == "allocated"
    assert observation_record.facts["metadata"] == {"device_scope": "local"}
    assert {edge.relationship for edge in observation_edges} == {
        MetadataRelationship.OBSERVED_DURING,
        MetadataRelationship.OBSERVED_ON,
    }
    assert structural_record.identity.entity_type == "resource_structural_fact"
    assert structural_record.facts["semantic_class"] == "structural"
    assert structural_record.facts["owner_identifier"] == "denoiser"
    assert profile_result.records[0].facts["semantic_class"] == "profile"
    assert profile_result.records[0].facts["derivation_version"] == "v1"
    assert profile_result.edges[1].relationship == "derived_from"
    assert accounting_result.records[0].facts["semantic_class"] == "accounting"
    assert accounting_result.records[0].facts["basis"] == "structural"
    assert gap_result.records[0].facts["semantic_class"] == "accounting_gap"
    assert gap_result.records[0].facts["reason"] == "no structural owner matched observed peak"
    assert gap_result.records[0].schema_version == METADATA_PAYLOAD_VERSION


@pytest.mark.unit
def test_resource_observation_frame_preserves_shared_context_and_measurement_identity() -> None:
    frame = ResourceObservationFrameFacts(
        frame_identifier="frame-1",
        run_identifier="run-1",
        event_name="sample",
        ts=123.0,
        collector_id="sampled_gpu",
        phase="training.epoch.0",
        rank=0,
        process_identifier="process-0",
        measurements=(
            ResourceObservationMeasurementFacts(
                measurement_identifier="frame-1:gpu-used",
                resource_kind="gpu_memory",
                measurement_kind="used",
                value=1024.0,
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

    result = build_resource_observation_frame_metadata(frame)
    frame_record, gpu_record, cpu_record = result.records

    assert frame_record.identity.entity_type == "resource_observation_frame"
    assert frame_record.facts["container_kind"] == "observation_frame"
    assert frame_record.facts["phase"] == "training.epoch.0"
    assert frame_record.facts["measurement_identifiers"] == ["frame-1:gpu-used", "frame-1:cpu-rss"]
    assert gpu_record.identity.identifier == "frame-1:gpu-used"
    assert gpu_record.facts["frame_identifier"] == "frame-1"
    assert "phase" not in gpu_record.facts
    assert cpu_record.identity.identifier == "frame-1:cpu-rss"
    assert {edge.relationship for edge in result.edges} == {
        MetadataRelationship.CONTAINED_IN,
        MetadataRelationship.OBSERVED_DURING,
        MetadataRelationship.OBSERVED_IN,
        MetadataRelationship.OBSERVED_ON,
        MetadataRelationship.PRODUCED_BY,
    }


@pytest.mark.unit
def test_resource_intelligence_facts_file_through_metadata_runtime() -> None:
    runtime = MetadataRuntime()
    observation = ResourceObservationFacts(
        observation_identifier="obs-1",
        run_identifier="run-1",
        resource_kind="cpu_memory",
        measurement_kind="rss",
        value=512.0,
        unit="mb",
        source="psutil.Process.memory_info",
    )
    structural = StructuralResourceFacts(
        structural_identifier="struct-1",
        run_identifier="run-1",
        owner_type="model_component",
        owner_identifier="text_encoder1",
        resource_kind="parameter_memory",
        quantity=42.0,
        unit="mb",
        basis="parameter_bytes",
        source="startup_component_memory",
    )
    profile = ResourceProfileFacts(
        profile_identifier="profile-1",
        run_identifier="run-1",
        profile_kind="session_summary",
        derivation_version="v1",
        values={"cpu_rss_peak_mb": 512.0},
        source_fact_references=(
            ResourceFactReference(
                entity_type="resource_observation",
                identifier="obs-1",
            ),
        ),
    )

    runtime.file(observation)
    runtime.file(structural)
    runtime.file(profile)

    snapshot = runtime.snapshot()

    assert snapshot.record_for(entity_type="resource_observation", identifier="obs-1") is not None
    assert snapshot.record_for(entity_type="resource_structural_fact", identifier="struct-1") is not None
    assert snapshot.record_for(entity_type="resource_profile", identifier="profile-1") is not None
    assert {edge.relationship for edge in snapshot.edges} == {
        MetadataRelationship.DESCRIBES,
        MetadataRelationship.OBSERVED_DURING,
        MetadataRelationship.OWNED_BY,
        MetadataRelationship.PROFILES,
        "derived_from",
    }


@pytest.mark.unit
def test_metadata_graph_relationships_cover_runtime_structural_and_evidence_scopes() -> None:
    frame = ResourceObservationFrameFacts(
        frame_identifier="frame-relationships",
        run_identifier="run-relationships",
        event_name="step_sample",
        collector_id="resource_monitor.sampled",
        collection_policy="sampled",
        phase="training.epoch.0",
        global_step=12,
        epoch=3,
        rank=1,
        world_size=2,
        host_identifier="host-a",
        process_identifier="pid-123",
        measurements=(
            ResourceObservationMeasurementFacts(
                measurement_identifier="frame-relationships:cuda-0-used",
                resource_kind="gpu_memory",
                measurement_kind="used",
                value=2048.0,
                unit="MiB",
                source="nvml",
                scope_type="device",
                device_identifier="cuda:0",
            ),
        ),
    )
    structural = StructuralResourceFacts(
        structural_identifier="struct-relationships",
        run_identifier="run-relationships",
        owner_type="model_component",
        owner_identifier="denoiser",
        resource_kind="parameter_memory",
        quantity=1024.0,
        unit="MiB",
        basis="parameter_bytes",
        source="startup_component_memory",
        component_key="unet",
        group_identifier="trainable-model",
    )
    source_ref = ResourceFactReference(
        entity_type="resource_observation",
        identifier="frame-relationships:cuda-0-used",
        relationship="derived_from",
    )
    profile = ResourceProfileFacts(
        profile_identifier="profile-relationships",
        run_identifier="run-relationships",
        profile_kind="session_peaks",
        derivation_version="v1",
        values={"gpu_used_peak_mib": 2048.0},
        source_fact_references=(source_ref,),
    )
    accounting = ResourceAccountingFacts(
        accounting_identifier="accounting-relationships",
        run_identifier="run-relationships",
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
                identifier="struct-relationships",
                relationship="supported_by",
            ),
        ),
    )
    gap = ResourceAccountingGapFacts(
        gap_identifier="gap-relationships",
        run_identifier="run-relationships",
        resource_kind="gpu_memory",
        quantity=512.0,
        unit="MiB",
        basis="observed_minus_accounted",
        derivation_method="session_peak",
        derivation_version="v1",
        source_fact_references=(source_ref,),
    )

    frame_edges = build_resource_observation_frame_metadata(frame).edges
    structural_edges = build_structural_resource_metadata(structural).edges
    profile_edges = build_resource_profile_metadata(profile).edges
    accounting_edges = build_resource_accounting_metadata(accounting).edges
    gap_edges = build_resource_accounting_gap_metadata(gap).edges

    frame_targets = {(edge.relationship, edge.target.entity_type, edge.target.identifier) for edge in frame_edges}
    assert (MetadataRelationship.OBSERVED_DURING, MetadataEntityType.RUN, "run-relationships") in frame_targets
    assert (MetadataRelationship.OBSERVED_ON, MetadataEntityType.HOST, "host-a") in frame_targets
    assert (MetadataRelationship.OBSERVED_IN, MetadataEntityType.PROCESS, "pid-123") in frame_targets
    assert (MetadataRelationship.OBSERVED_IN, MetadataEntityType.RANK, "run-relationships:rank:1") in frame_targets
    assert (
        MetadataRelationship.OBSERVED_DURING,
        MetadataEntityType.PHASE,
        "run-relationships:phase:training.epoch.0",
    ) in frame_targets
    assert (MetadataRelationship.OBSERVED_DURING, MetadataEntityType.STEP, "run-relationships:step:12") in frame_targets
    assert (
        MetadataRelationship.PRODUCED_BY,
        MetadataEntityType.COLLECTOR,
        "resource_monitor.sampled",
    ) in frame_targets
    assert (
        MetadataRelationship.OBSERVED_DURING,
        MetadataEntityType.EVENT,
        "run-relationships:event:step_sample",
    ) in frame_targets
    assert (
        MetadataRelationship.OBSERVED_ON,
        MetadataEntityType.DEVICE,
        "cuda:0",
    ) in frame_targets
    rank_edge = next(edge for edge in frame_edges if edge.target.entity_type == MetadataEntityType.RANK)
    step_edge = next(edge for edge in frame_edges if edge.target.entity_type == MetadataEntityType.STEP)
    assert rank_edge.facts == {"rank": 1, "world_size": 2}
    assert step_edge.facts == {"global_step": 12, "epoch": 3}

    structural_targets = {(edge.relationship, edge.target.entity_type, edge.target.identifier) for edge in structural_edges}
    assert (MetadataRelationship.DESCRIBES, MetadataEntityType.RUN, "run-relationships") in structural_targets
    assert (MetadataRelationship.OWNED_BY, "model_component", "denoiser") in structural_targets
    assert (MetadataRelationship.DESCRIBES, MetadataEntityType.COMPONENT, "unet") in structural_targets
    assert (MetadataRelationship.DESCRIBES, MetadataEntityType.GROUP, "trainable-model") in structural_targets

    assert {edge.relationship for edge in profile_edges} == {MetadataRelationship.PROFILES, "derived_from"}
    assert {edge.relationship for edge in accounting_edges} == {
        MetadataRelationship.ACCOUNTS_FOR,
        "supported_by",
    }
    assert {edge.relationship for edge in gap_edges} == {MetadataRelationship.GAP_FOR, "derived_from"}


@pytest.mark.unit
def test_metadata_graph_relationships_omit_unavailable_runtime_identities() -> None:
    result = build_resource_observation_frame_metadata(
        ResourceObservationFrameFacts(
            frame_identifier="frame-minimal",
            run_identifier="run-minimal",
            event_name="session_start",
            measurements=(
                ResourceObservationMeasurementFacts(
                    measurement_identifier="frame-minimal:cpu-rss",
                    resource_kind="cpu_memory",
                    measurement_kind="rss",
                    value=128.0,
                    unit="MiB",
                    source="psutil",
                    scope_type="process",
                ),
            ),
        )
    )

    assert {edge.relationship for edge in result.edges} == {
        MetadataRelationship.CONTAINED_IN,
        MetadataRelationship.OBSERVED_DURING,
    }


@pytest.mark.unit
def test_resource_profiles_and_accounting_require_source_references() -> None:
    runtime = MetadataRuntime()

    with pytest.raises(MetadataItemValidationError, match="source_fact_reference"):
        runtime.file(
            ResourceProfileFacts(
                profile_identifier="profile-1",
                run_identifier="run-1",
                profile_kind="session_summary",
                derivation_version="v1",
                values={"cpu_rss_peak_mb": 512.0},
                source_fact_references=(),
            )
        )

    with pytest.raises(MetadataItemValidationError, match="source_fact_reference"):
        runtime.file(
            ResourceAccountingFacts(
                accounting_identifier="acct-1",
                run_identifier="run-1",
                resource_kind="gpu_memory",
                quantity=256.0,
                unit="mb",
                owner_type="model_component",
                owner_identifier="denoiser",
                basis="structural",
                derivation_method="parameter_bytes",
                derivation_version="v1",
                source_fact_references=(),
            )
        )

    with pytest.raises(MetadataItemValidationError, match="source_fact_reference"):
        runtime.file(
            ResourceAccountingGapFacts(
                gap_identifier="gap-1",
                run_identifier="run-1",
                resource_kind="gpu_memory",
                quantity=64.0,
                unit="mb",
                basis="observed_minus_accounted",
                derivation_method="phase_summary",
                derivation_version="v1",
                source_fact_references=(),
            )
        )


@pytest.mark.unit
def test_resource_observation_frame_requires_unique_measurements() -> None:
    runtime = MetadataRuntime()
    measurement = ResourceObservationMeasurementFacts(
        measurement_identifier="duplicate",
        resource_kind="gpu_memory",
        measurement_kind="used",
        value=1.0,
        unit="MiB",
        source="nvml",
    )

    with pytest.raises(MetadataItemValidationError, match="unique"):
        runtime.file(
            ResourceObservationFrameFacts(
                frame_identifier="frame-1",
                run_identifier="run-1",
                event_name="sample",
                measurements=(measurement, measurement),
            )
        )
