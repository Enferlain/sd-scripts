"""Unit tests for resource-intelligence metadata facts."""

from __future__ import annotations

import pytest

from library.metadata import (
    METADATA_PAYLOAD_VERSION,
    MetadataItemValidationError,
    MetadataRuntime,
    ResourceAccountingFacts,
    ResourceAccountingGapFacts,
    ResourceFactReference,
    ResourceObservationFacts,
    ResourceProfileFacts,
    StructuralResourceFacts,
)
from library.metadata.emitters import (
    build_resource_accounting_gap_metadata,
    build_resource_accounting_metadata,
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
        "observed_during",
        "observed_on_device",
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
        "observed_during",
        "describes_run_resource",
        "profiles_run",
        "derived_from",
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
