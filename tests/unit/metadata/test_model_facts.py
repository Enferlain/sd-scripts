"""Typed model-family fact, identity, and validation contracts."""

from dataclasses import fields
import sqlite3

import pytest

from library.metadata import (
    build_model_component_identifier,
    build_model_realization_identifier,
    InMemoryMetadataBackend,
    MetadataGraphIndex,
    MetadataItemValidationError,
    MetadataRelationship,
    MetadataRuntime,
    ArtifactMetadataRecord,
    ModelArtifactFacts,
    ModelArtifactPresentation,
    ModelArtifactResolutionContext,
    ModelComponentMetadataRecord,
    ModelFamilyMetadataContribution,
    ModelFamilyContributionMetadataRecord,
    ModelFamilyMetadataField,
    ModelRealizationMetadataRecord,
    ModelRealizationFacts,
    RealizedModelComponentFacts,
    SQLiteMetadataStore,
    validate_metadata_item,
)


def _realization(run_identifier: str, *, family: str = "sdxl") -> ModelRealizationFacts:
    return ModelRealizationFacts.for_run(
        run_identifier=run_identifier,
        realization_key="target",
        family_identifier=family,
        model_version="base",
    )


def _component(
    realization: ModelRealizationFacts,
    *,
    key: str,
    order: int,
    roles: tuple[str, ...],
) -> RealizedModelComponentFacts:
    return RealizedModelComponentFacts.for_realization(
        run_identifier=realization.run_identifier,
        realization_identifier=realization.realization_identifier,
        component_key=key,
        public_name=key,
        declaration_order=order,
        roles=roles,
    )


@pytest.mark.unit
def test_qualified_model_identities_preserve_every_owning_scope() -> None:
    realization_identifier = build_model_realization_identifier("run/7", "target model")

    assert realization_identifier == "run/run%2F7/model/target%20model"
    assert build_model_component_identifier(realization_identifier, "text/encoder") == (
        "run/run%2F7/model/target%20model/component/text%2Fencoder"
    )


@pytest.mark.unit
def test_repeated_component_keys_are_distinct_across_runs_and_families() -> None:
    first_realization = _realization("run-1", family="sd")
    second_realization = _realization("run-2", family="future-family")
    first = _component(first_realization, key="denoiser", order=0, roles=("denoiser",))
    second = _component(second_realization, key="denoiser", order=0, roles=("denoiser",))

    assert first.component_key == second.component_key
    assert first.component_identifier != second.component_identifier
    validate_metadata_item(first)
    validate_metadata_item(second)


@pytest.mark.unit
def test_multiple_same_role_components_remain_separate_and_ordered() -> None:
    realization = _realization("run-3", family="future-family")
    components = (
        _component(realization, key="encoder_a", order=0, roles=("text_encoder",)),
        _component(realization, key="encoder_b", order=1, roles=("text_encoder",)),
    )
    runtime = MetadataRuntime()

    runtime.file_many((realization, *components))

    records = runtime.snapshot().records_for(entity_type="model_component")
    assert [record.facts["component_key"] for record in records] == ["encoder_a", "encoder_b"]
    assert [record.facts["declaration_order"] for record in records] == [0, 1]
    assert all(record.facts["roles"] == ["text_encoder"] for record in records)


@pytest.mark.unit
@pytest.mark.parametrize("duplicate_field", ["key", "order"])
def test_component_batch_rejects_duplicate_identity_or_order(duplicate_field: str) -> None:
    realization = _realization("run-4")
    first = _component(realization, key="first", order=0, roles=("text_encoder",))
    second = _component(
        realization,
        key="first" if duplicate_field == "key" else "second",
        order=1 if duplicate_field == "key" else 0,
        roles=("text_encoder",),
    )

    with pytest.raises(MetadataItemValidationError, match="must have unique"):
        MetadataRuntime().file_many((first, second))


@pytest.mark.unit
def test_model_realization_rejects_an_unqualified_or_mismatched_identity() -> None:
    facts = ModelRealizationFacts(
        run_identifier="run-5",
        realization_key="target",
        realization_identifier="target",
        family=_realization("unused").family,
    )

    with pytest.raises(MetadataItemValidationError, match="run-qualified identity constructor"):
        validate_metadata_item(facts)


@pytest.mark.unit
def test_realized_component_schema_has_no_live_module_field() -> None:
    assert "module" not in {field.name for field in fields(RealizedModelComponentFacts)}


@pytest.mark.unit
def test_artifact_context_is_family_neutral_and_artifact_facts_are_canonical() -> None:
    context = ModelArtifactResolutionContext(
        family_identifier="future-family",
        model_version="v1",
        artifact_identifier="adapter.safetensors",
        artifact_role="adapter",
        serialization_format="safetensors",
        resolution=(1024, 1024),
        created_at=1.0,
        presentation=ModelArtifactPresentation(title="Future Adapter"),
        prediction_type="flow",
    )
    facts = ModelArtifactFacts(
        artifact_identifier=context.artifact_identifier,
        family_identifier=context.family_identifier,
        artifact_role=context.artifact_role,
        artifact_format=context.serialization_format,
        architecture="future-v1/adapter",
        implementation="future-runtime",
        title=context.presentation.title or "Adapter",
        resolution="1024x1024",
        extension_fields={"future.sample_rate": "1"},
    )

    validate_metadata_item(facts)
    assert all(not field.startswith("modelspec.") for field in facts.extension_fields)


@pytest.mark.unit
def test_artifact_facts_reject_missing_required_claims_and_rendered_extensions() -> None:
    facts = ModelArtifactFacts(
        artifact_identifier="artifact",
        family_identifier="sd",
        artifact_role="adapter",
        artifact_format="safetensors",
        architecture="",
        implementation="diffusers",
        title="Adapter",
        resolution="512x512",
    )
    with pytest.raises(MetadataItemValidationError, match="architecture"):
        validate_metadata_item(facts)

    rendered_extension = ModelArtifactFacts(
        artifact_identifier="artifact",
        family_identifier="sd",
        artifact_role="adapter",
        artifact_format="safetensors",
        architecture="stable-diffusion-v1/lora",
        implementation="diffusers",
        title="Adapter",
        resolution="512x512",
        extension_fields={"modelspec.custom": "value"},
    )
    with pytest.raises(MetadataItemValidationError, match="canonical names"):
        validate_metadata_item(rendered_extension)


@pytest.mark.unit
def test_family_local_facts_require_an_explicit_versioned_canonical_contribution() -> None:
    realization = _realization("run-family", family="future-family")
    contribution = ModelFamilyMetadataContribution.for_realization(
        run_identifier=realization.run_identifier,
        realization_identifier=realization.realization_identifier,
        contribution_namespace="future.runtime",
        contribution_version="1",
        fields=(ModelFamilyMetadataField(name="attention_mask_enabled", value=True),),
    )

    validate_metadata_item(contribution)
    result = MetadataRuntime().file(contribution)

    assert isinstance(result.records[0], ModelFamilyContributionMetadataRecord)
    assert result.records[0].facts["attention_mask_enabled"] is True
    assert result.records[0].facts["contribution_namespace"] == "future.runtime"
    assert all(not key.startswith(("modelspec.", "ss_")) for key in result.records[0].facts)


@pytest.mark.unit
def test_family_local_facts_reject_rendered_compatibility_keys() -> None:
    realization = _realization("run-family-invalid", family="future-family")
    contribution = ModelFamilyMetadataContribution.for_realization(
        run_identifier=realization.run_identifier,
        realization_identifier=realization.realization_identifier,
        contribution_namespace="future.runtime",
        contribution_version="1",
        fields=(ModelFamilyMetadataField(name="ss_future_flag", value=True),),
    )

    with pytest.raises(MetadataItemValidationError, match="canonical names"):
        validate_metadata_item(contribution)


@pytest.mark.unit
def test_runtime_emits_qualified_realization_component_and_artifact_relationships() -> None:
    realization = _realization("run-9")
    component = _component(realization, key="denoiser", order=0, roles=("denoiser",))
    artifact = ModelArtifactFacts(
        artifact_identifier="adapter.safetensors",
        family_identifier="sdxl",
        artifact_role="adapter",
        artifact_format="safetensors",
        architecture="stable-diffusion-xl-v1-base/lora",
        implementation="https://github.com/Stability-AI/generative-models",
        title="Adapter",
        resolution="1024x1024",
        realization_identifier=realization.realization_identifier,
    )
    runtime = MetadataRuntime()

    results = runtime.file_many((realization, component, artifact))
    snapshot = runtime.snapshot()
    graph = MetadataGraphIndex.from_snapshot(snapshot)

    assert isinstance(results[0].records[0], ModelRealizationMetadataRecord)
    assert isinstance(results[1].records[0], ModelComponentMetadataRecord)
    assert isinstance(results[2].records[0], ArtifactMetadataRecord)
    realization_record = graph.record_for(
        entity_type="model_realization",
        identifier=realization.realization_identifier,
    )
    component_record = graph.record_for(
        entity_type="model_component",
        identifier=component.component_identifier,
    )
    artifact_record = graph.record_for(entity_type="model_artifact", identifier=artifact.artifact_identifier)
    assert realization_record is not None
    assert component_record is not None
    assert artifact_record is not None
    assert graph.edges_from(
        component_record.identity,
        relationship=MetadataRelationship.CONTAINED_IN,
        target_identifier=realization.realization_identifier,
    )
    assert graph.edges_from(
        artifact_record.identity,
        relationship=MetadataRelationship.DERIVED_FROM,
        target_identifier=realization.realization_identifier,
    )


@pytest.mark.unit
def test_model_records_and_relationships_round_trip_through_sqlite() -> None:
    connection = sqlite3.connect(":memory:")
    runtime = MetadataRuntime(
        backend=InMemoryMetadataBackend(
            store=SQLiteMetadataStore(connection),
        )
    )
    first_realization = _realization("run-sqlite-1", family="future-family")
    second_realization = _realization("run-sqlite-2", family="sdxl")
    first_component = _component(
        first_realization,
        key="denoiser",
        order=0,
        roles=("denoiser",),
    )
    second_component = _component(
        second_realization,
        key="denoiser",
        order=0,
        roles=("denoiser",),
    )
    contribution = ModelFamilyMetadataContribution.for_realization(
        run_identifier=first_realization.run_identifier,
        realization_identifier=first_realization.realization_identifier,
        contribution_namespace="future.runtime",
        contribution_version="1",
        fields=(ModelFamilyMetadataField(name="attention_mask_enabled", value=True),),
    )
    artifact = ModelArtifactFacts(
        artifact_identifier="future-adapter.safetensors",
        family_identifier="future-family",
        artifact_role="adapter",
        artifact_format="safetensors",
        architecture="future-v1/adapter",
        implementation="future-runtime",
        title="Future Adapter",
        resolution="1024x1024",
        realization_identifier=first_realization.realization_identifier,
    )

    runtime.file_many(
        (
            first_realization,
            first_component,
            contribution,
            artifact,
            second_realization,
            second_component,
        )
    )
    snapshot = runtime.snapshot()
    graph = MetadataGraphIndex.from_snapshot(snapshot)

    first_realization_record = graph.record_for(
        entity_type="model_realization",
        identifier=first_realization.realization_identifier,
    )
    second_realization_record = graph.record_for(
        entity_type="model_realization",
        identifier=second_realization.realization_identifier,
    )
    first_component_record = graph.record_for(
        entity_type="model_component",
        identifier=first_component.component_identifier,
    )
    second_component_record = graph.record_for(
        entity_type="model_component",
        identifier=second_component.component_identifier,
    )
    contribution_record = graph.record_for(
        entity_type="model_family_facts",
        identifier=contribution.contribution_identifier,
    )
    artifact_record = graph.record_for(
        entity_type="model_artifact",
        identifier=artifact.artifact_identifier,
    )

    assert isinstance(first_realization_record, ModelRealizationMetadataRecord)
    assert isinstance(second_realization_record, ModelRealizationMetadataRecord)
    assert isinstance(first_component_record, ModelComponentMetadataRecord)
    assert isinstance(second_component_record, ModelComponentMetadataRecord)
    assert isinstance(contribution_record, ModelFamilyContributionMetadataRecord)
    assert isinstance(artifact_record, ArtifactMetadataRecord)
    assert first_component_record.facts["component_key"] == "denoiser"
    assert second_component_record.facts["component_key"] == "denoiser"
    assert first_component_record.identity != second_component_record.identity
    assert first_component_record.facts["roles"] == ["denoiser"]

    assert graph.edges_from(
        first_realization_record.identity,
        relationship=MetadataRelationship.REALIZED_IN,
        target_identifier=first_realization.run_identifier,
    )
    assert graph.edges_from(
        second_realization_record.identity,
        relationship=MetadataRelationship.REALIZED_IN,
        target_identifier=second_realization.run_identifier,
    )
    assert graph.edges_from(
        first_component_record.identity,
        relationship=MetadataRelationship.CONTAINED_IN,
        target_identifier=first_realization.realization_identifier,
    )
    assert graph.edges_from(
        second_component_record.identity,
        relationship=MetadataRelationship.CONTAINED_IN,
        target_identifier=second_realization.realization_identifier,
    )
    assert graph.edges_from(
        contribution_record.identity,
        relationship=MetadataRelationship.DESCRIBES,
        target_identifier=first_realization.realization_identifier,
    )
    assert graph.edges_from(
        artifact_record.identity,
        relationship=MetadataRelationship.DERIVED_FROM,
        target_identifier=first_realization.realization_identifier,
    )


@pytest.mark.unit
def test_model_artifact_without_realization_omits_provenance_edge() -> None:
    artifact = ModelArtifactFacts(
        artifact_identifier="unlinked-adapter.safetensors",
        family_identifier="future-family",
        artifact_role="adapter",
        artifact_format="safetensors",
        architecture="future-v1/adapter",
        implementation="future-runtime",
        title="Unlinked Adapter",
        resolution="1024x1024",
    )

    result = MetadataRuntime().file(artifact)

    assert result.edges == ()
