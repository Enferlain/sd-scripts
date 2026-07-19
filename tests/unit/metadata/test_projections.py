"""Unit tests for metadata export projections."""

from dataclasses import replace

import pytest

from library.metadata import (
    ArtifactMetadataRecord,
    InMemoryMetadataBackend,
    KuroMetadataProjection,
    MetadataEdge,
    MetadataEntityType,
    MetadataIdentity,
    MetadataProjectionScopeError,
    MetadataProviderResult,
    MetadataRelationship,
    MetadataRequiredFact,
    MetadataRuntime,
    MetadataValidationError,
    ModelArtifactFacts,
    ModelFamilyMetadataContribution,
    ModelFamilyMetadataField,
    ModelRealizationFacts,
    ModelSpecCompatibilityProjection,
    RealizedModelComponentFacts,
    RunMetadataRecord,
    SafetensorsMetadataProjection,
    SsCompatibilityProjection,
)


@pytest.mark.unit
def test_kuro_projection_emits_schema_and_repo_owned_keys() -> None:
    backend = InMemoryMetadataBackend()
    backend.ingest(
        MetadataProviderResult.from_sequences(
            provider_id="tests.provider",
            records=[
                RunMetadataRecord(
                    identity=MetadataIdentity(entity_type="run", identifier="run-1", label="test run"),
                    producer="tests.training",
                    facts={
                        "seed": 42,
                        "ss_seed": 42,
                        "modelspec.title": "ignored by kuro",
                    },
                )
            ],
        )
    )

    result = KuroMetadataProjection().project(backend.snapshot())

    assert result.metadata["kuro.schema_version"] == "1"
    assert result.metadata["kuro.run.id"] == "run-1"
    assert result.metadata["kuro.run.label"] == "test run"
    assert result.metadata["kuro.run.seed"] == 42
    assert "ss_seed" not in result.metadata
    assert "modelspec.title" not in result.metadata


@pytest.mark.unit
def test_ss_compatibility_projection_preserves_legacy_concern_fields() -> None:
    backend = InMemoryMetadataBackend()
    backend.ingest(
        MetadataProviderResult.from_sequences(
            provider_id="tests.provider",
            records=[
                ArtifactMetadataRecord(
                    identity=MetadataIdentity(entity_type="artifact", identifier="checkpoint-1"),
                    producer="tests.checkpoint",
                    facts={
                        "ss_seed": 42,
                        "ss_steps": 10,
                        "modelspec.title": "Checkpoint",
                        "format": "safetensors",
                    },
                )
            ],
        )
    )
    snapshot = backend.snapshot()

    ss_result = SsCompatibilityProjection(minimum_keys=frozenset({"ss_seed"})).project(snapshot)

    assert ss_result.metadata == {"ss_seed": 42}


@pytest.mark.unit
def test_modelspec_projection_ignores_pre_rendered_legacy_facts() -> None:
    backend = InMemoryMetadataBackend()
    backend.ingest(
        MetadataProviderResult.from_sequences(
            provider_id="tests.legacy",
            records=[
                ArtifactMetadataRecord(
                    identity=MetadataIdentity(entity_type="artifact", identifier="checkpoint-1"),
                    producer="tests.legacy",
                    facts={"modelspec.title": "Legacy Checkpoint"},
                )
            ],
        )
    )

    result = ModelSpecCompatibilityProjection().project(backend.snapshot())

    assert result.metadata == {}


@pytest.mark.unit
def test_modelspec_projection_maps_canonical_artifact_facts_and_applies_extensions_last() -> None:
    runtime = MetadataRuntime()
    runtime.file(
        ModelArtifactFacts(
            artifact_identifier="adapter.safetensors",
            family_identifier="future",
            artifact_role="adapter",
            artifact_format="safetensors",
            architecture="future-v1/lora",
            implementation="future-runtime",
            title="Configured Title",
            resolution="1024x768",
            author="Metadata Tests",
            prediction_type=None,
            extension_fields={
                "custom_field": "custom-value",
                "sai_model_spec": "user-version",
                "title": "Extension Title",
            },
        )
    )

    result = ModelSpecCompatibilityProjection(artifact_identifier="adapter.safetensors").project(runtime.snapshot())

    assert result.metadata == {
        "modelspec.architecture": "future-v1/lora",
        "modelspec.implementation": "future-runtime",
        "modelspec.title": "Extension Title",
        "modelspec.resolution": "1024x768",
        "modelspec.sai_model_spec": "user-version",
        "modelspec.author": "Metadata Tests",
        "modelspec.custom_field": "custom-value",
    }
    assert "modelspec.prediction_type" not in result.metadata


@pytest.mark.unit
def test_modelspec_projection_rejects_missing_required_canonical_claim() -> None:
    backend = InMemoryMetadataBackend()
    backend.ingest(
        MetadataProviderResult.from_sequences(
            provider_id="tests.model",
            records=[
                ArtifactMetadataRecord(
                    identity=MetadataIdentity(
                        entity_type=MetadataEntityType.MODEL_ARTIFACT,
                        identifier="invalid.safetensors",
                    ),
                    producer="tests.model",
                    facts={
                        "implementation": "future-runtime",
                        "title": "Invalid",
                        "resolution": "512x512",
                    },
                )
            ],
        )
    )

    with pytest.raises(MetadataValidationError) as exc_info:
        ModelSpecCompatibilityProjection(artifact_identifier="invalid.safetensors").project(backend.snapshot())

    assert [fact.fact_key for fact in exc_info.value.missing_facts] == ["architecture"]


@pytest.mark.unit
def test_modelspec_projection_requires_scope_for_multiple_artifacts() -> None:
    runtime = MetadataRuntime()
    runtime.file_many(
        (
            _artifact("first.safetensors"),
            _artifact("second.safetensors"),
        )
    )

    with pytest.raises(MetadataProjectionScopeError, match="explicit artifact identifier"):
        ModelSpecCompatibilityProjection().project(runtime.snapshot())


@pytest.mark.unit
def test_explicit_artifact_scope_selects_newest_accepted_version_of_stable_identity() -> None:
    runtime = MetadataRuntime()
    original = _artifact("stable.safetensors")
    runtime.file_many((original, replace(original, title="Updated Checkpoint")))

    modelspec = ModelSpecCompatibilityProjection(artifact_identifier="stable.safetensors").project(runtime.snapshot())
    kuro = KuroMetadataProjection(artifact_identifier="stable.safetensors").project(runtime.snapshot())

    assert modelspec.metadata["modelspec.title"] == "Updated Checkpoint"
    assert kuro.metadata["kuro.model.artifact.title"] == "Updated Checkpoint"


@pytest.mark.unit
def test_scoped_kuro_projection_preserves_multiple_model_component_identities() -> None:
    items = _scoped_model_items()
    runtime = MetadataRuntime()
    runtime.file_many(items)

    result = KuroMetadataProjection(artifact_identifier="target.safetensors").project(runtime.snapshot())

    assert result.metadata["kuro.model.artifact.id"] == "target.safetensors"
    assert result.metadata["kuro.model.realization.family_identifier"] == "sd3"
    assert result.metadata["kuro.model.family.apply_lg_attn_mask"] is True
    component_key_fields = {
        key: value for key, value in result.metadata.items() if key.startswith("kuro.model.component.") and key.endswith(".component_key")
    }
    assert set(component_key_fields.values()) == {"text_encoder", "denoiser"}
    assert len(component_key_fields) == 2
    assert "other.safetensors" not in result.metadata.values()


@pytest.mark.unit
def test_scoped_kuro_projection_allows_artifact_without_realization_context() -> None:
    runtime = MetadataRuntime()
    runtime.file(_artifact("unlinked.safetensors"))

    result = KuroMetadataProjection(artifact_identifier="unlinked.safetensors").project(runtime.snapshot())

    assert result.metadata["kuro.model.artifact.id"] == "unlinked.safetensors"
    assert not any(key.startswith("kuro.model.realization.") for key in result.metadata)


@pytest.mark.unit
def test_scoped_kuro_projection_rejects_broken_realization_relationship() -> None:
    backend = InMemoryMetadataBackend()
    artifact_identity = MetadataIdentity(
        entity_type=MetadataEntityType.MODEL_ARTIFACT,
        identifier="broken.safetensors",
    )
    missing_realization = MetadataIdentity(
        entity_type=MetadataEntityType.MODEL_REALIZATION,
        identifier="run/missing/model/target",
    )
    backend.ingest(
        MetadataProviderResult.from_sequences(
            provider_id="tests.broken_model",
            records=[
                ArtifactMetadataRecord(
                    identity=artifact_identity,
                    producer="tests.broken_model",
                    facts={
                        "architecture": "future-v1",
                        "implementation": "future-runtime",
                        "title": "Broken",
                        "resolution": "512x512",
                    },
                )
            ],
            edges=[
                MetadataEdge(
                    source=artifact_identity,
                    target=missing_realization,
                    relationship=MetadataRelationship.DERIVED_FROM,
                    producer="tests.broken_model",
                )
            ],
        )
    )

    with pytest.raises(MetadataProjectionScopeError, match="missing realization"):
        KuroMetadataProjection(artifact_identifier="broken.safetensors").project(backend.snapshot())


@pytest.mark.unit
def test_model_family_ss_projection_maps_only_linked_canonical_contribution() -> None:
    runtime = MetadataRuntime()
    runtime.file_many(_scoped_model_items())

    result = SsCompatibilityProjection(artifact_identifier="target.safetensors").project(runtime.snapshot())

    assert result.metadata == {
        "ss_apply_lg_attn_mask": True,
        "ss_apply_t5_attn_mask": False,
    }


@pytest.mark.unit
def test_kuro_projection_is_deterministic_across_backend_filing_order() -> None:
    items = _scoped_model_items()
    first_runtime = MetadataRuntime()
    first_runtime.file_many(items)
    second_runtime = MetadataRuntime()
    second_runtime.file_many(tuple(reversed(items)))

    projection = KuroMetadataProjection(artifact_identifier="target.safetensors")

    assert projection.project(first_runtime.snapshot()) == projection.project(second_runtime.snapshot())


@pytest.mark.unit
def test_kuro_projection_requires_explicit_scope_for_multiple_model_artifacts() -> None:
    runtime = MetadataRuntime()
    runtime.file_many(_scoped_model_items())

    with pytest.raises(MetadataProjectionScopeError, match="include_all_records"):
        KuroMetadataProjection().project(runtime.snapshot())


@pytest.mark.unit
def test_explicit_broad_kuro_projection_qualifies_repeated_artifacts_deterministically() -> None:
    items = _scoped_model_items()
    first_runtime = MetadataRuntime()
    first_runtime.file_many(items)
    second_runtime = MetadataRuntime()
    second_runtime.file_many(tuple(reversed(items)))
    projection = KuroMetadataProjection(include_all_records=True)

    first = projection.project(first_runtime.snapshot())
    second = projection.project(second_runtime.snapshot())
    artifact_id_fields = {
        key: value for key, value in first.metadata.items() if key.startswith("kuro.model.artifact.") and key.endswith(".id")
    }

    assert set(artifact_id_fields.values()) == {
        "target.safetensors",
        "other.safetensors",
    }
    assert first == second


@pytest.mark.unit
def test_safetensors_projection_stringifies_only_after_semantic_projection() -> None:
    runtime = MetadataRuntime()
    runtime.file_many(_scoped_model_items())
    snapshot = runtime.snapshot()
    semantic = KuroMetadataProjection(artifact_identifier="target.safetensors").project(snapshot)
    safetensors = SafetensorsMetadataProjection.from_sequence(
        (
            KuroMetadataProjection(artifact_identifier="target.safetensors"),
            SsCompatibilityProjection(artifact_identifier="target.safetensors"),
        )
    ).project(snapshot)

    assert semantic.metadata["kuro.model.family.apply_lg_attn_mask"] is True
    assert safetensors.metadata["kuro.model.family.apply_lg_attn_mask"] == "True"
    assert safetensors.metadata["ss_apply_t5_attn_mask"] == "False"


@pytest.mark.unit
def test_safetensors_projection_validates_before_export() -> None:
    backend = InMemoryMetadataBackend()
    backend.ingest(
        MetadataProviderResult.from_sequences(
            provider_id="tests.provider",
            records=[
                ArtifactMetadataRecord(
                    identity=MetadataIdentity(entity_type="artifact", identifier="checkpoint-1"),
                    producer="tests.checkpoint",
                    facts={"format": "safetensors"},
                )
            ],
        )
    )
    projection = SafetensorsMetadataProjection.from_sequence(
        [KuroMetadataProjection()],
        required_facts=[
            MetadataRequiredFact(
                fact_key="run_id",
                entity_type="artifact",
                identifier="checkpoint-1",
                projection_id="tests.safetensors",
            )
        ],
    )

    with pytest.raises(MetadataValidationError):
        projection.project(backend.snapshot())


def _artifact(
    artifact_identifier: str,
    *,
    realization_identifier: str | None = None,
) -> ModelArtifactFacts:
    return ModelArtifactFacts(
        artifact_identifier=artifact_identifier,
        family_identifier="sd3",
        artifact_role="full_model",
        artifact_format="safetensors",
        architecture="stable-diffusion-3-medium",
        implementation="https://github.com/Stability-AI/sd3.5",
        title="Checkpoint",
        resolution="1024x1024",
        realization_identifier=realization_identifier,
    )


def _scoped_model_items() -> tuple[object, ...]:
    realization = ModelRealizationFacts.for_run(
        run_identifier="run-projection",
        realization_key="training-target",
        family_identifier="sd3",
        model_version="medium",
    )
    components = (
        RealizedModelComponentFacts.for_realization(
            run_identifier=realization.run_identifier,
            realization_identifier=realization.realization_identifier,
            component_key="text_encoder",
            public_name="Text Encoder",
            declaration_order=0,
            roles=("text_encoder",),
            present=True,
        ),
        RealizedModelComponentFacts.for_realization(
            run_identifier=realization.run_identifier,
            realization_identifier=realization.realization_identifier,
            component_key="denoiser",
            public_name="Denoiser",
            declaration_order=1,
            roles=("denoiser",),
            present=True,
        ),
    )
    contribution = ModelFamilyMetadataContribution.for_realization(
        run_identifier=realization.run_identifier,
        realization_identifier=realization.realization_identifier,
        contribution_namespace="sd3.checkpointing",
        contribution_version="1",
        fields=(
            ModelFamilyMetadataField(name="apply_lg_attn_mask", value=True),
            ModelFamilyMetadataField(name="apply_t5_attn_mask", value=False),
        ),
    )
    return (
        realization,
        *components,
        contribution,
        _artifact(
            "target.safetensors",
            realization_identifier=realization.realization_identifier,
        ),
        _artifact(
            "other.safetensors",
            realization_identifier=realization.realization_identifier,
        ),
    )
