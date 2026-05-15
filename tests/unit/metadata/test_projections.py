"""Unit tests for metadata export projections."""

import pytest

from library.metadata import (
    ArtifactMetadataRecord,
    InMemoryMetadataBackend,
    KuroMetadataProjection,
    MetadataIdentity,
    MetadataProviderResult,
    MetadataRequiredFact,
    MetadataValidationError,
    ModelSpecCompatibilityProjection,
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
    assert result.metadata["kuro.run.seed"] == "42"
    assert "ss_seed" not in result.metadata
    assert "modelspec.title" not in result.metadata


@pytest.mark.unit
def test_compatibility_projections_preserve_existing_export_key_shapes() -> None:
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
    modelspec_result = ModelSpecCompatibilityProjection().project(snapshot)

    assert ss_result.metadata == {"ss_seed": "42"}
    assert modelspec_result.metadata == {"modelspec.title": "Checkpoint"}


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
