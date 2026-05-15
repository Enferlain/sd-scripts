"""Unit tests for the metadata backbone core contracts."""

import pytest

from library.metadata import (
    ArtifactMetadataRecord,
    InMemoryMetadataBackend,
    MetadataEdge,
    MetadataEvent,
    MetadataIdentity,
    MetadataProviderResult,
    MetadataRequiredFact,
    MetadataValidationError,
    RunMetadataRecord,
)


@pytest.mark.unit
def test_metadata_identity_records_and_relationships_are_collected() -> None:
    run_identity = MetadataIdentity(entity_type="run", identifier="run-1")
    artifact_identity = MetadataIdentity(entity_type="artifact", identifier="checkpoint-1")
    run_record = RunMetadataRecord(
        identity=run_identity,
        producer="tests.training",
        facts={"session_id": 123, "seed": 42},
    )
    artifact_record = ArtifactMetadataRecord(
        identity=artifact_identity,
        producer="tests.checkpoint",
        facts={"format": "safetensors"},
    )
    event = MetadataEvent(
        event_type="artifact_written",
        identity=artifact_identity,
        producer="tests.checkpoint",
        facts={"step": 10},
    )
    edge = MetadataEdge(
        source=artifact_identity,
        target=run_identity,
        relationship="produced_by",
        producer="tests.checkpoint",
    )

    backend = InMemoryMetadataBackend()
    backend.ingest(
        MetadataProviderResult.from_sequences(
            provider_id="tests.provider",
            records=[run_record, artifact_record],
            events=[event],
            edges=[edge],
        )
    )

    snapshot = backend.snapshot()

    assert run_identity.key == "kuro:run:run-1"
    assert snapshot.record_for(entity_type="run", identifier="run-1") == run_record
    assert snapshot.records_for(entity_type="artifact") == (artifact_record,)
    assert snapshot.events == (event,)
    assert snapshot.edges == (edge,)


@pytest.mark.unit
def test_backend_validates_provider_required_facts() -> None:
    backend = InMemoryMetadataBackend()
    backend.ingest(
        MetadataProviderResult.from_sequences(
            provider_id="tests.provider",
            records=[
                RunMetadataRecord(
                    identity=MetadataIdentity(entity_type="run", identifier="run-1"),
                    producer="tests.training",
                    facts={"session_id": 123},
                )
            ],
            required_facts=[
                MetadataRequiredFact(
                    fact_key="seed",
                    entity_type="run",
                    identifier="run-1",
                    provider_id="tests.provider",
                )
            ],
        )
    )

    with pytest.raises(MetadataValidationError) as error_info:
        backend.validate()

    assert error_info.value.missing_facts[0].fact_key == "seed"
    assert error_info.value.missing_facts[0].owner == "tests.provider"


@pytest.mark.unit
def test_backend_validation_allows_absent_optional_facts() -> None:
    backend = InMemoryMetadataBackend()
    backend.ingest(
        MetadataProviderResult.from_sequences(
            provider_id="tests.provider",
            records=[
                RunMetadataRecord(
                    identity=MetadataIdentity(entity_type="run", identifier="run-1"),
                    producer="tests.training",
                    facts={"session_id": 123},
                )
            ],
        )
    )

    backend.validate()
