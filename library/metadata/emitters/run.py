"""Central emitters for accepted training-run and legacy model-spec facts."""

from __future__ import annotations

from library.metadata.dataclasses.model import ModelSpecFacts
from library.metadata.dataclasses.run import RunMetadataFacts
from library.metadata.providers import MetadataProviderResult
from library.metadata.records import MetadataIdentity, MetadataValue, ModelComponentMetadataRecord, RunMetadataRecord
from library.metadata.versions import METADATA_PAYLOAD_VERSION


def build_training_run_metadata(
    facts: RunMetadataFacts,
    *,
    provider_id: str = "training.run",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Emit one record for accepted training-run facts."""
    return MetadataProviderResult.from_sequences(
        provider_id=provider_id,
        schema_version=schema_version,
        records=[_build_training_run_record(facts, producer=provider_id)],
    )


def build_model_spec_metadata(
    facts: ModelSpecFacts,
    *,
    provider_id: str = "model.modelspec",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Emit one record for accepted legacy model-spec-compatible facts."""
    return MetadataProviderResult.from_sequences(
        provider_id=provider_id,
        schema_version=schema_version,
        records=[_build_model_spec_record(facts, producer=provider_id)],
    )


def _build_training_run_record(facts: RunMetadataFacts, *, producer: str) -> RunMetadataRecord:
    return RunMetadataRecord(
        identity=MetadataIdentity(
            entity_type="run",
            identifier=facts.run_identifier,
            schema_version=METADATA_PAYLOAD_VERSION,
        ),
        producer=producer,
        facts=dict(facts.metadata),
        schema_version=METADATA_PAYLOAD_VERSION,
    )


def _build_model_spec_record(facts: ModelSpecFacts, *, producer: str) -> ModelComponentMetadataRecord:
    metadata: dict[str, MetadataValue] = dict(facts.compatibility_metadata)
    if facts.architecture is not None:
        metadata["architecture"] = facts.architecture
    if facts.implementation is not None:
        metadata["implementation"] = facts.implementation
    if facts.prediction_type is not None:
        metadata["prediction_type"] = facts.prediction_type
    return ModelComponentMetadataRecord(
        identity=MetadataIdentity(
            entity_type="model",
            identifier=facts.model_identifier,
            schema_version=METADATA_PAYLOAD_VERSION,
        ),
        producer=producer,
        facts=metadata,
        schema_version=METADATA_PAYLOAD_VERSION,
    )


__all__ = ["build_model_spec_metadata", "build_training_run_metadata"]
