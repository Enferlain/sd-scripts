"""Central emitters for accepted training-run facts."""

from __future__ import annotations

from library.metadata.dataclasses.run import RunMetadataFacts
from library.metadata.providers import MetadataProviderResult
from library.metadata.records import MetadataIdentity, RunMetadataRecord
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


__all__ = ["build_training_run_metadata"]
