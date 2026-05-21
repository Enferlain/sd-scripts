"""Central emitters for checkpoint metadata assembly."""

from __future__ import annotations

from library.metadata.backends import InMemoryMetadataBackend
from library.metadata.dataclasses.artifact import CheckpointArtifactFacts
from library.metadata.dataclasses.model import ModelSpecFacts
from library.metadata.dataclasses.run import RunMetadataFacts
from library.metadata.emitters.run import build_model_spec_metadata, build_training_run_metadata
from library.metadata.projections import (
    KuroMetadataProjection,
    ModelSpecCompatibilityProjection,
    SafetensorsMetadataProjection,
    SsCompatibilityProjection,
)
from library.metadata.keys import SS_METADATA_MINIMUM_KEYS
from library.metadata.providers import MetadataProviderResult
from library.metadata.records import ArtifactMetadataRecord, MetadataIdentity, MetadataValue
from library.metadata.versions import METADATA_PAYLOAD_VERSION


def build_checkpoint_artifact_metadata(
    facts: CheckpointArtifactFacts,
    *,
    provider_id: str = "training.checkpoint",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build collected metadata for the checkpoint artifact boundary."""
    return MetadataProviderResult.from_sequences(
        provider_id=provider_id,
        schema_version=schema_version,
        records=[_build_checkpoint_artifact_record(facts, producer=provider_id)],
    )


def build_checkpoint_metadata(
    *,
    training_facts: RunMetadataFacts,
    minimum_training_facts: RunMetadataFacts,
    model_facts: ModelSpecFacts,
    no_metadata: bool,
    artifact_identifier: str = "checkpoint",
    step: int | None = None,
    epoch: int | None = None,
) -> dict[str, str]:
    """Build string artifact metadata through the metadata backbone."""
    run_facts = minimum_training_facts if no_metadata else training_facts
    checkpoint_facts = CheckpointArtifactFacts.for_checkpoint(
        artifact_identifier=artifact_identifier,
        no_metadata=no_metadata,
        step=step,
        epoch=epoch,
    )
    backend = InMemoryMetadataBackend()
    backend.ingest(build_training_run_metadata(run_facts))
    backend.ingest(build_model_spec_metadata(model_facts))
    backend.ingest(build_checkpoint_artifact_metadata(checkpoint_facts))
    backend.validate()

    projections = [
        SsCompatibilityProjection(
            minimum_keys=frozenset(SS_METADATA_MINIMUM_KEYS) if no_metadata else None,
        ),
        ModelSpecCompatibilityProjection(),
    ]
    if not no_metadata:
        projections.append(KuroMetadataProjection())
    return SafetensorsMetadataProjection.from_sequence(projections).project(backend.snapshot()).metadata


def _build_checkpoint_artifact_record(
    facts: CheckpointArtifactFacts,
    *,
    producer: str,
) -> ArtifactMetadataRecord:
    metadata: dict[str, MetadataValue] = {
        "kind": facts.kind,
        "format": facts.artifact_format,
        "metadata_policy": facts.metadata_policy,
    }
    if facts.step is not None:
        metadata["step"] = facts.step
    if facts.epoch is not None:
        metadata["epoch"] = facts.epoch
    return ArtifactMetadataRecord(
        identity=MetadataIdentity(
            entity_type="artifact",
            identifier=facts.artifact_identifier,
            schema_version=METADATA_PAYLOAD_VERSION,
        ),
        producer=producer,
        facts=metadata,
        schema_version=METADATA_PAYLOAD_VERSION,
    )
