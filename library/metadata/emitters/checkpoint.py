"""Central emitters for checkpoint metadata assembly."""

from __future__ import annotations

from library.metadata.backends import InMemoryMetadataBackend, MetadataSnapshot
from library.metadata.dataclasses.artifact import CheckpointArtifactFacts
from library.metadata.dataclasses.model import ModelArtifactFacts
from library.metadata.dataclasses.run import RunMetadataFacts
from library.metadata.emitters.model import build_model_artifact_metadata
from library.metadata.emitters.run import build_training_run_metadata
from library.metadata.providers import MetadataProviderResult
from library.metadata.records import (
    ArtifactMetadataRecord,
    MetadataIdentity,
    MetadataValue,
)
from library.metadata.values import stringify_metadata_mapping
from library.metadata.versions import METADATA_PAYLOAD_VERSION

from library.metadata.projections import (
    KuroMetadataProjection,
    MetadataProjection,
    ModelSpecCompatibilityProjection,
    SafetensorsMetadataProjection,
    scope_model_artifact_snapshot,
    SsCompatibilityProjection,
)


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


def build_model_artifact_export_metadata(facts: ModelArtifactFacts) -> dict[str, str]:
    """Project one standalone model artifact from canonical accepted facts.

    This is the bounded artifact-export boundary for supported paths that do
    not yet share the main trainer's long-lived metadata runtime, such as the
    transitional textual-inversion launchers.
    """
    backend = InMemoryMetadataBackend()
    backend.ingest(build_model_artifact_metadata(facts))
    backend.validate()
    projected = SafetensorsMetadataProjection.from_sequence(
        (
            KuroMetadataProjection(artifact_identifier=facts.artifact_identifier),
            ModelSpecCompatibilityProjection(artifact_identifier=facts.artifact_identifier),
        )
    ).project(backend.snapshot())
    return stringify_metadata_mapping(projected.metadata)


def build_checkpoint_metadata(
    *,
    snapshot: MetadataSnapshot,
    training_facts: RunMetadataFacts,
    no_metadata: bool,
    artifact_identifier: str = "checkpoint",
    artifact_format: str = "safetensors",
    step: int | None = None,
    epoch: int | None = None,
) -> dict[str, str]:
    """Project one accepted model artifact plus current checkpoint facts."""
    checkpoint_facts = CheckpointArtifactFacts.for_checkpoint(
        artifact_identifier=artifact_identifier,
        no_metadata=no_metadata,
        artifact_format=artifact_format,
        step=step,
        epoch=epoch,
    )
    model_snapshot = scope_model_artifact_snapshot(
        snapshot,
        artifact_identifier,
        include_related=not no_metadata,
    )
    backend = InMemoryMetadataBackend()
    backend.ingest(
        MetadataProviderResult.from_sequences(
            provider_id="training.checkpoint.model_scope",
            records=model_snapshot.records,
            edges=model_snapshot.edges,
        )
    )
    if not no_metadata:
        backend.ingest(build_training_run_metadata(training_facts))
        backend.ingest(build_checkpoint_artifact_metadata(checkpoint_facts))
    backend.validate()

    projections: list[MetadataProjection] = [
        KuroMetadataProjection(include_all_records=True),
    ]
    if not no_metadata:
        projections.append(SsCompatibilityProjection(artifact_identifier=artifact_identifier))
    projections.append(ModelSpecCompatibilityProjection(artifact_identifier=artifact_identifier))
    projected = SafetensorsMetadataProjection.from_sequence(projections).project(backend.snapshot())
    return stringify_metadata_mapping(projected.metadata)


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
