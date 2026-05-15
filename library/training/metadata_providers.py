"""Training-owned metadata providers for active checkpoint artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from library.constants import SS_METADATA_MINIMUM_KEYS
from library.metadata import (
    AdapterMetadataRecord,
    ArtifactMetadataRecord,
    InMemoryMetadataBackend,
    KuroMetadataProjection,
    MetadataIdentity,
    MetadataProviderResult,
    ModelComponentMetadataRecord,
    ModelSpecCompatibilityProjection,
    RunMetadataRecord,
    SafetensorsMetadataProjection,
    SsCompatibilityProjection,
)
from library.metadata.records import MetadataValue


@dataclass(frozen=True)
class TrainingRunMetadataProvider:
    """Provider wrapper for the existing `ss_*` training-run metadata dict."""

    metadata: Mapping[str, str]
    run_identifier: str
    provider_id: str = "training.run"
    schema_version: str = "1"

    def collect_metadata(self) -> MetadataProviderResult:
        record = RunMetadataRecord(
            identity=MetadataIdentity(entity_type="run", identifier=self.run_identifier),
            producer=self.provider_id,
            facts=dict(self.metadata),
            schema_version=self.schema_version,
        )
        return MetadataProviderResult.from_sequences(
            provider_id=self.provider_id,
            schema_version=self.schema_version,
            records=[record],
        )


@dataclass(frozen=True)
class ModelSpecMetadataProvider:
    """Provider wrapper for model-family `modelspec.*` compatibility metadata."""

    metadata: Mapping[str, str]
    model_identifier: str = "active-model"
    provider_id: str = "model.modelspec"
    schema_version: str = "1"

    def collect_metadata(self) -> MetadataProviderResult:
        facts: dict[str, MetadataValue] = dict(self.metadata)
        architecture = self.metadata.get("modelspec.architecture")
        implementation = self.metadata.get("modelspec.implementation")
        prediction_type = self.metadata.get("modelspec.prediction_type")
        if architecture is not None:
            facts["architecture"] = architecture
        if implementation is not None:
            facts["implementation"] = implementation
        if prediction_type is not None:
            facts["prediction_type"] = prediction_type

        record = ModelComponentMetadataRecord(
            identity=MetadataIdentity(entity_type="model", identifier=self.model_identifier),
            producer=self.provider_id,
            facts=facts,
            schema_version=self.schema_version,
        )
        return MetadataProviderResult.from_sequences(
            provider_id=self.provider_id,
            schema_version=self.schema_version,
            records=[record],
        )


@dataclass(frozen=True)
class AdapterArtifactMetadataProvider:
    """Provider wrapper for adapter facts already present in training metadata."""

    metadata: Mapping[str, str]
    adapter_identifier: str = "active-adapter"
    provider_id: str = "adapter.artifact"
    schema_version: str = "1"

    def collect_metadata(self) -> MetadataProviderResult:
        facts = {
            "adapter_module": self.metadata.get("ss_adapter_module"),
            "adapter_rank": self.metadata.get("ss_adapter_rank"),
            "adapter_alpha": self.metadata.get("ss_adapter_alpha"),
            "adapter_dropout": self.metadata.get("ss_adapter_neuron_dropout"),
            "scale_weight_norms": self.metadata.get("ss_scale_weight_norms"),
        }
        facts = {key: value for key, value in facts.items() if value is not None}
        if not facts:
            return MetadataProviderResult(provider_id=self.provider_id, schema_version=self.schema_version)

        record = AdapterMetadataRecord(
            identity=MetadataIdentity(entity_type="adapter", identifier=self.adapter_identifier),
            producer=self.provider_id,
            facts=facts,
            schema_version=self.schema_version,
        )
        return MetadataProviderResult.from_sequences(
            provider_id=self.provider_id,
            schema_version=self.schema_version,
            records=[record],
        )


@dataclass(frozen=True)
class CheckpointArtifactMetadataProvider:
    """Provider for the checkpoint artifact boundary itself."""

    artifact_identifier: str
    no_metadata: bool
    step: int | None = None
    epoch: int | None = None
    provider_id: str = "training.checkpoint"
    schema_version: str = "1"

    def collect_metadata(self) -> MetadataProviderResult:
        facts: dict[str, MetadataValue] = {
            "kind": "checkpoint",
            "format": "safetensors",
            "metadata_policy": "minimum" if self.no_metadata else "full",
        }
        if self.step is not None:
            facts["step"] = self.step
        if self.epoch is not None:
            facts["epoch"] = self.epoch

        record = ArtifactMetadataRecord(
            identity=MetadataIdentity(entity_type="artifact", identifier=self.artifact_identifier),
            producer=self.provider_id,
            facts=facts,
            schema_version=self.schema_version,
        )
        return MetadataProviderResult.from_sequences(
            provider_id=self.provider_id,
            schema_version=self.schema_version,
            records=[record],
        )


def build_checkpoint_metadata(
    *,
    training_metadata: Mapping[str, str],
    minimum_metadata: Mapping[str, str],
    model_metadata: Mapping[str, str],
    no_metadata: bool,
    run_identifier: str,
    artifact_identifier: str = "checkpoint",
    step: int | None = None,
    epoch: int | None = None,
) -> dict[str, str]:
    """Build string artifact metadata through the metadata backbone."""
    selected_training_metadata = minimum_metadata if no_metadata else training_metadata
    backend = InMemoryMetadataBackend()
    backend.ingest_all(
        [
            TrainingRunMetadataProvider(selected_training_metadata, run_identifier=run_identifier),
            ModelSpecMetadataProvider(model_metadata),
            AdapterArtifactMetadataProvider(selected_training_metadata),
            CheckpointArtifactMetadataProvider(
                artifact_identifier=artifact_identifier,
                no_metadata=no_metadata,
                step=step,
                epoch=epoch,
            ),
        ]
    )
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
