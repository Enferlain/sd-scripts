"""Central emitters for model realization, component, and artifact facts."""

from __future__ import annotations

from dataclasses import fields

from library.metadata.dataclasses.model import (
    ModelArtifactFacts,
    ModelFamilyMetadataContribution,
    ModelRealizationFacts,
    RealizedModelComponentFacts,
)
from library.metadata.graph import MetadataEntityType, MetadataRelationship
from library.metadata.providers import MetadataProviderResult
from library.metadata.records import (
    ArtifactMetadataRecord,
    MetadataEdge,
    MetadataIdentity,
    MetadataValue,
    ModelComponentMetadataRecord,
    ModelFamilyContributionMetadataRecord,
    ModelRealizationMetadataRecord,
)
from library.metadata.versions import METADATA_PAYLOAD_VERSION


def build_model_realization_metadata(
    facts: ModelRealizationFacts,
    *,
    provider_id: str = "model.realization",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build one run-scoped model realization record and run edge."""
    identity = _identity(
        entity_type=MetadataEntityType.MODEL_REALIZATION,
        identifier=facts.realization_identifier,
        label=facts.family.family_identifier,
        schema_version=schema_version,
    )
    record_facts: dict[str, MetadataValue] = {
        "run_identifier": facts.run_identifier,
        "realization_key": facts.realization_key,
        "family_identifier": facts.family.family_identifier,
        "family_declaration_version": facts.family.declaration_version,
    }
    if facts.model_version is not None:
        record_facts["model_version"] = facts.model_version
    record = ModelRealizationMetadataRecord(
        identity=identity,
        producer=provider_id,
        facts=record_facts,
        schema_version=schema_version,
    )
    run_identity = _identity(
        entity_type=MetadataEntityType.RUN,
        identifier=facts.run_identifier,
        schema_version=schema_version,
    )
    edge = MetadataEdge(
        source=identity,
        target=run_identity,
        relationship=MetadataRelationship.REALIZED_IN,
        producer=provider_id,
        schema_version=schema_version,
    )
    return MetadataProviderResult.from_sequences(
        provider_id=provider_id,
        schema_version=schema_version,
        records=[record],
        edges=[edge],
    )


def build_realized_model_component_metadata(
    facts: RealizedModelComponentFacts,
    *,
    provider_id: str = "model.realized_component",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build one qualified realized-component record and ownership edge."""
    identity = _identity(
        entity_type=MetadataEntityType.MODEL_COMPONENT,
        identifier=facts.component_identifier,
        label=facts.public_name,
        schema_version=schema_version,
    )
    record = ModelComponentMetadataRecord(
        identity=identity,
        producer=provider_id,
        facts={
            "run_identifier": facts.run_identifier,
            "realization_identifier": facts.realization_identifier,
            "component_key": facts.component_key,
            "public_name": facts.public_name,
            "declaration_order": facts.declaration_order,
            "roles": facts.roles,
            "capabilities": facts.capabilities,
            "present": facts.present,
        },
        schema_version=schema_version,
    )
    realization_identity = _identity(
        entity_type=MetadataEntityType.MODEL_REALIZATION,
        identifier=facts.realization_identifier,
        schema_version=schema_version,
    )
    edge = MetadataEdge(
        source=identity,
        target=realization_identity,
        relationship=MetadataRelationship.CONTAINED_IN,
        producer=provider_id,
        schema_version=schema_version,
    )
    return MetadataProviderResult.from_sequences(
        provider_id=provider_id,
        schema_version=schema_version,
        records=[record],
        edges=[edge],
    )


def build_model_artifact_metadata(
    facts: ModelArtifactFacts,
    *,
    provider_id: str = "model.artifact",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build canonical model facts for one output artifact."""
    identity = _identity(
        entity_type=MetadataEntityType.MODEL_ARTIFACT,
        identifier=facts.artifact_identifier,
        label=facts.title,
        schema_version=schema_version,
    )
    record_facts: dict[str, MetadataValue] = {
        field.name: value
        for field in fields(facts)
        if field.name not in {"artifact_identifier", "realization_identifier"}
        and (value := getattr(facts, field.name)) is not None
    }
    record_facts["extension_fields"] = dict(facts.extension_fields)
    record = ArtifactMetadataRecord(
        identity=identity,
        producer=provider_id,
        facts=record_facts,
        schema_version=schema_version,
    )
    edges: list[MetadataEdge] = []
    if facts.realization_identifier is not None:
        edges.append(
            MetadataEdge(
                source=identity,
                target=_identity(
                    entity_type=MetadataEntityType.MODEL_REALIZATION,
                    identifier=facts.realization_identifier,
                    schema_version=schema_version,
                ),
                relationship=MetadataRelationship.DERIVED_FROM,
                producer=provider_id,
                schema_version=schema_version,
            )
        )
    return MetadataProviderResult.from_sequences(
        provider_id=provider_id,
        schema_version=schema_version,
        records=[record],
        edges=edges,
    )


def build_model_family_contribution_metadata(
    facts: ModelFamilyMetadataContribution,
    *,
    provider_id: str = "model.family_contribution",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build a versioned family-local fact record without compatibility keys."""
    identity = _identity(
        entity_type=MetadataEntityType.MODEL_FAMILY_FACTS,
        identifier=facts.contribution_identifier,
        label=facts.contribution_namespace,
        schema_version=schema_version,
    )
    record = ModelFamilyContributionMetadataRecord(
        identity=identity,
        producer=provider_id,
        facts={
            "run_identifier": facts.run_identifier,
            "realization_identifier": facts.realization_identifier,
            "contribution_namespace": facts.contribution_namespace,
            "contribution_version": facts.contribution_version,
            **{field.name: field.value for field in facts.fields},
        },
        schema_version=schema_version,
    )
    edge = MetadataEdge(
        source=identity,
        target=_identity(
            entity_type=MetadataEntityType.MODEL_REALIZATION,
            identifier=facts.realization_identifier,
            schema_version=schema_version,
        ),
        relationship=MetadataRelationship.DESCRIBES,
        producer=provider_id,
        schema_version=schema_version,
    )
    return MetadataProviderResult.from_sequences(
        provider_id=provider_id,
        schema_version=schema_version,
        records=[record],
        edges=[edge],
    )


def _identity(
    *,
    entity_type: str,
    identifier: str,
    schema_version: str,
    label: str | None = None,
) -> MetadataIdentity:
    return MetadataIdentity(
        entity_type=entity_type,
        identifier=identifier,
        label=label,
        schema_version=schema_version,
    )
