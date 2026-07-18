"""Projection contracts for exported metadata shapes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import quote

from library.metadata.graph import MetadataEntityType, MetadataRelationship
from library.metadata.providers import MetadataRequiredFact
from library.metadata.records import MetadataRecord, MetadataValue
from library.metadata.validation import validate_required_facts
from library.metadata.values import stringify_metadata_mapping

from library.metadata.exports.resource import (
    project_resource_monitor_compatibility_event,
    project_resource_run_compatibility_events,
)

from library.metadata.keys import (
    KOHYA_SS_PREFIX,
    KURO_PREFIX,
    KURO_SCHEMA_VERSION,
    KURO_SCHEMA_VERSION_KEY,
    MODELSPEC_PREFIX,
    MODELSPEC_VERSION,
    MODELSPEC_VERSION_KEY,
)


__all__ = [
    "KuroMetadataProjection",
    "MetadataProjection",
    "MetadataProjectionScopeError",
    "ModelSpecCompatibilityProjection",
    "ProjectionResult",
    "SafetensorsMetadataProjection",
    "SsCompatibilityProjection",
    "project_resource_monitor_compatibility_event",
    "project_resource_run_compatibility_events",
]


class MetadataProjectionScopeError(ValueError):
    """Raised when a projection target is missing, conflicting, or ambiguous."""


@dataclass(frozen=True)
class ProjectionResult:
    """Projected metadata values before an optional string-only boundary."""

    metadata: dict[str, MetadataValue] = field(default_factory=dict)


class MetadataProjection(Protocol):
    """Converts typed metadata snapshots into external key/value metadata."""

    projection_id: str
    required_facts: tuple[MetadataRequiredFact, ...]

    def project(self, snapshot) -> ProjectionResult:
        """Project collected metadata into one external representation."""


def project_with_validation(projection: MetadataProjection, snapshot) -> ProjectionResult:
    """Validate projection requirements before producing exported metadata."""
    validate_required_facts(snapshot.records, projection.required_facts)
    return projection.project(snapshot)


@dataclass(frozen=True)
class KuroMetadataProjection:
    """Projection for repo-owned exported `kuro.*` metadata."""

    artifact_identifier: str | None = None
    include_all_records: bool = False
    projection_id: str = "kuro"
    required_facts: tuple[MetadataRequiredFact, ...] = ()

    def project(self, snapshot) -> ProjectionResult:
        validate_required_facts(snapshot.records, self.required_facts)
        records = _kuro_records_for_scope(
            snapshot,
            self.artifact_identifier,
            include_all_records=self.include_all_records,
            projection_id=self.projection_id,
        )
        supported_records = tuple(
            sorted(
                (record for record in records if _kuro_entity_prefix(record.identity.entity_type) is not None),
                key=lambda record: (record.identity.entity_type, record.identity.key),
            )
        )
        prefix_counts = Counter(_kuro_entity_prefix(record.identity.entity_type) for record in supported_records)
        metadata: dict[str, MetadataValue] = {KURO_SCHEMA_VERSION_KEY: KURO_SCHEMA_VERSION}
        for record in supported_records:
            base_prefix = _kuro_entity_prefix(record.identity.entity_type)
            assert base_prefix is not None
            entity_prefix = base_prefix if prefix_counts[base_prefix] == 1 else f"{base_prefix}.{_identity_key_segment(record)}"
            metadata[f"{entity_prefix}.id"] = record.identity.identifier
            metadata[f"{entity_prefix}.schema_version"] = record.identity.schema_version
            metadata[f"{entity_prefix}.producer"] = record.producer
            if _is_new_model_entity(record.identity.entity_type):
                metadata[f"{entity_prefix}.namespace"] = record.identity.namespace
            if record.identity.label is not None:
                metadata[f"{entity_prefix}.label"] = record.identity.label
            for key, value in sorted(record.facts.items()):
                if _is_compatibility_key(key):
                    continue
                projected_key = key if key.startswith(KURO_PREFIX) else f"{entity_prefix}.{key}"
                metadata[projected_key] = value
        return ProjectionResult(metadata)


@dataclass(frozen=True)
class SsCompatibilityProjection:
    """Projection for legacy Kohya `ss_*` compatibility metadata."""

    artifact_identifier: str | None = None
    projection_id: str = "kohya_ss"
    required_facts: tuple[MetadataRequiredFact, ...] = ()
    minimum_keys: frozenset[str] | None = None

    def project(self, snapshot) -> ProjectionResult:
        validate_required_facts(snapshot.records, self.required_facts)
        # Pre-prefixed facts are a bounded bridge for concern-owned legacy
        # metadata that does not yet carry artifact relationships. Canonical
        # family contributions below are artifact-scoped when a target is
        # supplied; later concern migrations replace this unscoped passthrough.
        metadata = _collect_prefixed_facts(snapshot.records, KOHYA_SS_PREFIX)
        family_records = (
            _scoped_model_records(snapshot, self.artifact_identifier, projection_id=self.projection_id)
            if self.artifact_identifier is not None
            else snapshot.records
        )
        for record in sorted(family_records, key=lambda item: item.identity.key):
            if record.identity.entity_type != MetadataEntityType.MODEL_FAMILY_FACTS:
                continue
            contribution_namespace = record.facts.get("contribution_namespace")
            contribution_version = record.facts.get("contribution_version")
            for fact_key, value in sorted(record.facts.items()):
                compatibility_key = _SS_FAMILY_FIELD_MAPPINGS.get((contribution_namespace, contribution_version, fact_key))
                if compatibility_key is None:
                    continue
                previous = metadata.get(compatibility_key)
                if previous is not None and previous != value:
                    raise _scope_error(
                        self.projection_id,
                        f"multiple family contributions claim {compatibility_key!r}",
                    )
                metadata[compatibility_key] = value
        if self.minimum_keys is not None:
            metadata = {key: value for key, value in metadata.items() if key in self.minimum_keys}
        return ProjectionResult(metadata)


@dataclass(frozen=True)
class ModelSpecCompatibilityProjection:
    """Projection for SAI Model Spec `modelspec.*` compatibility metadata."""

    artifact_identifier: str | None = None
    projection_id: str = "modelspec"
    required_facts: tuple[MetadataRequiredFact, ...] = ()

    def project(self, snapshot) -> ProjectionResult:
        validate_required_facts(snapshot.records, self.required_facts)
        artifact_record = _select_model_artifact_record(
            snapshot,
            self.artifact_identifier,
            projection_id=self.projection_id,
        )
        if artifact_record is None:
            # Transitional bridge for the live checkpoint path. Section 7
            # removes the pre-rendered ModelSpec producer and this fallback.
            return ProjectionResult(_collect_prefixed_facts(snapshot.records, MODELSPEC_PREFIX))

        required_facts = tuple(
            MetadataRequiredFact(
                fact_key=fact_key,
                entity_type=MetadataEntityType.MODEL_ARTIFACT,
                identifier=artifact_record.identity.identifier,
                namespace=artifact_record.identity.namespace,
                projection_id=self.projection_id,
            )
            for fact_key in _REQUIRED_MODELSPEC_FACTS
        )
        validate_required_facts(snapshot.records, required_facts)

        metadata: dict[str, MetadataValue] = {MODELSPEC_VERSION_KEY: MODELSPEC_VERSION}
        for fact_key in _MODELSPEC_FACT_KEYS:
            value = artifact_record.facts.get(fact_key)
            if value is not None:
                metadata[f"{MODELSPEC_PREFIX}{fact_key}"] = value

        extension_fields = artifact_record.facts.get("extension_fields", {})
        if not isinstance(extension_fields, Mapping):
            raise TypeError("Model artifact extension_fields must be a mapping before ModelSpec projection.")
        for key, value in sorted(extension_fields.items()):
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("Model artifact extension_fields must contain only string keys and values.")
            projected_key = key if key.startswith(MODELSPEC_PREFIX) else f"{MODELSPEC_PREFIX}{key}"
            metadata[projected_key] = value
        return ProjectionResult(metadata)


@dataclass(frozen=True)
class SafetensorsMetadataProjection:
    """Composite projection that produces one string-only safetensors metadata dict."""

    projections: tuple[MetadataProjection, ...]
    projection_id: str = "safetensors"
    required_facts: tuple[MetadataRequiredFact, ...] = ()

    @classmethod
    def from_sequence(
        cls,
        projections: Sequence[MetadataProjection],
        required_facts: Iterable[MetadataRequiredFact] = (),
    ) -> SafetensorsMetadataProjection:
        return cls(projections=tuple(projections), required_facts=tuple(required_facts))

    def project(self, snapshot) -> ProjectionResult:
        validate_required_facts(snapshot.records, self.required_facts)
        metadata: dict[str, MetadataValue] = {}
        for projection in self.projections:
            metadata.update(project_with_validation(projection, snapshot).metadata)
        string_metadata: dict[str, MetadataValue] = {}
        for key, value in stringify_metadata_mapping(metadata).items():
            string_metadata[key] = value
        return ProjectionResult(string_metadata)


def _collect_prefixed_facts(records, prefix: str) -> dict[str, MetadataValue]:
    metadata: dict[str, MetadataValue] = {}
    for record in records:
        metadata.update({key: value for key, value in record.facts.items() if key.startswith(prefix)})
    return metadata


def _is_compatibility_key(key: str) -> bool:
    return key.startswith(KOHYA_SS_PREFIX) or key.startswith(MODELSPEC_PREFIX)


def _kuro_entity_prefix(entity_type: str) -> str | None:
    normalized = entity_type.replace("_", "-")
    supported_entities = {
        "adapter": "kuro.adapter",
        "artifact": "kuro.artifact",
        "model": "kuro.model",
        "model-component": "kuro.model.component",
        "model-artifact": "kuro.model.artifact",
        "model-family-facts": "kuro.model.family",
        "model-realization": "kuro.model.realization",
        "run": "kuro.run",
    }
    return supported_entities.get(normalized)


_REQUIRED_MODELSPEC_FACTS = ("architecture", "implementation", "title", "resolution")

_MODELSPEC_FACT_KEYS = (
    "architecture",
    "implementation",
    "title",
    "resolution",
    "description",
    "author",
    "date",
    "hash_sha256",
    "implementation_version",
    "license",
    "usage_hint",
    "thumbnail",
    "tags",
    "merged_from",
    "trigger_phrase",
    "prediction_type",
    "timestep_range",
    "encoder_layer",
    "preprocessor",
    "is_negative_embedding",
    "unet_dtype",
    "vae_dtype",
)

_SS_FAMILY_FIELD_MAPPINGS = {
    ("sd3.checkpointing", "1", "apply_lg_attn_mask"): "ss_apply_lg_attn_mask",
    ("sd3.checkpointing", "1", "apply_t5_attn_mask"): "ss_apply_t5_attn_mask",
}


def _select_model_artifact_record(
    snapshot,
    artifact_identifier: str | None,
    *,
    projection_id: str,
) -> MetadataRecord | None:
    records = tuple(
        record
        for record in snapshot.records
        if record.identity.entity_type == MetadataEntityType.MODEL_ARTIFACT
        and (artifact_identifier is None or record.identity.identifier == artifact_identifier)
    )
    if not records:
        if artifact_identifier is None:
            return None
        raise _scope_error(projection_id, f"model artifact {artifact_identifier!r} is not present")
    if len(records) > 1:
        target = artifact_identifier if artifact_identifier is not None else "an explicit artifact identifier"
        raise _scope_error(projection_id, f"model artifact scope is ambiguous; select {target}")
    return records[0]


def _kuro_records_for_scope(
    snapshot,
    artifact_identifier: str | None,
    *,
    include_all_records: bool,
    projection_id: str,
) -> tuple[MetadataRecord, ...]:
    if artifact_identifier is not None and include_all_records:
        raise _scope_error(projection_id, "artifact_identifier and include_all_records are mutually exclusive")
    if include_all_records:
        return snapshot.records
    if artifact_identifier is not None:
        return _scoped_model_records(snapshot, artifact_identifier, projection_id=projection_id)

    model_artifacts = tuple(record for record in snapshot.records if record.identity.entity_type == MetadataEntityType.MODEL_ARTIFACT)
    if not model_artifacts:
        # Transitional bridge for legacy snapshots that do not yet contain a
        # canonical model-artifact record.
        return snapshot.records
    if len(model_artifacts) > 1:
        raise _scope_error(
            projection_id,
            "multiple model artifacts require artifact_identifier or include_all_records=True",
        )
    return _scoped_model_records(
        snapshot,
        model_artifacts[0].identity.identifier,
        projection_id=projection_id,
    )


def _scoped_model_records(
    snapshot,
    artifact_identifier: str,
    *,
    projection_id: str,
) -> tuple[MetadataRecord, ...]:
    artifact_record = _select_model_artifact_record(
        snapshot,
        artifact_identifier,
        projection_id=projection_id,
    )
    assert artifact_record is not None
    records_by_key = {record.identity.key: record for record in snapshot.records}
    realization_edges = tuple(
        edge
        for edge in snapshot.edges
        if edge.source.key == artifact_record.identity.key
        and edge.relationship == MetadataRelationship.DERIVED_FROM
        and edge.target.entity_type == MetadataEntityType.MODEL_REALIZATION
    )
    if len(realization_edges) > 1:
        raise _scope_error(projection_id, f"model artifact {artifact_identifier!r} has multiple realizations")

    selected = {artifact_record.identity.key: artifact_record}
    if not realization_edges:
        return (artifact_record,)

    realization_identity = realization_edges[0].target
    realization_record = records_by_key.get(realization_identity.key)
    if realization_record is None:
        raise _scope_error(
            projection_id,
            f"model artifact {artifact_identifier!r} references a missing realization",
        )
    selected[realization_record.identity.key] = realization_record

    for edge in snapshot.edges:
        if edge.target.key != realization_identity.key:
            continue
        if edge.relationship not in {
            MetadataRelationship.CONTAINED_IN,
            MetadataRelationship.DESCRIBES,
        }:
            continue
        record = records_by_key.get(edge.source.key)
        if record is not None and record.identity.entity_type in {
            MetadataEntityType.MODEL_COMPONENT,
            MetadataEntityType.MODEL_FAMILY_FACTS,
        }:
            selected[record.identity.key] = record
    return tuple(sorted(selected.values(), key=lambda record: (record.identity.entity_type, record.identity.key)))


def _is_new_model_entity(entity_type: str) -> bool:
    return entity_type in {
        MetadataEntityType.MODEL_ARTIFACT,
        MetadataEntityType.MODEL_COMPONENT,
        MetadataEntityType.MODEL_FAMILY_FACTS,
        MetadataEntityType.MODEL_REALIZATION,
    }


def _identity_key_segment(record: MetadataRecord) -> str:
    """Encode one qualified identity as a reversible external-key segment.

    URL quoting escapes structural separators such as ``:`` and ``/`` so a
    complete metadata identity cannot be confused with the surrounding
    ``kuro.*`` path. RFC 3986 unreserved characters remain readable.
    """
    return quote(record.identity.key, safe="")


def _scope_error(projection_id: str, description: str) -> MetadataProjectionScopeError:
    return MetadataProjectionScopeError(f"{projection_id} projection scope is invalid: {description}.")
