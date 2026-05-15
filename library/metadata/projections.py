"""Projection contracts for exported metadata shapes."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from library.metadata.keys import KOHYA_SS_PREFIX, KURO_PREFIX, KURO_SCHEMA_VERSION, KURO_SCHEMA_VERSION_KEY, MODELSPEC_PREFIX
from library.metadata.providers import MetadataRequiredFact
from library.metadata.records import MetadataValue
from library.metadata.validation import validate_required_facts


@dataclass(frozen=True)
class ProjectionResult:
    """String metadata ready for storage boundaries such as safetensors."""

    metadata: dict[str, str] = field(default_factory=dict)


class MetadataProjection(Protocol):
    """Converts typed metadata snapshots into external key/value metadata."""

    projection_id: str
    required_facts: tuple[MetadataRequiredFact, ...]

    def project(self, snapshot) -> ProjectionResult:
        """Project collected metadata into one external representation."""


def stringify_metadata_value(value: MetadataValue) -> str:
    """Convert a metadata value into a stable string for artifact metadata."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list | tuple | dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def stringify_metadata_mapping(values: Mapping[str, MetadataValue]) -> dict[str, str]:
    """Convert a metadata mapping to string-only metadata."""
    return {key: stringify_metadata_value(value) for key, value in values.items()}


def project_with_validation(projection: MetadataProjection, snapshot) -> ProjectionResult:
    """Validate projection requirements before producing exported metadata."""
    validate_required_facts(snapshot.records, projection.required_facts)
    return projection.project(snapshot)


@dataclass(frozen=True)
class KuroMetadataProjection:
    """Projection for repo-owned exported `kuro.*` metadata."""

    projection_id: str = "kuro"
    required_facts: tuple[MetadataRequiredFact, ...] = ()

    def project(self, snapshot) -> ProjectionResult:
        validate_required_facts(snapshot.records, self.required_facts)
        metadata = {KURO_SCHEMA_VERSION_KEY: KURO_SCHEMA_VERSION}
        for record in snapshot.records:
            entity_prefix = _kuro_entity_prefix(record.identity.entity_type)
            if entity_prefix is None:
                continue
            metadata[f"{entity_prefix}.id"] = record.identity.identifier
            metadata[f"{entity_prefix}.schema_version"] = record.identity.schema_version
            metadata[f"{entity_prefix}.producer"] = record.producer
            if record.identity.label is not None:
                metadata[f"{entity_prefix}.label"] = record.identity.label
            for key, value in record.facts.items():
                if _is_compatibility_key(key):
                    continue
                projected_key = key if key.startswith(KURO_PREFIX) else f"{entity_prefix}.{key}"
                metadata[projected_key] = stringify_metadata_value(value)
        return ProjectionResult(metadata)


@dataclass(frozen=True)
class SsCompatibilityProjection:
    """Projection for legacy Kohya `ss_*` compatibility metadata."""

    projection_id: str = "kohya_ss"
    required_facts: tuple[MetadataRequiredFact, ...] = ()
    minimum_keys: frozenset[str] | None = None

    def project(self, snapshot) -> ProjectionResult:
        validate_required_facts(snapshot.records, self.required_facts)
        metadata = _collect_prefixed_facts(snapshot.records, KOHYA_SS_PREFIX)
        if self.minimum_keys is not None:
            metadata = {key: value for key, value in metadata.items() if key in self.minimum_keys}
        return ProjectionResult(metadata)


@dataclass(frozen=True)
class ModelSpecCompatibilityProjection:
    """Projection for SAI Model Spec `modelspec.*` compatibility metadata."""

    projection_id: str = "modelspec"
    required_facts: tuple[MetadataRequiredFact, ...] = ()

    def project(self, snapshot) -> ProjectionResult:
        validate_required_facts(snapshot.records, self.required_facts)
        return ProjectionResult(_collect_prefixed_facts(snapshot.records, MODELSPEC_PREFIX))


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
        metadata: dict[str, str] = {}
        for projection in self.projections:
            metadata.update(project_with_validation(projection, snapshot).metadata)
        return ProjectionResult(metadata)


def _collect_prefixed_facts(records, prefix: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for record in records:
        metadata.update(
            {
                key: stringify_metadata_value(value)
                for key, value in record.facts.items()
                if key.startswith(prefix)
            }
        )
    return metadata


def _is_compatibility_key(key: str) -> bool:
    return key.startswith(KOHYA_SS_PREFIX) or key.startswith(MODELSPEC_PREFIX)


def _kuro_entity_prefix(entity_type: str) -> str | None:
    normalized = entity_type.replace("_", "-")
    supported_entities = {
        "adapter": "kuro.adapter",
        "artifact": "kuro.artifact",
        "model": "kuro.model",
        "model-component": "kuro.component",
        "run": "kuro.run",
    }
    return supported_entities.get(normalized)
