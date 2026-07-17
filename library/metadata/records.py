"""Typed metadata records, events, and relationships."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TypeAlias

from library.metadata.keys import DEFAULT_METADATA_NAMESPACE
from library.metadata.versions import METADATA_PAYLOAD_VERSION

PrimitiveMetadataValue: TypeAlias = str | int | float | bool | None
MetadataValue: TypeAlias = PrimitiveMetadataValue | list[object] | tuple[object, ...] | dict[str, object]


@dataclass(frozen=True)
class MetadataIdentity:
    """Identity of the entity a metadata record describes."""

    entity_type: str
    identifier: str
    namespace: str = DEFAULT_METADATA_NAMESPACE
    label: str | None = None
    schema_version: str = METADATA_PAYLOAD_VERSION

    @property
    def key(self) -> str:
        """Stable backend key for this identity."""
        return f"{self.namespace}:{self.entity_type}:{self.identifier}"


@dataclass(frozen=True)
class MetadataRecord:
    """Typed snapshot facts for one metadata entity."""

    identity: MetadataIdentity
    producer: str
    facts: dict[str, MetadataValue] = field(default_factory=dict)
    schema_version: str = METADATA_PAYLOAD_VERSION

    def has_fact(self, key: str) -> bool:
        """Return true when a fact exists and has a non-empty value."""
        value = self.facts.get(key)
        return value is not None and value != ""

    def with_facts(self, **facts: MetadataValue) -> MetadataRecord:
        """Return a copy with additional facts."""
        return replace(self, facts={**self.facts, **facts})


@dataclass(frozen=True)
class RunMetadataRecord(MetadataRecord):
    """Run/session-level metadata facts."""


@dataclass(frozen=True)
class ModelComponentMetadataRecord(MetadataRecord):
    """Loaded model-component metadata facts."""


@dataclass(frozen=True)
class ModelRealizationMetadataRecord(MetadataRecord):
    """Run-scoped model realization metadata facts."""


@dataclass(frozen=True)
class ModelFamilyContributionMetadataRecord(MetadataRecord):
    """Namespaced and versioned family-local metadata facts."""


@dataclass(frozen=True)
class AdapterMetadataRecord(MetadataRecord):
    """Adapter method or adapter artifact metadata facts."""


@dataclass(frozen=True)
class ArtifactMetadataRecord(MetadataRecord):
    """Durable artifact metadata facts."""


@dataclass(frozen=True)
class MetadataEvent:
    """Event describing something that happened to or around an entity."""

    event_type: str
    identity: MetadataIdentity
    producer: str
    facts: dict[str, MetadataValue] = field(default_factory=dict)
    schema_version: str = METADATA_PAYLOAD_VERSION


@dataclass(frozen=True)
class MetadataEdge:
    """Relationship between two metadata entities."""

    source: MetadataIdentity
    target: MetadataIdentity
    relationship: str
    producer: str
    facts: dict[str, MetadataValue] = field(default_factory=dict)
    schema_version: str = METADATA_PAYLOAD_VERSION
