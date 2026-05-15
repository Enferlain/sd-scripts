"""Provider contracts for domain-owned metadata producers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from library.metadata.records import MetadataEdge, MetadataEvent, MetadataRecord


@dataclass(frozen=True)
class MetadataRequiredFact:
    """A fact a provider or projection requires before an export boundary."""

    fact_key: str
    entity_type: str | None = None
    identifier: str | None = None
    namespace: str | None = None
    provider_id: str | None = None
    projection_id: str | None = None
    description: str | None = None

    @property
    def owner(self) -> str:
        """Return the producer/projection that declared the requirement."""
        return self.provider_id or self.projection_id or "metadata"


@dataclass(frozen=True)
class MetadataProviderResult:
    """Collected output from one domain-owned metadata provider."""

    provider_id: str
    schema_version: str = "1"
    records: tuple[MetadataRecord, ...] = field(default_factory=tuple)
    events: tuple[MetadataEvent, ...] = field(default_factory=tuple)
    edges: tuple[MetadataEdge, ...] = field(default_factory=tuple)
    required_facts: tuple[MetadataRequiredFact, ...] = field(default_factory=tuple)

    @classmethod
    def from_sequences(
        cls,
        *,
        provider_id: str,
        schema_version: str = "1",
        records: Sequence[MetadataRecord] = (),
        events: Sequence[MetadataEvent] = (),
        edges: Sequence[MetadataEdge] = (),
        required_facts: Sequence[MetadataRequiredFact] = (),
    ) -> MetadataProviderResult:
        """Create a provider result while freezing mutable input sequences."""
        return cls(
            provider_id=provider_id,
            schema_version=schema_version,
            records=tuple(records),
            events=tuple(events),
            edges=tuple(edges),
            required_facts=tuple(required_facts),
        )


class MetadataProvider(Protocol):
    """Domain-owned producer for typed metadata records and requirements."""

    provider_id: str
    schema_version: str

    def collect_metadata(self) -> MetadataProviderResult:
        """Collect this provider's current metadata facts."""
