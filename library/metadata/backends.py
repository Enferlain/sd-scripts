"""Metadata backend composition services."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from library.metadata.providers import MetadataProvider, MetadataProviderResult, MetadataRequiredFact
from library.metadata.records import MetadataEdge, MetadataEvent, MetadataRecord
from library.metadata.storage import InMemoryMetadataStore, MetadataStore
from library.metadata.validation import validate_required_facts


@dataclass(frozen=True)
class MetadataSnapshot:
    """Immutable-ish read view over collected metadata."""

    records: tuple[MetadataRecord, ...] = ()
    events: tuple[MetadataEvent, ...] = ()
    edges: tuple[MetadataEdge, ...] = ()
    required_facts: tuple[MetadataRequiredFact, ...] = ()

    def records_for(self, *, entity_type: str | None = None, namespace: str | None = None) -> tuple[MetadataRecord, ...]:
        """Return records filtered by identity attributes."""
        return tuple(
            record
            for record in self.records
            if (entity_type is None or record.identity.entity_type == entity_type)
            and (namespace is None or record.identity.namespace == namespace)
        )

    def record_for(self, *, entity_type: str, identifier: str, namespace: str | None = None) -> MetadataRecord | None:
        """Return the newest matching record if it exists."""
        for record in reversed(self.records):
            if record.identity.entity_type != entity_type:
                continue
            if record.identity.identifier != identifier:
                continue
            if namespace is not None and record.identity.namespace != namespace:
                continue
            return record
        return None


class MetadataBackend(Protocol):
    """Composition boundary used by callers that collect and project metadata."""

    def ingest(self, result: MetadataProviderResult) -> None:
        """Add provider output to the backend."""

    def ingest_provider(self, provider: MetadataProvider) -> MetadataProviderResult:
        """Collect and add one provider's output."""

    def snapshot(self) -> MetadataSnapshot:
        """Return a read view over collected metadata."""

    def validate(self, required_facts: Iterable[MetadataRequiredFact] = ()) -> None:
        """Validate collected metadata plus any additional requirements."""


class InMemoryMetadataBackend:
    """First-slice backend for tests and active checkpoint metadata composition."""

    def __init__(self, store: MetadataStore | None = None) -> None:
        self._store = store or InMemoryMetadataStore()
        self._required_facts: list[MetadataRequiredFact] = []

    def ingest(self, result: MetadataProviderResult) -> None:
        for record in result.records:
            self._store.save_record(record)
        for event in result.events:
            self._store.save_event(event)
        for edge in result.edges:
            self._store.save_edge(edge)
        self._required_facts.extend(result.required_facts)

    def ingest_provider(self, provider: MetadataProvider) -> MetadataProviderResult:
        result = provider.collect_metadata()
        self.ingest(result)
        return result

    def ingest_all(self, providers: Sequence[MetadataProvider]) -> list[MetadataProviderResult]:
        """Collect and add a sequence of providers."""
        return [self.ingest_provider(provider) for provider in providers]

    def snapshot(self) -> MetadataSnapshot:
        store_snapshot = self._store.snapshot()
        return MetadataSnapshot(
            records=store_snapshot.records,
            events=store_snapshot.events,
            edges=store_snapshot.edges,
            required_facts=tuple(dict.fromkeys(self._required_facts)),
        )

    def validate(self, required_facts: Iterable[MetadataRequiredFact] = ()) -> None:
        snapshot = self.snapshot()
        validate_required_facts(snapshot.records, (*snapshot.required_facts, *tuple(required_facts)))
