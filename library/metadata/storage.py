"""Storage interfaces for metadata records, events, and relationships."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from library.metadata.records import MetadataEdge, MetadataEvent, MetadataRecord


@dataclass(frozen=True)
class MetadataStoreSnapshot:
    """Stored metadata read view."""

    records: tuple[MetadataRecord, ...] = ()
    events: tuple[MetadataEvent, ...] = ()
    edges: tuple[MetadataEdge, ...] = ()


class MetadataStore(Protocol):
    """Persistence boundary for metadata backends."""

    def save_record(self, record: MetadataRecord) -> None:
        """Persist a metadata record."""

    def save_event(self, event: MetadataEvent) -> None:
        """Persist a metadata event."""

    def save_edge(self, edge: MetadataEdge) -> None:
        """Persist a metadata relationship."""

    def snapshot(self) -> MetadataStoreSnapshot:
        """Return stored metadata."""


class InMemoryMetadataStore:
    """Dependency-free store for tests and first-slice runtime composition."""

    def __init__(self) -> None:
        self._records: list[MetadataRecord] = []
        self._events: list[MetadataEvent] = []
        self._edges: list[MetadataEdge] = []

    def save_record(self, record: MetadataRecord) -> None:
        self._records.append(record)

    def save_event(self, event: MetadataEvent) -> None:
        self._events.append(event)

    def save_edge(self, edge: MetadataEdge) -> None:
        self._edges.append(edge)

    def snapshot(self) -> MetadataStoreSnapshot:
        return MetadataStoreSnapshot(
            records=tuple(self._records),
            events=tuple(self._events),
            edges=tuple(self._edges),
        )
