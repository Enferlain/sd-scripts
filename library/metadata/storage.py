"""Storage interfaces for metadata records, events, and relationships."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from library.metadata.records import (
    AdapterMetadataRecord,
    ArtifactMetadataRecord,
    MetadataEdge,
    MetadataEvent,
    MetadataIdentity,
    MetadataRecord,
    ModelComponentMetadataRecord,
    RunMetadataRecord,
)

SCHEMA_VERSION = 1


class MetadataSchemaVersionError(RuntimeError):
    """Raised when a metadata store schema version is unsupported."""


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


def _as_text(value: str | Path) -> str:
    return str(value)


def _metadata_to_json(metadata: dict[str, object]) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def _metadata_from_json(metadata_json: str) -> dict[str, object]:
    return dict(json.loads(metadata_json))


def _identity_from_row(row: sqlite3.Row, prefix: str) -> MetadataIdentity:
    return MetadataIdentity(
        entity_type=row[f"{prefix}_entity_type"],
        identifier=row[f"{prefix}_identifier"],
        namespace=row[f"{prefix}_namespace"],
        label=row[f"{prefix}_label"],
        schema_version=row[f"{prefix}_schema_version"],
    )


def _record_type(record: MetadataRecord) -> str:
    if isinstance(record, RunMetadataRecord):
        return "run"
    if isinstance(record, ModelComponentMetadataRecord):
        return "model"
    if isinstance(record, AdapterMetadataRecord):
        return "adapter"
    if isinstance(record, ArtifactMetadataRecord):
        return "artifact"
    return "record"


def _record_from_row(row: sqlite3.Row) -> MetadataRecord:
    identity = _identity_from_row(row, "record")
    record_type = row["record_type"]
    record_cls: type[MetadataRecord]
    if record_type == "run":
        record_cls = RunMetadataRecord
    elif record_type == "model":
        record_cls = ModelComponentMetadataRecord
    elif record_type == "adapter":
        record_cls = AdapterMetadataRecord
    elif record_type == "artifact":
        record_cls = ArtifactMetadataRecord
    else:
        record_cls = MetadataRecord
    return record_cls(
        identity=identity,
        producer=row["producer"],
        facts=_metadata_from_json(row["facts_json"]),
        schema_version=row["schema_version"],
    )


def _event_from_row(row: sqlite3.Row) -> MetadataEvent:
    return MetadataEvent(
        event_type=row["event_type"],
        identity=_identity_from_row(row, "event"),
        producer=row["producer"],
        facts=_metadata_from_json(row["facts_json"]),
        schema_version=row["schema_version"],
    )


def _edge_from_row(row: sqlite3.Row) -> MetadataEdge:
    return MetadataEdge(
        source=_identity_from_row(row, "source"),
        target=_identity_from_row(row, "target"),
        relationship=row["relationship"],
        producer=row["producer"],
        facts=_metadata_from_json(row["facts_json"]),
        schema_version=row["schema_version"],
    )


def _migrate_schema(connection: sqlite3.Connection) -> None:
    current_version = connection.execute("PRAGMA user_version").fetchone()[0]
    if current_version > SCHEMA_VERSION:
        raise MetadataSchemaVersionError(
            f"Unsupported metadata schema version {current_version}; expected at most {SCHEMA_VERSION}"
        )
    if current_version == SCHEMA_VERSION:
        return

    with connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_type TEXT NOT NULL,
                record_entity_type TEXT NOT NULL,
                record_identifier TEXT NOT NULL,
                record_namespace TEXT NOT NULL,
                record_label TEXT,
                record_schema_version TEXT NOT NULL,
                producer TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                facts_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_entity_type TEXT NOT NULL,
                event_identifier TEXT NOT NULL,
                event_namespace TEXT NOT NULL,
                event_label TEXT,
                event_schema_version TEXT NOT NULL,
                producer TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                facts_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity_type TEXT NOT NULL,
                source_identifier TEXT NOT NULL,
                source_namespace TEXT NOT NULL,
                source_label TEXT,
                source_schema_version TEXT NOT NULL,
                target_entity_type TEXT NOT NULL,
                target_identifier TEXT NOT NULL,
                target_namespace TEXT NOT NULL,
                target_label TEXT,
                target_schema_version TEXT NOT NULL,
                relationship TEXT NOT NULL,
                producer TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                facts_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metadata_records_lookup
            ON metadata_records(record_entity_type, record_identifier, record_namespace, id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metadata_events_lookup
            ON metadata_events(event_entity_type, event_identifier, event_namespace, id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metadata_edges_lookup
            ON metadata_edges(source_entity_type, source_identifier, source_namespace, id)
            """
        )
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


class SQLiteMetadataStore:
    """SQLite-backed metadata store with versioned schema initialization.

    Schema versioning uses SQLite ``PRAGMA user_version``. Version 1 creates the
    initial record, event, and edge tables and preserves insertion-order
    snapshots.
    """

    def __init__(self, database: str | Path | sqlite3.Connection) -> None:
        if isinstance(database, sqlite3.Connection):
            self._connection = database
            self._owns_connection = False
        else:
            self._connection = sqlite3.connect(_as_text(database))
            self._owns_connection = True
        self._connection.row_factory = sqlite3.Row
        _migrate_schema(self._connection)

    @property
    def schema_version(self) -> int:
        """Return the supported schema version for this store."""
        return SCHEMA_VERSION

    def close(self) -> None:
        """Close the owned SQLite connection, if any."""
        if self._owns_connection:
            self._connection.close()
            self._owns_connection = False

    def save_record(self, record: MetadataRecord) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO metadata_records (
                    record_type,
                    record_entity_type,
                    record_identifier,
                    record_namespace,
                    record_label,
                    record_schema_version,
                    producer,
                    schema_version,
                    facts_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _record_type(record),
                    record.identity.entity_type,
                    record.identity.identifier,
                    record.identity.namespace,
                    record.identity.label,
                    record.identity.schema_version,
                    record.producer,
                    record.schema_version,
                    _metadata_to_json(record.facts),
                ),
            )

    def save_event(self, event: MetadataEvent) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO metadata_events (
                    event_type,
                    event_entity_type,
                    event_identifier,
                    event_namespace,
                    event_label,
                    event_schema_version,
                    producer,
                    schema_version,
                    facts_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_type,
                    event.identity.entity_type,
                    event.identity.identifier,
                    event.identity.namespace,
                    event.identity.label,
                    event.identity.schema_version,
                    event.producer,
                    event.schema_version,
                    _metadata_to_json(event.facts),
                ),
            )

    def save_edge(self, edge: MetadataEdge) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO metadata_edges (
                    source_entity_type,
                    source_identifier,
                    source_namespace,
                    source_label,
                    source_schema_version,
                    target_entity_type,
                    target_identifier,
                    target_namespace,
                    target_label,
                    target_schema_version,
                    relationship,
                    producer,
                    schema_version,
                    facts_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.source.entity_type,
                    edge.source.identifier,
                    edge.source.namespace,
                    edge.source.label,
                    edge.source.schema_version,
                    edge.target.entity_type,
                    edge.target.identifier,
                    edge.target.namespace,
                    edge.target.label,
                    edge.target.schema_version,
                    edge.relationship,
                    edge.producer,
                    edge.schema_version,
                    _metadata_to_json(edge.facts),
                ),
            )

    def snapshot(self) -> MetadataStoreSnapshot:
        records = tuple(
            _record_from_row(row)
            for row in self._connection.execute(
                """
                SELECT *
                FROM metadata_records
                ORDER BY id
                """
            )
        )
        events = tuple(
            _event_from_row(row)
            for row in self._connection.execute(
                """
                SELECT *
                FROM metadata_events
                ORDER BY id
                """
            )
        )
        edges = tuple(
            _edge_from_row(row)
            for row in self._connection.execute(
                """
                SELECT *
                FROM metadata_edges
                ORDER BY id
                """
            )
        )
        return MetadataStoreSnapshot(records=records, events=events, edges=edges)
