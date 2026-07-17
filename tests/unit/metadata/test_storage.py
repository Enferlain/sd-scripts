"""Unit tests for metadata storage backends."""

import sqlite3

import pytest

from library.metadata import (
    ArtifactMetadataRecord,
    InMemoryMetadataBackend,
    MetadataEdge,
    MetadataEvent,
    MetadataIdentity,
    MetadataProviderResult,
    MetadataStorageDecodeError,
    RunMetadataRecord,
    SQLiteMetadataStore,
)


@pytest.mark.unit
def test_sqlite_metadata_store_applies_schema_migration() -> None:
    connection = sqlite3.connect(":memory:")

    store = SQLiteMetadataStore(connection)

    assert store.schema_version == 1
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    assert {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'metadata_%'"
        )
    } == {"metadata_records", "metadata_events", "metadata_edges"}


@pytest.mark.unit
def test_sqlite_metadata_store_round_trips_records_events_and_edges() -> None:
    store = SQLiteMetadataStore(sqlite3.connect(":memory:"))

    run_identity = MetadataIdentity(entity_type="run", identifier="run-1", label="training run")
    artifact_identity = MetadataIdentity(
        entity_type="artifact",
        identifier="checkpoint-1",
        label="checkpoint artifact",
    )
    run_record = RunMetadataRecord(
        identity=run_identity,
        producer="tests.training",
        facts={"session_id": "123", "seed": 42, "flags": [True, False]},
    )
    artifact_record = ArtifactMetadataRecord(
        identity=artifact_identity,
        producer="tests.checkpoint",
        facts={"format": "safetensors", "payload": {"kind": "adapter"}},
    )
    event = MetadataEvent(
        event_type="artifact_written",
        identity=artifact_identity,
        producer="tests.checkpoint",
        facts={"step": 10, "epoch": 2},
    )
    edge = MetadataEdge(
        source=artifact_identity,
        target=run_identity,
        relationship="produced_by",
        producer="tests.checkpoint",
        facts={"primary": True},
    )

    store.save_record(run_record)
    store.save_record(artifact_record)
    store.save_event(event)
    store.save_edge(edge)

    snapshot = store.snapshot()

    assert snapshot.records == (run_record, artifact_record)
    assert snapshot.events == (event,)
    assert snapshot.edges == (edge,)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("facts_json", "message"),
    [
        ("not-json", "contain invalid JSON"),
        ("[]", "must decode to a JSON object"),
    ],
)
def test_sqlite_metadata_store_rejects_malformed_persisted_facts(
    facts_json: str,
    message: str,
) -> None:
    connection = sqlite3.connect(":memory:")
    store = SQLiteMetadataStore(connection)
    store.save_record(
        RunMetadataRecord(
            identity=MetadataIdentity(entity_type="run", identifier="run-corrupt"),
            producer="tests.training",
            facts={"session_id": "123"},
        )
    )
    with connection:
        connection.execute(
            "UPDATE metadata_records SET facts_json = ?",
            (facts_json,),
        )

    with pytest.raises(MetadataStorageDecodeError, match=message):
        store.snapshot()


@pytest.mark.unit
def test_in_memory_backend_can_use_sqlite_store() -> None:
    connection = sqlite3.connect(":memory:")
    backend = InMemoryMetadataBackend(store=SQLiteMetadataStore(connection))

    backend.ingest(
        MetadataProviderResult.from_sequences(
            provider_id="tests.provider",
            records=[
                RunMetadataRecord(
                    identity=MetadataIdentity(entity_type="run", identifier="run-1"),
                    producer="tests.training",
                    facts={"session_id": "123"},
                )
            ],
        )
    )

    snapshot = backend.snapshot()

    assert snapshot.record_for(entity_type="run", identifier="run-1") is not None
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
