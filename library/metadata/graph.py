"""Shared metadata graph vocabulary and construction helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from library.metadata.versions import METADATA_PAYLOAD_VERSION

from library.metadata.records import (
    MetadataEdge,
    MetadataIdentity,
    MetadataRecord,
    MetadataValue,
)


class MetadataGraphSnapshot(Protocol):
    """Public snapshot shape needed by metadata graph read helpers."""

    records: tuple[MetadataRecord, ...]
    edges: tuple[MetadataEdge, ...]

    def record_for(self, *, entity_type: str, identifier: str, namespace: str | None = None) -> MetadataRecord | None:
        """Return the newest matching record if it exists."""


@dataclass(frozen=True, slots=True)
class MetadataGraphIndex:
    """Reusable identity and edge index over one metadata snapshot."""

    snapshot: MetadataGraphSnapshot
    _records_by_key: Mapping[str, MetadataRecord] = field(repr=False)
    _records_by_type_identifier: Mapping[tuple[str, str], MetadataRecord] = field(repr=False)
    _edges_by_source: Mapping[str, tuple[MetadataEdge, ...]] = field(repr=False)
    _edges_by_target: Mapping[str, tuple[MetadataEdge, ...]] = field(repr=False)

    @classmethod
    def from_snapshot(cls, snapshot: MetadataGraphSnapshot | MetadataGraphIndex) -> MetadataGraphIndex:
        """Build lookup maps in one pass over snapshot records and edges."""
        if isinstance(snapshot, cls):
            return snapshot

        records_by_key: dict[str, MetadataRecord] = {}
        records_by_type_identifier: dict[tuple[str, str], MetadataRecord] = {}
        for record in snapshot.records:
            records_by_key[record.identity.key] = record
            records_by_type_identifier[(record.identity.entity_type, record.identity.identifier)] = record

        edges_by_source: defaultdict[str, list[MetadataEdge]] = defaultdict(list)
        edges_by_target: defaultdict[str, list[MetadataEdge]] = defaultdict(list)
        for edge in snapshot.edges:
            edges_by_source[edge.source.key].append(edge)
            edges_by_target[edge.target.key].append(edge)

        return cls(
            snapshot=snapshot,
            _records_by_key=records_by_key,
            _records_by_type_identifier=records_by_type_identifier,
            _edges_by_source={key: tuple(edges) for key, edges in edges_by_source.items()},
            _edges_by_target={key: tuple(edges) for key, edges in edges_by_target.items()},
        )

    @property
    def records(self) -> tuple[MetadataRecord, ...]:
        """Records retained by the indexed snapshot."""
        return self.snapshot.records

    @property
    def edges(self) -> tuple[MetadataEdge, ...]:
        """Edges retained by the indexed snapshot."""
        return self.snapshot.edges

    def record_for(
        self,
        *,
        entity_type: str,
        identifier: str,
        namespace: str | None = None,
    ) -> MetadataRecord | None:
        """Return a qualified match or the newest cross-namespace match when namespace is omitted."""
        if namespace is None:
            return self._records_by_type_identifier.get((entity_type, identifier))
        return self._records_by_key.get(f"{namespace}:{entity_type}:{identifier}")

    def edges_from(
        self,
        identity: MetadataIdentity,
        *,
        relationship: str | None = None,
        target_type: str | None = None,
        target_identifier: str | None = None,
        target_namespace: str | None = None,
    ) -> tuple[MetadataEdge, ...]:
        """Return indexed edges leaving an identity."""
        return tuple(
            edge
            for edge in self._edges_by_source.get(identity.key, ())
            if _matches_edge(
                edge,
                relationship=relationship,
                target_type=target_type,
                target_identifier=target_identifier,
                target_namespace=target_namespace,
            )
        )

    def edges_to(
        self,
        identity: MetadataIdentity,
        *,
        relationship: str | None = None,
        source_type: str | None = None,
        source_identifier: str | None = None,
        source_namespace: str | None = None,
    ) -> tuple[MetadataEdge, ...]:
        """Return indexed edges pointing at an identity."""
        return tuple(
            edge
            for edge in self._edges_by_target.get(identity.key, ())
            if _matches_edge(
                edge,
                relationship=relationship,
                source_type=source_type,
                source_identifier=source_identifier,
                source_namespace=source_namespace,
            )
        )


class MetadataEntityType(StrEnum):
    """Well-known metadata graph entity types shared across concerns.

    Domain emitters may still use custom entity-type strings when a scope is
    valid but not part of the shared vocabulary yet.
    """

    ARTIFACT = "artifact"
    COLLECTOR = "collector"
    COMPONENT = "component"
    DEVICE = "device"
    EVENT = "event"
    GROUP = "group"
    HOST = "host"
    PHASE = "phase"
    PROCESS = "process"
    RANK = "rank"
    RUN = "run"
    MODEL_REALIZATION = "model_realization"
    MODEL_COMPONENT = "model_component"
    MODEL_ARTIFACT = "model_artifact"
    MODEL_FAMILY_FACTS = "model_family_facts"
    STEP = "step"


class MetadataRelationship(StrEnum):
    """Well-known metadata graph relationship verbs shared across concerns.

    Domain emitters may still use custom relationship strings for evidence or
    compatibility edges that are intentionally more specific than these verbs.
    """

    ACCOUNTS_FOR = "accounts_for"
    CONTAINED_IN = "contained_in"
    DERIVED_FROM = "derived_from"
    DESCRIBES = "describes"
    GAP_FOR = "gap_for"
    OBSERVED_DURING = "observed_during"
    OBSERVED_IN = "observed_in"
    OBSERVED_ON = "observed_on"
    OWNED_BY = "owned_by"
    PRODUCED_BY = "produced_by"
    PROFILES = "profiles"
    REALIZED_IN = "realized_in"


def metadata_identity(
    *,
    entity_type: str,
    identifier: str,
    label: str | None = None,
    namespace: str | None = None,
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataIdentity:
    """Build a metadata identity from shared graph terms or custom strings."""
    kwargs: dict[str, str] = {}
    if namespace is not None:
        kwargs["namespace"] = namespace
    return MetadataIdentity(
        entity_type=entity_type,
        identifier=identifier,
        label=label,
        schema_version=schema_version,
        **kwargs,
    )


def metadata_edge(
    *,
    source: MetadataIdentity,
    target: MetadataIdentity,
    relationship: str,
    producer: str,
    facts: dict[str, MetadataValue] | None = None,
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataEdge:
    """Build a metadata edge from shared graph terms or custom strings."""
    return MetadataEdge(
        source=source,
        target=target,
        relationship=relationship,
        producer=producer,
        facts={} if facts is None else dict(facts),
        schema_version=schema_version,
    )


def edges_from(
    snapshot: MetadataGraphSnapshot | MetadataGraphIndex,
    identity: MetadataIdentity,
    *,
    relationship: str | None = None,
    target_type: str | None = None,
    target_identifier: str | None = None,
    target_namespace: str | None = None,
) -> tuple[MetadataEdge, ...]:
    """Return graph edges leaving an identity, optionally narrowed by target."""
    if isinstance(snapshot, MetadataGraphIndex):
        return snapshot.edges_from(
            identity,
            relationship=relationship,
            target_type=target_type,
            target_identifier=target_identifier,
            target_namespace=target_namespace,
        )
    return tuple(
        edge
        for edge in snapshot.edges
        if _same_identity(edge.source, identity)
        and _matches_edge(
            edge,
            relationship=relationship,
            target_type=target_type,
            target_identifier=target_identifier,
            target_namespace=target_namespace,
        )
    )


def edges_to(
    snapshot: MetadataGraphSnapshot | MetadataGraphIndex,
    identity: MetadataIdentity,
    *,
    relationship: str | None = None,
    source_type: str | None = None,
    source_identifier: str | None = None,
    source_namespace: str | None = None,
) -> tuple[MetadataEdge, ...]:
    """Return graph edges pointing at an identity, optionally narrowed by source."""
    if isinstance(snapshot, MetadataGraphIndex):
        return snapshot.edges_to(
            identity,
            relationship=relationship,
            source_type=source_type,
            source_identifier=source_identifier,
            source_namespace=source_namespace,
        )
    return tuple(
        edge
        for edge in snapshot.edges
        if _same_identity(edge.target, identity)
        and _matches_edge(
            edge,
            relationship=relationship,
            source_type=source_type,
            source_identifier=source_identifier,
            source_namespace=source_namespace,
        )
    )


def target_identities(
    snapshot: MetadataGraphSnapshot | MetadataGraphIndex,
    identity: MetadataIdentity,
    *,
    relationship: str | None = None,
    target_type: str | None = None,
) -> tuple[MetadataIdentity, ...]:
    """Return unique target identities reached from an identity."""
    return _unique_identities(
        edge.target
        for edge in edges_from(
            snapshot,
            identity,
            relationship=relationship,
            target_type=target_type,
        )
    )


def source_identities(
    snapshot: MetadataGraphSnapshot | MetadataGraphIndex,
    identity: MetadataIdentity,
    *,
    relationship: str | None = None,
    source_type: str | None = None,
) -> tuple[MetadataIdentity, ...]:
    """Return unique source identities pointing at an identity."""
    return _unique_identities(
        edge.source
        for edge in edges_to(
            snapshot,
            identity,
            relationship=relationship,
            source_type=source_type,
        )
    )


def records_by_identity(
    snapshot: MetadataGraphSnapshot | MetadataGraphIndex,
    identities: Iterable[MetadataIdentity],
) -> tuple[MetadataRecord, ...]:
    """Resolve identities to newest matching records, preserving input order."""
    records: list[MetadataRecord] = []
    seen: set[str] = set()
    for identity in identities:
        if identity.key in seen:
            continue
        seen.add(identity.key)
        record = snapshot.record_for(
            entity_type=identity.entity_type,
            identifier=identity.identifier,
            namespace=identity.namespace,
        )
        if record is not None:
            records.append(record)
    return tuple(records)


def _matches_edge(
    edge: MetadataEdge,
    *,
    relationship: str | None = None,
    source_type: str | None = None,
    source_identifier: str | None = None,
    source_namespace: str | None = None,
    target_type: str | None = None,
    target_identifier: str | None = None,
    target_namespace: str | None = None,
) -> bool:
    return (
        (relationship is None or edge.relationship == relationship)
        and (source_type is None or edge.source.entity_type == source_type)
        and (source_identifier is None or edge.source.identifier == source_identifier)
        and (source_namespace is None or edge.source.namespace == source_namespace)
        and (target_type is None or edge.target.entity_type == target_type)
        and (target_identifier is None or edge.target.identifier == target_identifier)
        and (target_namespace is None or edge.target.namespace == target_namespace)
    )


def _same_identity(left: MetadataIdentity, right: MetadataIdentity) -> bool:
    return left.key == right.key


def _unique_identities(identities: Iterable[MetadataIdentity]) -> tuple[MetadataIdentity, ...]:
    unique: list[MetadataIdentity] = []
    seen: set[str] = set()
    for identity in identities:
        if identity.key in seen:
            continue
        seen.add(identity.key)
        unique.append(identity)
    return tuple(unique)


__all__ = [
    "MetadataEntityType",
    "MetadataGraphIndex",
    "MetadataGraphSnapshot",
    "MetadataRelationship",
    "edges_from",
    "edges_to",
    "metadata_edge",
    "metadata_identity",
    "records_by_identity",
    "source_identities",
    "target_identities",
]
