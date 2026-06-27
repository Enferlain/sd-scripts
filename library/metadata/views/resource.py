"""Resource-intelligence read views over metadata snapshots."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast

from library.metadata.records import MetadataEdge, MetadataIdentity, MetadataRecord

from library.metadata.graph import (
    edges_from,
    metadata_identity,
    MetadataEntityType,
    MetadataGraphIndex,
    MetadataGraphSnapshot,
    MetadataRelationship,
    records_by_identity,
    source_identities,
)


_RESOURCE_ACCOUNTING_ENTITY = "resource_accounting"
_RESOURCE_ACCOUNTING_GAP_ENTITY = "resource_accounting_gap"
_RESOURCE_COLLECTOR_STATUS_ENTITY = "resource_collector_status"
_RESOURCE_OBSERVATION_ENTITY = "resource_observation"
_RESOURCE_OBSERVATION_FRAME_ENTITY = "resource_observation_frame"
_RESOURCE_PROFILE_ENTITY = "resource_profile"
_RESOURCE_STRUCTURAL_ENTITY = "resource_structural_fact"


@dataclass(frozen=True, slots=True)
class ResourceRunView:
    """Queryable accepted resource facts for one run."""

    snapshot: MetadataGraphSnapshot
    run_identity: MetadataIdentity
    _graph: MetadataGraphIndex = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_graph", MetadataGraphIndex.from_snapshot(self.snapshot))

    @classmethod
    def from_snapshot(
        cls,
        snapshot: MetadataGraphSnapshot,
        *,
        run_identifier: str,
        namespace: str | None = None,
    ) -> ResourceRunView:
        """Build a resource-run view from a public metadata snapshot."""
        return cls(
            snapshot=snapshot,
            run_identity=metadata_identity(
                entity_type=MetadataEntityType.RUN,
                identifier=run_identifier,
                namespace=namespace,
            ),
        )

    @property
    def run_identifier(self) -> str:
        """Identifier of the run represented by this view."""
        return self.run_identity.identifier

    def observation_frames(self) -> tuple[MetadataRecord, ...]:
        """Return observation-frame records linked to this run."""
        return self._records_linked_to_run(
            entity_type=_RESOURCE_OBSERVATION_FRAME_ENTITY,
            relationship=MetadataRelationship.OBSERVED_DURING,
        )

    def observations(self) -> tuple[MetadataRecord, ...]:
        """Return scalar observations and frame-contained measurements for this run."""
        direct_observations = self._records_linked_to_run(
            entity_type=_RESOURCE_OBSERVATION_ENTITY,
            relationship=MetadataRelationship.OBSERVED_DURING,
        )
        frame_measurements = records_by_identity(
            self._graph,
            (
                identity
                for frame in self.observation_frames()
                for identity in source_identities(
                    self._graph,
                    frame.identity,
                    relationship=MetadataRelationship.CONTAINED_IN,
                    source_type=_RESOURCE_OBSERVATION_ENTITY,
                )
            ),
        )
        return self._records_in_snapshot_order((*direct_observations, *frame_measurements))

    def structural_facts(self) -> tuple[MetadataRecord, ...]:
        """Return structural resource facts linked to this run."""
        return self._records_linked_to_run(
            entity_type=_RESOURCE_STRUCTURAL_ENTITY,
            relationship=MetadataRelationship.DESCRIBES,
        )

    def profiles(self) -> tuple[MetadataRecord, ...]:
        """Return resource profiles linked to this run."""
        return self._records_linked_to_run(
            entity_type=_RESOURCE_PROFILE_ENTITY,
            relationship=MetadataRelationship.PROFILES,
        )

    def accounting_statements(self) -> tuple[MetadataRecord, ...]:
        """Return resource accounting statements linked to this run."""
        return self._records_linked_to_run(
            entity_type=_RESOURCE_ACCOUNTING_ENTITY,
            relationship=MetadataRelationship.ACCOUNTS_FOR,
        )

    def accounting_gaps(self) -> tuple[MetadataRecord, ...]:
        """Return unresolved resource accounting gaps linked to this run."""
        return self._records_linked_to_run(
            entity_type=_RESOURCE_ACCOUNTING_GAP_ENTITY,
            relationship=MetadataRelationship.GAP_FOR,
        )

    def collector_statuses(self) -> tuple[MetadataRecord, ...]:
        """Return operational collector-status records linked to this run."""
        return self._records_linked_to_run(
            entity_type=_RESOURCE_COLLECTOR_STATUS_ENTITY,
            relationship=MetadataRelationship.OBSERVED_DURING,
        )

    def degraded_collector_statuses(self) -> tuple[MetadataRecord, ...]:
        """Return degraded collector-status records linked to this run."""
        return self._records_in_snapshot_order(
            status
            for status in self.collector_statuses()
            if status.facts.get("degraded") is True
        )

    def collector_statuses_for_collector(self, collector_id: str) -> tuple[MetadataRecord, ...]:
        """Return run collector-status records for one collector identity."""
        collector_identity = metadata_identity(
            entity_type=MetadataEntityType.COLLECTOR,
            identifier=collector_id,
            namespace=self.run_identity.namespace,
        )
        return self._records_in_view(
            records_by_identity(
                self._graph,
                source_identities(
                    self._graph,
                    collector_identity,
                    relationship=MetadataRelationship.DESCRIBES,
                    source_type=_RESOURCE_COLLECTOR_STATUS_ENTITY,
                ),
            ),
            allowed_records=self.collector_statuses(),
        )

    def collector_statuses_for_rank(self, rank: int) -> tuple[MetadataRecord, ...]:
        """Return run collector-status records for one distributed rank."""
        return self._records_in_snapshot_order(
            status
            for status in self.collector_statuses()
            if status.facts.get("rank") == rank
        )

    def observation_frames_for_phase(self, phase: str) -> tuple[MetadataRecord, ...]:
        """Return run observation frames linked to a phase name."""
        return self._records_in_snapshot_order(
            frame
            for frame in self.observation_frames()
            if any(
                edge.facts.get("phase") == phase
                for edge in edges_from(
                    self._graph,
                    frame.identity,
                    relationship=MetadataRelationship.OBSERVED_DURING,
                    target_type=MetadataEntityType.PHASE,
                )
            )
        )

    def observation_frames_for_step(self, global_step: int) -> tuple[MetadataRecord, ...]:
        """Return run observation frames linked to a global step."""
        return self._records_in_snapshot_order(
            frame
            for frame in self.observation_frames()
            if any(
                edge.facts.get("global_step") == global_step
                for edge in edges_from(
                    self._graph,
                    frame.identity,
                    relationship=MetadataRelationship.OBSERVED_DURING,
                    target_type=MetadataEntityType.STEP,
                )
            )
        )

    def observation_frames_for_rank(self, rank: int) -> tuple[MetadataRecord, ...]:
        """Return run observation frames linked to a distributed rank."""
        return self._records_in_snapshot_order(
            frame
            for frame in self.observation_frames()
            if any(
                edge.facts.get("rank") == rank
                for edge in edges_from(
                    self._graph,
                    frame.identity,
                    relationship=MetadataRelationship.OBSERVED_IN,
                    target_type=MetadataEntityType.RANK,
                )
            )
        )

    def observations_for_device(self, device_identifier: str) -> tuple[MetadataRecord, ...]:
        """Return run observations linked to a device identity."""
        device_identity = metadata_identity(
            entity_type=MetadataEntityType.DEVICE,
            identifier=device_identifier,
            namespace=self.run_identity.namespace,
        )
        return self._records_in_view(
            records_by_identity(
                self._graph,
                source_identities(
                    self._graph,
                    device_identity,
                    relationship=MetadataRelationship.OBSERVED_ON,
                    source_type=_RESOURCE_OBSERVATION_ENTITY,
                ),
            ),
            allowed_records=self.observations(),
        )

    def structural_facts_for_component(self, component_identifier: str) -> tuple[MetadataRecord, ...]:
        """Return run structural facts linked to a domain-provided component identifier."""
        component_identity = metadata_identity(
            entity_type=MetadataEntityType.COMPONENT,
            identifier=component_identifier,
            namespace=self.run_identity.namespace,
        )
        return self._records_in_view(
            records_by_identity(
                self._graph,
                source_identities(
                    self._graph,
                    component_identity,
                    relationship=MetadataRelationship.DESCRIBES,
                    source_type=_RESOURCE_STRUCTURAL_ENTITY,
                ),
            ),
            allowed_records=self.structural_facts(),
        )

    def measurements_for_frame(self, frame: MetadataRecord) -> tuple[MetadataRecord, ...]:
        """Return observation measurements contained in a frame record."""
        if frame.identity.entity_type != _RESOURCE_OBSERVATION_FRAME_ENTITY:
            return ()

        belongs_to_run = any(
            edge.target.key == self.run_identity.key
            for edge in edges_from(
                self._graph,
                frame.identity,
                relationship=MetadataRelationship.OBSERVED_DURING,
                target_type=MetadataEntityType.RUN,
            )
        )
        if not belongs_to_run:
            return ()

        measurements = records_by_identity(
            self._graph,
            source_identities(
                self._graph,
                frame.identity,
                relationship=MetadataRelationship.CONTAINED_IN,
                source_type=_RESOURCE_OBSERVATION_ENTITY,
            ),
        )
        measurement_identifiers = frame.facts.get("measurement_identifiers")
        if not isinstance(measurement_identifiers, list):
            return measurements

        records_by_identifier = {
            measurement.identity.identifier: measurement for measurement in measurements
        }
        ordered = [
            records_by_identifier.pop(identifier)
            for identifier in measurement_identifiers
            if isinstance(identifier, str) and identifier in records_by_identifier
        ]
        ordered.extend(
            measurement
            for measurement in measurements
            if measurement.identity.identifier in records_by_identifier
        )
        return tuple(ordered)

    def source_records_for(self, record: MetadataRecord) -> tuple[MetadataRecord, ...]:
        """Resolve explicitly declared source records, including cross-run and artifact evidence."""
        references = _source_fact_references(record)
        referenced_identities = tuple(
            edge.target
            for edge in edges_from(self._graph, record.identity)
            if _edge_matches_source_reference(edge, references, default_namespace=record.identity.namespace)
        )
        return records_by_identity(self._graph, referenced_identities)

    def resource_records(self) -> tuple[MetadataRecord, ...]:
        """Return all resource fact records visible for this run."""
        return self._records_in_snapshot_order(
            (
                *self.observation_frames(),
                *self.observations(),
                *self.structural_facts(),
                *self.profiles(),
                *self.accounting_statements(),
                *self.accounting_gaps(),
            )
        )

    def _records_linked_to_run(self, *, entity_type: str, relationship: str) -> tuple[MetadataRecord, ...]:
        return records_by_identity(
            self._graph,
            source_identities(
                self._graph,
                self.run_identity,
                relationship=relationship,
                source_type=entity_type,
            ),
        )

    def _records_in_view(
        self,
        records: tuple[MetadataRecord, ...],
        *,
        allowed_records: tuple[MetadataRecord, ...],
    ) -> tuple[MetadataRecord, ...]:
        allowed_keys = {record.identity.key for record in allowed_records}
        return self._records_in_snapshot_order(
            record
            for record in records
            if record.identity.key in allowed_keys
        )

    def _records_in_snapshot_order(self, records: Iterable[MetadataRecord]) -> tuple[MetadataRecord, ...]:
        """Return selected records in their original snapshot order."""
        keys = {record.identity.key for record in records}
        return tuple(record for record in self.snapshot.records if record.identity.key in keys)


def _source_fact_references(record: MetadataRecord) -> tuple[dict[str, object], ...]:
    references = record.facts.get("source_fact_references")
    if not isinstance(references, list):
        return ()
    return tuple(cast(dict[str, object], reference) for reference in references if isinstance(reference, dict))


def _edge_matches_source_reference(
    edge: MetadataEdge,
    references: tuple[dict[str, object], ...],
    *,
    default_namespace: str,
) -> bool:
    return any(
        reference.get("entity_type") == edge.target.entity_type
        and reference.get("identifier") == edge.target.identifier
        and reference.get("relationship") == edge.relationship
        and reference.get("namespace", default_namespace) == edge.target.namespace
        for reference in references
    )


__all__ = [
    "ResourceRunView",
]
