"""Resource-intelligence read views over metadata snapshots."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from library.metadata.records import MetadataIdentity, MetadataRecord

from library.metadata.graph import (
    edges_from,
    metadata_identity,
    MetadataEntityType,
    MetadataGraphSnapshot,
    MetadataRelationship,
    records_by_identity,
    source_identities,
)


_RESOURCE_ACCOUNTING_ENTITY = "resource_accounting"
_RESOURCE_ACCOUNTING_GAP_ENTITY = "resource_accounting_gap"
_RESOURCE_OBSERVATION_ENTITY = "resource_observation"
_RESOURCE_OBSERVATION_FRAME_ENTITY = "resource_observation_frame"
_RESOURCE_PROFILE_ENTITY = "resource_profile"
_RESOURCE_STRUCTURAL_ENTITY = "resource_structural_fact"
_RESOURCE_FACT_ENTITY_TYPES = {
    _RESOURCE_ACCOUNTING_ENTITY,
    _RESOURCE_ACCOUNTING_GAP_ENTITY,
    _RESOURCE_OBSERVATION_ENTITY,
    _RESOURCE_OBSERVATION_FRAME_ENTITY,
    _RESOURCE_PROFILE_ENTITY,
    _RESOURCE_STRUCTURAL_ENTITY,
}


@dataclass(frozen=True, slots=True)
class ResourceRunView:
    """Queryable accepted resource facts for one run."""

    snapshot: MetadataGraphSnapshot
    run_identity: MetadataIdentity

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
            self.snapshot,
            (
                identity
                for frame in self.observation_frames()
                for identity in source_identities(
                    self.snapshot,
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

    def observation_frames_for_phase(self, phase: str) -> tuple[MetadataRecord, ...]:
        """Return run observation frames linked to a phase name."""
        return self._records_in_snapshot_order(
            frame
            for frame in self.observation_frames()
            if any(
                edge.facts.get("phase") == phase
                for edge in edges_from(
                    self.snapshot,
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
                    self.snapshot,
                    frame.identity,
                    relationship=MetadataRelationship.OBSERVED_DURING,
                    target_type=MetadataEntityType.STEP,
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
                self.snapshot,
                source_identities(
                    self.snapshot,
                    device_identity,
                    relationship=MetadataRelationship.OBSERVED_ON,
                    source_type=_RESOURCE_OBSERVATION_ENTITY,
                ),
            ),
            allowed_records=self.observations(),
        )

    def measurements_for_frame(self, frame: MetadataRecord) -> tuple[MetadataRecord, ...]:
        """Return observation measurements contained in a frame record."""
        return self._records_in_view(
            records_by_identity(
                self.snapshot,
                source_identities(
                    self.snapshot,
                    frame.identity,
                    relationship=MetadataRelationship.CONTAINED_IN,
                    source_type=_RESOURCE_OBSERVATION_ENTITY,
                ),
            ),
            allowed_records=self.observations(),
        )

    def source_records_for(self, record: MetadataRecord) -> tuple[MetadataRecord, ...]:
        """Return evidence records referenced as targets by a derived resource record."""
        referenced_identities = tuple(
            edge.target
            for edge in edges_from(self.snapshot, record.identity)
            if edge.target.entity_type in _RESOURCE_FACT_ENTITY_TYPES
        )
        return self._records_in_view(
            records_by_identity(self.snapshot, referenced_identities),
            allowed_records=self.resource_records(),
        )

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
            self.snapshot,
            source_identities(
                self.snapshot,
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


__all__ = [
    "ResourceRunView",
]
