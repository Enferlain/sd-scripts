"""Resource-intelligence read views over metadata snapshots."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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
class ResourceProfileValueComparison:
    """Comparison for one value key shared by, or missing from, two profiles."""

    key: str
    baseline_value: object | None
    candidate_value: object | None
    status: str
    delta: float | None = None
    ratio: float | None = None
    regression_threshold: float | None = None
    regression: bool = False


@dataclass(frozen=True, slots=True)
class ResourceProfileComparison:
    """Neutral comparison between two stored resource profiles."""

    baseline_profile: MetadataIdentity
    candidate_profile: MetadataIdentity
    profile_kind: str | None
    derivation_version: str | None
    compatible: bool
    values: tuple[ResourceProfileValueComparison, ...]
    reason: str | None = None

    def regressions(self) -> tuple[ResourceProfileValueComparison, ...]:
        """Return value comparisons that exceeded caller-provided regression thresholds."""
        return tuple(value for value in self.values if value.regression)


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

    def profiles_for(
        self,
        *,
        profile_kind: str | None = None,
        derivation_version: str | None = None,
    ) -> tuple[MetadataRecord, ...]:
        """Return profiles matching the requested kind and derivation version."""
        return self._records_in_snapshot_order(
            profile
            for profile in self.profiles()
            if (profile_kind is None or profile.facts.get("profile_kind") == profile_kind)
            and (derivation_version is None or profile.facts.get("derivation_version") == derivation_version)
        )

    def latest_profile(
        self,
        *,
        profile_kind: str,
        derivation_version: str | None = None,
    ) -> MetadataRecord | None:
        """Return the latest matching profile, preferring numeric generated_at when present."""
        profiles = self.profiles_for(profile_kind=profile_kind, derivation_version=derivation_version)
        if not profiles:
            return None
        generated_profiles = tuple(profile for profile in profiles if _numeric_profile_fact(profile, "generated_at") is not None)
        if generated_profiles:
            return max(
                enumerate(generated_profiles),
                key=lambda item: (
                    cast(float, _numeric_profile_fact(item[1], "generated_at")),
                    item[0],
                ),
            )[1]
        return profiles[-1]

    def compare_profile_to(
        self,
        baseline_view: ResourceRunView,
        *,
        profile_kind: str,
        derivation_version: str | None = None,
        regression_thresholds: Mapping[str, float] | None = None,
    ) -> ResourceProfileComparison | None:
        """Compare this run's latest matching profile against a baseline run profile."""
        baseline_profile = baseline_view.latest_profile(
            profile_kind=profile_kind,
            derivation_version=derivation_version,
        )
        candidate_profile = self.latest_profile(
            profile_kind=profile_kind,
            derivation_version=derivation_version,
        )
        if baseline_profile is None or candidate_profile is None:
            return None
        return compare_resource_profiles(
            baseline_profile,
            candidate_profile,
            regression_thresholds=regression_thresholds,
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
        return self._records_in_snapshot_order(status for status in self.collector_statuses() if status.facts.get("degraded") is True)

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
        return self._records_in_snapshot_order(status for status in self.collector_statuses() if status.facts.get("rank") == rank)

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

        records_by_identifier = {measurement.identity.identifier: measurement for measurement in measurements}
        ordered = [
            records_by_identifier.pop(identifier)
            for identifier in measurement_identifiers
            if isinstance(identifier, str) and identifier in records_by_identifier
        ]
        ordered.extend(measurement for measurement in measurements if measurement.identity.identifier in records_by_identifier)
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
        return self._records_in_snapshot_order(record for record in records if record.identity.key in allowed_keys)

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


def compare_resource_profiles(
    baseline_profile: MetadataRecord,
    candidate_profile: MetadataRecord,
    *,
    regression_thresholds: Mapping[str, float] | None = None,
) -> ResourceProfileComparison:
    """Compare two stored profiles without deriving new profile meaning.

    Regression thresholds are absolute positive deltas. A value is marked as a
    regression only when ``candidate - baseline`` exceeds the supplied threshold
    for that value key.
    """
    baseline_kind = _string_profile_fact(baseline_profile, "profile_kind")
    candidate_kind = _string_profile_fact(candidate_profile, "profile_kind")
    baseline_version = _string_profile_fact(baseline_profile, "derivation_version")
    candidate_version = _string_profile_fact(candidate_profile, "derivation_version")
    if baseline_kind != candidate_kind:
        return ResourceProfileComparison(
            baseline_profile=baseline_profile.identity,
            candidate_profile=candidate_profile.identity,
            profile_kind=None,
            derivation_version=None,
            compatible=False,
            values=(),
            reason="profile_kind_mismatch",
        )
    if baseline_version != candidate_version:
        return ResourceProfileComparison(
            baseline_profile=baseline_profile.identity,
            candidate_profile=candidate_profile.identity,
            profile_kind=baseline_kind,
            derivation_version=None,
            compatible=False,
            values=(),
            reason="derivation_version_mismatch",
        )

    thresholds = {} if regression_thresholds is None else dict(regression_thresholds)
    baseline_values = _profile_values(baseline_profile)
    candidate_values = _profile_values(candidate_profile)
    value_keys = sorted((*baseline_values.keys(), *candidate_values.keys()))
    unique_value_keys = tuple(dict.fromkeys(value_keys))
    return ResourceProfileComparison(
        baseline_profile=baseline_profile.identity,
        candidate_profile=candidate_profile.identity,
        profile_kind=baseline_kind,
        derivation_version=baseline_version,
        compatible=True,
        values=tuple(
            _compare_profile_value(
                key,
                baseline_values=baseline_values,
                candidate_values=candidate_values,
                regression_threshold=thresholds.get(key),
            )
            for key in unique_value_keys
        ),
    )


def _compare_profile_value(
    key: str,
    *,
    baseline_values: Mapping[str, object],
    candidate_values: Mapping[str, object],
    regression_threshold: float | None,
) -> ResourceProfileValueComparison:
    missing = object()
    baseline_value = baseline_values.get(key, missing)
    candidate_value = candidate_values.get(key, missing)
    if baseline_value is missing:
        return ResourceProfileValueComparison(
            key=key,
            baseline_value=None,
            candidate_value=candidate_value,
            status="missing_baseline",
        )
    if candidate_value is missing:
        return ResourceProfileValueComparison(
            key=key,
            baseline_value=baseline_value,
            candidate_value=None,
            status="missing_candidate",
        )

    baseline_numeric = _numeric_value(baseline_value)
    candidate_numeric = _numeric_value(candidate_value)
    if baseline_numeric is None or candidate_numeric is None:
        return ResourceProfileValueComparison(
            key=key,
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            status="non_numeric",
        )

    delta = candidate_numeric - baseline_numeric
    ratio = None if baseline_numeric == 0.0 else candidate_numeric / baseline_numeric
    regression = regression_threshold is not None and delta > regression_threshold
    return ResourceProfileValueComparison(
        key=key,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        status="numeric",
        delta=delta,
        ratio=ratio,
        regression_threshold=regression_threshold,
        regression=regression,
    )


def _profile_values(profile: MetadataRecord) -> dict[str, object]:
    values = profile.facts.get("values")
    if not isinstance(values, dict):
        return {}
    return {str(key): value for key, value in values.items()}


def _numeric_profile_fact(profile: MetadataRecord, key: str) -> float | None:
    return _numeric_value(profile.facts.get(key))


def _numeric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _string_profile_fact(profile: MetadataRecord, key: str) -> str | None:
    value = profile.facts.get(key)
    return value if isinstance(value, str) else None


__all__ = [
    "compare_resource_profiles",
    "ResourceProfileComparison",
    "ResourceProfileValueComparison",
    "ResourceRunView",
]
