"""Typed resource-intelligence metadata fact shapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from library.metadata.records import MetadataValue


@dataclass(frozen=True, slots=True)
class ResourceFactReference:
    """Reference to another accepted metadata fact used as evidence."""

    entity_type: str
    identifier: str
    relationship: str = "derived_from"
    namespace: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceObservationFacts:
    """Measured resource value at a known runtime scope."""

    observation_identifier: str
    run_identifier: str
    resource_kind: str
    measurement_kind: str
    value: float
    unit: str
    source: str
    ts: float | None = None
    collector_id: str | None = None
    scope_type: str | None = None
    phase: str | None = None
    global_step: int | None = None
    epoch: int | None = None
    rank: int | None = None
    world_size: int | None = None
    host_identifier: str | None = None
    process_identifier: str | None = None
    device_identifier: str | None = None
    quality: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceObservationMeasurementFacts:
    """One individually addressable measurement within an observation frame."""

    measurement_identifier: str
    resource_kind: str
    measurement_kind: str
    value: float
    unit: str
    source: str
    collector_id: str | None = None
    scope_type: str | None = None
    device_identifier: str | None = None
    quality: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceObservationFrameFacts:
    """Shared runtime and collection context for co-collected observations."""

    frame_identifier: str
    run_identifier: str
    event_name: str
    measurements: tuple[ResourceObservationMeasurementFacts, ...]
    ts: float | None = None
    collector_id: str | None = None
    collection_policy: str | None = None
    phase: str | None = None
    global_step: int | None = None
    epoch: int | None = None
    rank: int | None = None
    world_size: int | None = None
    host_identifier: str | None = None
    process_identifier: str | None = None
    quality: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceCollectorStatusFacts:
    """Operational status for a resource collector invocation."""

    status_identifier: str
    run_identifier: str
    collector_id: str
    status: str
    degraded: bool
    reason: str
    event_name: str | None = None
    ts: float | None = None
    rank: int | None = None
    world_size: int | None = None
    message: str | None = None
    fallback_collector_id: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StructuralResourceFacts:
    """Known resource-bearing object or state size."""

    structural_identifier: str
    run_identifier: str
    owner_type: str
    owner_identifier: str
    resource_kind: str
    quantity: float
    unit: str
    basis: str
    source: str
    component_key: str | None = None
    group_identifier: str | None = None
    validity_scope: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceProfileFacts:
    """Versioned derived profile for one run or workload shape."""

    profile_identifier: str
    run_identifier: str
    profile_kind: str
    derivation_version: str
    values: Mapping[str, MetadataValue]
    source_fact_references: tuple[ResourceFactReference, ...]
    generated_at: float | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceAccountingFacts:
    """Evidence-constrained resource accounting statement."""

    accounting_identifier: str
    run_identifier: str
    resource_kind: str
    quantity: float
    unit: str
    owner_type: str
    owner_identifier: str
    basis: str
    derivation_method: str
    derivation_version: str
    source_fact_references: tuple[ResourceFactReference, ...]
    validity_scope: str | None = None
    window_start: str | None = None
    window_end: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceAccountingGapFacts:
    """Explicit unresolved gap left by resource accounting."""

    gap_identifier: str
    run_identifier: str
    resource_kind: str
    quantity: float
    unit: str
    basis: str
    derivation_method: str
    derivation_version: str
    source_fact_references: tuple[ResourceFactReference, ...]
    scope: str | None = None
    reason: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)
