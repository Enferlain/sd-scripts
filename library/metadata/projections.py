"""Projection contracts for exported metadata shapes."""

from __future__ import annotations

import json

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from library.metadata.dataclasses.observability import ResourceMonitorFacts
from library.metadata.dataclasses.resource import ResourceObservationFrameFacts
from library.metadata.providers import MetadataRequiredFact
from library.metadata.records import MetadataRecord, MetadataValue
from library.metadata.validation import validate_required_facts
from library.metadata.views.resource import ResourceRunView

from library.metadata.keys import (
    KOHYA_SS_PREFIX,
    KURO_PREFIX,
    KURO_SCHEMA_VERSION,
    KURO_SCHEMA_VERSION_KEY,
    MODELSPEC_PREFIX,
)


_RESOURCE_MONITOR_COMPATIBILITY_KEYS = (
    "ts",
    "event",
    "rank",
    "world_size",
    "mode",
    "device_scope",
    "run_identifier",
    "config_name",
    "git_sha",
    "git_dirty",
    "global_step",
    "epoch",
    "phase",
    "duration_ms",
    "gpu_allocated_mb",
    "gpu_allocated_by_device_mb",
    "gpu_reserved_mb",
    "gpu_reserved_by_device_mb",
    "gpu_peak_allocated_mb",
    "gpu_peak_allocated_by_device_mb",
    "gpu_used_mb",
    "gpu_used_by_device_mb",
    "cpu_rss_mb",
    "cpu_vms_mb",
    "steps_per_sec",
    "samples_per_sec",
    "dropped_samples",
    "collection_ms",
    "deep_alloc_retries",
    "deep_ooms",
    "deep_active_mb",
    "deep_reserved_mb",
    "deep_inactive_split_mb",
    "deep_window_active",
)

_RESOURCE_FRAME_METADATA_KEYS = (
    "device_scope",
    "config_name",
    "git_sha",
    "git_dirty",
    "dropped_samples",
    "deep_window_active",
)

_RESOURCE_SCOPED_MEASUREMENT_KEYS = {
    ("gpu_memory", "allocated", "device_aggregate"): "gpu_allocated_mb",
    ("gpu_memory", "reserved", "device_aggregate"): "gpu_reserved_mb",
    ("gpu_memory", "peak_allocated", "device_aggregate"): "gpu_peak_allocated_mb",
    ("gpu_memory", "used_visible", "device_aggregate"): "gpu_used_mb",
    ("cpu_memory", "rss", "process"): "cpu_rss_mb",
    ("cpu_memory", "vms", "process"): "cpu_vms_mb",
    ("runtime", "duration", "event"): "duration_ms",
    ("throughput", "steps_per_second", "step_window"): "steps_per_sec",
    ("throughput", "samples_per_second", "step_window"): "samples_per_sec",
    ("collector_cost", "collection_duration", "collector"): "collection_ms",
    ("cuda_allocator_diagnostic", "alloc_retries", "diagnostic_window"): "deep_alloc_retries",
    ("cuda_allocator_diagnostic", "ooms", "diagnostic_window"): "deep_ooms",
    ("gpu_memory", "deep_active", "diagnostic_window"): "deep_active_mb",
    ("gpu_memory", "deep_reserved", "diagnostic_window"): "deep_reserved_mb",
    ("gpu_memory", "deep_inactive_split", "diagnostic_window"): "deep_inactive_split_mb",
}

_RESOURCE_DEVICE_MEASUREMENT_KEYS = {
    ("gpu_memory", "allocated"): "gpu_allocated_by_device_mb",
    ("gpu_memory", "reserved"): "gpu_reserved_by_device_mb",
    ("gpu_memory", "peak_allocated"): "gpu_peak_allocated_by_device_mb",
    ("gpu_memory", "used_visible"): "gpu_used_by_device_mb",
}

_RESOURCE_COUNT_FIELDS = {"deep_alloc_retries", "deep_ooms"}


@dataclass(frozen=True)
class ProjectionResult:
    """String metadata ready for storage boundaries such as safetensors."""

    metadata: dict[str, str] = field(default_factory=dict)


class MetadataProjection(Protocol):
    """Converts typed metadata snapshots into external key/value metadata."""

    projection_id: str
    required_facts: tuple[MetadataRequiredFact, ...]

    def project(self, snapshot) -> ProjectionResult:
        """Project collected metadata into one external representation."""


def stringify_metadata_value(value: MetadataValue) -> str:
    """Convert a metadata value into a stable string for artifact metadata."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list | tuple | dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def stringify_metadata_mapping(values: Mapping[str, MetadataValue]) -> dict[str, str]:
    """Convert a metadata mapping to string-only metadata."""
    return {key: stringify_metadata_value(value) for key, value in values.items()}


def project_with_validation(projection: MetadataProjection, snapshot) -> ProjectionResult:
    """Validate projection requirements before producing exported metadata."""
    validate_required_facts(snapshot.records, projection.required_facts)
    return projection.project(snapshot)


@dataclass(frozen=True)
class KuroMetadataProjection:
    """Projection for repo-owned exported `kuro.*` metadata."""

    projection_id: str = "kuro"
    required_facts: tuple[MetadataRequiredFact, ...] = ()

    def project(self, snapshot) -> ProjectionResult:
        validate_required_facts(snapshot.records, self.required_facts)
        metadata = {KURO_SCHEMA_VERSION_KEY: KURO_SCHEMA_VERSION}
        for record in snapshot.records:
            entity_prefix = _kuro_entity_prefix(record.identity.entity_type)
            if entity_prefix is None:
                continue
            metadata[f"{entity_prefix}.id"] = record.identity.identifier
            metadata[f"{entity_prefix}.schema_version"] = record.identity.schema_version
            metadata[f"{entity_prefix}.producer"] = record.producer
            if record.identity.label is not None:
                metadata[f"{entity_prefix}.label"] = record.identity.label
            for key, value in record.facts.items():
                if _is_compatibility_key(key):
                    continue
                projected_key = key if key.startswith(KURO_PREFIX) else f"{entity_prefix}.{key}"
                metadata[projected_key] = stringify_metadata_value(value)
        return ProjectionResult(metadata)


@dataclass(frozen=True)
class SsCompatibilityProjection:
    """Projection for legacy Kohya `ss_*` compatibility metadata."""

    projection_id: str = "kohya_ss"
    required_facts: tuple[MetadataRequiredFact, ...] = ()
    minimum_keys: frozenset[str] | None = None

    def project(self, snapshot) -> ProjectionResult:
        validate_required_facts(snapshot.records, self.required_facts)
        metadata = _collect_prefixed_facts(snapshot.records, KOHYA_SS_PREFIX)
        if self.minimum_keys is not None:
            metadata = {key: value for key, value in metadata.items() if key in self.minimum_keys}
        return ProjectionResult(metadata)


@dataclass(frozen=True)
class ModelSpecCompatibilityProjection:
    """Projection for SAI Model Spec `modelspec.*` compatibility metadata."""

    projection_id: str = "modelspec"
    required_facts: tuple[MetadataRequiredFact, ...] = ()

    def project(self, snapshot) -> ProjectionResult:
        validate_required_facts(snapshot.records, self.required_facts)
        return ProjectionResult(_collect_prefixed_facts(snapshot.records, MODELSPEC_PREFIX))


@dataclass(frozen=True)
class SafetensorsMetadataProjection:
    """Composite projection that produces one string-only safetensors metadata dict."""

    projections: tuple[MetadataProjection, ...]
    projection_id: str = "safetensors"
    required_facts: tuple[MetadataRequiredFact, ...] = ()

    @classmethod
    def from_sequence(
        cls,
        projections: Sequence[MetadataProjection],
        required_facts: Iterable[MetadataRequiredFact] = (),
    ) -> SafetensorsMetadataProjection:
        return cls(projections=tuple(projections), required_facts=tuple(required_facts))

    def project(self, snapshot) -> ProjectionResult:
        validate_required_facts(snapshot.records, self.required_facts)
        metadata: dict[str, str] = {}
        for projection in self.projections:
            metadata.update(project_with_validation(projection, snapshot).metadata)
        return ProjectionResult(metadata)


def project_resource_monitor_compatibility_event(
    facts: ResourceMonitorFacts | ResourceObservationFrameFacts,
) -> dict[str, Any]:
    """Project one accepted resource fact boundary into the current flat event shape."""
    if isinstance(facts, ResourceMonitorFacts):
        return _project_resource_monitor_facts(facts)

    event = _empty_resource_monitor_event()
    event.update(
        {
            "ts": facts.ts,
            "event": facts.event_name,
            "rank": facts.rank,
            "world_size": facts.world_size,
            "mode": facts.collection_policy,
            "run_identifier": facts.run_identifier,
            "global_step": facts.global_step,
            "epoch": facts.epoch,
            "phase": facts.phase,
        }
    )
    _project_resource_frame_metadata(event, facts.metadata)
    for measurement in facts.measurements:
        _project_resource_measurement(
            event,
            resource_kind=measurement.resource_kind,
            measurement_kind=measurement.measurement_kind,
            scope_type=measurement.scope_type,
            device_identifier=measurement.device_identifier,
            value=measurement.value,
        )
    return event


def project_resource_run_compatibility_events(view: ResourceRunView) -> tuple[dict[str, Any], ...]:
    """Project a resource-run view into ordered flat compatibility events."""
    return tuple(_project_resource_frame_record(view, frame) for frame in view.observation_frames())


def _project_resource_monitor_facts(facts: ResourceMonitorFacts) -> dict[str, Any]:
    event = _empty_resource_monitor_event()
    for key in _RESOURCE_MONITOR_COMPATIBILITY_KEYS:
        event[key] = facts.event_name if key == "event" else getattr(facts, key)
    return event


def _project_resource_frame_record(view: ResourceRunView, frame: MetadataRecord) -> dict[str, Any]:
    event = _empty_resource_monitor_event()
    event.update(
        {
            "ts": frame.facts.get("ts"),
            "event": frame.facts.get("event_name"),
            "rank": frame.facts.get("rank"),
            "world_size": frame.facts.get("world_size"),
            "mode": frame.facts.get("collection_policy"),
            "run_identifier": frame.facts.get("run_identifier"),
            "global_step": frame.facts.get("global_step"),
            "epoch": frame.facts.get("epoch"),
            "phase": frame.facts.get("phase"),
        }
    )
    frame_metadata = frame.facts.get("metadata")
    if isinstance(frame_metadata, Mapping):
        _project_resource_frame_metadata(event, cast(Mapping[str, object], frame_metadata))

    for measurement in view.measurements_for_frame(frame):
        resource_kind = measurement.facts.get("resource_kind")
        measurement_kind = measurement.facts.get("measurement_kind")
        value = measurement.facts.get("value")
        if not isinstance(resource_kind, str) or not isinstance(measurement_kind, str):
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        scope_type = measurement.facts.get("scope_type")
        device_identifier = measurement.facts.get("device_identifier")
        _project_resource_measurement(
            event,
            resource_kind=resource_kind,
            measurement_kind=measurement_kind,
            scope_type=scope_type if isinstance(scope_type, str) else None,
            device_identifier=device_identifier if isinstance(device_identifier, str) else None,
            value=float(value),
        )
    return event


def _project_resource_frame_metadata(event: dict[str, Any], metadata: Mapping[str, object]) -> None:
    for key in _RESOURCE_FRAME_METADATA_KEYS:
        if key in metadata:
            event[key] = metadata[key]


def _project_resource_measurement(
    event: dict[str, Any],
    *,
    resource_kind: str,
    measurement_kind: str,
    scope_type: str | None,
    device_identifier: str | None,
    value: float,
) -> None:
    if scope_type == "device":
        field_name = _RESOURCE_DEVICE_MEASUREMENT_KEYS.get((resource_kind, measurement_kind))
        if field_name is None or device_identifier is None:
            return
        by_device = event[field_name]
        if not isinstance(by_device, dict):
            by_device = {}
            event[field_name] = by_device
        by_device[_resource_device_key(device_identifier)] = value
        return

    field_name = _RESOURCE_SCOPED_MEASUREMENT_KEYS.get((resource_kind, measurement_kind, scope_type))
    if field_name is None:
        return
    event[field_name] = int(value) if field_name in _RESOURCE_COUNT_FIELDS and value.is_integer() else value


def _empty_resource_monitor_event() -> dict[str, Any]:
    return dict.fromkeys(_RESOURCE_MONITOR_COMPATIBILITY_KEYS)


def _resource_device_key(device_identifier: str) -> str:
    return device_identifier.removeprefix("cuda:") if device_identifier.startswith("cuda:") else device_identifier


def _collect_prefixed_facts(records, prefix: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for record in records:
        metadata.update(
            {
                key: stringify_metadata_value(value)
                for key, value in record.facts.items()
                if key.startswith(prefix)
            }
        )
    return metadata


def _is_compatibility_key(key: str) -> bool:
    return key.startswith(KOHYA_SS_PREFIX) or key.startswith(MODELSPEC_PREFIX)


def _kuro_entity_prefix(entity_type: str) -> str | None:
    normalized = entity_type.replace("_", "-")
    supported_entities = {
        "adapter": "kuro.adapter",
        "artifact": "kuro.artifact",
        "model": "kuro.model",
        "model-component": "kuro.component",
        "run": "kuro.run",
    }
    return supported_entities.get(normalized)
