"""Versioned resource export projections from accepted metadata facts."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from library.metadata.dataclasses.observability import ResourceMonitorFacts
from library.metadata.dataclasses.resource import ResourceObservationFrameFacts
from library.metadata.records import MetadataIdentity, MetadataRecord
from library.metadata.views.resource import ResourceRunView


RESOURCE_EXPORT_SCHEMA_VERSION = "1"
RESOURCE_MONITOR_JSONL_SCHEMA = "resource_monitor.compatibility_jsonl"
RESOURCE_REPORT_EXPORT_SCHEMA = "resource_intelligence.report"
RESOURCE_PROFILE_EXPORT_SCHEMA = "resource_intelligence.profile"
RESOURCE_ACCOUNTING_EXPORT_SCHEMA = "resource_intelligence.accounting"

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


@dataclass(frozen=True, slots=True)
class ResourceExportProjection:
    """One versioned resource export plus the accepted records it projects."""

    schema_name: str
    run_identifier: str | None
    payload: dict[str, Any]
    schema_version: str = RESOURCE_EXPORT_SCHEMA_VERSION
    source_identities: tuple[MetadataIdentity, ...] = field(default_factory=tuple)

    def document(self) -> dict[str, Any]:
        """Return a standalone export document with an explicit schema envelope."""
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "run_identifier": self.run_identifier,
            **self.payload,
        }


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


def project_resource_report(
    view: ResourceRunView,
    *,
    jsonl_path: str | None = None,
    input_source: str = "metadata_view",
    total_jsonl_event_count: int | None = None,
) -> ResourceExportProjection:
    """Project accepted observations into the resource section of a run report."""
    frames = view.observation_frames()
    events = tuple(_project_resource_frame_record(view, frame) for frame in frames)
    profiles = view.profiles()
    return _project_resource_report_events(
        events,
        run_identifier=view.run_identifier,
        jsonl_path=jsonl_path,
        input_source=input_source,
        total_resource_event_count=len(events),
        total_jsonl_event_count=total_jsonl_event_count,
        profile_views=tuple(_project_derived_record(view, profile) for profile in profiles),
        source_identities=_unique_identities(
            (
                *(record.identity for record in (*frames, *view.observations())),
                *_derived_export_source_identities(view, profiles),
            )
        ),
    )


def project_resource_report_compatibility_events(
    events: Sequence[Mapping[str, Any]],
    *,
    run_identifier: str | None,
    jsonl_path: str | None,
    input_source: str,
    total_resource_event_count: int | None = None,
    total_jsonl_event_count: int | None = None,
) -> ResourceExportProjection:
    """Project explicit compatibility events when accepted observation frames are unavailable."""
    normalized_events = tuple(dict(event) for event in events)
    return _project_resource_report_events(
        normalized_events,
        run_identifier=run_identifier,
        jsonl_path=jsonl_path,
        input_source=input_source,
        total_resource_event_count=len(normalized_events) if total_resource_event_count is None else total_resource_event_count,
        total_jsonl_event_count=total_jsonl_event_count,
        profile_views=(),
    )


def project_resource_profile_export(view: ResourceRunView) -> ResourceExportProjection:
    """Project already-derived durable resource profiles without deriving new values."""
    profiles = view.profiles()
    projected_profiles = tuple(_project_derived_record(view, record) for record in profiles)
    return ResourceExportProjection(
        schema_name=RESOURCE_PROFILE_EXPORT_SCHEMA,
        run_identifier=view.run_identifier,
        payload={"profiles": list(projected_profiles)},
        source_identities=_derived_export_source_identities(view, profiles),
    )


def project_resource_accounting_export(view: ResourceRunView) -> ResourceExportProjection:
    """Project accepted accounting statements and gaps as visibly distinct collections."""
    statements = view.accounting_statements()
    gaps = view.accounting_gaps()
    records = (*statements, *gaps)
    return ResourceExportProjection(
        schema_name=RESOURCE_ACCOUNTING_EXPORT_SCHEMA,
        run_identifier=view.run_identifier,
        payload={
            "statements": [_project_derived_record(view, record) for record in statements],
            "gaps": [_project_derived_record(view, record) for record in gaps],
        },
        source_identities=_derived_export_source_identities(view, records),
    )


def _project_resource_report_events(
    events: Sequence[dict[str, Any]],
    *,
    run_identifier: str | None,
    jsonl_path: str | None,
    input_source: str,
    total_resource_event_count: int,
    total_jsonl_event_count: int | None,
    profile_views: Sequence[dict[str, Any]] = (),
    source_identities: tuple[MetadataIdentity, ...] = (),
) -> ResourceExportProjection:
    event_list = list(events)
    phase_rows = _pair_phase_events(event_list)
    session_start = next((event for event in event_list if event.get("event") == "session_start"), None)
    session_end = next((event for event in reversed(event_list) if event.get("event") == "session_end"), None)
    gpu_used_peak_session_mb = max(
        (float(event["gpu_used_mb"]) for event in event_list if event.get("gpu_used_mb") is not None),
        default=None,
    )
    gpu_used_peak_session_by_device_mb = _peak_device_map(event_list, "gpu_used_by_device_mb")
    return ResourceExportProjection(
        schema_name=RESOURCE_REPORT_EXPORT_SCHEMA,
        run_identifier=run_identifier,
        payload={
            "jsonl_path": jsonl_path,
            "input_source": input_source,
            "event_count": len(event_list),
            "total_resource_event_count": total_resource_event_count,
            "total_jsonl_event_count": total_jsonl_event_count,
            "session_start": session_start,
            "session_end": session_end,
            "gpu_used_peak_session_mb": gpu_used_peak_session_mb,
            "gpu_used_peak_session_by_device_mb": gpu_used_peak_session_by_device_mb,
            "profile_views": list(profile_views),
            "phases": phase_rows,
            "debug": _build_resource_debug_summary(
                session_start=session_start,
                session_end=session_end,
                gpu_used_peak_session_by_device_mb=gpu_used_peak_session_by_device_mb,
                phase_rows=phase_rows,
            ),
        },
        source_identities=source_identities,
    )


def _project_derived_record(view: ResourceRunView, record: MetadataRecord) -> dict[str, Any]:
    return {
        "identity": _identity_payload(record.identity),
        "producer": record.producer,
        "facts": dict(record.facts),
        "resolved_source_identities": [_identity_payload(source.identity) for source in view.source_records_for(record)],
    }


def _derived_export_source_identities(
    view: ResourceRunView,
    records: Sequence[MetadataRecord],
) -> tuple[MetadataIdentity, ...]:
    return _unique_identities(
        identity for record in records for identity in (record.identity, *(source.identity for source in view.source_records_for(record)))
    )


def _unique_identities(identities: Iterable[MetadataIdentity]) -> tuple[MetadataIdentity, ...]:
    seen: set[str] = set()
    result: list[MetadataIdentity] = []
    for identity in identities:
        if identity.key in seen:
            continue
        seen.add(identity.key)
        result.append(identity)
    return tuple(result)


def _identity_payload(identity: MetadataIdentity) -> dict[str, Any]:
    return {
        "entity_type": identity.entity_type,
        "identifier": identity.identifier,
        "namespace": identity.namespace,
        "label": identity.label,
        "schema_version": identity.schema_version,
    }


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
        if not isinstance(value, int | float) or isinstance(value, bool):
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


def _pair_phase_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase_starts: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    phase_rows: list[dict[str, Any]] = []
    for event in events:
        phase_name = event.get("phase")
        if not phase_name:
            continue
        if event.get("event") == "phase_start":
            phase_starts[phase_name].append(event)
            continue
        if event.get("event") != "phase_end":
            continue
        start = phase_starts[phase_name].pop() if phase_starts[phase_name] else {}
        start_ts = start.get("ts")
        end_ts = event.get("ts")
        phase_samples = [
            sample
            for sample in events
            if sample.get("event") == "step_sample"
            and sample.get("phase") is None
            and sample.get("gpu_used_mb") is not None
            and isinstance(sample.get("ts"), int | float)
            and isinstance(start_ts, int | float)
            and isinstance(end_ts, int | float)
            and start_ts <= sample["ts"] <= end_ts
        ]
        gpu_used_peak_mb = max((float(sample["gpu_used_mb"]) for sample in phase_samples), default=None)
        gpu_used_peak_by_device_mb = _peak_device_map(phase_samples, "gpu_used_by_device_mb")
        phase_rows.append(
            {
                "phase": phase_name,
                "duration_s": (event.get("duration_ms") or 0.0) / 1000.0 if event.get("duration_ms") is not None else None,
                "gpu_allocated_start_mb": start.get("gpu_allocated_mb"),
                "gpu_allocated_start_by_device_mb": start.get("gpu_allocated_by_device_mb"),
                "gpu_allocated_end_mb": event.get("gpu_allocated_mb"),
                "gpu_allocated_end_by_device_mb": event.get("gpu_allocated_by_device_mb"),
                "gpu_reserved_start_mb": start.get("gpu_reserved_mb"),
                "gpu_reserved_start_by_device_mb": start.get("gpu_reserved_by_device_mb"),
                "gpu_reserved_end_mb": event.get("gpu_reserved_mb"),
                "gpu_reserved_end_by_device_mb": event.get("gpu_reserved_by_device_mb"),
                "gpu_peak_allocated_mb": event.get("gpu_peak_allocated_mb"),
                "gpu_peak_allocated_by_device_mb": event.get("gpu_peak_allocated_by_device_mb"),
                "gpu_used_peak_mb": gpu_used_peak_mb,
                "gpu_used_peak_by_device_mb": gpu_used_peak_by_device_mb,
                "cpu_rss_start_mb": start.get("cpu_rss_mb"),
                "cpu_rss_end_mb": event.get("cpu_rss_mb"),
                "cpu_vms_start_mb": start.get("cpu_vms_mb"),
                "cpu_vms_end_mb": event.get("cpu_vms_mb"),
            }
        )
    return phase_rows


def _peak_device_map(events: list[dict[str, Any]], field_name: str) -> dict[str, float] | None:
    peaks: dict[str, float] = {}
    for event in events:
        raw_map = event.get(field_name)
        if not isinstance(raw_map, dict):
            continue
        for device, value in raw_map.items():
            if not isinstance(device, str) or not isinstance(value, int | float):
                continue
            numeric_value = float(value)
            previous = peaks.get(device)
            if previous is None or numeric_value > previous:
                peaks[device] = numeric_value
    return peaks or None


def _device_map_to_rows(device_map: dict[str, float] | None, *, value_key: str) -> list[dict[str, Any]]:
    if not isinstance(device_map, dict):
        return []
    rows: list[dict[str, Any]] = []
    for device, value in sorted(device_map.items()):
        if not isinstance(device, str) or not isinstance(value, int | float):
            continue
        rows.append({"device": device, value_key: float(value)})
    return rows


def _delta_mb(start_value: Any, end_value: Any) -> float | None:
    if not isinstance(start_value, int | float) or not isinstance(end_value, int | float):
        return None
    return float(end_value) - float(start_value)


def _describe_delta(metric_name: str, delta_mb: float | None, *, threshold_mb: float = 1.0) -> str | None:
    if delta_mb is None or abs(delta_mb) < threshold_mb:
        return None
    direction = "higher" if delta_mb > 0 else "lower"
    return f"{metric_name} ended {abs(delta_mb):.0f} MB {direction} than it started."


def _build_resource_observations(
    *,
    gpu_allocated_start_mb: Any,
    gpu_allocated_end_mb: Any,
    gpu_reserved_start_mb: Any,
    gpu_reserved_end_mb: Any,
    gpu_peak_allocated_mb: Any,
    gpu_used_peak_mb: Any,
    cpu_rss_start_mb: Any,
    cpu_rss_end_mb: Any,
    cpu_vms_start_mb: Any,
    cpu_vms_end_mb: Any,
    threshold_mb: float = 1.0,
) -> list[str]:
    observations: list[str] = []
    gpu_allocated_delta_mb = _delta_mb(gpu_allocated_start_mb, gpu_allocated_end_mb)
    gpu_reserved_delta_mb = _delta_mb(gpu_reserved_start_mb, gpu_reserved_end_mb)
    cpu_rss_delta_mb = _delta_mb(cpu_rss_start_mb, cpu_rss_end_mb)
    cpu_vms_delta_mb = _delta_mb(cpu_vms_start_mb, cpu_vms_end_mb)

    for metric_name, delta_mb in (
        ("GPU allocated", gpu_allocated_delta_mb),
        ("GPU reserved", gpu_reserved_delta_mb),
        ("CPU RSS", cpu_rss_delta_mb),
        ("CPU VMS", cpu_vms_delta_mb),
    ):
        description = _describe_delta(metric_name, delta_mb, threshold_mb=threshold_mb)
        if description is not None:
            observations.append(description)

    if (
        isinstance(gpu_peak_allocated_mb, int | float)
        and isinstance(gpu_allocated_start_mb, int | float)
        and isinstance(gpu_allocated_end_mb, int | float)
        and float(gpu_peak_allocated_mb) > max(float(gpu_allocated_start_mb), float(gpu_allocated_end_mb)) + threshold_mb
    ):
        observations.append("GPU peak allocated exceeded both start and end snapshots, indicating a transient in-phase allocator peak.")

    if (
        isinstance(gpu_used_peak_mb, int | float)
        and isinstance(gpu_allocated_start_mb, int | float)
        and isinstance(gpu_allocated_end_mb, int | float)
        and float(gpu_used_peak_mb) > max(float(gpu_allocated_start_mb), float(gpu_allocated_end_mb)) + threshold_mb
    ):
        observations.append(
            "Sampled GPU used peaked above the allocator snapshots; this shows higher visible device usage during the window, not component-level ownership."
        )

    if (
        gpu_reserved_delta_mb is not None
        and gpu_allocated_delta_mb is not None
        and gpu_reserved_delta_mb > gpu_allocated_delta_mb + threshold_mb
    ):
        observations.append(
            "GPU reserved grew more than GPU allocated, so allocator capacity expanded beyond the end-of-window live allocation level."
        )
    return observations


def _build_phase_change_rows(phase_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "phase": row.get("phase"),
            "gpu_allocated_delta_mb": _delta_mb(row.get("gpu_allocated_start_mb"), row.get("gpu_allocated_end_mb")),
            "gpu_reserved_delta_mb": _delta_mb(row.get("gpu_reserved_start_mb"), row.get("gpu_reserved_end_mb")),
            "cpu_rss_delta_mb": _delta_mb(row.get("cpu_rss_start_mb"), row.get("cpu_rss_end_mb")),
            "cpu_vms_delta_mb": _delta_mb(row.get("cpu_vms_start_mb"), row.get("cpu_vms_end_mb")),
            "observations": _build_resource_observations(
                gpu_allocated_start_mb=row.get("gpu_allocated_start_mb"),
                gpu_allocated_end_mb=row.get("gpu_allocated_end_mb"),
                gpu_reserved_start_mb=row.get("gpu_reserved_start_mb"),
                gpu_reserved_end_mb=row.get("gpu_reserved_end_mb"),
                gpu_peak_allocated_mb=row.get("gpu_peak_allocated_mb"),
                gpu_used_peak_mb=row.get("gpu_used_peak_mb"),
                cpu_rss_start_mb=row.get("cpu_rss_start_mb"),
                cpu_rss_end_mb=row.get("cpu_rss_end_mb"),
                cpu_vms_start_mb=row.get("cpu_vms_start_mb"),
                cpu_vms_end_mb=row.get("cpu_vms_end_mb"),
            ),
        }
        for row in phase_rows
    ]


def _build_resource_debug_summary(
    *,
    session_start: dict[str, Any] | None,
    session_end: dict[str, Any] | None,
    gpu_used_peak_session_by_device_mb: dict[str, float] | None,
    phase_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    session_change_summary = {
        "gpu_allocated_delta_mb": _delta_mb(
            None if session_start is None else session_start.get("gpu_allocated_mb"),
            None if session_end is None else session_end.get("gpu_allocated_mb"),
        ),
        "gpu_reserved_delta_mb": _delta_mb(
            None if session_start is None else session_start.get("gpu_reserved_mb"),
            None if session_end is None else session_end.get("gpu_reserved_mb"),
        ),
        "cpu_rss_delta_mb": _delta_mb(
            None if session_start is None else session_start.get("cpu_rss_mb"),
            None if session_end is None else session_end.get("cpu_rss_mb"),
        ),
        "cpu_vms_delta_mb": _delta_mb(
            None if session_start is None else session_start.get("cpu_vms_mb"),
            None if session_end is None else session_end.get("cpu_vms_mb"),
        ),
        "observations": _build_resource_observations(
            gpu_allocated_start_mb=None if session_start is None else session_start.get("gpu_allocated_mb"),
            gpu_allocated_end_mb=None if session_end is None else session_end.get("gpu_allocated_mb"),
            gpu_reserved_start_mb=None if session_start is None else session_start.get("gpu_reserved_mb"),
            gpu_reserved_end_mb=None if session_end is None else session_end.get("gpu_reserved_mb"),
            gpu_peak_allocated_mb=None if session_end is None else session_end.get("gpu_peak_allocated_mb"),
            gpu_used_peak_mb=None if session_end is None else session_end.get("gpu_used_mb"),
            cpu_rss_start_mb=None if session_start is None else session_start.get("cpu_rss_mb"),
            cpu_rss_end_mb=None if session_end is None else session_end.get("cpu_rss_mb"),
            cpu_vms_start_mb=None if session_start is None else session_start.get("cpu_vms_mb"),
            cpu_vms_end_mb=None if session_end is None else session_end.get("cpu_vms_mb"),
        ),
    }
    return {
        "surface_split": {
            "live_metadata_fields": [
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
            ],
            "report_debug_only_fields": [
                "session_gpu_used_peak_by_device_rows",
                "phase_device_rows",
                "session_change_summary",
                "phase_change_rows",
            ],
        },
        "session_change_summary": session_change_summary,
        "phase_change_rows": _build_phase_change_rows(phase_rows),
        "session_gpu_used_peak_by_device_rows": _device_map_to_rows(
            gpu_used_peak_session_by_device_mb,
            value_key="gpu_used_peak_mb",
        ),
        "phase_device_rows": [
            {
                "phase": row.get("phase"),
                "gpu_allocated_start_by_device_mb": row.get("gpu_allocated_start_by_device_mb"),
                "gpu_allocated_end_by_device_mb": row.get("gpu_allocated_end_by_device_mb"),
                "gpu_reserved_start_by_device_mb": row.get("gpu_reserved_start_by_device_mb"),
                "gpu_reserved_end_by_device_mb": row.get("gpu_reserved_end_by_device_mb"),
                "gpu_peak_allocated_by_device_mb": row.get("gpu_peak_allocated_by_device_mb"),
                "gpu_used_peak_by_device_mb": row.get("gpu_used_peak_by_device_mb"),
            }
            for row in phase_rows
            if any(
                row.get(field) is not None
                for field in (
                    "gpu_allocated_start_by_device_mb",
                    "gpu_allocated_end_by_device_mb",
                    "gpu_reserved_start_by_device_mb",
                    "gpu_reserved_end_by_device_mb",
                    "gpu_peak_allocated_by_device_mb",
                    "gpu_used_peak_by_device_mb",
                )
            )
        ],
        "session_allocator_by_device": {
            "gpu_allocated_start_by_device_mb": None if session_start is None else session_start.get("gpu_allocated_by_device_mb"),
            "gpu_allocated_end_by_device_mb": None if session_end is None else session_end.get("gpu_allocated_by_device_mb"),
            "gpu_reserved_start_by_device_mb": None if session_start is None else session_start.get("gpu_reserved_by_device_mb"),
            "gpu_reserved_end_by_device_mb": None if session_end is None else session_end.get("gpu_reserved_by_device_mb"),
            "gpu_peak_allocated_end_by_device_mb": None if session_end is None else session_end.get("gpu_peak_allocated_by_device_mb"),
        },
    }
