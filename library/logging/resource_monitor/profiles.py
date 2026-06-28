"""Resource-domain derivation for durable resource profiles."""

from __future__ import annotations

from collections.abc import Iterable

from library.metadata.dataclasses.resource import ResourceFactReference, ResourceProfileFacts
from library.metadata.graph import MetadataRelationship
from library.metadata.records import MetadataRecord, MetadataValue
from library.metadata.views import ResourceRunView


RUN_RESOURCE_SUMMARY_PROFILE_KIND = "run_resource_summary"
RUN_RESOURCE_SUMMARY_DERIVATION_VERSION = "run_resource_summary_v1"


def build_run_resource_profile(
    view: ResourceRunView,
    *,
    generated_at: float | None = None,
) -> ResourceProfileFacts | None:
    """Derive the first versioned run resource profile from accepted facts."""
    frames = view.observation_frames()
    observations = view.observations()
    structural_facts = view.structural_facts()
    if not frames and not observations and not structural_facts:
        return None

    frame_by_identifier = {frame.identity.identifier: frame for frame in frames}
    values: dict[str, MetadataValue] = {
        "observation_frame_count": len(frames),
        "observation_measurement_count": len(observations),
        "structural_fact_count": len(structural_facts),
    }

    phase_names = _phase_names(frames)
    if phase_names:
        values["phase_count"] = len(phase_names)
        phase_name_values: list[object] = list(phase_names)
        values["phase_names"] = phase_name_values

    _add_boundary_values(values, observations, frame_by_identifier)
    _add_peak_values(values, observations)
    _add_structural_totals(values, structural_facts)

    source_records = _profile_source_records(
        frames=frames,
        observations=observations,
        structural_facts=structural_facts,
    )
    source_references = _source_references(source_records)

    return ResourceProfileFacts(
        profile_identifier=(f"{view.run_identifier}:profile:{RUN_RESOURCE_SUMMARY_PROFILE_KIND}:{RUN_RESOURCE_SUMMARY_DERIVATION_VERSION}"),
        run_identifier=view.run_identifier,
        profile_kind=RUN_RESOURCE_SUMMARY_PROFILE_KIND,
        derivation_version=RUN_RESOURCE_SUMMARY_DERIVATION_VERSION,
        values=values,
        source_fact_references=source_references,
        generated_at=generated_at,
        metadata={
            "derivation_owner": "library.logging.resource_monitor",
            "profile_scope": "run",
        },
    )


def _phase_names(frames: Iterable[MetadataRecord]) -> list[str]:
    names: list[str] = []
    for frame in frames:
        phase = frame.facts.get("phase")
        if isinstance(phase, str) and phase not in names:
            names.append(phase)
    return names


def _add_boundary_values(
    values: dict[str, MetadataValue],
    observations: Iterable[MetadataRecord],
    frame_by_identifier: dict[str, MetadataRecord],
) -> None:
    session_start_cpu_rss = _first_measurement(
        observations,
        frame_by_identifier=frame_by_identifier,
        event_name="session_start",
        resource_kind="cpu_memory",
        measurement_kind="rss",
    )
    session_end_cpu_rss = _last_measurement(
        observations,
        frame_by_identifier=frame_by_identifier,
        event_name="session_end",
        resource_kind="cpu_memory",
        measurement_kind="rss",
    )
    session_start_cpu_vms = _first_measurement(
        observations,
        frame_by_identifier=frame_by_identifier,
        event_name="session_start",
        resource_kind="cpu_memory",
        measurement_kind="vms",
    )
    session_end_cpu_vms = _last_measurement(
        observations,
        frame_by_identifier=frame_by_identifier,
        event_name="session_end",
        resource_kind="cpu_memory",
        measurement_kind="vms",
    )
    session_duration = _last_measurement(
        observations,
        frame_by_identifier=frame_by_identifier,
        event_name="session_end",
        resource_kind="runtime",
        measurement_kind="duration",
    )

    if session_start_cpu_rss is not None:
        values["cpu_rss_start_mib"] = _numeric_fact(session_start_cpu_rss, "value")
    if session_end_cpu_rss is not None:
        values["cpu_rss_end_mib"] = _numeric_fact(session_end_cpu_rss, "value")
    if session_start_cpu_rss is not None and session_end_cpu_rss is not None:
        values["cpu_rss_delta_mib"] = _numeric_fact(session_end_cpu_rss, "value") - _numeric_fact(
            session_start_cpu_rss,
            "value",
        )

    if session_start_cpu_vms is not None:
        values["cpu_vms_start_mib"] = _numeric_fact(session_start_cpu_vms, "value")
    if session_end_cpu_vms is not None:
        values["cpu_vms_end_mib"] = _numeric_fact(session_end_cpu_vms, "value")
    if session_start_cpu_vms is not None and session_end_cpu_vms is not None:
        values["cpu_vms_delta_mib"] = _numeric_fact(session_end_cpu_vms, "value") - _numeric_fact(
            session_start_cpu_vms,
            "value",
        )

    if session_duration is not None:
        values["session_duration_s"] = _numeric_fact(session_duration, "value") / 1000.0


def _add_peak_values(values: dict[str, MetadataValue], observations: Iterable[MetadataRecord]) -> None:
    gpu_used_records = tuple(
        record
        for record in observations
        if _matches_measurement(
            record,
            resource_kind="gpu_memory",
            measurement_kind="used_visible",
            scope_type="device_aggregate",
        )
    )
    gpu_used_device_records = tuple(
        record
        for record in observations
        if _matches_measurement(
            record,
            resource_kind="gpu_memory",
            measurement_kind="used_visible",
            scope_type="device",
        )
    )
    gpu_allocated_peak_records = tuple(
        record
        for record in observations
        if _matches_measurement(
            record,
            resource_kind="gpu_memory",
            measurement_kind="peak_allocated",
            scope_type="device_aggregate",
        )
    )
    gpu_allocated_peak_device_records = tuple(
        record
        for record in observations
        if _matches_measurement(
            record,
            resource_kind="gpu_memory",
            measurement_kind="peak_allocated",
            scope_type="device",
        )
    )
    gpu_reserved_records = tuple(
        record
        for record in observations
        if _matches_measurement(
            record,
            resource_kind="gpu_memory",
            measurement_kind="reserved",
            scope_type="device_aggregate",
        )
    )
    gpu_reserved_device_records = tuple(
        record
        for record in observations
        if _matches_measurement(
            record,
            resource_kind="gpu_memory",
            measurement_kind="reserved",
            scope_type="device",
        )
    )

    _add_peak_value(values, "gpu_used_peak_mib", gpu_used_records)
    _add_peak_value(values, "gpu_allocated_peak_mib", gpu_allocated_peak_records)
    _add_peak_value(values, "gpu_reserved_peak_mib", gpu_reserved_records)
    _add_device_peak_values(values, "gpu_used_peak_by_device_mib", gpu_used_device_records)
    _add_device_peak_values(values, "gpu_allocated_peak_by_device_mib", gpu_allocated_peak_device_records)
    _add_device_peak_values(values, "gpu_reserved_peak_by_device_mib", gpu_reserved_device_records)


def _add_structural_totals(
    values: dict[str, MetadataValue],
    structural_facts: Iterable[MetadataRecord],
) -> None:
    structural_records = tuple(structural_facts)
    _add_structural_total(
        values,
        structural_records,
        value_key="loaded_parameter_memory_mib",
        resource_kind="parameter_memory",
    )
    _add_structural_total(
        values,
        structural_records,
        value_key="trainable_parameter_memory_mib",
        resource_kind="trainable_parameter_memory",
    )
    _add_structural_total(
        values,
        structural_records,
        value_key="gradient_memory_estimate_mib",
        resource_kind="gradient_memory",
    )
    _add_structural_total(
        values,
        structural_records,
        value_key="optimizer_state_memory_estimate_mib",
        resource_kind="optimizer_state_memory",
    )


def _profile_source_records(
    *,
    frames: tuple[MetadataRecord, ...],
    observations: tuple[MetadataRecord, ...],
    structural_facts: tuple[MetadataRecord, ...],
) -> tuple[MetadataRecord, ...]:
    # The profile includes accepted-record counts, so every accepted record in
    # these domains is evidence for at least one durable profile value.
    return _unique_records((*frames, *observations, *structural_facts))


def _source_references(records: Iterable[MetadataRecord]) -> tuple[ResourceFactReference, ...]:
    return tuple(
        ResourceFactReference(
            entity_type=record.identity.entity_type,
            identifier=record.identity.identifier,
            relationship=MetadataRelationship.DERIVED_FROM,
            namespace=record.identity.namespace,
        )
        for record in records
    )


def _unique_records(records: Iterable[MetadataRecord]) -> tuple[MetadataRecord, ...]:
    unique: dict[str, MetadataRecord] = {}
    for record in records:
        unique.setdefault(record.identity.key, record)
    return tuple(unique.values())


def _first_measurement(
    observations: Iterable[MetadataRecord],
    *,
    frame_by_identifier: dict[str, MetadataRecord],
    event_name: str,
    resource_kind: str,
    measurement_kind: str,
) -> MetadataRecord | None:
    return next(
        (
            record
            for record in observations
            if _matches_event_measurement(
                record,
                frame_by_identifier=frame_by_identifier,
                event_name=event_name,
                resource_kind=resource_kind,
                measurement_kind=measurement_kind,
            )
        ),
        None,
    )


def _last_measurement(
    observations: Iterable[MetadataRecord],
    *,
    frame_by_identifier: dict[str, MetadataRecord],
    event_name: str,
    resource_kind: str,
    measurement_kind: str,
) -> MetadataRecord | None:
    matching = tuple(
        record
        for record in observations
        if _matches_event_measurement(
            record,
            frame_by_identifier=frame_by_identifier,
            event_name=event_name,
            resource_kind=resource_kind,
            measurement_kind=measurement_kind,
        )
    )
    return matching[-1] if matching else None


def _matches_event_measurement(
    record: MetadataRecord,
    *,
    frame_by_identifier: dict[str, MetadataRecord],
    event_name: str,
    resource_kind: str,
    measurement_kind: str,
) -> bool:
    frame_identifier = record.facts.get("frame_identifier")
    if not isinstance(frame_identifier, str):
        return False
    frame = frame_by_identifier.get(frame_identifier)
    return (
        frame is not None
        and frame.facts.get("event_name") == event_name
        and _matches_measurement(record, resource_kind=resource_kind, measurement_kind=measurement_kind)
    )


def _matches_measurement(
    record: MetadataRecord,
    *,
    resource_kind: str,
    measurement_kind: str,
    scope_type: str | None = None,
) -> bool:
    if record.facts.get("resource_kind") != resource_kind:
        return False
    if record.facts.get("measurement_kind") != measurement_kind:
        return False
    return scope_type is None or record.facts.get("scope_type") == scope_type


def _add_peak_value(
    values: dict[str, MetadataValue],
    value_key: str,
    records: tuple[MetadataRecord, ...],
) -> None:
    if not records:
        return
    values[value_key] = max(_numeric_fact(record, "value") for record in records)


def _add_device_peak_values(
    values: dict[str, MetadataValue],
    value_key: str,
    records: tuple[MetadataRecord, ...],
) -> None:
    device_peaks = _device_peak_values(records)
    if device_peaks:
        device_peak_values: dict[str, object] = dict(device_peaks)
        values[value_key] = device_peak_values


def _device_peak_values(records: Iterable[MetadataRecord]) -> dict[str, float]:
    peaks: dict[str, float] = {}
    for record in records:
        device_identifier = record.facts.get("device_identifier")
        if not isinstance(device_identifier, str):
            continue
        peaks[device_identifier] = max(
            peaks.get(device_identifier, 0.0),
            _numeric_fact(record, "value"),
        )
    return peaks


def _add_structural_total(
    values: dict[str, MetadataValue],
    structural_facts: Iterable[MetadataRecord],
    *,
    value_key: str,
    resource_kind: str,
) -> None:
    records = tuple(
        record for record in structural_facts if record.facts.get("resource_kind") == resource_kind and record.facts.get("unit") == "MiB"
    )
    if records:
        values[value_key] = sum(_numeric_fact(record, "quantity") for record in records)


def _numeric_fact(record: MetadataRecord, key: str) -> float:
    value = record.facts.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


__all__ = [
    "RUN_RESOURCE_SUMMARY_DERIVATION_VERSION",
    "RUN_RESOURCE_SUMMARY_PROFILE_KIND",
    "build_run_resource_profile",
]
