"""Resource-domain production of accepted typed facts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from library.metadata.dataclasses.observability import ResourceMonitorFacts
from library.metadata.dataclasses.resource import (
    ResourceObservationFrameFacts,
    ResourceObservationMeasurementFacts,
)
from library.metadata.exports.resource import (
    project_resource_monitor_compatibility_event,
)
from library.metadata.records import MetadataValue


@dataclass(frozen=True, slots=True)
class ResourceMonitorProducedFacts:
    """Accepted facts produced from one resource-monitor event boundary."""

    compatibility: ResourceMonitorFacts | None
    observation_frame: ResourceObservationFrameFacts | None

    def __post_init__(self) -> None:
        if self.compatibility is None and self.observation_frame is None:
            raise ValueError("Resource monitor production requires a canonical frame or compatibility fallback.")
        if self.compatibility is not None and self.observation_frame is not None:
            raise ValueError("Resource monitor production cannot contain both a canonical frame and compatibility fallback.")

    def as_metadata_items(self) -> tuple[ResourceMonitorFacts | ResourceObservationFrameFacts, ...]:
        """Return the canonical frame or its narrow compatibility fallback."""
        if self.observation_frame is not None:
            return (self.observation_frame,)
        if self.compatibility is not None:
            return (self.compatibility,)
        raise AssertionError("ResourceMonitorProducedFacts invariant violated.")


def project_resource_monitor_jsonl_event(facts: ResourceMonitorProducedFacts) -> dict[str, Any]:
    """Project the current resource-monitor JSONL shape from produced facts."""
    source_facts = facts.observation_frame or facts.compatibility
    if source_facts is None:
        raise ValueError("Resource monitor facts require an observation frame or compatibility fallback.")
    return project_resource_monitor_compatibility_event(source_facts)


def build_compatibility_resource_monitor_facts(
    event_payload: Mapping[str, Any],
    *,
    run_identifier: str,
) -> ResourceMonitorFacts:
    """Build the transitional bundled fact from one current monitor event."""
    return ResourceMonitorFacts(
        run_identifier=run_identifier,
        event_name=str(event_payload["event"]),
        ts=float(event_payload["ts"]),
        rank=event_payload["rank"],
        world_size=event_payload["world_size"],
        mode=event_payload["mode"],
        device_scope=event_payload["device_scope"],
        config_name=event_payload["config_name"],
        git_sha=event_payload["git_sha"],
        git_dirty=event_payload["git_dirty"],
        global_step=event_payload["global_step"],
        epoch=event_payload["epoch"],
        phase=event_payload["phase"],
        duration_ms=event_payload["duration_ms"],
        gpu_allocated_mb=event_payload["gpu_allocated_mb"],
        gpu_allocated_by_device_mb=event_payload["gpu_allocated_by_device_mb"],
        gpu_reserved_mb=event_payload["gpu_reserved_mb"],
        gpu_reserved_by_device_mb=event_payload["gpu_reserved_by_device_mb"],
        gpu_peak_allocated_mb=event_payload["gpu_peak_allocated_mb"],
        gpu_peak_allocated_by_device_mb=event_payload["gpu_peak_allocated_by_device_mb"],
        gpu_used_mb=event_payload["gpu_used_mb"],
        gpu_used_by_device_mb=event_payload["gpu_used_by_device_mb"],
        cpu_rss_mb=event_payload["cpu_rss_mb"],
        cpu_vms_mb=event_payload["cpu_vms_mb"],
        steps_per_sec=event_payload["steps_per_sec"],
        samples_per_sec=event_payload["samples_per_sec"],
        dropped_samples=event_payload["dropped_samples"],
        collection_ms=event_payload["collection_ms"],
        deep_alloc_retries=event_payload["deep_alloc_retries"],
        deep_ooms=event_payload["deep_ooms"],
        deep_active_mb=event_payload["deep_active_mb"],
        deep_reserved_mb=event_payload["deep_reserved_mb"],
        deep_inactive_split_mb=event_payload["deep_inactive_split_mb"],
        deep_window_active=event_payload["deep_window_active"],
    )


def build_resource_monitor_produced_facts(
    event_payload: Mapping[str, Any],
    *,
    run_identifier: str,
    sequence: int,
) -> ResourceMonitorProducedFacts:
    """Build a canonical observation frame or the narrow compatibility fallback."""
    observation_frame = build_resource_observation_frame_facts(
        event_payload,
        run_identifier=run_identifier,
        sequence=sequence,
    )
    return ResourceMonitorProducedFacts(
        compatibility=None
        if observation_frame is not None
        else build_compatibility_resource_monitor_facts(
            event_payload,
            run_identifier=run_identifier,
        ),
        observation_frame=observation_frame,
    )


def build_resource_observation_frame_facts(
    event_payload: Mapping[str, Any],
    *,
    run_identifier: str,
    sequence: int,
) -> ResourceObservationFrameFacts | None:
    """Build canonical co-collected observations from one monitor event."""
    event_name = str(event_payload["event"])
    frame_identifier = _frame_identifier(event_name=event_name, run_identifier=run_identifier, event_payload=event_payload, sequence=sequence)
    measurements = tuple(_build_measurements(frame_identifier=frame_identifier, event_payload=event_payload))
    if not measurements:
        return None

    metadata = _frame_metadata(event_payload)
    return ResourceObservationFrameFacts(
        frame_identifier=frame_identifier,
        run_identifier=run_identifier,
        event_name=event_name,
        measurements=measurements,
        ts=_float_or_none(event_payload.get("ts")),
        collector_id=_collector_id(event_payload),
        collection_policy=_string_or_none(event_payload.get("mode")),
        phase=_string_or_none(event_payload.get("phase")),
        global_step=_int_or_none(event_payload.get("global_step")),
        epoch=_int_or_none(event_payload.get("epoch")),
        rank=_int_or_none(event_payload.get("rank")),
        world_size=_int_or_none(event_payload.get("world_size")),
        metadata=metadata,
    )


def _build_measurements(
    *,
    frame_identifier: str,
    event_payload: Mapping[str, Any],
) -> tuple[ResourceObservationMeasurementFacts, ...]:
    measurements: list[ResourceObservationMeasurementFacts] = []
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="gpu_allocated_mb",
        event_payload=event_payload,
        resource_kind="gpu_memory",
        measurement_kind="allocated",
        unit="MiB",
        source="torch_cuda_allocator",
        scope_type="device_aggregate",
    )
    _append_device_measurements(
        measurements,
        frame_identifier=frame_identifier,
        key="gpu_allocated_by_device_mb",
        event_payload=event_payload,
        resource_kind="gpu_memory",
        measurement_kind="allocated",
        unit="MiB",
        source="torch_cuda_allocator",
    )
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="gpu_reserved_mb",
        event_payload=event_payload,
        resource_kind="gpu_memory",
        measurement_kind="reserved",
        unit="MiB",
        source="torch_cuda_allocator",
        scope_type="device_aggregate",
    )
    _append_device_measurements(
        measurements,
        frame_identifier=frame_identifier,
        key="gpu_reserved_by_device_mb",
        event_payload=event_payload,
        resource_kind="gpu_memory",
        measurement_kind="reserved",
        unit="MiB",
        source="torch_cuda_allocator",
    )
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="gpu_peak_allocated_mb",
        event_payload=event_payload,
        resource_kind="gpu_memory",
        measurement_kind="peak_allocated",
        unit="MiB",
        source="torch_cuda_allocator",
        scope_type="device_aggregate",
        metadata={"window_reset_boundary": str(event_payload["event"])},
    )
    _append_device_measurements(
        measurements,
        frame_identifier=frame_identifier,
        key="gpu_peak_allocated_by_device_mb",
        event_payload=event_payload,
        resource_kind="gpu_memory",
        measurement_kind="peak_allocated",
        unit="MiB",
        source="torch_cuda_allocator",
        metadata={"window_reset_boundary": str(event_payload["event"])},
    )
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="gpu_used_mb",
        event_payload=event_payload,
        resource_kind="gpu_memory",
        measurement_kind="used_visible",
        unit="MiB",
        source="resource_monitor.device_memory",
        scope_type="device_aggregate",
        quality="source_may_be_nvml_or_torch_fallback",
    )
    _append_device_measurements(
        measurements,
        frame_identifier=frame_identifier,
        key="gpu_used_by_device_mb",
        event_payload=event_payload,
        resource_kind="gpu_memory",
        measurement_kind="used_visible",
        unit="MiB",
        source="resource_monitor.device_memory",
        quality="source_may_be_nvml_or_torch_fallback",
    )
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="cpu_rss_mb",
        event_payload=event_payload,
        resource_kind="cpu_memory",
        measurement_kind="rss",
        unit="MiB",
        source="psutil",
        scope_type="process",
    )
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="cpu_vms_mb",
        event_payload=event_payload,
        resource_kind="cpu_memory",
        measurement_kind="vms",
        unit="MiB",
        source="psutil",
        scope_type="process",
    )
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="duration_ms",
        event_payload=event_payload,
        resource_kind="runtime",
        measurement_kind="duration",
        unit="ms",
        source="resource_monitor",
        scope_type="event",
    )
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="steps_per_sec",
        event_payload=event_payload,
        resource_kind="throughput",
        measurement_kind="steps_per_second",
        unit="steps/s",
        source="resource_monitor.step_timer",
        scope_type="step_window",
    )
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="samples_per_sec",
        event_payload=event_payload,
        resource_kind="throughput",
        measurement_kind="samples_per_second",
        unit="samples/s",
        source="resource_monitor.step_timer",
        scope_type="step_window",
    )
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="collection_ms",
        event_payload=event_payload,
        resource_kind="collector_cost",
        measurement_kind="collection_duration",
        unit="ms",
        source="resource_monitor",
        scope_type="collector",
    )
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="deep_alloc_retries",
        event_payload=event_payload,
        resource_kind="cuda_allocator_diagnostic",
        measurement_kind="alloc_retries",
        unit="count",
        source="torch_cuda_allocator.memory_stats",
        scope_type="diagnostic_window",
    )
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="deep_ooms",
        event_payload=event_payload,
        resource_kind="cuda_allocator_diagnostic",
        measurement_kind="ooms",
        unit="count",
        source="torch_cuda_allocator.memory_stats",
        scope_type="diagnostic_window",
    )
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="deep_active_mb",
        event_payload=event_payload,
        resource_kind="gpu_memory",
        measurement_kind="deep_active",
        unit="MiB",
        source="torch_cuda_allocator.memory_stats",
        scope_type="diagnostic_window",
    )
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="deep_reserved_mb",
        event_payload=event_payload,
        resource_kind="gpu_memory",
        measurement_kind="deep_reserved",
        unit="MiB",
        source="torch_cuda_allocator.memory_stats",
        scope_type="diagnostic_window",
    )
    _append_measurement(
        measurements,
        frame_identifier=frame_identifier,
        key="deep_inactive_split_mb",
        event_payload=event_payload,
        resource_kind="gpu_memory",
        measurement_kind="deep_inactive_split",
        unit="MiB",
        source="torch_cuda_allocator.memory_stats",
        scope_type="diagnostic_window",
    )
    return tuple(measurements)


def _append_measurement(
    measurements: list[ResourceObservationMeasurementFacts],
    *,
    frame_identifier: str,
    key: str,
    event_payload: Mapping[str, Any],
    resource_kind: str,
    measurement_kind: str,
    unit: str,
    source: str,
    scope_type: str,
    quality: str | None = None,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> None:
    value = _float_or_none(event_payload.get(key))
    if value is None:
        return
    measurements.append(
        ResourceObservationMeasurementFacts(
            measurement_identifier=f"{frame_identifier}:{key}",
            resource_kind=resource_kind,
            measurement_kind=measurement_kind,
            value=value,
            unit=unit,
            source=source,
            scope_type=scope_type,
            quality=quality,
            metadata={} if metadata is None else dict(metadata),
        )
    )


def _append_device_measurements(
    measurements: list[ResourceObservationMeasurementFacts],
    *,
    frame_identifier: str,
    key: str,
    event_payload: Mapping[str, Any],
    resource_kind: str,
    measurement_kind: str,
    unit: str,
    source: str,
    quality: str | None = None,
    metadata: Mapping[str, MetadataValue] | None = None,
) -> None:
    values = event_payload.get(key)
    if not isinstance(values, Mapping):
        return
    for device, raw_value in sorted(values.items(), key=lambda item: _device_sort_key(item[0])):
        value = _float_or_none(raw_value)
        if value is None:
            continue
        device_identifier = _device_identifier(device)
        measurements.append(
            ResourceObservationMeasurementFacts(
                measurement_identifier=f"{frame_identifier}:{key}:{device_identifier}",
                resource_kind=resource_kind,
                measurement_kind=measurement_kind,
                value=value,
                unit=unit,
                source=source,
                scope_type="device",
                device_identifier=device_identifier,
                quality=quality,
                metadata={} if metadata is None else dict(metadata),
            )
        )


def _frame_identifier(
    *,
    event_name: str,
    run_identifier: str,
    event_payload: Mapping[str, Any],
    sequence: int,
) -> str:
    rank = event_payload.get("rank", "unknown")
    return f"{run_identifier}:resource_frame:{rank}:{sequence}:{event_name}"


def _collector_id(event_payload: Mapping[str, Any]) -> str:
    if (
        event_payload.get("event") == "step_sample"
        and event_payload.get("collection_ms") is not None
        and event_payload.get("global_step") is None
    ):
        return "resource_monitor.sampler"
    if event_payload.get("deep_window_active") is not None:
        return "resource_monitor.deep"
    return "resource_monitor"


def _frame_metadata(event_payload: Mapping[str, Any]) -> dict[str, MetadataValue]:
    metadata: dict[str, MetadataValue] = {}
    for key in ("device_scope", "config_name", "git_sha", "git_dirty", "dropped_samples", "deep_window_active"):
        value = event_payload.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def _device_identifier(device: object) -> str:
    value = str(device)
    if value.startswith("cuda:"):
        return value
    return f"cuda:{value}"


def _device_sort_key(device: object) -> tuple[int, int | str]:
    value = str(device)
    if value.startswith("cuda:"):
        value = value.removeprefix("cuda:")
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    return None


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
