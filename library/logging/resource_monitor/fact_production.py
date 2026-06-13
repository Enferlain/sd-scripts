"""Resource-domain production of accepted typed facts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from library.metadata.dataclasses.observability import ResourceMonitorFacts


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
