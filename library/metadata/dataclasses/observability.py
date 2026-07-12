"""Typed observability and report metadata fact shapes.

These facts keep logging/report/resource-monitor boundaries explicit without
making the logging package own the canonical metadata schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from library.metadata.records import MetadataValue


@dataclass(frozen=True, slots=True)
class LoggedArtifactFacts:
    """Facts for a file registered through the logging layer."""

    path: str
    kind: str
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)
    run_identifier: str | None = None


@dataclass(frozen=True, slots=True)
class RunLifecycleFacts:
    """Facts for a run lifecycle event emitted by logging or training."""

    run_identifier: str
    event_type: str
    run_name: str | None = None
    status: str | None = None
    mode_name: str | None = None
    strategy_name: str | None = None
    optimizer_name: str | None = None
    config_name: str | None = None
    global_step: int | None = None
    epoch: int | None = None
    duration_ms: float | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceMonitorFacts:
    """Compatibility fallback for monitor boundaries without observations."""

    run_identifier: str
    event_name: str
    ts: float | None = None
    rank: int | None = None
    world_size: int | None = None
    mode: str | None = None
    device_scope: str | None = None
    config_name: str | None = None
    git_sha: str | None = None
    git_dirty: bool | None = None
    global_step: int | None = None
    epoch: int | None = None
    phase: str | None = None
    duration_ms: float | None = None
    gpu_allocated_mb: float | None = None
    gpu_allocated_by_device_mb: Mapping[str, float] | None = None
    gpu_reserved_mb: float | None = None
    gpu_reserved_by_device_mb: Mapping[str, float] | None = None
    gpu_peak_allocated_mb: float | None = None
    gpu_peak_allocated_by_device_mb: Mapping[str, float] | None = None
    gpu_used_mb: float | None = None
    gpu_used_by_device_mb: Mapping[str, float] | None = None
    cpu_rss_mb: float | None = None
    cpu_vms_mb: float | None = None
    steps_per_sec: float | None = None
    samples_per_sec: float | None = None
    dropped_samples: int | None = None
    collection_ms: float | None = None
    deep_alloc_retries: int | None = None
    deep_ooms: int | None = None
    deep_active_mb: float | None = None
    deep_reserved_mb: float | None = None
    deep_inactive_split_mb: float | None = None
    deep_window_active: bool | None = None


@dataclass(frozen=True, slots=True)
class RunReportFacts:
    """Facts for a benchmark/run report artifact boundary."""

    report_identifier: str
    run_identifier: str
    status: str
    generated_at: float
    output_name: str | None = None
    output_dir: str | None = None
    mode_name: str | None = None
    strategy_name: str | None = None
    optimizer_name: str | None = None
    global_step: int | None = None
    num_train_epochs: int | None = None
    resource_event_count: int | None = None
    phase_count: int | None = None
    include_full_config: bool | None = None
    resource_jsonl_path: str | None = None


@dataclass(frozen=True, slots=True)
class AnalyticsSnapshotFacts:
    """Facts for backend-neutral analytics/debug snapshots."""

    snapshot_identifier: str
    snapshot_kind: str
    source: str
    payload: Mapping[str, MetadataValue] = field(default_factory=dict)
    generated_at: float | None = None
    run_identifier: str | None = None
