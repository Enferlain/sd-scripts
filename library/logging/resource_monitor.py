"""
Config-driven resource monitoring for trainer phases and optimization steps.

The monitor is intentionally lightweight on the hot path:
- `off` mode uses a strict no-op implementation.
- `basic` mode collects cheap allocator/RSS counters.
- `sampled`/`deep` add a daemon sampler thread for higher-fidelity device memory.
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TextIO

import psutil
import torch
from tqdm.auto import tqdm

from library.logging.summaries import DiagnosticRow
from library.utils.hash_utils import get_git_is_dirty, get_git_revision_hash

if TYPE_CHECKING:
    from accelerate import Accelerator

    from library.config.dataclasses.output import ResourceMonitorConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Snapshot:
    """Point-in-time resource snapshot."""

    gpu_allocated_mb: float | None
    gpu_reserved_mb: float | None
    gpu_peak_allocated_mb: float | None
    cpu_rss_mb: float | None


@dataclass
class _PhaseStartState:
    """Captured state for phase duration and sampled peak logging."""

    started_at: float
    start_snapshot: _Snapshot
    sampled_peak_gpu_used_mb: float | None = None


@dataclass(frozen=True)
class _SampledMetrics:
    """A single background-sampler datapoint."""

    ts: float
    gpu_used_mb: float | None
    cpu_rss_mb: float | None
    collection_ms: float | None


@dataclass(frozen=True)
class _DeepCounters:
    """Deep allocator diagnostics collected from torch.cuda.memory_stats()."""

    alloc_retries: int | None
    ooms: int | None
    active_mb: float | None
    reserved_mb: float | None
    inactive_split_mb: float | None
    collection_ms: float | None


@dataclass(frozen=True)
class _StartupComponentMemory:
    label: str
    param_bytes_total: int
    param_bytes_trainable: int


class ResourceMonitor(Protocol):
    """Resource monitor contract used by Trainer and phases."""

    def start_session(self) -> None: ...

    def end_session(self) -> None: ...

    def phase_start(self, name: str) -> None: ...

    def phase_end(self, name: str) -> None: ...

    def step_end(self, global_step: int, epoch: int) -> None: ...

    def emit_startup_component_memory(
        self,
        components: Mapping[str, Any] | Iterable[tuple[str, Any]] | Iterable[DiagnosticRow] | None,
        optimizer_name: str,
        *,
        deepspeed_enabled: bool = False,
        deepspeed_zero_stage: int | None = None,
    ) -> None: ...

    @property
    def jsonl_path(self) -> Path | None: ...


class NoOpResourceMonitor:
    """Strict no-op resource monitor used when mode is off."""

    def start_session(self) -> None:
        return

    def end_session(self) -> None:
        return

    def phase_start(self, name: str) -> None:
        return

    def phase_end(self, name: str) -> None:
        return

    def step_end(self, global_step: int, epoch: int) -> None:
        return

    def emit_startup_component_memory(
        self,
        components: Mapping[str, Any] | Iterable[tuple[str, Any]] | Iterable[DiagnosticRow] | None,
        optimizer_name: str,
        *,
        deepspeed_enabled: bool = False,
        deepspeed_zero_stage: int | None = None,
    ) -> None:
        _ = deepspeed_enabled, deepspeed_zero_stage
        return

    @property
    def jsonl_path(self) -> Path | None:
        return None


class BasicResourceMonitor:
    """Low-overhead monitor for phase summaries and periodic step snapshots."""

    def __init__(
        self,
        *,
        accelerator: Accelerator,
        resource_monitor_config: ResourceMonitorConfig,
        output_jsonl_path: Path | None,
        run_id: str | int | None = None,
        config_name: str | None = None,
        git_sha: str | None = None,
        git_dirty: bool | None = None,
    ):
        self._accelerator = accelerator
        self._resource_monitor_config = resource_monitor_config
        self._rank_scope = resource_monitor_config.rank_scope
        self._phase_summary = resource_monitor_config.phase_summary
        self._component_breakdown = resource_monitor_config.component_breakdown
        self._log_every_n_steps = resource_monitor_config.log_every_n_steps
        self._mode = resource_monitor_config.mode
        self._device_scope = resource_monitor_config.device_scope
        self._max_collection_ms = resource_monitor_config.max_collection_ms

        self._rank = int(getattr(accelerator, "process_index", 0))
        self._world_size = int(getattr(accelerator, "num_processes", 1))

        self._process = psutil.Process()
        self._session_started_at: float | None = None
        self._phase_states: dict[str, _PhaseStartState] = {}
        self._last_step_sample: tuple[int, float] | None = None
        self._session_ended = False

        self._latest_gpu_used_mb: float | None = None
        self._latest_collection_ms: float | None = None
        self._dropped_samples = 0
        self._latest_deep_counters: _DeepCounters | None = None
        self._latest_deep_window_active: bool | None = None

        self._jsonl_path = output_jsonl_path
        self._jsonl_file: TextIO | None = None
        self._jsonl_events_since_flush = 0
        self._jsonl_flush_mode = resource_monitor_config.jsonl_flush_mode
        self._jsonl_flush_every_n_events = resource_monitor_config.jsonl_flush_every_n_events

        self._run_id = self._normalize_metadata_value(run_id)
        self._config_name = self._normalize_metadata_value(config_name)
        self._git_sha = self._normalize_metadata_value(git_sha)
        self._git_dirty = git_dirty if isinstance(git_dirty, bool) else None

        self._warned_once: set[str] = set()

    @property
    def jsonl_path(self) -> Path | None:
        return self._jsonl_path

    def _normalize_metadata_value(self, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return str(value)

    def _should_emit_this_rank(self) -> bool:
        if self._rank_scope == "all":
            return True
        return self._accelerator.is_main_process

    def _is_cuda_visible(self) -> bool:
        return torch.cuda.is_available()

    def _warn_once(self, key: str, message: str, *args: object) -> None:
        if key in self._warned_once:
            return
        self._warned_once.add(key)
        logger.warning(message, *args)

    def _log_info_external(self, message: str, *args: object) -> None:
        with tqdm.external_write_mode(file=sys.stderr):
            logger.info(message, *args)

    def _print_block_external(self, message: str) -> None:
        with tqdm.external_write_mode(file=sys.stderr):
            print(message, file=sys.stderr)

    def _collect_snapshot(self, *, reset_peak: bool = False) -> _Snapshot:
        gpu_allocated_mb: float | None = None
        gpu_reserved_mb: float | None = None
        gpu_peak_allocated_mb: float | None = None

        if self._is_cuda_visible():
            try:
                if reset_peak:
                    torch.cuda.reset_peak_memory_stats()
                gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                gpu_peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            except Exception as exc:  # pragma: no cover - backend-specific failure path
                self._warn_once("snapshot_cuda", "resource monitor CUDA snapshot failed: %s", exc)

        cpu_rss_mb: float | None = None
        try:
            cpu_rss_mb = self._process.memory_info().rss / (1024 * 1024)
        except Exception as exc:  # pragma: no cover - platform-specific failure path
            self._warn_once("snapshot_cpu", "resource monitor CPU RSS snapshot failed: %s", exc)

        return _Snapshot(
            gpu_allocated_mb=gpu_allocated_mb,
            gpu_reserved_mb=gpu_reserved_mb,
            gpu_peak_allocated_mb=gpu_peak_allocated_mb,
            cpu_rss_mb=cpu_rss_mb,
        )

    def _format_gpu(self, value: float | None) -> str:
        return "n/a" if value is None else f"{value:.0f}MB"

    def _format_cpu(self, value: float | None) -> str:
        return "n/a" if value is None else f"{value:.0f}MB"

    def _normalize_component_items(
        self,
        components: Mapping[str, Any] | Iterable[tuple[str, Any]] | Iterable[DiagnosticRow] | None,
    ) -> list[tuple[str, Any] | DiagnosticRow]:
        if not components:
            return []

        if isinstance(components, Mapping):
            raw_items = list(components.items())
        else:
            raw_items = list(components)

        normalized: list[tuple[str, Any] | DiagnosticRow] = []
        for item in raw_items:
            if isinstance(item, DiagnosticRow):
                normalized.append(item)
                continue
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            name, module = item
            if not isinstance(name, str) or module is None:
                continue
            normalized.append((name, module))
        return normalized

    def _collect_startup_component_memory(
        self,
        components: Mapping[str, Any] | Iterable[tuple[str, Any]] | Iterable[DiagnosticRow] | None,
    ) -> list[_StartupComponentMemory]:
        component_items = self._normalize_component_items(components)
        startup_rows: list[_StartupComponentMemory] = []

        for item in component_items:
            if isinstance(item, DiagnosticRow):
                startup_rows.append(
                    _StartupComponentMemory(
                        label=item.label,
                        param_bytes_total=item.param_bytes_total,
                        param_bytes_trainable=item.param_bytes_trainable,
                    )
                )
                continue

            name, module = item
            param_bytes_total = 0
            param_bytes_trainable = 0
            for p in module.parameters():
                bytes_count = p.numel() * p.element_size()
                param_bytes_total += bytes_count
                if p.requires_grad:
                    param_bytes_trainable += bytes_count
            startup_rows.append(
                _StartupComponentMemory(
                    label=name,
                    param_bytes_total=param_bytes_total,
                    param_bytes_trainable=param_bytes_trainable,
                )
            )

        return startup_rows

    def _resolve_flush_mode(self) -> str:
        if self._jsonl_flush_mode != "auto":
            return self._jsonl_flush_mode
        return "line" if self._mode in {"sampled", "deep"} else "batch"

    def _open_jsonl_stream(self) -> None:
        if self._jsonl_file is not None or self._jsonl_path is None:
            return
        try:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_file = self._jsonl_path.open("a", encoding="utf-8")
        except Exception as exc:
            self._jsonl_file = None
            self._warn_once("jsonl_open", "resource monitor JSONL disabled; failed to open %s (%s)", self._jsonl_path, exc)

    def _close_jsonl_stream(self) -> None:
        if self._jsonl_file is None:
            return
        try:
            self._jsonl_file.flush()
            self._jsonl_file.close()
        except Exception as exc:  # pragma: no cover - filesystem specific
            self._warn_once("jsonl_close", "resource monitor JSONL close failed: %s", exc)
        finally:
            self._jsonl_file = None

    def _write_jsonl_event(self, event: dict[str, Any], *, force_flush: bool = False) -> None:
        if self._jsonl_file is None:
            return

        try:
            self._jsonl_file.write(json.dumps(event, ensure_ascii=True) + "\n")
            self._jsonl_events_since_flush += 1

            flush_mode = self._resolve_flush_mode()
            should_flush = force_flush or flush_mode == "line"
            if flush_mode == "batch" and self._jsonl_events_since_flush >= self._jsonl_flush_every_n_events:
                should_flush = True

            if should_flush:
                self._jsonl_file.flush()
                self._jsonl_events_since_flush = 0
        except Exception as exc:  # pragma: no cover - filesystem specific
            self._warn_once("jsonl_write", "resource monitor JSONL write failed: %s", exc)

    def _build_event(
        self,
        *,
        event: str,
        global_step: int | None,
        epoch: int | None,
        phase: str | None,
        duration_ms: float | None,
        steps_per_sec: float | None,
        snapshot: _Snapshot | None,
        gpu_used_mb: float | None,
        cpu_rss_mb: float | None,
        dropped_samples: int | None,
        collection_ms: float | None,
        deep_alloc_retries: int | None,
        deep_ooms: int | None,
        deep_active_mb: float | None,
        deep_reserved_mb: float | None,
        deep_inactive_split_mb: float | None,
        deep_window_active: bool | None,
        ts: float | None = None,
    ) -> dict[str, Any]:
        return {
            "ts": time.time() if ts is None else ts,
            "event": event,
            "rank": self._rank,
            "world_size": self._world_size,
            "mode": self._mode,
            "device_scope": self._device_scope,
            "run_id": self._run_id,
            "config_name": self._config_name,
            "git_sha": self._git_sha,
            "git_dirty": self._git_dirty,
            "global_step": global_step,
            "epoch": epoch,
            "phase": phase,
            "duration_ms": duration_ms,
            "gpu_allocated_mb": None if snapshot is None else snapshot.gpu_allocated_mb,
            "gpu_reserved_mb": None if snapshot is None else snapshot.gpu_reserved_mb,
            "gpu_peak_allocated_mb": None if snapshot is None else snapshot.gpu_peak_allocated_mb,
            "gpu_used_mb": gpu_used_mb,
            "cpu_rss_mb": cpu_rss_mb,
            "steps_per_sec": steps_per_sec,
            "samples_per_sec": None,
            "dropped_samples": dropped_samples,
            "collection_ms": collection_ms,
            "deep_alloc_retries": deep_alloc_retries,
            "deep_ooms": deep_ooms,
            "deep_active_mb": deep_active_mb,
            "deep_reserved_mb": deep_reserved_mb,
            "deep_inactive_split_mb": deep_inactive_split_mb,
            "deep_window_active": deep_window_active,
        }

    def _emit_event(
        self,
        *,
        event: str,
        global_step: int | None = None,
        epoch: int | None = None,
        phase: str | None = None,
        duration_ms: float | None = None,
        steps_per_sec: float | None = None,
        snapshot: _Snapshot | None = None,
        gpu_used_mb: float | None = None,
        cpu_rss_mb: float | None = None,
        dropped_samples: int | None = None,
        collection_ms: float | None = None,
        deep_counters: _DeepCounters | None = None,
        deep_window_active: bool | None = None,
        force_flush: bool = False,
        ts: float | None = None,
    ) -> None:
        if self._jsonl_file is None:
            return

        counters = self._latest_deep_counters if deep_counters is None else deep_counters
        window_active = self._latest_deep_window_active if deep_window_active is None else deep_window_active

        event_payload = self._build_event(
            event=event,
            global_step=global_step,
            epoch=epoch,
            phase=phase,
            duration_ms=duration_ms,
            steps_per_sec=steps_per_sec,
            snapshot=snapshot,
            gpu_used_mb=self._latest_gpu_used_mb if gpu_used_mb is None else gpu_used_mb,
            cpu_rss_mb=(snapshot.cpu_rss_mb if snapshot is not None else cpu_rss_mb),
            dropped_samples=dropped_samples,
            collection_ms=self._latest_collection_ms if collection_ms is None else collection_ms,
            deep_alloc_retries=None if counters is None else counters.alloc_retries,
            deep_ooms=None if counters is None else counters.ooms,
            deep_active_mb=None if counters is None else counters.active_mb,
            deep_reserved_mb=None if counters is None else counters.reserved_mb,
            deep_inactive_split_mb=None if counters is None else counters.inactive_split_mb,
            deep_window_active=window_active,
            ts=ts,
        )
        self._write_jsonl_event(event_payload, force_flush=force_flush)

    def _record_sampled_metrics(self, sample: _SampledMetrics) -> None:
        self._latest_gpu_used_mb = sample.gpu_used_mb
        self._latest_collection_ms = sample.collection_ms

        for state in self._phase_states.values():
            if sample.gpu_used_mb is None:
                continue
            if state.sampled_peak_gpu_used_mb is None or sample.gpu_used_mb > state.sampled_peak_gpu_used_mb:
                state.sampled_peak_gpu_used_mb = sample.gpu_used_mb

    def _process_sampler_updates(self, max_items: int | None = 128) -> None:
        _ = max_items

    def _enforce_collection_budget(self, collection_ms: float | None, *, source: str) -> None:
        if collection_ms is None:
            return
        if self._max_collection_ms <= 0:
            return
        if collection_ms <= self._max_collection_ms:
            return
        self._warn_once(
            f"collection_budget_{source}",
            "resource monitor %s collection time %.2fms exceeded budget %.2fms",
            source,
            collection_ms,
            self._max_collection_ms,
        )

    def start_session(self) -> None:
        try:
            if not self._should_emit_this_rank():
                return
            if self._session_started_at is not None:
                return

            self._session_started_at = time.perf_counter()
            start_snapshot = self._collect_snapshot(reset_peak=False)
            self._open_jsonl_stream()

            self._log_info_external(
                "Resource monitor started: mode=%s, rank_scope=%s, step_log_every=%s",
                self._mode,
                self._rank_scope,
                self._log_every_n_steps,
            )

            self._emit_event(event="session_start", snapshot=start_snapshot)
        except Exception as exc:  # pragma: no cover - defensive safety net
            self._warn_once("start_session", "resource monitor start_session failed: %s", exc)

    def end_session(self) -> None:
        try:
            if self._session_ended or not self._should_emit_this_rank():
                return
            self._process_sampler_updates(max_items=None)

            if self._session_started_at is None:
                self._session_ended = True
                self._close_jsonl_stream()
                return

            end_snapshot = self._collect_snapshot(reset_peak=False)
            duration = time.perf_counter() - self._session_started_at

            self._log_info_external(
                ("Resource session summary: duration=%.2fs, gpu_allocated=%s, gpu_reserved=%s, gpu_peak=%s, gpu_used=%s, cpu_rss=%s"),
                duration,
                self._format_gpu(end_snapshot.gpu_allocated_mb),
                self._format_gpu(end_snapshot.gpu_reserved_mb),
                self._format_gpu(end_snapshot.gpu_peak_allocated_mb),
                self._format_gpu(self._latest_gpu_used_mb),
                self._format_cpu(end_snapshot.cpu_rss_mb),
            )

            self._emit_event(
                event="session_end",
                duration_ms=duration * 1000,
                snapshot=end_snapshot,
                dropped_samples=self._dropped_samples if self._mode in {"sampled", "deep"} else None,
                force_flush=True,
            )

            self._session_ended = True
            self._phase_states.clear()
        except Exception as exc:  # pragma: no cover - defensive safety net
            self._warn_once("end_session", "resource monitor end_session failed: %s", exc)
        finally:
            self._close_jsonl_stream()

    def phase_start(self, name: str) -> None:
        try:
            if not self._should_emit_this_rank() or not self._phase_summary:
                return

            self._process_sampler_updates(max_items=128)
            start_snapshot = self._collect_snapshot(reset_peak=True)
            self._phase_states[name] = _PhaseStartState(
                started_at=time.perf_counter(),
                start_snapshot=start_snapshot,
                sampled_peak_gpu_used_mb=self._latest_gpu_used_mb,
            )
            self._emit_event(event="phase_start", phase=name, snapshot=start_snapshot)
        except Exception as exc:  # pragma: no cover - defensive safety net
            self._warn_once("phase_start", "resource monitor phase_start failed (%s): %s", name, exc)

    def phase_end(self, name: str) -> None:
        try:
            if not self._should_emit_this_rank() or not self._phase_summary:
                return

            self._process_sampler_updates(max_items=1024)
            start_state = self._phase_states.pop(name, None)
            if start_state is None:
                return

            end_snapshot = self._collect_snapshot(reset_peak=False)
            duration = time.perf_counter() - start_state.started_at

            sampled_peak_suffix = ""
            if start_state.sampled_peak_gpu_used_mb is not None:
                sampled_peak_suffix = f", gpu_used_peak={self._format_gpu(start_state.sampled_peak_gpu_used_mb)}"

            self._log_info_external(
                ("Resource phase[%s]: duration=%.2fs, gpu_allocated=%s->%s, gpu_reserved=%s->%s, gpu_peak=%s%s, cpu_rss=%s->%s"),
                name,
                duration,
                self._format_gpu(start_state.start_snapshot.gpu_allocated_mb),
                self._format_gpu(end_snapshot.gpu_allocated_mb),
                self._format_gpu(start_state.start_snapshot.gpu_reserved_mb),
                self._format_gpu(end_snapshot.gpu_reserved_mb),
                self._format_gpu(end_snapshot.gpu_peak_allocated_mb),
                sampled_peak_suffix,
                self._format_cpu(start_state.start_snapshot.cpu_rss_mb),
                self._format_cpu(end_snapshot.cpu_rss_mb),
            )

            self._emit_event(
                event="phase_end",
                phase=name,
                duration_ms=duration * 1000,
                snapshot=end_snapshot,
                dropped_samples=self._dropped_samples if self._mode in {"sampled", "deep"} else None,
                force_flush=True,
            )
        except Exception as exc:  # pragma: no cover - defensive safety net
            self._warn_once("phase_end", "resource monitor phase_end failed (%s): %s", name, exc)

    def step_end(self, global_step: int, epoch: int) -> None:
        try:
            if not self._should_emit_this_rank():
                return

            self._process_sampler_updates(max_items=128)
            if self._log_every_n_steps <= 0:
                return
            if global_step % self._log_every_n_steps != 0:
                return

            now = time.perf_counter()
            steps_per_sec: float | None = None
            if self._last_step_sample is not None:
                last_step, last_t = self._last_step_sample
                if global_step > last_step and now > last_t:
                    steps_per_sec = (global_step - last_step) / (now - last_t)
            self._last_step_sample = (global_step, now)

            snapshot = self._collect_snapshot(reset_peak=False)
            if steps_per_sec is None:
                self._log_info_external(
                    "Resource step[%s|epoch=%s]: gpu_allocated=%s, gpu_reserved=%s, gpu_used=%s, cpu_rss=%s",
                    global_step,
                    epoch,
                    self._format_gpu(snapshot.gpu_allocated_mb),
                    self._format_gpu(snapshot.gpu_reserved_mb),
                    self._format_gpu(self._latest_gpu_used_mb),
                    self._format_cpu(snapshot.cpu_rss_mb),
                )
            else:
                self._log_info_external(
                    "Resource step[%s|epoch=%s]: %.2f steps/s, gpu_allocated=%s, gpu_reserved=%s, gpu_used=%s, cpu_rss=%s",
                    global_step,
                    epoch,
                    steps_per_sec,
                    self._format_gpu(snapshot.gpu_allocated_mb),
                    self._format_gpu(snapshot.gpu_reserved_mb),
                    self._format_gpu(self._latest_gpu_used_mb),
                    self._format_cpu(snapshot.cpu_rss_mb),
                )

            self._emit_event(
                event="step_sample",
                global_step=global_step,
                epoch=epoch,
                steps_per_sec=steps_per_sec,
                snapshot=snapshot,
                dropped_samples=self._dropped_samples if self._mode in {"sampled", "deep"} else None,
            )
        except Exception as exc:  # pragma: no cover - defensive safety net
            self._warn_once("step_end", "resource monitor step_end failed (step=%s): %s", global_step, exc)

    def emit_startup_component_memory(
        self,
        components: Mapping[str, Any] | Iterable[tuple[str, Any]] | Iterable[DiagnosticRow] | None,
        optimizer_name: str,
        *,
        deepspeed_enabled: bool = False,
        deepspeed_zero_stage: int | None = None,
    ) -> None:
        try:
            if not self._should_emit_this_rank():
                return
            if not self._component_breakdown:
                return
            startup_rows = self._collect_startup_component_memory(components)
            if not startup_rows:
                return

            lines = ["Resource startup breakdown:"]
            total_param_bytes = sum(row.param_bytes_total for row in startup_rows)
            total_trainable_bytes = sum(row.param_bytes_trainable for row in startup_rows)
            lines.append("  loaded model weights:")
            total_frozen_bytes = max(total_param_bytes - total_trainable_bytes, 0)

            table_rows: list[tuple[str, str, str, str, str]] = []
            for row in startup_rows:
                loaded_mb = row.param_bytes_total / (1024 * 1024)
                trainable_mb = row.param_bytes_trainable / (1024 * 1024)
                frozen_mb = max(loaded_mb - trainable_mb, 0.0)
                share_percent = (row.param_bytes_total / total_param_bytes * 100) if total_param_bytes > 0 else 0.0
                table_rows.append(
                    (
                        row.label,
                        f"{loaded_mb:.1f}MB",
                        f"{trainable_mb:.1f}MB",
                        f"{frozen_mb:.1f}MB",
                        f"{share_percent:.1f}%",
                    )
                )

            table_rows.append(
                (
                    "total",
                    f"{total_param_bytes / (1024 * 1024):.1f}MB",
                    f"{total_trainable_bytes / (1024 * 1024):.1f}MB",
                    f"{total_frozen_bytes / (1024 * 1024):.1f}MB",
                    "100.0%",
                )
            )

            component_width = max(len("component"), *(len(row[0]) for row in table_rows))
            loaded_width = max(len("loaded"), *(len(row[1]) for row in table_rows))
            trainable_width = max(len("trainable"), *(len(row[2]) for row in table_rows))
            frozen_width = max(len("frozen"), *(len(row[3]) for row in table_rows))
            share_width = max(len("share"), *(len(row[4]) for row in table_rows))

            lines.append(
                "    "
                f"{'component':<{component_width}} | "
                f"{'loaded':>{loaded_width}} | "
                f"{'trainable':>{trainable_width}} | "
                f"{'frozen':>{frozen_width}} | "
                f"{'share':>{share_width}}"
            )
            lines.append(
                "    "
                f"{'-' * component_width}-+-"
                f"{'-' * loaded_width}-+-"
                f"{'-' * trainable_width}-+-"
                f"{'-' * frozen_width}-+-"
                f"{'-' * share_width}"
            )
            for component, loaded, trainable, frozen, share in table_rows:
                lines.append(
                    "    "
                    f"{component:<{component_width}} | "
                    f"{loaded:>{loaded_width}} | "
                    f"{trainable:>{trainable_width}} | "
                    f"{frozen:>{frozen_width}} | "
                    f"{share:>{share_width}}"
                )

            optimizer_state_multiplier = 2 if "adam" in optimizer_name.lower() or "lion" in optimizer_name.lower() else 1
            optimizer_state_bytes = total_trainable_bytes * optimizer_state_multiplier
            lines.append("  training state:")
            lines.append(f"    - gradients (est): {total_trainable_bytes / (1024 * 1024):.1f}MB")
            lines.append(f"    - optimizer_state (est): {optimizer_state_bytes / (1024 * 1024):.1f}MB [{optimizer_name}]")
            lines.append(f"    - total (est): {(total_param_bytes + total_trainable_bytes + optimizer_state_bytes) / (1024 * 1024):.1f}MB")
            if deepspeed_enabled:
                zero_stage_label = "n/a" if deepspeed_zero_stage is None else str(deepspeed_zero_stage)
                lines.append(
                    "    - DeepSpeed/ZeRO caveat: effective per-rank footprint may be lower/higher than these estimates "
                    f"due to state partitioning/offload (zero_stage={zero_stage_label})."
                )
            self._print_block_external("\n" + "\n".join(lines))
        except Exception as exc:  # pragma: no cover - defensive safety net
            self._warn_once("startup_component_memory", "resource monitor startup component estimate failed: %s", exc)


class SampledResourceMonitor(BasicResourceMonitor):
    """Sampled/deep monitor with a daemon sampler thread and bounded queue."""

    def __init__(
        self,
        *,
        accelerator: Accelerator,
        resource_monitor_config: ResourceMonitorConfig,
        output_jsonl_path: Path | None,
        run_id: str | int | None = None,
        config_name: str | None = None,
        git_sha: str | None = None,
        git_dirty: bool | None = None,
    ):
        super().__init__(
            accelerator=accelerator,
            resource_monitor_config=resource_monitor_config,
            output_jsonl_path=output_jsonl_path,
            run_id=run_id,
            config_name=config_name,
            git_sha=git_sha,
            git_dirty=git_dirty,
        )

        self._sample_interval_sec = resource_monitor_config.sample_interval_sec
        self._queue_maxsize = resource_monitor_config.queue_maxsize
        self._drop_policy = resource_monitor_config.drop_policy
        self._deep_window_steps = resource_monitor_config.deep_window_steps
        self._deep_window_seconds = resource_monitor_config.deep_window_seconds

        self._sample_queue: queue.Queue[_SampledMetrics] = queue.Queue(maxsize=self._queue_maxsize)
        self._sampler_thread: threading.Thread | None = None
        self._sampler_stop_event = threading.Event()
        self._sampler_exception: Exception | None = None
        self._sampler_exception_reported = False

        self._deep_window_started_at: float | None = None
        self._deep_window_start_step: int | None = None
        self._deep_window_sample_count = 0
        self._deep_window_baseline_alloc_retries: int | None = None
        self._deep_window_baseline_ooms: int | None = None
        self._deep_window_last_alloc_retries: int | None = None
        self._deep_window_last_ooms: int | None = None
        self._deep_window_peak_inactive_split_mb: float | None = None
        self._deep_window_summary_emitted = False

        self._nvml_module: Any | None = None
        self._nvml_unavailable = False

    def _collect_gpu_used_via_nvml_mb(self) -> float | None:
        if not self._is_cuda_visible() or self._nvml_unavailable:
            return None

        if self._nvml_module is None:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml_module = pynvml
            except Exception as exc:
                self._nvml_unavailable = True
                self._warn_once("nvml_unavailable", "resource monitor NVML unavailable; using fallback metrics (%s)", exc)
                return None

        assert self._nvml_module is not None

        try:
            if self._device_scope == "all_visible":
                used_bytes = 0
                for index in range(torch.cuda.device_count()):
                    handle = self._nvml_module.nvmlDeviceGetHandleByIndex(index)
                    info = self._nvml_module.nvmlDeviceGetMemoryInfo(handle)
                    used_bytes += int(info.used)
            else:
                index = torch.cuda.current_device()
                handle = self._nvml_module.nvmlDeviceGetHandleByIndex(index)
                info = self._nvml_module.nvmlDeviceGetMemoryInfo(handle)
                used_bytes = int(info.used)
            return used_bytes / (1024 * 1024)
        except Exception as exc:
            self._warn_once("nvml_collect", "resource monitor NVML collection failed; using fallback metrics (%s)", exc)
            return None

    def _collect_gpu_used_via_torch_mb(self) -> float | None:
        if not self._is_cuda_visible():
            return None

        def _used_for_device(index: int) -> int:
            try:
                free, total = torch.cuda.mem_get_info(index)
            except TypeError:
                with torch.cuda.device(index):
                    free, total = torch.cuda.mem_get_info()
            return max(int(total - free), 0)

        try:
            if self._device_scope == "all_visible":
                used_bytes = sum(_used_for_device(index) for index in range(torch.cuda.device_count()))
            else:
                used_bytes = _used_for_device(torch.cuda.current_device())
            return used_bytes / (1024 * 1024)
        except Exception as exc:  # pragma: no cover - backend-specific
            self._warn_once("torch_mem_get_info", "resource monitor torch mem_get_info fallback failed: %s", exc)
            return None

    def _collect_sample_metrics(self) -> _SampledMetrics:
        started = time.perf_counter()

        gpu_used_mb = self._collect_gpu_used_via_nvml_mb()
        if gpu_used_mb is None:
            gpu_used_mb = self._collect_gpu_used_via_torch_mb()

        cpu_rss_mb: float | None = None
        try:
            cpu_rss_mb = self._process.memory_info().rss / (1024 * 1024)
        except Exception as exc:  # pragma: no cover - platform specific
            self._warn_once("sample_cpu", "resource monitor sampled CPU RSS collection failed: %s", exc)

        collection_ms = (time.perf_counter() - started) * 1000
        return _SampledMetrics(ts=time.time(), gpu_used_mb=gpu_used_mb, cpu_rss_mb=cpu_rss_mb, collection_ms=collection_ms)

    def _enqueue_sample(self, sample: _SampledMetrics) -> None:
        try:
            self._sample_queue.put_nowait(sample)
            return
        except queue.Full:
            pass

        if self._drop_policy == "drop_newest":
            self._dropped_samples += 1
            return

        if self._drop_policy == "drop_oldest":
            with suppress(queue.Empty):
                self._sample_queue.get_nowait()
            self._dropped_samples += 1
            try:
                self._sample_queue.put_nowait(sample)
            except queue.Full:
                self._dropped_samples += 1
            return

        # block policy: sampler may block briefly, hot path never calls this.
        try:
            timeout = max(0.05, min(0.5, self._sample_interval_sec))
            self._sample_queue.put(sample, timeout=timeout)
        except queue.Full:
            self._dropped_samples += 1

    def _report_sampler_exception_once(self) -> None:
        if self._sampler_exception is None or self._sampler_exception_reported:
            return
        self._sampler_exception_reported = True
        self._warn_once(
            "sampler_exception",
            "resource monitor sampler stopped after error; monitor continues in degraded mode (%s)",
            self._sampler_exception,
        )

    def _process_sampler_updates(self, max_items: int | None = 128) -> None:
        processed = 0
        while max_items is None or processed < max_items:
            try:
                sample = self._sample_queue.get_nowait()
            except queue.Empty:
                break

            self._record_sampled_metrics(sample)
            self._enforce_collection_budget(sample.collection_ms, source="sampler")

            # Emit periodic sampled datapoints as schema-compliant step_sample events
            # with null global_step/epoch.
            self._emit_event(
                event="step_sample",
                gpu_used_mb=sample.gpu_used_mb,
                cpu_rss_mb=sample.cpu_rss_mb,
                collection_ms=sample.collection_ms,
                dropped_samples=self._dropped_samples,
                ts=sample.ts,
            )
            processed += 1

        self._report_sampler_exception_once()

    def _sampler_loop(self) -> None:
        while not self._sampler_stop_event.wait(self._sample_interval_sec):
            try:
                sample = self._collect_sample_metrics()
                self._enqueue_sample(sample)
            except Exception as exc:  # pragma: no cover - defensive runtime safety
                self._sampler_exception = exc
                return

    def _start_sampler(self) -> None:
        if self._sampler_thread is not None:
            return

        self._sampler_stop_event.clear()
        self._sampler_thread = threading.Thread(target=self._sampler_loop, name="resource-monitor-sampler", daemon=True)
        self._sampler_thread.start()

    def _stop_sampler(self) -> None:
        thread = self._sampler_thread
        if thread is None:
            return

        self._sampler_stop_event.set()
        thread.join(timeout=max(0.2, 2 * self._sample_interval_sec))
        if thread.is_alive():
            self._warn_once("sampler_join", "resource monitor sampler thread did not stop before timeout")
        self._sampler_thread = None

    def _bytes_to_mb(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value) / (1024 * 1024)
        except (TypeError, ValueError):
            return None

    def _ensure_deep_window_started(self, global_step: int) -> None:
        if self._deep_window_started_at is None:
            self._deep_window_started_at = time.perf_counter()
        if self._deep_window_start_step is None:
            self._deep_window_start_step = global_step

    def _deep_window_age_steps(self, global_step: int) -> int | None:
        if self._deep_window_start_step is None:
            return None
        return (global_step - self._deep_window_start_step) + 1

    def _is_deep_window_active(self, global_step: int) -> bool:
        if self._mode != "deep":
            return False

        step_limit_enabled = self._deep_window_steps > 0
        time_limit_enabled = self._deep_window_seconds > 0

        if not step_limit_enabled and not time_limit_enabled:
            return True

        if self._deep_window_started_at is None or self._deep_window_start_step is None:
            return False

        within_steps = True
        if step_limit_enabled:
            age_steps = self._deep_window_age_steps(global_step)
            within_steps = age_steps is not None and age_steps <= self._deep_window_steps

        within_time = True
        if time_limit_enabled:
            elapsed = time.perf_counter() - self._deep_window_started_at
            within_time = elapsed <= self._deep_window_seconds

        return within_steps and within_time

    def _emit_deep_window_summary(self, *, reason: str) -> None:
        if self._mode != "deep" or self._deep_window_summary_emitted:
            return

        self._deep_window_summary_emitted = True
        alloc_delta: int | None = None
        oom_delta: int | None = None

        if self._deep_window_baseline_alloc_retries is not None and self._deep_window_last_alloc_retries is not None:
            alloc_delta = max(self._deep_window_last_alloc_retries - self._deep_window_baseline_alloc_retries, 0)
        if self._deep_window_baseline_ooms is not None and self._deep_window_last_ooms is not None:
            oom_delta = max(self._deep_window_last_ooms - self._deep_window_baseline_ooms, 0)

        self._log_info_external(
            "Resource deep window summary (%s): samples=%s, alloc_retries_delta=%s, ooms_delta=%s, inactive_split_peak=%s",
            reason,
            self._deep_window_sample_count,
            "n/a" if alloc_delta is None else alloc_delta,
            "n/a" if oom_delta is None else oom_delta,
            self._format_gpu(self._deep_window_peak_inactive_split_mb),
        )

    def _collect_deep_counters(self, global_step: int) -> tuple[_DeepCounters | None, bool]:
        if self._mode != "deep":
            self._latest_deep_counters = None
            self._latest_deep_window_active = None
            return None, False

        self._ensure_deep_window_started(global_step)
        window_active = self._is_deep_window_active(global_step)
        self._latest_deep_window_active = window_active
        if not window_active:
            self._latest_deep_counters = None
            self._emit_deep_window_summary(reason="window_limit")
            return None, False

        if not self._is_cuda_visible():
            self._latest_deep_counters = None
            return None, True

        started = time.perf_counter()
        try:
            stats = torch.cuda.memory_stats()
        except Exception as exc:  # pragma: no cover - backend-specific
            self._warn_once("deep_stats", "resource monitor deep memory_stats collection failed: %s", exc)
            return None, True

        counters = _DeepCounters(
            alloc_retries=int(stats.get("num_alloc_retries", 0)),
            ooms=int(stats.get("num_ooms", 0)),
            active_mb=self._bytes_to_mb(stats.get("active_bytes.all.current")),
            reserved_mb=self._bytes_to_mb(stats.get("reserved_bytes.all.current")),
            inactive_split_mb=self._bytes_to_mb(stats.get("inactive_split_bytes.all.current")),
            collection_ms=(time.perf_counter() - started) * 1000,
        )
        self._latest_deep_counters = counters
        self._enforce_collection_budget(counters.collection_ms, source="deep")

        self._deep_window_sample_count += 1
        if self._deep_window_baseline_alloc_retries is None:
            self._deep_window_baseline_alloc_retries = counters.alloc_retries
        if self._deep_window_baseline_ooms is None:
            self._deep_window_baseline_ooms = counters.ooms
        self._deep_window_last_alloc_retries = counters.alloc_retries
        self._deep_window_last_ooms = counters.ooms
        if counters.inactive_split_mb is not None and (
            self._deep_window_peak_inactive_split_mb is None or counters.inactive_split_mb > self._deep_window_peak_inactive_split_mb
        ):
            self._deep_window_peak_inactive_split_mb = counters.inactive_split_mb

        if self._log_every_n_steps > 0 and global_step % self._log_every_n_steps == 0:
            self._log_info_external(
                "Resource deep[%s]: alloc_retries=%s, ooms=%s, active=%s, reserved=%s, inactive_split=%s",
                global_step,
                counters.alloc_retries,
                counters.ooms,
                self._format_gpu(counters.active_mb),
                self._format_gpu(counters.reserved_mb),
                self._format_gpu(counters.inactive_split_mb),
            )
        return counters, True

    def start_session(self) -> None:
        super().start_session()
        if not self._should_emit_this_rank() or self._session_started_at is None:
            return

        self._deep_window_started_at = time.perf_counter()
        self._deep_window_start_step = None
        self._deep_window_sample_count = 0
        self._deep_window_baseline_alloc_retries = None
        self._deep_window_baseline_ooms = None
        self._deep_window_last_alloc_retries = None
        self._deep_window_last_ooms = None
        self._deep_window_peak_inactive_split_mb = None
        self._deep_window_summary_emitted = False
        self._start_sampler()

    def phase_start(self, name: str) -> None:
        self._process_sampler_updates(max_items=256)
        super().phase_start(name)

    def phase_end(self, name: str) -> None:
        self._process_sampler_updates(max_items=1024)
        super().phase_end(name)

    def step_end(self, global_step: int, epoch: int) -> None:
        self._process_sampler_updates(max_items=256)
        deep_counters, deep_window_active = self._collect_deep_counters(global_step)
        if deep_counters is not None:
            self._latest_deep_counters = deep_counters
        elif self._mode == "deep":
            self._latest_deep_counters = None
        self._latest_deep_window_active = deep_window_active if self._mode == "deep" else None
        super().step_end(global_step, epoch)

    def end_session(self) -> None:
        self._stop_sampler()
        self._process_sampler_updates(max_items=None)
        self._report_sampler_exception_once()
        self._emit_deep_window_summary(reason="session_end")
        super().end_session()


def _resolve_output_jsonl_path(resource_monitor_config: ResourceMonitorConfig, output_dir: str | Path | None) -> Path | None:
    output_jsonl = getattr(resource_monitor_config, "output_jsonl", None)
    if output_jsonl is None:
        return None
    if not isinstance(output_jsonl, str):
        return None

    stripped = output_jsonl.strip()
    if not stripped:
        return None

    resolved = Path(stripped)
    if not resolved.is_absolute():
        base = Path(output_dir) if output_dir is not None else Path(".")
        resolved = base / resolved
    return resolved


def create_resource_monitor(
    *,
    accelerator: Accelerator,
    resource_monitor_config: ResourceMonitorConfig,
    output_dir: str | Path | None = None,
    run_id: str | int | None = None,
    config_name: str | None = None,
    git_sha: str | None = None,
    git_dirty: bool | None = None,
) -> ResourceMonitor:
    """Create a monitor instance from typed logging config."""
    enabled = getattr(resource_monitor_config, "enabled", False)
    if not isinstance(enabled, bool):
        enabled = False

    mode = getattr(resource_monitor_config, "mode", "off")
    if not isinstance(mode, str):
        mode = "off"

    if not enabled:
        return NoOpResourceMonitor()

    if mode == "off":
        return NoOpResourceMonitor()

    output_jsonl_path = _resolve_output_jsonl_path(resource_monitor_config, output_dir)
    resolved_git_sha = git_sha if git_sha is not None else get_git_revision_hash()
    resolved_git_dirty = git_dirty if isinstance(git_dirty, bool) else get_git_is_dirty()

    if mode in {"sampled", "deep"}:
        return SampledResourceMonitor(
            accelerator=accelerator,
            resource_monitor_config=resource_monitor_config,
            output_jsonl_path=output_jsonl_path,
            run_id=run_id,
            config_name=config_name,
            git_sha=resolved_git_sha,
            git_dirty=resolved_git_dirty,
        )

    return BasicResourceMonitor(
        accelerator=accelerator,
        resource_monitor_config=resource_monitor_config,
        output_jsonl_path=output_jsonl_path,
        run_id=run_id,
        config_name=config_name,
        git_sha=resolved_git_sha,
        git_dirty=resolved_git_dirty,
    )
