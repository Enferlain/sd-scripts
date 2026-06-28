"""
Config-driven resource monitoring for trainer phases and optimization steps.

The monitor is intentionally lightweight on the hot path:
- `off` mode uses a strict no-op implementation.
- `basic` mode collects cheap allocator/RSS counters.
- `sampled`/`deep` add a daemon sampler thread for higher-fidelity device memory.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import psutil
from tqdm.auto import tqdm

from library.logging.phase_tags import is_training_epoch_phase
from library.logging.summaries import DiagnosticRow
from library.metadata.dataclasses.resource import ResourceCollectorStatusFacts
from library.metadata.runtime import MetadataRuntime
from library.metadata.views import ResourceRunView
from library.utils.hash_utils import get_git_is_dirty, get_git_revision_hash

from .collect import (
    _DeepCounters,
    _SampledMetrics,
    _Snapshot,
    build_resource_collection_policy,
    ResourceCollectionMixin,
)
from .console_summary import (
    build_phase_resource_console_summary,
    build_session_resource_console_summary,
    build_step_resource_console_summary,
    render_phase_resource_console_summary,
    render_session_resource_console_summary,
    render_step_resource_console_summary,
)
from .events import ResourceEventMixin
from .profiles import build_run_resource_profile
from .startup import ResourceStartupMixin

if TYPE_CHECKING:
    from accelerate import Accelerator

    from library.config.dataclasses.output import ResourceMonitorConfig


logger = logging.getLogger(__name__)

DEFAULT_CONSOLE_PHASE_SUMMARY_PHASES = frozenset(
    {
        "startup.dataset_manifest",
        "startup.metadata",
        "cache.latents",
        "cache.text_encoder",
        "training.prep.trainables",
        "training.prep.accelerator",
        "checkpoint.save",
    }
)


def _normalize_phase_summary_mode(value: object) -> str:
    if isinstance(value, bool):
        return "verbose" if value else "off"
    if value is None:
        return "off"
    stripped = str(value).strip().lower()
    if stripped == "true":
        return "verbose"
    if stripped == "false":
        return "off"
    if stripped:
        return stripped
    return "off"


@dataclass
class _PhaseStartState:
    """Captured state for phase duration and sampled peak logging."""

    started_at: float
    start_snapshot: _Snapshot
    sampled_peak_gpu_used_mb: float | None = None


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
        optimizer_state_multiplier: float | None = None,
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
        optimizer_state_multiplier: float | None = None,
    ) -> None:
        _ = deepspeed_enabled, deepspeed_zero_stage, optimizer_state_multiplier
        return

    @property
    def jsonl_path(self) -> Path | None:
        return None


class BasicResourceMonitor(ResourceStartupMixin, ResourceEventMixin, ResourceCollectionMixin):
    """Low-overhead monitor for phase summaries and periodic step snapshots."""

    def __init__(
        self,
        *,
        accelerator: Accelerator,
        resource_monitor_config: ResourceMonitorConfig,
        output_jsonl_path: Path | None,
        run_identifier: str | int | None = None,
        config_name: str | None = None,
        git_sha: str | None = None,
        git_dirty: bool | None = None,
        metadata_runtime: MetadataRuntime | None = None,
    ):
        self._accelerator = accelerator
        self._resource_monitor_config = resource_monitor_config
        self._rank_scope = resource_monitor_config.rank_scope
        self._phase_summary = _normalize_phase_summary_mode(resource_monitor_config.phase_summary)
        self._component_breakdown = resource_monitor_config.component_breakdown
        self._log_every_n_steps = resource_monitor_config.log_every_n_steps
        self._mode = resource_monitor_config.mode
        self._device_scope = resource_monitor_config.device_scope
        self._max_collection_ms = resource_monitor_config.max_collection_ms
        self._collection_policy = build_resource_collection_policy(self._mode)

        self._rank = int(getattr(accelerator, "process_index", 0))
        self._world_size = int(getattr(accelerator, "num_processes", 1))

        self._process = psutil.Process()
        self._session_started_at: float | None = None
        self._phase_states: dict[str, _PhaseStartState] = {}
        self._last_step_sample: tuple[int, float] | None = None
        self._session_ended = False

        self._latest_gpu_used_mb: float | None = None
        self._latest_gpu_used_by_device_mb: dict[str, float] | None = None
        self._latest_collection_ms: float | None = None
        self._dropped_samples = 0
        self._latest_deep_counters: _DeepCounters | None = None
        self._latest_deep_window_active: bool | None = None
        self._pending_step_deep_counters: _DeepCounters | None = None
        self._pending_step_deep_window_active: bool | None = None

        self._jsonl_path = output_jsonl_path
        self._jsonl_file = None
        self._jsonl_events_since_flush = 0
        self._jsonl_flush_mode = resource_monitor_config.jsonl_flush_mode
        self._jsonl_flush_every_n_events = resource_monitor_config.jsonl_flush_every_n_events
        self._metadata_runtime = metadata_runtime
        self._resource_fact_sequence = 0
        self._resource_status_sequence = 0

        self._run_identifier = self._normalize_metadata_value(run_identifier)
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

    def _should_log_phase_summary(self, phase_name: str) -> bool:
        if self._phase_summary == "off":
            return False
        if self._phase_summary == "verbose":
            return True
        if phase_name in DEFAULT_CONSOLE_PHASE_SUMMARY_PHASES:
            return True
        return is_training_epoch_phase(phase_name)

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

    def _format_gpu(self, value: float | None) -> str:
        return "n/a" if value is None else f"{value:.0f}MB"

    def _format_cpu(self, value: float | None) -> str:
        return "n/a" if value is None else f"{value:.0f}MB"

    def _record_sampled_metrics(self, sample: _SampledMetrics) -> None:
        self._latest_gpu_used_mb = sample.gpu_used_mb
        self._latest_gpu_used_by_device_mb = sample.gpu_used_by_device_mb
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
        self._record_collector_status(
            collector_id=f"resource_monitor.{source}",
            status="budget_exceeded",
            degraded=True,
            reason="collection_budget_exceeded",
            message=f"collection time {collection_ms:.2f}ms exceeded budget {self._max_collection_ms:.2f}ms",
            metadata={
                "collection_ms": collection_ms,
                "max_collection_ms": self._max_collection_ms,
            },
        )

    def _record_collector_status(
        self,
        *,
        collector_id: str,
        status: str,
        degraded: bool,
        reason: str,
        message: str | None = None,
        fallback_collector_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        metadata_runtime = getattr(self, "_metadata_runtime", None)
        run_identifier = getattr(self, "_run_identifier", None)
        if metadata_runtime is None or run_identifier is None:
            return

        sequence = self._resource_status_sequence
        self._resource_status_sequence = sequence + 1
        status_facts = ResourceCollectorStatusFacts(
            status_identifier=f"{run_identifier}:collector_status:{self._rank}:{sequence}:{collector_id}:{status}",
            run_identifier=run_identifier,
            collector_id=collector_id,
            status=status,
            degraded=degraded,
            reason=reason,
            ts=time.time(),
            rank=self._rank,
            world_size=self._world_size,
            message=message,
            fallback_collector_id=fallback_collector_id,
            metadata={} if metadata is None else dict(metadata),
        )
        metadata_runtime.file(status_facts)

    def _file_run_resource_profile(self, *, generated_at: float | None = None) -> None:
        metadata_runtime = self._metadata_runtime
        run_identifier = self._run_identifier
        if metadata_runtime is None or run_identifier is None:
            return
        try:
            view = ResourceRunView.from_snapshot(metadata_runtime.snapshot(), run_identifier=run_identifier)
            profile = build_run_resource_profile(view, generated_at=generated_at)
            if profile is not None:
                metadata_runtime.file(profile)
        except Exception as exc:  # pragma: no cover - profile filing must not affect shutdown
            self._warn_once("resource_profile", "resource profile derivation failed: %s", exc)

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

            session_summary = build_session_resource_console_summary(
                duration_s=duration,
                snapshot=end_snapshot,
                gpu_used_mb=self._latest_gpu_used_mb,
            )
            session_log_message = render_session_resource_console_summary(session_summary)
            self._log_info_external(session_log_message.message, *session_log_message.args)

            self._emit_event(
                event="session_end",
                duration_ms=duration * 1000,
                snapshot=end_snapshot,
                dropped_samples=self._dropped_samples if self._mode in {"sampled", "deep"} else None,
                force_flush=True,
            )
            self._file_run_resource_profile(generated_at=time.time())

            self._session_ended = True
            self._phase_states.clear()
        except Exception as exc:  # pragma: no cover - defensive safety net
            self._warn_once("end_session", "resource monitor end_session failed: %s", exc)
        finally:
            self._close_jsonl_stream()

    def phase_start(self, name: str) -> None:
        try:
            if not self._should_emit_this_rank():
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
            if not self._should_emit_this_rank():
                return

            self._process_sampler_updates(max_items=1024)
            start_state = self._phase_states.pop(name, None)
            if start_state is None:
                return

            end_snapshot = self._collect_snapshot(reset_peak=False)
            duration = time.perf_counter() - start_state.started_at

            if self._should_log_phase_summary(name):
                phase_summary = build_phase_resource_console_summary(
                    phase_name=name,
                    duration_s=duration,
                    start_snapshot=start_state.start_snapshot,
                    end_snapshot=end_snapshot,
                    sampled_peak_gpu_used_mb=start_state.sampled_peak_gpu_used_mb,
                )
                phase_log_message = render_phase_resource_console_summary(phase_summary)
                self._log_info_external(phase_log_message.message, *phase_log_message.args)

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
            step_summary = build_step_resource_console_summary(
                global_step=global_step,
                epoch=epoch,
                snapshot=snapshot,
                gpu_used_mb=self._latest_gpu_used_mb,
                steps_per_sec=steps_per_sec,
            )
            step_log_message = render_step_resource_console_summary(step_summary)
            self._log_info_external(step_log_message.message, *step_log_message.args)

            self._emit_event(
                event="step_sample",
                global_step=global_step,
                epoch=epoch,
                steps_per_sec=steps_per_sec,
                snapshot=snapshot,
                deep_counters=self._pending_step_deep_counters,
                deep_window_active=self._pending_step_deep_window_active,
                dropped_samples=self._dropped_samples if self._mode in {"sampled", "deep"} else None,
            )
        except Exception as exc:  # pragma: no cover - defensive safety net
            self._warn_once("step_end", "resource monitor step_end failed (step=%s): %s", global_step, exc)


class SampledResourceMonitor(BasicResourceMonitor):
    """Sampled/deep monitor with a daemon sampler thread and bounded queue."""

    def __init__(
        self,
        *,
        accelerator: Accelerator,
        resource_monitor_config: ResourceMonitorConfig,
        output_jsonl_path: Path | None,
        run_identifier: str | int | None = None,
        config_name: str | None = None,
        git_sha: str | None = None,
        git_dirty: bool | None = None,
        metadata_runtime: MetadataRuntime | None = None,
    ):
        super().__init__(
            accelerator=accelerator,
            resource_monitor_config=resource_monitor_config,
            output_jsonl_path=output_jsonl_path,
            run_identifier=run_identifier,
            config_name=config_name,
            git_sha=git_sha,
            git_dirty=git_dirty,
            metadata_runtime=metadata_runtime,
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
            self._emit_event(
                event="step_sample",
                gpu_used_mb=sample.gpu_used_mb,
                gpu_used_by_device_mb=sample.gpu_used_by_device_mb,
                gpu_used_source=sample.gpu_used_source,
                gpu_used_quality=sample.gpu_used_quality,
                cpu_rss_mb=sample.cpu_rss_mb,
                cpu_vms_mb=sample.cpu_vms_mb,
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
        self._pending_step_deep_counters = deep_counters
        self._pending_step_deep_window_active = deep_window_active if self._mode == "deep" else None
        try:
            super().step_end(global_step, epoch)
        finally:
            self._pending_step_deep_counters = None
            self._pending_step_deep_window_active = None

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
    run_identifier: str | int | None = None,
    config_name: str | None = None,
    git_sha: str | None = None,
    git_dirty: bool | None = None,
    metadata_runtime: MetadataRuntime | None = None,
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
            run_identifier=run_identifier,
            config_name=config_name,
            git_sha=resolved_git_sha,
            git_dirty=resolved_git_dirty,
            metadata_runtime=metadata_runtime,
        )

    return BasicResourceMonitor(
        accelerator=accelerator,
        resource_monitor_config=resource_monitor_config,
        output_jsonl_path=output_jsonl_path,
        run_identifier=run_identifier,
        config_name=config_name,
        git_sha=resolved_git_sha,
        git_dirty=resolved_git_dirty,
        metadata_runtime=metadata_runtime,
    )
