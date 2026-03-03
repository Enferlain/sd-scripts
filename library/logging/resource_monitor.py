"""
Config-driven resource monitoring for trainer phases and optimization steps.

The monitor is intentionally lightweight on the hot path:
- `off` mode uses a strict no-op implementation.
- non-`off` modes collect cheap allocator/RSS counters without mandatory
  synchronization.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import psutil
import torch

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
    cpu_rss_mb: float


@dataclass(frozen=True)
class _PhaseStartState:
    """Captured state for phase duration/memory delta logging."""

    started_at: float
    start_snapshot: _Snapshot


class ResourceMonitor(Protocol):
    """Resource monitor contract used by Trainer and phases."""

    def start_session(self) -> None: ...

    def end_session(self) -> None: ...

    def phase_start(self, name: str) -> None: ...

    def phase_end(self, name: str) -> None: ...

    def step_end(self, global_step: int, epoch: int) -> None: ...

    def emit_startup_component_memory(
        self, components: Mapping[str, Any] | Iterable[tuple[str, Any]] | None, optimizer_name: str
    ) -> None: ...


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
        self, components: Mapping[str, Any] | Iterable[tuple[str, Any]] | None, optimizer_name: str
    ) -> None:
        return


class BasicResourceMonitor:
    """Low-overhead monitor for phase summaries and periodic step snapshots."""

    def __init__(self, *, accelerator: Accelerator, resource_monitor_config: ResourceMonitorConfig):
        self._accelerator = accelerator
        self._resource_monitor_config = resource_monitor_config
        self._rank_scope = resource_monitor_config.rank_scope
        self._phase_summary = resource_monitor_config.phase_summary
        self._component_breakdown = resource_monitor_config.component_breakdown
        self._log_every_n_steps = resource_monitor_config.log_every_n_steps
        self._mode = resource_monitor_config.mode

        self._process = psutil.Process()
        self._session_started_at: float | None = None
        self._session_start_snapshot: _Snapshot | None = None
        self._phase_states: dict[str, _PhaseStartState] = {}
        self._last_step_sample: tuple[int, float] | None = None
        self._session_ended = False

    def _should_emit_this_rank(self) -> bool:
        if self._rank_scope == "all":
            return True
        return self._accelerator.is_main_process

    def _is_cuda_visible(self) -> bool:
        return torch.cuda.is_available()

    def _collect_snapshot(self, *, reset_peak: bool = False) -> _Snapshot:
        if self._is_cuda_visible():
            if reset_peak:
                torch.cuda.reset_peak_memory_stats()
            gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            gpu_peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            gpu_allocated_mb = None
            gpu_reserved_mb = None
            gpu_peak_allocated_mb = None

        cpu_rss_mb = self._process.memory_info().rss / (1024 * 1024)
        return _Snapshot(
            gpu_allocated_mb=gpu_allocated_mb,
            gpu_reserved_mb=gpu_reserved_mb,
            gpu_peak_allocated_mb=gpu_peak_allocated_mb,
            cpu_rss_mb=cpu_rss_mb,
        )

    def _format_gpu(self, value: float | None) -> str:
        return "n/a" if value is None else f"{value:.0f}MB"

    def _normalize_component_items(
        self, components: Mapping[str, Any] | Iterable[tuple[str, Any]] | None
    ) -> list[tuple[str, Any]]:
        if not components:
            return []

        if isinstance(components, Mapping):
            raw_items = list(components.items())
        else:
            raw_items = list(components)

        normalized: list[tuple[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            name, module = item
            if not isinstance(name, str) or module is None:
                continue
            normalized.append((name, module))
        return normalized

    def start_session(self) -> None:
        if not self._should_emit_this_rank():
            return
        if self._session_started_at is not None:
            return

        self._session_started_at = time.perf_counter()
        self._session_start_snapshot = self._collect_snapshot(reset_peak=False)
        logger.info(
            "Resource monitor started: mode=%s, rank_scope=%s, step_log_every=%s",
            self._mode,
            self._rank_scope,
            self._log_every_n_steps,
        )

    def end_session(self) -> None:
        if self._session_ended or not self._should_emit_this_rank():
            return
        if self._session_started_at is None:
            self._session_ended = True
            return

        end_snapshot = self._collect_snapshot(reset_peak=False)
        duration = time.perf_counter() - self._session_started_at

        logger.info(
            (
                "Resource session summary: duration=%.2fs, "
                "gpu_allocated=%s, gpu_reserved=%s, gpu_peak=%s, cpu_rss=%.0fMB"
            ),
            duration,
            self._format_gpu(end_snapshot.gpu_allocated_mb),
            self._format_gpu(end_snapshot.gpu_reserved_mb),
            self._format_gpu(end_snapshot.gpu_peak_allocated_mb),
            end_snapshot.cpu_rss_mb,
        )
        self._session_ended = True

    def phase_start(self, name: str) -> None:
        if not self._should_emit_this_rank() or not self._phase_summary:
            return

        start_snapshot = self._collect_snapshot(reset_peak=True)
        self._phase_states[name] = _PhaseStartState(started_at=time.perf_counter(), start_snapshot=start_snapshot)

    def phase_end(self, name: str) -> None:
        if not self._should_emit_this_rank() or not self._phase_summary:
            return

        start_state = self._phase_states.pop(name, None)
        if start_state is None:
            return

        end_snapshot = self._collect_snapshot(reset_peak=False)
        duration = time.perf_counter() - start_state.started_at
        logger.info(
            (
                "Resource phase[%s]: duration=%.2fs, "
                "gpu_allocated=%s->%s, gpu_reserved=%s->%s, gpu_peak=%s, "
                "cpu_rss=%.0fMB->%.0fMB"
            ),
            name,
            duration,
            self._format_gpu(start_state.start_snapshot.gpu_allocated_mb),
            self._format_gpu(end_snapshot.gpu_allocated_mb),
            self._format_gpu(start_state.start_snapshot.gpu_reserved_mb),
            self._format_gpu(end_snapshot.gpu_reserved_mb),
            self._format_gpu(end_snapshot.gpu_peak_allocated_mb),
            start_state.start_snapshot.cpu_rss_mb,
            end_snapshot.cpu_rss_mb,
        )

    def step_end(self, global_step: int, epoch: int) -> None:
        if not self._should_emit_this_rank():
            return
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
            logger.info(
                "Resource step[%s|epoch=%s]: gpu_allocated=%s, gpu_reserved=%s, cpu_rss=%.0fMB",
                global_step,
                epoch,
                self._format_gpu(snapshot.gpu_allocated_mb),
                self._format_gpu(snapshot.gpu_reserved_mb),
                snapshot.cpu_rss_mb,
            )
        else:
            logger.info(
                "Resource step[%s|epoch=%s]: %.2f steps/s, gpu_allocated=%s, gpu_reserved=%s, cpu_rss=%.0fMB",
                global_step,
                epoch,
                steps_per_sec,
                self._format_gpu(snapshot.gpu_allocated_mb),
                self._format_gpu(snapshot.gpu_reserved_mb),
                snapshot.cpu_rss_mb,
            )

    def emit_startup_component_memory(
        self, components: Mapping[str, Any] | Iterable[tuple[str, Any]] | None, optimizer_name: str
    ) -> None:
        if not self._should_emit_this_rank():
            return
        if not self._component_breakdown:
            return
        component_items = self._normalize_component_items(components)
        if not component_items:
            return

        lines = ["Resource startup estimates (parameter memory):"]
        total_param_bytes = 0
        total_trainable_bytes = 0
        for name, module in component_items:
            param_bytes = 0
            trainable_bytes = 0
            for p in module.parameters():
                bytes_count = p.numel() * p.element_size()
                param_bytes += bytes_count
                if p.requires_grad:
                    trainable_bytes += bytes_count

            total_param_bytes += param_bytes
            total_trainable_bytes += trainable_bytes
            lines.append(
                f"  - {name}: params={param_bytes / (1024 * 1024):.1f}MB, trainable={trainable_bytes / (1024 * 1024):.1f}MB"
            )

        optimizer_state_multiplier = 2 if "adam" in optimizer_name.lower() or "lion" in optimizer_name.lower() else 1
        optimizer_state_bytes = total_trainable_bytes * optimizer_state_multiplier
        lines.append(f"  - gradients (est): {total_trainable_bytes / (1024 * 1024):.1f}MB")
        lines.append(f"  - optimizer_state (est): {optimizer_state_bytes / (1024 * 1024):.1f}MB [{optimizer_name}]")
        lines.append(
            "  - total (est): "
            f"{(total_param_bytes + total_trainable_bytes + optimizer_state_bytes) / (1024 * 1024):.1f}MB"
        )
        logger.info("\n".join(lines))


def create_resource_monitor(
    *,
    accelerator: Accelerator,
    resource_monitor_config: ResourceMonitorConfig,
    output_dir: str | Path | None = None,
) -> ResourceMonitor:
    """Create a monitor instance from typed logging config."""
    _ = output_dir  # Reserved for future JSONL artifact placement.

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

    if mode in {"sampled", "deep"}:
        logger.warning(
            "resource_monitor.mode=%s currently uses basic collector behavior; sampled/deep collectors are planned next",
            mode,
        )

    return BasicResourceMonitor(accelerator=accelerator, resource_monitor_config=resource_monitor_config)
