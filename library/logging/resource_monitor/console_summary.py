"""Resource-domain console summaries for live monitor output."""

from __future__ import annotations

from dataclasses import dataclass

from .collect import _Snapshot


@dataclass(frozen=True, slots=True)
class ResourceConsoleLogMessage:
    """A rendered logger call that remains presentation-only."""

    message: str
    args: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class SessionResourceConsoleSummary:
    """Resource-domain values needed for the session console summary."""

    duration_s: float
    snapshot: _Snapshot
    gpu_used_mb: float | None


@dataclass(frozen=True, slots=True)
class PhaseResourceConsoleSummary:
    """Resource-domain values needed for the phase console summary."""

    phase_name: str
    duration_s: float
    start_snapshot: _Snapshot
    end_snapshot: _Snapshot
    sampled_peak_gpu_used_mb: float | None


@dataclass(frozen=True, slots=True)
class StepResourceConsoleSummary:
    """Resource-domain values needed for the step console summary."""

    global_step: int
    epoch: int
    snapshot: _Snapshot
    gpu_used_mb: float | None
    steps_per_sec: float | None


def build_session_resource_console_summary(
    *,
    duration_s: float,
    snapshot: _Snapshot,
    gpu_used_mb: float | None,
) -> SessionResourceConsoleSummary:
    """Build the presentation input for a session resource summary."""
    return SessionResourceConsoleSummary(duration_s=duration_s, snapshot=snapshot, gpu_used_mb=gpu_used_mb)


def build_phase_resource_console_summary(
    *,
    phase_name: str,
    duration_s: float,
    start_snapshot: _Snapshot,
    end_snapshot: _Snapshot,
    sampled_peak_gpu_used_mb: float | None,
) -> PhaseResourceConsoleSummary:
    """Build the presentation input for a phase resource summary."""
    return PhaseResourceConsoleSummary(
        phase_name=phase_name,
        duration_s=duration_s,
        start_snapshot=start_snapshot,
        end_snapshot=end_snapshot,
        sampled_peak_gpu_used_mb=sampled_peak_gpu_used_mb,
    )


def build_step_resource_console_summary(
    *,
    global_step: int,
    epoch: int,
    snapshot: _Snapshot,
    gpu_used_mb: float | None,
    steps_per_sec: float | None,
) -> StepResourceConsoleSummary:
    """Build the presentation input for a step resource summary."""
    return StepResourceConsoleSummary(
        global_step=global_step,
        epoch=epoch,
        snapshot=snapshot,
        gpu_used_mb=gpu_used_mb,
        steps_per_sec=steps_per_sec,
    )


def render_session_resource_console_summary(summary: SessionResourceConsoleSummary) -> ResourceConsoleLogMessage:
    """Render the existing session summary logger call from a domain summary."""
    return ResourceConsoleLogMessage(
        message="Resource session summary: duration=%.2fs, gpu_allocated=%s, gpu_reserved=%s, gpu_peak=%s, gpu_used=%s, cpu_rss=%s",
        args=(
            summary.duration_s,
            _format_memory_mb(summary.snapshot.gpu_allocated_mb),
            _format_memory_mb(summary.snapshot.gpu_reserved_mb),
            _format_memory_mb(summary.snapshot.gpu_peak_allocated_mb),
            _format_memory_mb(summary.gpu_used_mb),
            _format_memory_mb(summary.snapshot.cpu_rss_mb),
        ),
    )


def render_phase_resource_console_summary(summary: PhaseResourceConsoleSummary) -> ResourceConsoleLogMessage:
    """Render the existing phase summary logger call from a domain summary."""
    sampled_peak_suffix = ""
    if summary.sampled_peak_gpu_used_mb is not None:
        sampled_peak_suffix = f", gpu_used_peak={_format_memory_mb(summary.sampled_peak_gpu_used_mb)}"
    return ResourceConsoleLogMessage(
        message="Resource phase[%s]: duration=%.2fs, gpu_allocated=%s->%s, gpu_reserved=%s->%s, gpu_peak=%s%s, cpu_rss=%s->%s",
        args=(
            summary.phase_name,
            summary.duration_s,
            _format_memory_mb(summary.start_snapshot.gpu_allocated_mb),
            _format_memory_mb(summary.end_snapshot.gpu_allocated_mb),
            _format_memory_mb(summary.start_snapshot.gpu_reserved_mb),
            _format_memory_mb(summary.end_snapshot.gpu_reserved_mb),
            _format_memory_mb(summary.end_snapshot.gpu_peak_allocated_mb),
            sampled_peak_suffix,
            _format_memory_mb(summary.start_snapshot.cpu_rss_mb),
            _format_memory_mb(summary.end_snapshot.cpu_rss_mb),
        ),
    )


def render_step_resource_console_summary(summary: StepResourceConsoleSummary) -> ResourceConsoleLogMessage:
    """Render the existing step summary logger call from a domain summary."""
    if summary.steps_per_sec is None:
        return ResourceConsoleLogMessage(
            message="Resource step[%s|epoch=%s]: gpu_allocated=%s, gpu_reserved=%s, gpu_used=%s, cpu_rss=%s",
            args=(
                summary.global_step,
                summary.epoch,
                _format_memory_mb(summary.snapshot.gpu_allocated_mb),
                _format_memory_mb(summary.snapshot.gpu_reserved_mb),
                _format_memory_mb(summary.gpu_used_mb),
                _format_memory_mb(summary.snapshot.cpu_rss_mb),
            ),
        )
    return ResourceConsoleLogMessage(
        message="Resource step[%s|epoch=%s]: %.2f steps/s, gpu_allocated=%s, gpu_reserved=%s, gpu_used=%s, cpu_rss=%s",
        args=(
            summary.global_step,
            summary.epoch,
            summary.steps_per_sec,
            _format_memory_mb(summary.snapshot.gpu_allocated_mb),
            _format_memory_mb(summary.snapshot.gpu_reserved_mb),
            _format_memory_mb(summary.gpu_used_mb),
            _format_memory_mb(summary.snapshot.cpu_rss_mb),
        ),
    )


def _format_memory_mb(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}MB"
