"""Unit tests for the lightweight runtime trace recorder."""

from __future__ import annotations

from library.logging.runtime_trace import RuntimeTrace


class _Clock:
    def __init__(self, initial: float = 0.0):
        self.value = initial

    def __call__(self) -> float:
        return self.value


class _RaisingClock:
    def __init__(self):
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self.calls == 1:
            return 0.0
        raise RuntimeError("clock failed")


def test_runtime_trace_records_phase_rows_and_events_with_launch_relative_offsets():
    clock = _Clock()
    trace = RuntimeTrace(launched_perf=0.0, clock=clock)

    trace.phase_start("startup.metadata")
    clock.value = 1.25
    trace.phase_end("startup.metadata")
    clock.value = 2.0
    trace.event("training.progress_bar.started")

    summary = trace.summary()
    assert summary["phases"] == [
        {
            "tag": "startup.metadata",
            "start_offset_s": 0.0,
            "end_offset_s": 1.25,
            "duration_s": 1.25,
        }
    ]
    assert summary["events"] == [{"tag": "training.progress_bar.started", "offset_s": 2.0}]


def test_runtime_trace_aggregates_repeated_serial_phases():
    clock = _Clock()
    trace = RuntimeTrace(launched_perf=0.0, clock=clock)

    trace.phase_start("training.prep.optimizer_groups")
    clock.value = 1.0
    trace.phase_end("training.prep.optimizer_groups")
    clock.value = 2.0
    trace.phase_start("training.prep.optimizer_groups")
    clock.value = 4.5
    trace.phase_end("training.prep.optimizer_groups")

    summary = trace.summary()
    assert len(summary["phases"]) == 2
    assert summary["phase_totals"]["training.prep.optimizer_groups"] == 3.5


def test_runtime_trace_handles_unmatched_starts_and_ends_without_crashing():
    clock = _Clock()
    trace = RuntimeTrace(launched_perf=0.0, clock=clock)

    trace.phase_end("training.prep.optimizer_groups")
    trace.phase_start("training.prep.optimizer_groups")
    trace.phase_start("training.prep.optimizer_groups")
    clock.value = 3.0
    summary = trace.summary()

    assert summary["phases"] == []
    assert summary["open_phases"] == [{"tag": "training.prep.optimizer_groups", "start_offset_s": 0.0}]


def test_runtime_trace_preserves_open_phase_if_phase_end_recording_fails():
    trace = RuntimeTrace(launched_perf=0.0, clock=_RaisingClock())

    trace.phase_start("training.prep.optimizer_groups")

    try:
        trace.phase_end("training.prep.optimizer_groups")
    except RuntimeError as exc:
        assert str(exc) == "clock failed"
    else:  # pragma: no cover - defensive expectation
        raise AssertionError("phase_end should propagate unexpected clock failures")

    summary = trace.summary()
    assert summary["phases"] == []
    assert summary["open_phases"] == [{"tag": "training.prep.optimizer_groups", "start_offset_s": 0.0}]


def test_runtime_trace_computes_milestone_deltas_and_finish_offset():
    clock = _Clock()
    trace = RuntimeTrace(launched_perf=0.0, clock=clock)

    clock.value = 1.0
    trace.event("training.progress_bar.started")
    clock.value = 2.0
    trace.event("training.first_step.started")
    clock.value = 3.5
    trace.event("training.first_step.synced")
    clock.value = 5.0
    trace.finish()

    milestones = trace.summary()["milestones"]
    assert milestones["time_to_progress_bar_s"] == 1.0
    assert milestones["time_to_first_step_started_s"] == 2.0
    assert milestones["time_to_first_synced_step_s"] == 3.5
    assert milestones["time_to_run_ended_s"] == 5.0
