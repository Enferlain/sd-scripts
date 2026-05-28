"""Lightweight runtime trace recorder for startup/training regressions."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from library.logging.phase_tags import (
    EVENT_TRAINING_FIRST_STEP_STARTED,
    EVENT_TRAINING_FIRST_STEP_SYNCED,
    EVENT_TRAINING_PROGRESS_BAR_STARTED,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RuntimeTracePhaseRow:
    """One completed duration-bearing phase span."""

    tag: str
    start_offset_s: float
    end_offset_s: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class RuntimeTraceEventRow:
    """One point-in-time milestone event."""

    tag: str
    offset_s: float


@dataclass(slots=True)
class RuntimeTrace:
    """Low-overhead monotonic runtime trace recorder.

    This recorder assumes single-threaded access from the trainer/runtime path.
    It is intentionally lightweight and does not provide synchronization.
    """

    launched_perf: float
    clock: Callable[[], float] = time.perf_counter
    _open_phases: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _phase_rows: list[RuntimeTracePhaseRow] = field(default_factory=list, init=False, repr=False)
    _event_rows: list[RuntimeTraceEventRow] = field(default_factory=list, init=False, repr=False)
    _phase_totals: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _warned_keys: set[str] = field(default_factory=set, init=False, repr=False)
    _finished_offset_s: float | None = field(default=None, init=False, repr=False)

    def _offset_s(self, perf_value: float) -> float:
        return max(0.0, perf_value - self.launched_perf)

    def _warn_once(self, key: str, message: str, *args: object) -> None:
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        logger.warning(message, *args)

    def phase_start(self, tag: str) -> None:
        """Record the start of one named phase span."""
        if tag in self._open_phases:
            self._warn_once(f"duplicate_start:{tag}", "Runtime trace phase %s was started twice without ending; ignoring duplicate start.", tag)
            return
        self._open_phases[tag] = self.clock()

    def phase_end(self, tag: str) -> None:
        """Record the end of one named phase span."""
        started_at = self._open_phases.get(tag)
        if started_at is None:
            self._warn_once(f"missing_start:{tag}", "Runtime trace phase %s ended without a matching start; ignoring unmatched end.", tag)
            return

        ended_at = self.clock()
        start_offset_s = self._offset_s(started_at)
        end_offset_s = self._offset_s(ended_at)
        duration_s = max(0.0, end_offset_s - start_offset_s)
        self._phase_rows.append(
            RuntimeTracePhaseRow(
                tag=tag,
                start_offset_s=start_offset_s,
                end_offset_s=end_offset_s,
                duration_s=duration_s,
            )
        )
        self._phase_totals[tag] = self._phase_totals.get(tag, 0.0) + duration_s
        del self._open_phases[tag]

    def event(self, tag: str) -> None:
        """Record one point-in-time runtime milestone."""
        self._event_rows.append(RuntimeTraceEventRow(tag=tag, offset_s=self._offset_s(self.clock())))

    def finish(self) -> None:
        """Freeze the final run-end offset.

        This records when the run exited, regardless of success or failure, so
        failed runs still have comparable launch-to-end timing.
        """
        if self._finished_offset_s is not None:
            return
        self._finished_offset_s = self._offset_s(self.clock())

    def _first_event_offset(self, tag: str) -> float | None:
        for event in self._event_rows:
            if event.tag == tag:
                return event.offset_s
        return None

    def summary(self) -> dict[str, object]:
        """Return a machine-readable trace summary."""
        milestones = {
            "time_to_progress_bar_s": self._first_event_offset(EVENT_TRAINING_PROGRESS_BAR_STARTED),
            "time_to_first_step_started_s": self._first_event_offset(EVENT_TRAINING_FIRST_STEP_STARTED),
            "time_to_first_synced_step_s": self._first_event_offset(EVENT_TRAINING_FIRST_STEP_SYNCED),
            "time_to_run_ended_s": self._finished_offset_s,
        }

        return {
            "phases": [
                {
                    "tag": row.tag,
                    "start_offset_s": row.start_offset_s,
                    "end_offset_s": row.end_offset_s,
                    "duration_s": row.duration_s,
                }
                for row in self._phase_rows
            ],
            "events": [{"tag": row.tag, "offset_s": row.offset_s} for row in self._event_rows],
            "phase_totals": dict(self._phase_totals),
            "milestones": milestones,
            "open_phases": [
                {"tag": tag, "start_offset_s": self._offset_s(started_at)}
                for tag, started_at in self._open_phases.items()
            ],
        }


__all__ = [
    "RuntimeTrace",
    "RuntimeTraceEventRow",
    "RuntimeTracePhaseRow",
]
