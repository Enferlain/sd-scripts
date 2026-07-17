from __future__ import annotations

import os
import signal
import sys
import time
from contextlib import contextmanager
from types import FrameType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from library.training.runners.trainer import Trainer


DOUBLE_CTRL_C_COOLDOWN_SECONDS = 5.0


def _write_stderr_immediately(message: str) -> None:
    """Write without waiting for Rich, tqdm, logging, or buffered text I/O."""
    payload = f"{message}\n".encode("utf-8", errors="replace")
    try:
        os.write(2, payload)
    except OSError:
        print(message, file=sys.stderr, flush=True)


@contextmanager
def install_double_ctrl_c_guard(trainer: Trainer) -> Iterator[None]:
    """Guard the first Ctrl+C while preserving normal Python interruption.

    Behavior:
    - the first Ctrl+C prints a warning and lets training continue
    - a second Ctrl+C within ``DOUBLE_CTRL_C_COOLDOWN_SECONDS`` restores the
      original SIGINT handler and raises ``KeyboardInterrupt``
    - later Ctrl+C presses use ordinary Python behavior and may interrupt cleanup

    Notes:
    - signal delivery may appear delayed while Python is busy in long-running C/CUDA work
    - interrupt cleanup still relies on the trainer's unconditional ``finally`` path
    """
    del trainer  # Kept in the public API for existing call sites.

    previous_handler = signal.getsignal(signal.SIGINT)
    first_interrupt_at: float | None = None
    guard_accepted = False

    def _handle_sigint(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        nonlocal first_interrupt_at, guard_accepted

        # Normally the original handler has already been restored before
        # another SIGINT can arrive. This only covers a pending re-entrant call.
        if guard_accepted:
            raise KeyboardInterrupt

        now = time.monotonic()

        if (
            first_interrupt_at is None
            or (now - first_interrupt_at) > DOUBLE_CTRL_C_COOLDOWN_SECONDS
        ):
            first_interrupt_at = now
            _write_stderr_immediately(
                f"[interrupt] press Ctrl+C again within "
                f"{int(DOUBLE_CTRL_C_COOLDOWN_SECONDS)}s to interrupt training"
            )
            return

        guard_accepted = True
        signal.signal(signal.SIGINT, previous_handler)
        _write_stderr_immediately(
            "[interrupt] interruption accepted; shutting down cleanly "
            "(press Ctrl+C again to interrupt cleanup)"
        )
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        yield
    finally:
        # Idempotent if the second press already restored it.
        signal.signal(signal.SIGINT, previous_handler)
