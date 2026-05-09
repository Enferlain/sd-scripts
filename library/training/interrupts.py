from __future__ import annotations

import signal
import sys
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from library.training.runners.trainer import Trainer


DOUBLE_CTRL_C_COOLDOWN_SECONDS = 5.0


def _emit_interrupt_warning(trainer: Trainer) -> None:
    message = (
        f"[interrupt] press Ctrl+C again within "
        f"{int(DOUBLE_CTRL_C_COOLDOWN_SECONDS)}s to interrupt training"
    )
    console = getattr(trainer, "_console", None)
    if console is not None:
        console.print_external(message)
        return
    print(message, file=sys.stderr)


@contextmanager
def install_double_ctrl_c_guard(trainer: Trainer) -> Iterator[None]:
    """Require a second Ctrl+C before interrupting the active training run."""

    previous_handler = signal.getsignal(signal.SIGINT)
    last_interrupt_at: float | None = None

    def _handle_sigint(signum, frame) -> None:
        nonlocal last_interrupt_at
        now = time.monotonic()

        if (
            last_interrupt_at is None
            or (now - last_interrupt_at) > DOUBLE_CTRL_C_COOLDOWN_SECONDS
        ):
            last_interrupt_at = now
            _emit_interrupt_warning(trainer)
            return

        signal.signal(signal.SIGINT, previous_handler)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_handler)
