from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any

from tqdm.auto import tqdm

from library.logging.summaries import TrainingStartupSummary, render_training_startup_summary


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, level.upper())


@dataclass(slots=True)
class MainProcessConsole:
    """Repo-owned console emitter for canonical human-facing training output."""

    is_main_process: bool = True
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _rich_console: Any | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            from rich.console import Console

            self._rich_console = Console(stderr=True)
        except ImportError:
            self._rich_console = None

    def log(self, message: str, *args, level: str | int = "info", tag: str | None = None, stacklevel: int = 2) -> None:
        if not self.is_main_process:
            return
        rendered = f"[{tag}] {message}" if tag is not None else message
        self.logger.log(_resolve_level(level), rendered, *args, stacklevel=stacklevel)

    def log_external(self, message: str, *args, level: str | int = "info", tag: str | None = None, stacklevel: int = 2) -> None:
        if not self.is_main_process:
            return
        with tqdm.external_write_mode(file=sys.stderr):
            self.log(message, *args, level=level, tag=tag, stacklevel=stacklevel + 1)

    def print_external(self, message: str) -> None:
        if not self.is_main_process:
            return
        with tqdm.external_write_mode(file=sys.stderr):
            print(message, file=sys.stderr)

    def log_block(self, body: str, *, level: str | int = "info", stacklevel: int = 2) -> None:
        if not self.is_main_process:
            return
        del level, stacklevel
        if self._rich_console is not None:
            self._rich_console.print(body)
            return
        print(body, file=sys.stderr)

    def log_startup_summary(self, summary: TrainingStartupSummary, *, stacklevel: int = 2) -> None:
        self.log_block(render_training_startup_summary(summary), stacklevel=stacklevel)
