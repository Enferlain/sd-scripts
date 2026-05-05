from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum

import torch


class StrategyPhase(StrEnum):
    """Named strategy execution phases exposed through strategy context."""

    TRAIN = "train"
    VALIDATION = "validation"
    SAMPLING = "sampling"


@dataclass(frozen=True, slots=True)
class TrainingContext:
    """Training-loop facts visible while strategy work is executing."""

    global_step: int | None = None
    is_train: bool | None = None


@dataclass(frozen=True, slots=True)
class DenoiserContext:
    """Denoiser-forward facts visible while a strategy denoiser operation runs."""

    timesteps: torch.Tensor | None = None
    sample_indices: tuple[int, ...] | None = None
    batch_size: int | None = None


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Scoped runtime facts published by strategy execution."""

    phase: StrategyPhase | None = None
    model_family: str | None = None
    training: TrainingContext | None = None
    denoiser: DenoiserContext | None = None


_CURRENT_STRATEGY_CONTEXT: ContextVar[StrategyContext | None] = ContextVar(
    "current_strategy_context",
    default=None,
)


def current_strategy_context() -> StrategyContext | None:
    """Return the active strategy context, if strategy work published one."""

    return _CURRENT_STRATEGY_CONTEXT.get()


def require_strategy_context() -> StrategyContext:
    """Return the active strategy context or fail for consumers that require it."""

    context = current_strategy_context()
    if context is None:
        raise RuntimeError("No strategy context is active.")
    return context


@contextmanager
def publish_strategy_context(context: StrategyContext) -> Iterator[StrategyContext]:
    """Publish strategy context for a scoped operation and restore prior state."""

    token = _CURRENT_STRATEGY_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_STRATEGY_CONTEXT.reset(token)
