"""Bounded host-thread placement, not a second run description or scheduler.

Only selected synchronous CPU calls enter here. Their coordinator retains the
lease until host work and cleanup really finish. Cancellation requests a stop;
it cannot kill Python/native work or certify that no effects occurred.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import torch


class NotEntered(RuntimeError):
    """Dispatch was interrupted/rejected without entering selected work."""

    def __init__(self, cause: BaseException):
        super().__init__(str(cause))
        self.cause = cause


class PhysicalFailed(RuntimeError):
    """Failure around selected work, with any coordinator interruption intact."""

    def __init__(self, cause: BaseException, cancellation: asyncio.CancelledError | None):
        super().__init__(str(cause))
        self.cause, self.cancellation = cause, cancellation


@dataclass(frozen=True)
class Finished[T]:
    value: T
    cancellation: asyncio.CancelledError | None


@dataclass(frozen=True)
class CpuContext:
    """Supported numerical TLS plus this target's uncached autocast policy.

    Torch's weight cache is shared across threads and keyed by source tensor,
    not the requested dtype. Separate worker contexts must not reuse it.
    """

    gradients: bool
    inference: bool
    autocast: bool
    dtype: torch.dtype

    @classmethod
    def capture(cls) -> CpuContext:
        return cls(
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
            torch.is_autocast_enabled("cpu"),
            torch.get_autocast_dtype("cpu"),
        )

    def invoke[T](self, call: Callable[[], T]) -> T:
        with (
            torch.inference_mode(self.inference),
            torch.set_grad_enabled(self.gradients),
            torch.autocast("cpu", enabled=self.autocast, dtype=self.dtype, cache_enabled=False),
        ):
            return call()


class Workers:
    def __init__(self, limit: int):
        if type(limit) is not int or limit < 1:
            raise ValueError("Worker limit must be a positive integer")
        self.limit = limit
        self.executor: ThreadPoolExecutor | None = None
        self.slots: asyncio.Semaphore | None = None
        self.active = 0

    @contextmanager
    def lifetime(self) -> Generator[None]:
        if self.executor is not None:
            raise RuntimeError("Worker lifetime is already active")
        self.slots = asyncio.Semaphore(self.limit)
        # No executor/threads exist in an unpublished preparation attempt.
        with ThreadPoolExecutor(max_workers=self.limit, thread_name_prefix="candidate-work") as executor:
            self.executor = executor
            primary = None
            try:
                yield
            except BaseException as error:
                primary = error
                raise
            finally:
                # Every execute() drains its physical future, even through
                # repeated cancellation. Root/TaskGroup exit joins those calls.
                self.executor = None
                self.slots = None
                if self.active:
                    failure = RuntimeError("Worker root exited with undrained submissions")
                    if primary is not None:
                        raise BaseExceptionGroup("Run failure and invalid worker lifetime", (primary, failure)) from primary
                    raise failure

    async def execute[T](self, call: Callable[[], T], stop: threading.Event) -> Finished[T]:
        executor, slots = self.executor, self.slots
        if executor is None or slots is None:
            raise NotEntered(RuntimeError("Worker lifetime is not active"))
        python_context, numerical_context = contextvars.copy_context(), CpuContext.capture()
        try:
            await slots.acquire()
        except asyncio.CancelledError as error:
            raise NotEntered(error) from error
        self.active += 1
        try:
            try:
                physical = cast(Future[T], executor.submit(python_context.run, numerical_context.invoke, call))
            except BaseException as error:
                raise NotEntered(error) from error
            pending = asyncio.wrap_future(physical)
            cancelled = None
            while True:
                if cancelled is not None and pending.done():
                    # Also observe a physical exception (including one named
                    # CancelledError); it must not become an infinite drain.
                    try:
                        return Finished(pending.result(), cancelled)
                    except BaseException as error:
                        raise PhysicalFailed(error, cancelled) from error
                try:
                    value = await asyncio.shield(pending)
                    return Finished(value, cancelled)
                except asyncio.CancelledError as error:
                    cancelled = cancelled or error
                    stop.set()
                    if physical.cancel():
                        # A successfully cancelled physical future never ran.
                        pending.cancel()
                        raise NotEntered(cancelled) from cancelled
                    # Running or already-completed work still needs observation.
                    # Do not cancel the asyncio wrapper or release the slot.
                except BaseException as error:
                    raise PhysicalFailed(error, cancelled) from error
        finally:
            self.active -= 1
            slots.release()
