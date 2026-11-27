"""The run's kernel: facts, failure and closing, waiting and waking, tasks and the handoff to workers (D15).

Every other runtime component works through it. It runs on one event loop.
A wait names the keys whose changes can make its condition hold, so waking
scales with what changed; when every live activity waits and no selected
work runs, the run has stalled. The handoff of selected work to a worker
thread is in one place, :meth:`Kernel.submit`, so a scheduler built for the
runtime can replace it (sd-scripts-syv.40).
"""

import asyncio
import random
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from library.iro.runtime.state import Cancelled, Fact, Identity, RunFailure
from library.iro.seams import Completion

Observer = Callable[[Fact], None]

# Every waiter also waits under this key, so failure and cancellation wake everyone.
EVERYONE = ("everyone",)


class Kernel:
    """What every runtime component shares: the observer, the run's failures, its waits and its tasks.

    ``explore`` perturbs the schedule for conformance testing: at every wait,
    and before a call group is submitted, the run yields to other activities
    a random number of times. ``checked`` makes a wait whose condition came to hold without a
    change on its keys fail the run.
    """

    def __init__(self, observe: Observer | None, explore: random.Random | None, checked: bool):
        self.observe = observe or (lambda fact: None)
        self.explore = explore
        self.checked = checked
        self.cancelled = False
        self.failed = False
        self.closed = threading.Event()  # set as admission closes, for work on worker threads: a call group's guard before each call
        self.outstanding: list[Completion] = []  # the run's physical work not yet seen complete: its settlement waits for it (D15)
        self.errors: list[str] = []
        self._failures: list[BaseException] = []
        self._stall: RunFailure | None = None
        self._waiting = 0
        self.joins: set[asyncio.Future] = set()
        self.working = 0
        self.live = 1
        self._waiters: dict[tuple, set[asyncio.Future]] = defaultdict(set)
        self._watched: dict[asyncio.Future, Callable[[], bool]] = {}  # checked mode: each wait's condition
        self._wake_check_due = False
        self.executor: ThreadPoolExecutor | None = None
        self.tasks: set[asyncio.Task] = set()

    def fact(self, owner: str, invocation: Identity, kind: str, *values: object) -> None:
        self.observe(Fact(owner, invocation, kind, values))

    @property
    def closing(self) -> bool:
        return self.cancelled or self.failed

    # Failure and closing.

    def note_failure(self, failed: BaseException) -> None:
        if not isinstance(failed, Cancelled) and not any(f is failed for f in self._failures):
            self._failures.append(failed)
            self.errors.append(str(failed))
            self.failed = True
            self.closed.set()
            self.wake_all()

    def missed(self, what: str) -> None:
        """Checked mode found what lowering excluded: the run fails, and goes on settling with the full result."""
        self.note_failure(RunFailure(f"lowering excluded what the full check found: {what}"))

    def admission_point(self) -> None:
        if self.closing:
            raise Cancelled("the run was cancelled" if self.cancelled else "a sibling failed, so admission closed")

    def cancel(self) -> None:
        self.cancelled = True
        self.closed.set()
        self.wake_all()

    # The run's physical work.

    def queued(self, *completions: Completion | None) -> None:
        """Keeps the completions of physical work the run queued until they are done; those done already are dropped."""
        self.outstanding = [c for c in self.outstanding if not c.done()]
        self.outstanding.extend(c for c in completions if c is not None)

    async def settle_work(self) -> list[str]:
        """At the run's end however it ended: waits, off the coordinator, for the physical work the run queued; why any failed."""
        pending, self.outstanding = self.outstanding, []
        return await self.submit(_wait_all, pending) if pending else []

    # Waiting and waking.

    def changed(self, *keys: tuple) -> None:
        """Wakes the waiters on these keys; no keys wakes nobody."""
        woken: set[asyncio.Future] = set()
        for key in keys:
            woken |= self._waiters.pop(key, set())
        self._wake(woken)
        if self.checked and not self._wake_check_due:
            self._wake_check_due = True
            asyncio.get_running_loop().call_soon(self.check_wakes)

    def check_wakes(self) -> bool:
        """Checked mode, once the run yields: a wait whose condition holds must have been woken."""
        self._wake_check_due = False
        if self.closing:
            return False
        for future, ready in list(self._watched.items()):
            if not future.done() and ready():
                self.missed("a wait's condition came to hold without a change on the keys it waits on")
                return True
        return False

    def wake_all(self) -> None:
        """Wakes every waiter: for failure, cancellation and a stall, which every activity must see."""
        woken = set().union(*self._waiters.values()) if self._waiters else set()
        self._waiters.clear()
        self._wake(woken)

    def _wake(self, woken: set[asyncio.Future]) -> None:
        for future in woken:
            if not future.done():
                future.set_result(None)
                self._waiting -= 1

    async def until(self, ready: Callable[[], bool], *keys: tuple) -> None:
        """Waits until ``ready``, woken by a change on one of ``keys``, so waking scales with what changed.

        When every live activity waits and no selected work runs, the run has stalled.
        """
        keys = (*keys, EVERYONE)
        while True:
            if self.explore is not None:
                for _ in range(self.explore.randrange(3)):
                    await asyncio.sleep(0)
            if ready():
                return
            if self._stall is not None:
                raise self._stall
            future = asyncio.get_running_loop().create_future()
            for key in keys:
                self._waiters[key].add(future)
            if self.checked:
                self._watched[future] = ready
            self._waiting += 1
            self.check_stall()
            try:
                if self._stall is not None:
                    raise self._stall
                await future
            finally:
                if not future.done():
                    future.cancel()
                    self._waiting -= 1
                self._watched.pop(future, None)
                for key in keys:
                    waiting = self._waiters.get(key)
                    if waiting is not None:
                        waiting.discard(future)
                        if not waiting:
                            del self._waiters[key]

    def check_stall(self) -> None:
        joining = sum(1 for join in self.joins if not join.done())
        if self._stall is None and self._waiting + joining >= self.live and self.working == 0:
            if self.checked and self.check_wakes():
                return  # not a stall: a lost wake, which fails the run with its own reason
            waited = sorted({str(key) for key in self._waiters if key != EVERYONE})
            self._stall = RunFailure(
                f"no progress: every activity waits for another and no selected work runs; waiting on {', '.join(waited) or 'nothing'}"
            )
            self.note_failure(self._stall)
            self.wake_all()

    async def jitter(self) -> None:
        """Explored runs yield here a random number of times; the call path does before submitting a group."""
        if self.explore is not None:
            for _ in range(self.explore.randrange(3)):
                await asyncio.sleep(0)

    # Tasks and the handoff to workers.

    def spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coroutine)
        self.tasks.add(task)
        self.live += 1
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        self.live -= 1
        # After the callbacks already queued, so a parent's join has seen its last branch finish.
        asyncio.get_running_loop().call_soon(self.check_stall)

    async def join(self, tasks: list[asyncio.Task]) -> list[Any]:
        """Waits for every task, their failures among the results; the stall check counts the waiting parent."""
        joined = asyncio.gather(*tasks, return_exceptions=True)
        self.joins.add(joined)
        try:
            return await joined
        finally:
            self.joins.discard(joined)

    async def submit(self, work: Callable[..., Any], *args: object) -> Any:
        """Runs selected work on a worker thread and waits for it without blocking; the run counts it as working."""
        assert self.executor is not None
        self.working += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(self.executor, work, *args)
        finally:
            self.working -= 1


def _wait_all(completions: list[Completion]) -> list[str]:
    """Waits for each completion on a worker thread; what each that failed raised."""
    failures = []
    for completion in completions:
        try:
            completion.wait()
        except Exception as failed:
            failures.append(f"the device work the run queued failed to complete: {failed}")
    return failures


__all__ = ["EVERYONE", "Kernel", "Observer"]
