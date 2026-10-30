"""One loop-confined scalar feedback target with typed local stages.

The owner counts tickets until feedback acknowledgement, across every local
handoff and selected call. A mailbox slot stays owned through publication and
feedback. Control runs independently of a producer blocked on that slot.
Selected coroutines are never cancelled by coordinator interruption: their
binding ABI promises physical completion before returning/raising.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from .local_runtime import LocalMetrics, Stage, invoke_stage
from .prepare import Prepared, ResolvedCall
from .payloads import INTEGER, RESERVED_OUTCOMES, LocalOutcomeError


async def drain[T](completion: asyncio.Future[T], interrupted: Callable[[], None]) -> tuple[T, asyncio.CancelledError | None]:
    """Observe an owned lifetime despite repeated coordinator cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while not completion.done():
        try:
            await asyncio.shield(completion)
        except asyncio.CancelledError as error:
            if completion.cancelled():
                break
            cancellation = cancellation or error
            interrupted()
        except BaseException:
            break  # Observe the completed physical failure via result().
    return completion.result(), cancellation


@dataclass(frozen=True)
class CallFact:
    origin: str
    role: str
    input: Any
    output: Any
    outcome: str
    entered: int
    physical_end: int
    released: int


@dataclass(frozen=True)
class Decision:
    kind: str
    key: int
    revision: int
    bias: int
    clock: int


@dataclass(frozen=True)
class TicketFact:
    key: int
    source_revision: int
    policy_revision: int
    policy_bias: int
    value: int
    selected: int
    calls: tuple[CallFact, ...]
    publication_ack: int | None
    feedback_ack: int | None
    terminal: str
    retired: int


@dataclass(frozen=True)
class Report:
    status: str
    tickets: tuple[TicketFact, ...]
    decisions: tuple[Decision, ...]
    high_water: int
    stop_clock: int
    released: int
    errors: tuple[str, ...]
    diagnostics: tuple[tuple[str, int, int], ...]
    local_dispatches: int = 0


@dataclass
class _Ticket:
    key: int
    policy_revision: int
    policy_bias: int
    value: int
    selected: int
    calls: list[CallFact] = field(default_factory=list)
    publication_ack: int | None = None
    feedback_ack: int | None = None


@dataclass
class ScalarFrame:
    owner: ScalarSession
    ticket: _Ticket
    origin: str
    role: str
    value: Any
    result: object = None
    outcome: str = "entered_failed"
    entered: int = 0
    ended: int = 0
    observed_input: Any = None

    def __enter__(self) -> ScalarFrame:
        contract = self.owner.prepared.local_contracts.get(self.origin)
        if contract is not None and not contract.input.validate(self.value):
            raise ValueError(f"{self.origin}: invalid input schema before entry")
        self.observed_input = contract.input.observe(self.value) if contract is not None else self.value
        self.entered = self.owner._fact("entered:" + self.origin, self.ticket.key)
        self.owner._active_calls += 1
        return self

    def physical_end(self) -> None:
        self.ended = self.owner._fact("physical_end:" + self.origin, self.ticket.key)

    def validate(self, result: Any) -> Any:
        self.result = result if type(result) is int else None
        self.outcome = "invalid_output"
        contract = self.owner.prepared.local_contracts.get(self.origin)
        codec = contract.output if contract is not None else INTEGER
        if not codec.validate(result) or (self.role == "publish" and result != self.value):
            raise ValueError(f"{self.origin}: invalid returned scalar/receipt after entry")
        self.result = codec.observe(result)
        self.outcome = "returned"
        return result

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None) -> None:
        undeclared = False
        contract = self.owner.prepared.local_contracts.get(self.origin)
        if isinstance(error, LocalOutcomeError):
            undeclared = (
                not isinstance(error.outcome, str)
                or error.outcome in RESERVED_OUTCOMES
                or contract is None
                or error.outcome not in contract.outcomes
            )
            self.outcome = "undeclared_outcome" if undeclared else error.outcome
        self.owner._active_calls -= 1
        released = self.owner._fact("call_released:" + self.origin, self.ticket.key)
        self.ticket.calls.append(
            CallFact(
                self.origin,
                self.role,
                self.observed_input,
                self.result,
                self.outcome,
                self.entered,
                self.ended,
                released,
            )
        )
        if undeclared:
            raise ValueError(f"{self.origin}: undeclared selected outcome") from error


class ScalarSession:
    """Explicit open-source lifecycle; callers request stop independently."""

    def __init__(self, prepared: Prepared, *, diagnostic_limit: int = 32):
        self.prepared = prepared
        self.local_metrics = LocalMetrics()
        self._changed = asyncio.Event()
        self._feedback: asyncio.Queue[tuple[_Ticket, int, asyncio.Future[None]] | None] = asyncio.Queue()
        self._active: dict[int, _Ticket] = {}
        self._mailbox: deque[tuple[_Ticket, int]] = deque()
        self._completed: list[TicketFact] = []
        self._decisions: list[Decision] = []
        self._diagnostics: deque[tuple[str, int, int]] = deque(maxlen=diagnostic_limit)
        self._errors: list[str] = []
        self._clock = self._cursor = self._revision = self._bias = 0
        self._producer_done = self._aborting = self._stopping = self._started = False
        self._stop_clock = self._active_calls = self._high_water = 0
        self.publication_waiting: set[int] = set()
        self.report: Report | None = None

    def _fact(self, name: str, key: int = -1) -> int:
        self._clock += 1
        self._diagnostics.append((name, key, self._clock))
        self._changed.set()
        return self._clock

    async def wait_for(self, predicate: Callable[[], bool]) -> None:
        # All state is confined to this event loop, including stop(). No await
        # separates predicate/clear; no producer can change state in between.
        while not predicate():
            self._changed.clear()
            await self._changed.wait()

    @property
    def feedback_count(self) -> int:
        return self._revision

    @property
    def active_calls(self) -> int:
        return self._active_calls

    @property
    def outstanding(self) -> int:
        return len(self._active)

    @property
    def buffered(self) -> int:
        return len(self._mailbox)

    def stop(self) -> None:
        if not self._stopping:
            self._stopping = True
            self._stop_clock = self._fact("selection_closed")

    def _failed(self, ticket: _Ticket, error: BaseException) -> None:
        self._errors.append(f"ticket {ticket.key}: {type(error).__name__}: {error}")
        self.stop()

    def _retire(self, ticket: _Ticket, terminal: str) -> None:
        removed = self._active.pop(ticket.key)
        if removed is not ticket:
            raise RuntimeError("ticket retirement lost active ownership")
        retired = self._fact(terminal, ticket.key)
        self._completed.append(
            TicketFact(
                ticket.key,
                0,
                ticket.policy_revision,
                ticket.policy_bias,
                ticket.value,
                ticket.selected,
                tuple(ticket.calls),
                ticket.publication_ack,
                ticket.feedback_ack,
                terminal,
                retired,
            )
        )

    async def _invoke(self, ticket: _Ticket, value: int, call: ResolvedCall) -> int:
        return await invoke_stage(
            Stage(call.origin, call.binding.invoke),
            value,
            lambda source, incoming: ScalarFrame(self, ticket, source, call.binding.role, incoming),
        )

    async def _producer(self) -> None:
        ticket: _Ticket | None = None
        try:
            while True:
                await self.wait_for(lambda: self._stopping or len(self._active) < self.prepared.horizon)
                if self._stopping:
                    break
                key, self._cursor = self._cursor, self._cursor + 1
                selected = self._fact("selected", key)
                ticket = _Ticket(key, self._revision, self._bias, key + self._bias, selected)
                self._active[key] = ticket
                self._high_water = max(self._high_water, len(self._active))
                self._decisions.append(Decision("selected", key, self._revision, self._bias, selected))
                value = await self.prepared.local.run(
                    ticket.value,
                    lambda source, incoming, selected=ticket: ScalarFrame(self, selected, source, "map", incoming),
                    self.local_metrics,
                )
                if len(self._mailbox) == self.prepared.capacity:
                    self.publication_waiting.add(key)
                    self._fact("publication_waiting", key)
                await self.wait_for(lambda: self._aborting or len(self._mailbox) < self.prepared.capacity)
                self.publication_waiting.discard(key)
                if self._aborting:
                    self._retire(ticket, "abandoned")
                    ticket = None
                    break
                self._mailbox.append((ticket, value))
                self._fact("handoff_published", key)
                ticket = None  # The mailbox now owns this ticket.
        except BaseException as error:
            if ticket is None:
                raise
            self._failed(ticket, error)
            self._retire(ticket, "failed")
        finally:
            self._producer_done = True
            self._fact("producer_closed")

    async def _consumer(self) -> None:
        ticket: _Ticket | None = None
        try:
            while True:
                await self.wait_for(lambda: bool(self._mailbox) or self._producer_done)
                if not self._mailbox:
                    break
                ticket, value = self._mailbox[0]  # Retain capacity until feedback acknowledgement.
                score = await self._invoke(ticket, value, self.prepared.score)
                await self._invoke(ticket, score, self.prepared.publish)
                ticket.publication_ack = self._fact("publication_ack", ticket.key)
                acknowledgement: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                self._feedback.put_nowait((ticket, score, acknowledgement))
                await acknowledgement
                self._mailbox.popleft()
                self._fact("handoff_released", ticket.key)
                ticket = None
        except BaseException as error:
            if ticket is None:
                raise
            self._failed(ticket, error)
            self._aborting = True
            self._retire(ticket, "failed")
            self._mailbox.popleft()
            while self._mailbox:
                abandoned, _ = self._mailbox.popleft()
                self._retire(abandoned, "abandoned")
            self._fact("consumer_aborted")
        finally:
            self._feedback.put_nowait(None)

    def _apply_feedback(self, ticket: _Ticket, score: int) -> None:
        if self._active.get(ticket.key) is not ticket or ticket.publication_ack is None or ticket.feedback_ack is not None:
            raise ValueError(f"ticket {ticket.key}: feedback lacks a live, unacknowledged publication")
        self._revision += 1
        self._bias = score % 5
        ticket.feedback_ack = self._fact("feedback_ack", ticket.key)
        self._decisions.append(Decision("feedback", ticket.key, self._revision, self._bias, ticket.feedback_ack))
        self._retire(ticket, "acknowledged")

    async def _control(self) -> None:
        while (message := await self._feedback.get()) is not None:
            ticket, score, acknowledgement = message
            try:
                self._apply_feedback(ticket, score)
            except BaseException as error:
                acknowledgement.set_exception(error)
            else:
                acknowledgement.set_result(None)

    async def run(self) -> Report:
        if self._started:
            raise RuntimeError("a prepared session instance has one lifetime")
        self._started = True
        tasks = [asyncio.create_task(call()) for call in (self._producer, self._consumer, self._control)]
        completion = asyncio.gather(*tasks)
        # Shield the entire owner lifetime, including calls and cleanup. A
        # second cancellation must not release a still-running physical use.
        _, cancelled = await drain(completion, self.stop)
        assert not self._active and self._active_calls == 0 and not self._mailbox
        released = self._fact("session_released")
        self.report = Report(
            "cancelled" if cancelled else "failed" if self._errors else "stopped",
            tuple(sorted(self._completed, key=lambda fact: fact.key)),
            tuple(self._decisions),
            self._high_water,
            self._stop_clock,
            released,
            tuple(self._errors),
            tuple(self._diagnostics),
            self.local_metrics.dispatches,
        )
        if cancelled:
            raise cancelled
        return self.report
