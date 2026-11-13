"""The coordinator: one per process, executing a prepared run (D15, D16).

It realizes every lifecycle protocol of the accepted strategy as the
reference semantics defines it, and reports the same owner-visible facts, so
each realization is checked against the reference. It runs on one event
loop and never blocks on selected work: calls run on worker threads through
the implementation and device seams, a fused group as one submission, and
the coordinator waits for their completion without blocking.

It executes the run's lowered form (:mod:`library.iro.prepare.lowered`): a
body's values sit in numbered slots and each step is bound to its fixed data,
so a round dispatches on step kinds and never reads the strategy again. The
coordinator keeps runtime control state only: leases and their protection,
retained results, claims, requests, stops and device turns. Owners hold the
state leases reach.

Bookkeeping is bounded: an invocation's records live while it or its scope
is live, productions' observed versions while they produce, a participant's
pending device completions until later device work on it finds them done,
and facts go to the observer as they happen.

In checked mode, for conformance tests, every check that lowering narrows
also runs in full, and the run fails if the full check finds what lowering
excluded.
"""

import asyncio
import random
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from library.iro.operation_contracts import AccessMode, Admission, Boundary, LifecycleProtocol, ResourceUse, VersionSelection
from library.iro.prepare import HOST, Prepared
from library.iro.prepare.lowered import (
    BORROW,
    DISPOSE,
    FORK,
    INVOKE,
    RELATIONSHIP,
    RELEASE,
    REQUEST,
    RESOURCE,
    STOP,
    STREAM,
    TAKE,
    UNREALIZED,
    Body,
    Call,
    Calls,
    HandlerInfo,
    Invoke,
    Named,
    RequestStep,
    Settling,
    Step,
    StopStep,
    StreamStep,
    lower,
)
from library.iro.runtime.state import (
    TRANSITIONS,
    Cancelled,
    Fact,
    HandlerState,
    Held,
    Identity,
    Outcome,
    Product,
    Protection,
    Request,
    RunFailure,
    State,
    Stream,
    Ticket,
)
from library.iro.seams import CallContext, Completion, Device, Implementation, Owner, Use, Version

Observer = Callable[[Fact], None]

# Every waiter also waits under this key, so failure and cancellation wake everyone.
_EVERYONE = ("everyone",)


@dataclass
class _Turn:
    """One device's turns: the admission holding it and those waiting, in request order (D15)."""

    holder: Identity | None = None
    depth: int = 0
    waiting: list[Identity] = field(default_factory=list)


@dataclass(eq=False)
class _Admission:
    """A device admission in progress: it ends when its work has ended and its retained results are released."""

    device: str
    identity: Identity
    body: Identity
    products: set[int] = field(default_factory=set)
    work_ended: bool = False


class _Records:
    """Which statements started, and which are still live, in each scope: what a wait for a source to settle reads."""

    def __init__(self) -> None:
        self.started: dict[tuple[str, Identity], int] = {}
        self.live: dict[tuple[str, Identity], int] = defaultdict(int)
        self.by_scope: dict[Identity, set[tuple[str, Identity]]] = defaultdict(set)

    def begin(self, owner: str, scopes: list[Identity]) -> None:
        for scope in scopes:
            key = (owner, scope)
            self.started[key] = 1
            self.by_scope[scope].add(key)
            self.live[key] += 1

    def retire(self, owner: str, identity: Identity, scopes: list[Identity]) -> bool:
        """Drops a finished invocation; true when a scope enclosing it has nothing of its statement left live."""
        settled = False
        for scope in scopes:
            key = (owner, scope)
            self.live[key] -= 1
            if not self.live[key]:
                del self.live[key]
                # A waiter shares a scope that encloses the invocation; its own scope ending (every
                # iteration of a repeat) settles nothing anyone waits for.
                settled = settled or scope != identity
        return settled

    def close(self, scope: Identity) -> None:
        """The scope's invocation ended: the records kept in it go."""
        if self.by_scope:
            for key in self.by_scope.pop(scope, ()):
                self.started.pop(key, None)

    def empty(self) -> bool:
        return not (self.started or self.live or self.by_scope)

    def size(self) -> int:
        return len(self.started) + len(self.live) + len(self.by_scope)

    def settled(self, chain: tuple[str, ...], scope: Identity) -> bool:
        started = False
        for owner in chain:
            if self.live.get((owner, scope)):
                return False
            started = started or (owner, scope) in self.started
        return started


class Coordinator:
    """Executes one prepared run against its owners, implementations and devices."""

    def __init__(
        self,
        prepared: Prepared,
        owners: Mapping[str, Owner],
        installed: Mapping[str, Implementation],
        devices: Mapping[str, Device],
        *,
        observe: Observer | None = None,
        explore: random.Random | None = None,
        checked: bool = False,
    ):
        """``explore`` perturbs the schedule for conformance testing: at every
        wait and submission the coordinator yields to other activities a
        random number of times. It changes when things happen, never what may happen.

        ``checked`` runs every check lowering narrows in full beside the
        lowered one, and fails the run if the full check finds anything the
        lowered one excluded.
        """
        self.prepared = prepared
        self.explore = explore
        self.checked = checked
        self.lowered = lower(prepared, installed)
        self.owners, self.installed, self.devices = dict(owners), dict(installed), dict(devices)
        if HOST.name not in self.devices:
            raise ValueError("the coordinator needs the host device")
        self.observe = observe or (lambda fact: None)
        # Live runtime state.
        self.protection: dict[str, Protection] = defaultdict(Protection)
        self.leases: dict[int, Held] = {}
        self.products: dict[tuple[str, int], Product] = {}
        self.streams: dict[str, Stream] = {}
        self.handlers: dict[str, HandlerState] = defaultdict(HandlerState)
        self.outcomes: dict[Identity, Outcome] = {}
        self.request_of: dict[Identity, Request] = {}
        self.emissions: dict[Identity, list[tuple[RequestStep, Identity, tuple[object, ...]]]] = defaultdict(list)
        self.stops: set[tuple[Identity, str]] = set()
        self.call_counts: dict[str, int] = defaultdict(int)
        self.turns: dict[str, _Turn] = defaultdict(_Turn)
        self.admissions: dict[int, _Admission] = {}
        self._admission_of: dict[Identity, _Admission] = {}
        self._productions: dict[Identity, list[Version]] = {}
        self._busy: set[str] = set()
        self._ordinals: dict[Identity, dict[str, int]] = defaultdict(dict)
        self._waited = self.lowered.waited
        self._records = _Records()
        self._all_records = _Records()  # checked mode: every invocation's records, to compare with the lowered ones
        self._next_token = 0
        self.cancelled = False
        self._failed = False
        self.errors: list[str] = []
        self._failures: list[BaseException] = []
        self._stall: RunFailure | None = None
        self._waiting = 0
        self._joins: set[asyncio.Future] = set()
        self._working = 0
        self._live = 1
        self._waiters: dict[tuple, set[asyncio.Future]] = defaultdict(set)
        self._watched: dict[asyncio.Future, Callable[[], bool]] = {}  # checked mode: each wait's condition
        self._wake_check_due = False
        self._executor: ThreadPoolExecutor | None = None
        self._tasks: set[asyncio.Task] = set()
        self._statements: dict[str, Callable[[Any, tuple[object, ...], Identity], Awaitable[tuple[object, ...]]]] = {
            INVOKE: self._invoke,
            RESOURCE: self._resource,
            RELATIONSHIP: self._relationship,
            STREAM: self._stream,
            RELEASE: self._release,
            TAKE: self._take_step,
            BORROW: self._borrow,
            DISPOSE: self._dispose,
            FORK: self._fork,
            REQUEST: self._request,
            STOP: self._stop,
            UNREALIZED: self._unrealized,
        }
        self._protocols: dict[LifecycleProtocol, Callable[[Invoke, tuple[object, ...], Identity], Awaitable[tuple[object, ...]]]] = {
            LifecycleProtocol.SEQUENCE: self._sequence,
            LifecycleProtocol.CHOICE: self._choice,
            LifecycleProtocol.PARALLEL: self._parallel,
            LifecycleProtocol.REPEAT: self._repeat,
            LifecycleProtocol.ACCESS: self._access,
            LifecycleProtocol.FILTER: self._filter,
            LifecycleProtocol.PRODUCER: self._producer,
            LifecycleProtocol.HANDLER: self._handler,
        }

    # Facts, lifecycle and waiting.

    def _fact(self, owner: str, invocation: Identity, kind: str, *values: object) -> None:
        self.observe(Fact(owner, invocation, kind, values))

    @property
    def closing(self) -> bool:
        return self.cancelled or self._failed

    def _changed(self, *keys: tuple) -> None:
        """Wakes the waiters on these keys; no keys wakes nobody."""
        woken: set[asyncio.Future] = set()
        for key in keys:
            woken |= self._waiters.pop(key, set())
        self._wake(woken)
        if self.checked and not self._wake_check_due:
            self._wake_check_due = True
            asyncio.get_running_loop().call_soon(self._check_wakes)

    def _check_wakes(self) -> bool:
        """Checked mode, once the coordinator yields: a wait whose condition holds must have been woken."""
        self._wake_check_due = False
        if self.closing:
            return False
        for future, ready in list(self._watched.items()):
            if not future.done() and ready():
                self._missed("a wait's condition came to hold without a change on the keys it waits on")
                return True
        return False

    def _wake_all(self) -> None:
        """Wakes every waiter: for failure, cancellation and a stall, which every activity must see."""
        woken = set().union(*self._waiters.values()) if self._waiters else set()
        self._waiters.clear()
        self._wake(woken)

    def _wake(self, woken: set[asyncio.Future]) -> None:
        for future in woken:
            if not future.done():
                future.set_result(None)
                self._waiting -= 1

    async def _until(self, ready: Callable[[], bool], *keys: tuple) -> None:
        """Waits until ``ready``, woken by a change on one of ``keys``, so waking scales with what changed.

        When every live activity waits and no selected work runs, the run has stalled.
        """
        keys = (*keys, _EVERYONE)
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
            self._check_stall()
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

    def _check_stall(self) -> None:
        joining = sum(1 for join in self._joins if not join.done())
        if self._stall is None and self._waiting + joining >= self._live and self._working == 0:
            if self.checked and self._check_wakes():
                return  # not a stall: a lost wake, which fails the run with its own reason
            waited = sorted({str(key) for key in self._waiters if key != _EVERYONE})
            self._stall = RunFailure(
                f"no progress: every activity waits for another and no selected work runs; waiting on {', '.join(waited) or 'nothing'}"
            )
            self._note_failure(self._stall)
            self._wake_all()

    def _note_failure(self, failed: BaseException) -> None:
        if not isinstance(failed, Cancelled) and not any(f is failed for f in self._failures):
            self._failures.append(failed)
            self.errors.append(str(failed))
            self._failed = True
            self._wake_all()

    def _missed(self, what: str) -> None:
        """Checked mode found what lowering excluded: the run fails, and goes on settling with the full result."""
        self._note_failure(RunFailure(f"lowering excluded what the full check found: {what}"))

    def _admission_point(self) -> None:
        if self.closing:
            raise Cancelled("the run was cancelled" if self.cancelled else "a sibling failed, so admission closed")

    def _identity(self, parent: Identity, owner: str) -> Identity:
        ordinals = self._ordinals[parent]
        ordinal = ordinals.get(owner, 0)
        ordinals[owner] = ordinal + 1
        return (*parent, (owner, ordinal))

    def _begin(self, identity: Identity, owner: str) -> None:
        self.outcomes[identity] = Outcome(owner, identity)
        commons = self._waited.get(owner)
        if commons is not None:
            self._records.begin(owner, _scopes(identity, commons))
        if self.checked:
            self._all_records.begin(owner, _scopes(identity))

    def _state(self, identity: Identity, state: State) -> None:
        outcome = self.outcomes[identity]
        if state not in TRANSITIONS[outcome.state]:
            raise RunFailure(f"{outcome.owner}: {outcome.state} cannot become {state}")
        outcome.state = state
        self._fact(outcome.owner, identity, "state", str(state))
        if state == State.TERMINAL:
            self._fact(outcome.owner, identity, "outcome", outcome.status, outcome.error)
            self._retire(identity, outcome.owner)

    def _retire(self, identity: Identity, owner: str) -> None:
        """Bookkeeping for a finished invocation, and for the scope it closed, is dropped (bounded bookkeeping)."""
        del self.outcomes[identity]
        commons = self._waited.get(owner)
        settled = commons is not None and self._records.retire(owner, identity, _scopes(identity, commons))
        if self._ordinals:
            self._ordinals.pop(identity, None)
        self._records.close(identity)
        if self.checked:
            self._all_records.retire(owner, identity, _scopes(identity))
            self._all_records.close(identity)
        if self.stops:
            self.stops = {(scope, target) for scope, target in self.stops if scope != identity}
        if settled:
            self._changed(("settled", owner))

    # Running.

    async def run(self) -> Outcome:
        """Runs the strategy to settlement and returns the root's outcome."""
        with ThreadPoolExecutor(max_workers=self.prepared.target.workers, thread_name_prefix="iro-work") as executor:
            self._executor = executor
            name = self.lowered.path
            root = self._identity((), name)
            self._begin(root, name)
            outcome = self.outcomes[root]
            for state in (State.PENDING, State.ADMITTED, State.RUNNING):
                self._state(root, state)
            try:
                try:
                    await self._body(self.lowered.root, (), root)
                except RunFailure:
                    raise
                except Exception as error:
                    raise RunFailure(f"{name}: {error!r}") from error
                outcome.status = "succeeded"
            except RunFailure as failed:
                self._note_failure(failed)
                outcome.status = "cancelled" if self.cancelled or isinstance(failed, Cancelled) else "failed"
                outcome.error = str(failed)
            finally:
                self._state(root, State.CLOSING)
                self._state(root, State.SETTLING)
                await self._until(lambda: self.closing or not self._pending_under(root), ("requests",))
                self._settle_requests(root)
                if self.checked and self._failed and not self.cancelled and outcome.status != "failed":
                    # A miss closes admission without raising where it was found, so the root saw only that.
                    outcome.status, outcome.error = "failed", "; ".join(self.errors)
                self._state(root, State.TERMINAL)
        return outcome

    def cancel(self) -> None:
        """Run cancellation: admission closes while entered work settles."""
        self.cancelled = True
        self._wake_all()

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coroutine)
        self._tasks.add(task)
        self._live += 1
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        self._live -= 1
        # After the callbacks already queued, so a parent's join has seen its last branch finish.
        asyncio.get_running_loop().call_soon(self._check_stall)

    # Bodies.

    async def _body(self, body: Body, args: tuple[object, ...] | list[object], identity: Identity) -> tuple[object, ...]:
        if len(args) != body.entries:
            raise RunFailure(f"{body.path}: a body receives {body.entries} entries, given {len(args)}")
        frame = [*args, *body.template]
        returned: tuple[object, ...] = ()
        failed: BaseException | None = None
        try:
            for step in body.steps:
                if isinstance(step, Calls):
                    await self._calls(step, frame, identity)
                    continue
                result = await self._statements[step.kind](step, tuple(map(frame.__getitem__, step.operands)), identity)
                if len(result) != len(step.results):
                    raise RunFailure(f"{step.path}: gave {len(result)} results for {len(step.results)}")
                for slot, value in zip(step.results, result, strict=True):
                    frame[slot] = value
            returned = tuple(map(frame.__getitem__, body.returns))
            return returned
        except BaseException as error:
            failed = error
            raise
        finally:
            if body.retained or body.claims or self.checked:
                self._end_checks(body, frame, returned, identity, failed)

    def _end_checks(
        self, body: Body, frame: list[object], returned: tuple[object, ...], identity: Identity, failed: BaseException | None
    ) -> None:
        """What a body leaves when it ends: retained results are released and claims settled, and leaving them fails it (D16)."""
        given = {id(v) for v in returned}
        leaked: RunFailure | None = None
        for slot in reversed(body.retained):
            value = frame[slot]
            if isinstance(value, Product) and value.active and id(value) not in given:
                if failed is None and leaked is None:
                    leaked = RunFailure(f"{value.key[0]}: a retained result was never released")
                self._release_product(value, "cancelled" if self.cancelled else "failed")
        for slot in body.claims:
            value = frame[slot]
            if (
                isinstance(value, Ticket)
                and value.owner == identity
                and value.disposition is None
                and not value.forks
                and id(value) not in given
            ):
                if failed is None and leaked is None:
                    leaked = RunFailure(f"{value.stream}/{value.key}: a claim was never settled")
                self._settle(value, "cancelled" if self.cancelled or isinstance(failed, Cancelled) else "failed", identity)
        if self.checked:
            for slot, value in enumerate(frame):
                if isinstance(value, Product) and slot in body.results and slot not in body.retained:
                    self._missed(f"{body.path}: a retained result in a slot the end check skips")
                if isinstance(value, Ticket) and slot not in body.claims:
                    self._missed(f"{body.path}: a claim in a slot the end check skips")
        if leaked is not None:
            raise leaked

    # Statements.

    async def _resource(self, step: Named, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        if step.word not in self.owners:
            raise RunFailure(f"{step.path}: no owner holds @{step.word}")
        return (step.word,)

    async def _relationship(self, step: Step, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        return ((args[0], args[1]),)

    async def _stream(self, step: StreamStep, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self.streams[step.name] = Stream(step.name, step.flow)
        return (step.name,)

    async def _release(self, step: Step, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self._admission_point()
        product = args[0]
        if not isinstance(product, Product):
            raise RunFailure(f"{step.path}: releases something that is not a retained result")
        self._release_product(product, "released")
        return ()

    async def _take_step(self, step: Step, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self._admission_point()
        stream = self.streams[str(args[0])]
        while True:
            await self._until(lambda: self.closing or bool(stream.ready) or stream.closed, ("stream", stream.name))
            self._admission_point()
            ticket = self._take(step, stream, identity)
            if ticket is not None:
                return (ticket,)

    async def _borrow(self, step: Step, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self._admission_point()
        return (deepcopy(self._owned(step, args[0], identity).payload),)

    async def _dispose(self, step: Named, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self._admission_point()
        self._settle(self._owned(step, args[0], identity), step.word, identity)
        return ()

    async def _fork(self, step: Step, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self._admission_point()
        ticket = self._owned(step, args[0], identity)
        ticket.forks = [Ticket(ticket.stream, ticket.key, identity, ticket.payload, ticket.versions, ticket) for _ in step.results]
        ticket.owner = None
        self._fact(ticket.stream, identity, "forked", ticket.key, len(ticket.forks))
        return tuple(ticket.forks)

    async def _request(self, step: RequestStep, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self._admission_point()
        self._arm_request(step, identity, args)
        return ()

    async def _stop(self, step: StopStep, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self._admission_point()
        settling = step.settling
        await self._until(lambda: self.closing or self._source_settled(settling, identity), *settling.keys)
        self._admission_point()
        self.stops.add((_scope(step.scope, identity), step.target))
        self._fact(step.target, identity, "stop_requested", step.source)
        self._changed(("stop", step.target))
        return ()

    async def _unrealized(self, step: Step, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self._admission_point()
        raise RunFailure(step.problem or f"{step.path}: the runtime does not realize it")

    def _owned(self, step: Step, value: object, identity: Identity) -> Ticket:
        if not isinstance(value, Ticket) or value.owner != identity or value.disposition is not None or value.forks:
            raise RunFailure(f"{step.path}: the claim is not held here")
        return value

    # Invocations.

    async def _invoke(
        self, step: Invoke, args: tuple[object, ...], parent: Identity, *, body: list[object] | None = None
    ) -> tuple[object, ...]:
        identity = self._identity(parent, step.path)
        self._begin(identity, step.path)
        outcome = self.outcomes[identity]
        self._state(identity, State.PENDING)
        if self.request_of and parent in self.request_of:
            self.request_of[identity] = self.request_of[parent]
        try:
            self._admission_point()
            self._state(identity, State.ADMITTED)
            self._state(identity, State.RUNNING)
            if step.captures_claims or self.checked:
                for ticket in (v for v in (body if body is not None else args) if isinstance(v, Ticket)):
                    if not step.captures_claims:
                        self._missed(f"{step.path}: a captured claim")
                    if ticket.owner != parent or ticket.disposition is not None:
                        raise RunFailure(f"{step.path}: a captured claim is not held by the capturing body")
                    ticket.owner = identity
            try:
                if body is not None:
                    result = await self._body(step.bodies[0], body, identity)
                else:
                    result = await self._protocol(step, args, identity)
            except RunFailure:
                raise
            except Exception as error:
                raise RunFailure(f"{step.path}: {error!r}") from error
            if step.owns_claims or self.checked:
                for value in result:
                    if isinstance(value, Ticket):
                        if not step.owns_claims:
                            self._missed(f"{step.path}: a claim among its results")
                        value.owner = parent
            outcome.status = "succeeded"
            return result
        except RunFailure as failed:
            outcome.status = "cancelled" if isinstance(failed, Cancelled) else "failed"
            outcome.error = str(failed)
            self._note_failure(failed)
            raise
        finally:
            self._state(identity, State.CLOSING)
            self._state(identity, State.SETTLING)
            if step.owns_claims or self.checked:
                for stream in self.streams.values():
                    for ticket in tuple(stream.outstanding.values()):
                        for held in _leaves(ticket):
                            if held.owner == identity and held.disposition is None:
                                if not step.owns_claims:
                                    self._missed(f"{step.path}: a claim it still owned when it settled")
                                self._settle(held, "cancelled" if self.cancelled else "failed", identity)
            if self.emissions:
                for request_step, source, request_args in self.emissions.pop(identity, ()):
                    await self._emit(request_step, source, request_args, outcome.status)
            if self.request_of:
                self.request_of.pop(identity, None)
            self._state(identity, State.TERMINAL)

    def _entry(self, step: Invoke, args: tuple[object, ...], supplied: tuple[object, ...] = (), *, role: int = 0) -> list[object]:
        entry = step.entries[role]
        values: list[object] = [None] * entry.size
        for argument, operand in entry.captures:
            values[argument] = args[operand]
        if len(entry.supplied) != len(supplied):
            raise RunFailure(f"{step.path}: the protocol supplies {len(supplied)} entries, the body takes {len(entry.supplied)}")
        for position, value in zip(entry.supplied, supplied, strict=True):
            values[position] = value
        return values

    async def _protocol(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        if step.problem is not None:
            raise RunFailure(step.problem)
        return await self._protocols[step.protocol](step, args, identity)

    async def _sequence(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        return await self._body(step.bodies[0], self._entry(step, args), identity)

    async def _choice(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        chosen = args[0]
        if step.optional:
            if chosen is not None:
                return await self._body(step.bodies[0], self._entry(step, args, (chosen,)), identity)
            return await self._body(step.bodies[1], self._entry(step, args, role=1), identity)
        if type(chosen) is not bool:
            raise RunFailure(f"{step.path}: a choice chooses on a bool")
        role = 0 if chosen else 1
        return await self._body(step.bodies[role], self._entry(step, args, role=role), identity)

    async def _parallel(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        branches = []
        for index, body in enumerate(step.bodies):
            branch = (*identity, (step.branches[index], 0))
            if identity in self.request_of:
                self.request_of[branch] = self.request_of[identity]
            used = step.branch_claims[index]
            for position, value in enumerate(args):
                if isinstance(value, Ticket) and position in used:
                    value.owner = branch
            branches.append(self._spawn(self._branch(body, self._entry(step, args, role=index), branch)))
        join = asyncio.gather(*branches, return_exceptions=True)
        self._joins.add(join)
        try:
            results = await join
        finally:
            self._joins.discard(join)
        errors = [r for r in results if isinstance(r, BaseException)]
        if self.cancelled:
            raise Cancelled("the run was cancelled")
        if errors:
            raise next((e for e in errors if not isinstance(e, Cancelled)), errors[0])
        return ()

    async def _branch(self, body: Body, args: list[object], branch: Identity) -> tuple[object, ...]:
        try:
            try:
                return await self._body(body, args, branch)
            except RunFailure:
                raise
            except Exception as error:
                raise RunFailure(f"{branch[-1][0]}: {error!r}") from error
        except RunFailure as failed:
            self._note_failure(failed)
            raise
        finally:
            self.request_of.pop(branch, None)
            self._ordinals.pop(branch, None)

    async def _repeat(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        assert step.limit is not None  # lowering gave a repeat by policy its problem
        split = len(args) - step.carried
        captures, carries = args[:split], args[split:]
        for iteration in range(step.limit):
            carries = await self._invoke(step, (), identity, body=self._entry(step, (*captures, *carries), (iteration, *carries)))
        return carries

    async def _access(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        uses = step.uses
        device = step.device
        # Leases and the device turn are taken together: nothing waits for one while holding the other (D14).
        await self._turn_queue(device, identity, lambda: all(self._can_lease(name, use) for name, use in uses), step.keys)
        leases = self._acquire(step, identity)
        if device is not None:
            self._grant_turn(device, identity, identity[:-1])
        try:
            return await self._body(step.bodies[0], self._entry(step, args, leases), identity)
        finally:
            for lease in leases:
                lease.body_ended = True
            for lease in leases:
                self._release_lease_if_done(lease)
            if device is not None:
                self._work_ended(identity)

    async def _filter(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        ticket = self._owned(step, args[0], identity)
        disposition = "failed"
        try:
            kept = await self._body(step.bodies[0], self._entry(step, args, (deepcopy(ticket.payload),)), identity)
            if len(kept) != 1 or type(kept[0]) is not bool:
                raise RunFailure(f"{step.path}: a filter's body yields one bool")
            disposition = "forwarded" if kept[0] else "filtered_out"
            return (deepcopy(ticket.payload) if kept[0] else None,)
        except Cancelled:
            disposition = "cancelled"
            raise
        finally:
            self._settle(ticket, disposition, identity)

    async def _producer(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        name = str(args[0])
        stream = self.streams[name]
        keys = (("stream", name), ("stop", step.path))
        outstanding, ready = stream.flow.max_outstanding, stream.flow.ready_bound
        try:
            index = 0
            while not self._stopped(step.path, identity) and not self.closing:
                await self._until(
                    lambda: self.closing or self._stopped(step.path, identity) or len(stream.outstanding) < outstanding,
                    *keys,
                )
                self._admission_point()
                if self._stopped(step.path, identity):
                    break
                ticket = Ticket(name, stream.next_key, identity)
                stream.next_key += 1
                stream.outstanding[ticket.key] = ticket
                production = (*identity, (step.path, index))
                self._productions[identity] = []
                try:
                    (value,) = await self._invoke(step, (), identity, body=self._entry(step, args, (index,)))
                except RunFailure as failed:
                    if ticket.disposition is None:
                        self._settle(ticket, "cancelled" if isinstance(failed, Cancelled) else "failed", production)
                    raise
                finally:
                    observed = self._productions.pop(identity, [])
                index += 1
                ticket.versions = tuple(dict.fromkeys(observed))
                if step.optional and value is None:
                    stream.outstanding.pop(ticket.key)
                    self._fact(name, identity, "nothing_produced", ticket.key)
                    self._changed(("stream", name))
                    continue
                try:
                    await self._until(
                        lambda: self.closing or self._stopped(step.path, identity) or len(stream.ready) < ready,
                        *keys,
                    )
                    self._admission_point()
                except RunFailure as failed:
                    if ticket.disposition is None:
                        self._settle(ticket, "cancelled" if isinstance(failed, Cancelled) else "failed", identity)
                    raise
                ticket.payload, ticket.owner = deepcopy(value), None
                if self._stopped(step.path, identity):
                    self._settle(ticket, "cancelled", identity)
                else:
                    stream.ready.append(ticket)
                    self._fact(name, identity, "published", ticket.key, ticket.versions)
                self._changed(("stream", name))
        finally:
            stream.closed = True
            self._changed(("stream", name))
            # Wait for consumers to settle what they took; what the producer still holds settles at the drain.
            await self._until(
                lambda: all(
                    leaf.owner is None or leaf.owner == identity
                    for t in stream.outstanding.values()
                    if t not in stream.ready
                    for leaf in _leaves(t)
                ),
                ("stream", name),
            )
            for ticket in tuple(stream.outstanding.values()):
                if ticket in stream.ready or ticket.owner == identity:
                    self._settle(ticket, "cancelled", identity)
        return ()

    async def _handler(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        handler = self.handlers[step.path]
        sources = step.sources
        key = ("handler", step.path)
        while True:
            await self._until(
                lambda: self.closing or bool(handler.pending) or all(self._source_settled(s, identity) for s in sources), *step.keys
            )
            if self.closing:
                while handler.pending:
                    request = handler.pending.popleft()
                    self._end_reservations(request)
                    self._fact(step.path, identity, "request_disposed", request.ordinal, "cancelled")
                self._changed(key, ("requests",))
                return ()
            if not handler.pending:
                return ()
            handler.active += 1
            request = handler.pending.popleft()
            self.request_of[identity] = request
            disposition = "failed"
            try:
                await self._invoke(step, (), identity, body=self._entry(step, args, (deepcopy(request.payload),)))
                disposition = "completed"
            except RunFailure as failed:
                disposition = "cancelled" if isinstance(failed, Cancelled) else "failed"
                raise
            finally:
                handler.active -= 1
                self._end_reservations(request)
                self._fact(step.path, identity, "request_settled", request.ordinal, disposition)
                del self.request_of[identity]
                self._changed(key, ("requests",))

    # Calls: a fused group is one submission to a worker thread on its device.

    async def _calls(self, step: Calls, frame: list[object], identity: Identity) -> None:
        self._admission_point()
        if step.problem is not None:
            raise RunFailure(step.problem)
        device = self.devices.get(step.group.device)
        if device is None:
            raise RunFailure(f"{step.path}: the coordinator has no device {step.group.device}")
        own_admission = not step.host and self._admission_for(identity, step.group.device) is None
        if own_admission:
            await self._turn_queue(step.group.device, identity)
            self._grant_turn(step.group.device, identity, identity)
        borrowed: list[Product] = []
        leased: list[Held] = []
        try:
            arguments: list[list[object]] = []
            program = []
            for call in step.calls:
                values = [frame[s] for s in call.slots] if not call.locals else [frame[s] if s >= 0 else None for s in call.slots]
                given = values
                if call.leases or call.retained:
                    given = list(values)
                    for position in call.leases:
                        held = values[position]
                        assert isinstance(held, Held)
                        if not held.active:
                            raise RunFailure(f"{call.path}: a lease is used after its release")
                        given[position] = held.handle
                        leased.append(held)
                    for position in call.retained:
                        product = values[position]
                        assert isinstance(product, Product)
                        if not product.active:
                            raise RunFailure(f"{call.path}: a retained result is used after its release")
                        product.borrowers += 1
                        borrowed.append(product)
                        given[position] = product.value
                if self.checked:
                    self._check_arguments(call, values)
                arguments.append(values)
                program.append((call.implementation, given, call.locals, len(call.results)))
            if len(leased) > 1:
                leased = list(dict.fromkeys(leased))
            if self.explore is not None:
                for _ in range(self.explore.randrange(3)):
                    await asyncio.sleep(0)
            # Calls not declared safe for concurrent use run one at a time per participant (D15).
            busy: set[str] = set()
            if not step.safe and leased:
                busy = {lease.resource for lease in leased}
                await self._until(lambda: self.closing or not busy & self._busy, *(("resource", r) for r in busy))
                self._admission_point()
            after = [c for lease in leased for c in self.protection[lease.resource].pending]
            after += [p.completion for p in borrowed if p.completion is not None]
            reaches = tuple(lease.handle for lease in leased)
            assert self._executor is not None
            self._busy |= busy
            self._working += 1
            try:
                reports, completions = await asyncio.get_running_loop().run_in_executor(
                    self._executor, _run_group, device, step.group.context, after, reaches, program, step.wait
                )
            finally:
                self._working -= 1
                self._busy -= busy
            if not step.wait:
                # Later work on these participants follows each call's completion (D15).
                queued = [c for c in completions if c is not None]
                for lease in leased:
                    pending = self.protection[lease.resource].pending
                    pending[:] = [c for c in pending if not c.done()]
                    pending.extend(queued)
            made: list[tuple[object, ...]] = []
            for index, call in enumerate(step.calls):
                if index >= len(reports):
                    break
                report = reports[index]
                ordinal = self.call_counts[call.path]
                self.call_counts[call.path] = ordinal + 1
                self._fact(call.path, identity, "call_entered", ordinal)
                if isinstance(report, BaseException):
                    self._fact(call.path, identity, "call_failed", ordinal, str(report))
                    raise RunFailure(f"{call.path}: {report}") from report
                results: tuple[object, ...] = report
                if len(results) != len(call.results):
                    self._fact(call.path, identity, "call_invalid", ordinal)
                    raise RunFailure(f"{call.path}: returned {len(results)} results for {len(call.results)}")
                resolved = arguments[index]
                if call.updates or call.retains:
                    resolved = list(resolved)
                    for position, earlier, result in call.locals:
                        resolved[position] = made[earlier][result]
                    for position in call.updates:
                        self._updated(call.path, resolved[position], identity)
                request = self.request_of.get(identity) if self.request_of else None
                shown = ("retained",) if call.retains else results
                self._fact(call.path, identity, "call_completed", ordinal, shown, request.payload if request else None)
                if call.retains:
                    product = self._retain(call, ordinal, results[0], resolved, identity, step.group.device)
                    product.completion = None if step.wait else completions[index]
                    results = (product,)
                made.append(results)
                for slot, value in zip(call.results, results, strict=True):
                    frame[slot] = value
        finally:
            for product in borrowed:
                product.borrowers -= 1
            if own_admission:
                self._work_ended(identity)
            if leased:
                self._changed(*(("resource", lease.resource) for lease in leased))

    def _check_arguments(self, call: Call, values: list[object]) -> None:
        for position, value in enumerate(values):
            if isinstance(value, Held) and position not in call.leases:
                self._missed(f"{call.path}: a lease passed where its type says none can be")
            if isinstance(value, Product) and position not in call.retained:
                self._missed(f"{call.path}: a retained result passed where its type says none can be")

    def _updated(self, call: str, held: object, identity: Identity) -> None:
        """Records an owner update, which only the current writer's lease can make.

        A call that updates owner state is never fused with another, so each
        update is observed after the one call that made it. The fact carries
        the version; the value stays with its owner.
        """
        if not isinstance(held, Held) or held.mode not in (AccessMode.OWNER_UPDATE, AccessMode.GRANT):
            raise RunFailure(f"{call}: updates state it holds no owner_update lease for")
        if self.protection[held.resource].writer != held.token:
            raise RunFailure(f"{call}: a stale owner update of @{held.resource}")
        now = self.owners[held.resource].current(held.resource)
        if now == held.version:
            return
        if (now.incarnation, now.lineage) != (held.version.incarnation, held.version.lineage) or now.ordinal <= held.version.ordinal:
            raise RunFailure(f"{call}: @{held.resource} moved to an incomparable or older version")
        held.version = now
        self._fact(held.resource, identity, "updated", now)

    def _retain(self, call: Call, ordinal: int, value: object, args: list[object], identity: Identity, device: str) -> Product:
        retained = [args[i] for i in call.retains]
        parents = tuple(v for v in retained if isinstance(v, Product))
        leases = {id(v): v for v in retained if isinstance(v, Held)}
        for parent in parents:
            leases.update((id(lease), lease) for lease in parent.leases)
        product = Product((call.path, ordinal), value, tuple(leases.values()), parents)
        self.products[product.key] = product
        admission = self._admission_for(identity, device) if device != HOST.name else self._admission_for(identity)
        if admission is not None:
            admission.products.add(id(product))
        self._fact(call.path, identity, "product_retained", ordinal, tuple(lease.version for lease in product.leases))
        return product

    # Device turns (D15): one admission at a time per device, in request order.

    def _admission_for(self, identity: Identity, device: str | None = None) -> _Admission | None:
        """The innermost admission on the device covering work of this invocation.

        An admission covers the work it was taken for and, until the retained
        results it produced are released, the rest of the body that holds them
        (D15): a window member's forward is in an access, its backward after it.
        """
        found: _Admission | None = None
        for admission in self._admission_of.values():
            if device is not None and admission.device != device:
                continue
            inside = identity[: len(admission.identity)] == admission.identity and not admission.work_ended
            after = bool(admission.products) and identity[: len(admission.body)] == admission.body
            if (inside or after) and (found is None or len(admission.identity) > len(found.identity)):
                found = admission
        return found

    def _turn_ready(self, device: str, identity: Identity) -> bool:
        turn = self.turns[device]
        if turn.holder is not None:
            return self._admission_for(identity, device) is not None
        return bool(turn.waiting) and turn.waiting[0] == identity

    async def _turn_queue(
        self, device: str | None, identity: Identity, also: Callable[[], bool] = lambda: True, keys: tuple[tuple, ...] = ()
    ) -> None:
        """Waits, in request order, until the device turn and ``also`` are both available; ``keys`` wake ``also``."""
        if device is None:
            await self._until(lambda: self.closing or also(), *keys)
            self._admission_point()
            return
        turn = self.turns[device]
        turn.waiting.append(identity)
        try:
            await self._until(lambda: self.closing or (self._turn_ready(device, identity) and also()), ("turn", device), *keys)
        finally:
            turn.waiting.remove(identity)
            self._changed(("turn", device))
        self._admission_point()

    def _grant_turn(self, device: str, identity: Identity, body: Identity) -> None:
        turn = self.turns[device]
        if turn.holder is None:
            turn.holder, turn.depth = identity, 1
        else:
            turn.depth += 1
        admission = _Admission(device, identity, body)
        self._admission_of[identity] = admission
        self.admissions[id(admission)] = admission

    def _work_ended(self, identity: Identity) -> None:
        admission = self._admission_of.get(identity)
        if admission is None:
            return
        admission.work_ended = True
        self._end_admission_if_done(admission)

    def _end_admission_if_done(self, admission: _Admission) -> None:
        if not admission.work_ended or admission.products or id(admission) not in self.admissions:
            return
        del self.admissions[id(admission)]
        if self._admission_of.get(admission.identity) is admission:
            del self._admission_of[admission.identity]
        turn = self.turns[admission.device]
        turn.depth -= 1
        if turn.depth == 0:
            turn.holder = None
        self._changed(("turn", admission.device))

    # Claims.

    def _settle(self, ticket: Ticket, disposition: str, identity: Identity) -> None:
        if ticket.disposition is not None:
            raise RunFailure(f"{ticket.stream}/{ticket.key}: a claim is settled twice")
        ticket.disposition = (disposition,)
        self._fact(ticket.stream, identity, "disposition", ticket.key, disposition)
        while ticket.parent is not None:
            parent = ticket.parent
            if any(fork.disposition is None for fork in parent.forks):
                self._changed(("stream", ticket.stream))
                return
            parent.disposition = tuple(d for fork in parent.forks for d in fork.disposition or ())
            ticket = parent
        stream = self.streams[ticket.stream]
        if stream.outstanding.get(ticket.key) is not ticket:
            raise RunFailure(f"{ticket.stream}/{ticket.key}: settled without its reservation")
        stream.outstanding.pop(ticket.key)
        if ticket in stream.ready:
            stream.ready.remove(ticket)
        self._fact(ticket.stream, (), "settled", ticket.key, ticket.disposition)
        self._changed(("stream", ticket.stream))

    def _take(self, step: Step, stream: Stream, identity: Identity) -> Ticket | None:
        if not stream.ready:
            raise RunFailure(f"{step.path}: @{stream.name} closed before this take")
        ticket = stream.ready.popleft()
        admitted = self._admit(stream, ticket)
        self._fact(step.path, identity, "admission", ticket.key, ticket.versions, admitted)
        ticket.owner = identity
        self._changed(("stream", ticket.stream))
        if not admitted:
            self._settle(ticket, "rejected", identity)
            return None
        return ticket

    def _admit(self, stream: Stream, ticket: Ticket) -> bool:
        policy = stream.flow.versions
        observed: dict[str, Version] = {}
        for version in ticket.versions:
            if observed.setdefault(version.owner, version) != version:
                return False
        if policy.admission == Admission.ANY_WITH_PROVENANCE:
            return True
        allowed = policy.max_lag if policy.admission == Admission.BOUNDED_LAG else 0
        for owner in policy.owners:
            seen, now = observed.get(owner), self.owners[owner].current(owner)
            if seen is None or (seen.incarnation, seen.lineage) != (now.incarnation, now.lineage) or seen.ordinal > now.ordinal:
                return False
            if now.ordinal - seen.ordinal > allowed:
                return False
        return True

    # Leases and retained results.

    def _can_lease(self, name: str, use: ResourceUse) -> bool:
        protection = self.protection[name]
        if use.mode == AccessMode.SNAPSHOT and use.selection in (VersionSelection.PUBLISHED_INITIAL, VersionSelection.AT_REQUEST):
            return True
        if use.mode in (AccessMode.SNAPSHOT, AccessMode.PROTECTED_CURRENT):
            return protection.writer is None
        return protection.writer is None and not protection.readers and not protection.reservations

    def _acquire(self, step: Invoke, identity: Identity) -> tuple[Held, ...]:
        held: list[Held] = []
        try:
            return self._acquire_each(step, identity, held)
        except BaseException:
            for lease in held:
                lease.body_ended = True
                self._release_lease_if_done(lease)
            raise

    def _acquire_each(self, step: Invoke, identity: Identity, held: list[Held]) -> tuple[Held, ...]:
        request = self.request_of.get(identity) if self.request_of else None
        for name, use in step.uses:
            if use.mode == AccessMode.GRANT:
                raise RunFailure(f"{step.path}: grants come with training's words")
            if use.selection == VersionSelection.AT_REQUEST and request is None:
                raise RunFailure(f"{step.path}: at_request outside requested work")
            if use.mode == AccessMode.SNAPSHOT and use.selection == VersionSelection.AT_REQUEST:
                assert request is not None
                lease = request.snapshots.pop(name)
                lease.body_ended = False
            else:
                reserved = {name: request.reserved[name]} if request is not None and name in request.reserved else None
                (granted,) = self.owners[name].lease([Use(name, use.mode, use.selection, use.route, use.view)], reserved)
                if reserved is not None and granted.version != reserved[name]:
                    granted.release()
                    raise RunFailure(f"{step.path}: @{name} moved past the version its request reserved")
                lease = self._hold(name, use.mode, granted)
            protection = self.protection[name]
            if use.mode == AccessMode.PROTECTED_CURRENT:
                protection.readers.add(lease.token)
            elif use.mode == AccessMode.OWNER_UPDATE:
                protection.writer = lease.token
            for production, observed in self._productions.items():
                if identity[: len(production)] == production:
                    observed.append(lease.version)
            self._fact(step.path, identity, "leased", name, str(use.mode), lease.version)
            held.append(lease)
        return tuple(held)

    def _hold(self, name: str, mode: AccessMode, granted) -> Held:
        lease = Held(self._next_token, name, mode, granted, granted.version)
        self._next_token += 1
        self.leases[lease.token] = lease
        return lease

    def _release_product(self, product: Product, how: str) -> None:
        if self.products.get(product.key) is not product or not product.active:
            raise RunFailure(f"{product.key[0]}: a retained result is released twice")
        if product.borrowers or any(any(p is product for p in other.parents) for other in self.products.values() if other.active):
            raise RunFailure(f"{product.key[0]}: released while a dependent use is still running")
        product.active = False
        del self.products[product.key]
        self._fact(product.key[0], (), "product_released", product.key[1], how)
        for lease in product.leases:
            self._release_lease_if_done(lease)
        for admission in list(self.admissions.values()):
            if id(product) in admission.products:
                admission.products.discard(id(product))
                self._end_admission_if_done(admission)

    def _release_lease_if_done(self, lease: Held) -> None:
        if not lease.active or not lease.body_ended:
            return
        if any(lease in product.leases for product in self.products.values()):
            return
        protection = self.protection[lease.resource]
        protection.readers.discard(lease.token)
        if protection.writer == lease.token:
            protection.writer = None
        lease.active = False
        lease.lease.release()
        del self.leases[lease.token]
        self._fact(lease.resource, (), "lease_released", lease.version)
        self._changed(("resource", lease.resource))

    # Requests.

    def _arm_request(self, step: RequestStep, identity: Identity, args: tuple[object, ...]) -> None:
        running = [
            identity[:length]
            for length in range(1, len(identity) + 1)
            if identity[length - 1][0] == step.source
            and (outcome := self.outcomes.get(identity[:length])) is not None
            and outcome.state == State.RUNNING
        ]
        if not running:
            raise RunFailure(f"{step.path}: a request is made inside its source")
        self.emissions[running[-1]].append((step, running[-1], args))

    async def _emit(self, step: RequestStep, source: Identity, args: tuple[object, ...], status: str) -> None:
        if step.boundary == Boundary.COMPLETED:
            raise RunFailure(f"{step.path}: requests at completion are not realized yet")
        if (step.boundary == Boundary.SETTLED_SUCCESS and status != "succeeded") or self.closing:
            self._fact(step.handler_name, source, "request_disposed", "unfulfilled")
            return
        target = step.handler
        handler = self.handlers[target.path]
        reads = [name for name, _ in target.at_request]
        await self._until(
            lambda: (
                self.closing
                or (
                    not any(self.protection[name].writer is not None for name in reads)
                    and len(handler.pending) + handler.active < target.bound
                )
            ),
            ("handler", target.path),
            *(("resource", name) for name in reads),
        )
        if self.closing:
            self._fact(step.handler_name, source, "request_disposed", "cancelled")
            return
        request = Request(target.path, handler.next_ordinal, source, deepcopy(args[0]))
        handler.next_ordinal += 1
        self._reserve(request, target)
        handler.pending.append(request)
        self._fact(target.path, source, "request_pending", request.ordinal, request.payload)
        self._changed(("handler", target.path), ("requests",))

    def _reserve(self, request: Request, handler: HandlerInfo) -> None:
        """Holds the version at the request for each at_request use: a snapshot is taken now, a live read reserved."""
        for name, use in handler.at_request:
            owner = self.owners[name]
            if use.mode == AccessMode.SNAPSHOT:
                (granted,) = owner.lease([Use(name, use.mode, VersionSelection.CURRENT_AT_START, use.route, use.view)])
                lease = self._hold(name, use.mode, granted)
                request.snapshots[name] = lease
                version = lease.version
            else:
                version = owner.current(name)
                request.reserved[name] = version
                self.protection[name].reservations.add(id(request))
            self._fact(name, request.cause, "reserved", version, str(use.mode))

    def _end_reservations(self, request: Request) -> None:
        for name in request.reserved:
            self.protection[name].reservations.discard(id(request))
        names = list(request.reserved)
        request.reserved.clear()
        for lease in request.snapshots.values():
            lease.body_ended = True
            self._release_lease_if_done(lease)
        request.snapshots.clear()
        self._changed(*(("resource", name) for name in names))

    def _pending_under(self, identity: Identity) -> list[Request]:
        return [r for h in self.handlers.values() for r in h.pending if r.cause[: len(identity)] == identity]

    def _settle_requests(self, identity: Identity) -> None:
        for request in self._pending_under(identity):
            self.handlers[request.handler].pending.remove(request)
            self._end_reservations(request)
            self._fact(request.handler, identity, "request_disposed", request.ordinal, "cancelled")
            self._changed(("handler", request.handler), ("requests",))

    # Scopes.

    def _stopped(self, target: str, identity: Identity) -> bool:
        return bool(self.stops) and any(t == target and identity[: len(scope)] == scope for scope, t in self.stops)

    def _source_settled(self, settling: Settling, identity: Identity) -> bool:
        """Whether the source, and every statement between it and the shared scope, has run and ended there."""
        scope = _scope(settling.common, identity)
        settled = self._records.settled(settling.chain, scope)
        if self.checked and self._all_records.settled(settling.chain, scope) != settled:
            self._missed(f"{settling.source}: the scope records lowering kept disagree with the full records")
        return settled


def _run_group(
    device: Device,
    context: CallContext,
    after: list[Completion],
    reaches: tuple[object, ...],
    program: list[tuple[Implementation, list[object], tuple[tuple[int, int, int], ...], int]],
    wait: bool,
) -> tuple[list[Any], list[Completion | None]]:
    """Runs a group's calls in order inside the device's context for them, on a worker thread.

    Each call's report is its results or what it raised, beside its
    completion; the group stops at the first call that fails or gives the
    wrong number of results, so no later call reads a result that is not
    there. Entering the context, which follows earlier work, or leaving it
    failing fails the call it was entered for. With ``wait`` the worker waits
    for the work's completion, as for host work or results that leave device
    work; otherwise later device work follows each completion in the device's
    own order.
    """
    reports: list[Any] = []
    completions: list[Completion | None] = []
    try:
        with device.context(context, after, reaches):
            for implementation, given, passed, count in program:
                if passed:
                    given = list(given)
                    for position, earlier, result in passed:
                        given[position] = reports[earlier][result]
                try:
                    results, completion = implementation(device, *given)
                except Exception as failed:
                    reports.append(failed)
                    completions.append(None)
                    break
                results = tuple(results)
                reports.append(results)
                completions.append(completion)
                if len(results) != count:
                    break
    except Exception as failed:
        if not reports:
            reports.append(failed)
            completions.append(None)
        elif not isinstance(reports[-1], BaseException):
            reports[-1] = failed
    if wait:
        for index, completion in enumerate(completions):
            if completion is None:
                continue
            try:
                completion.wait()
            except Exception as failed:
                reports[index] = failed  # physical completion failed: the call failed
    return reports, completions


def _scopes(identity: Identity, commons: frozenset[str] | None = None) -> list[Identity]:
    """The invocations enclosing an identity, itself included: the scopes a source can settle in.

    A parallel branch is not an invocation, so it is never a scope. With
    ``commons``, only the scopes a wait can ask about: the outermost, and
    those of the statements a waiter shares with this one.
    """
    return [
        identity[:length]
        for length in range(1, len(identity) + 1)
        if "#" not in identity[length - 1][0] and (commons is None or length == 1 or identity[length - 1][0] in commons)
    ]


def _scope(common: str, identity: Identity) -> Identity:
    """The invocation of ``common`` that encloses ``identity``: the innermost, else the outermost scope."""
    for length in range(len(identity), 0, -1):
        if identity[length - 1][0] == common:
            return identity[:length]
    return identity[:1]


def _leaves(ticket: Ticket) -> list[Ticket]:
    if not ticket.forks:
        return [ticket]
    return [leaf for fork in ticket.forks for leaf in _leaves(fork)]


__all__ = ["Coordinator", "Observer"]
