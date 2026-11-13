"""The coordinator: one per process, executing a prepared run (D15, D16).

It realizes every lifecycle protocol of the accepted strategy as the
reference semantics defines it, and reports the same owner-visible facts, so
each realization is checked against the reference. It runs on one event
loop and never blocks on selected work: calls run on worker threads through
the implementation and device seams, a fused group as one submission, and
the coordinator waits for their completion without blocking.

Each body is lowered once into steps, so executing it dispatches on nothing
but the step's own kind. The coordinator keeps runtime control state only:
leases and their protection, retained results, claims, requests, stops and
device turns. Owners hold the state leases reach.

Bookkeeping is bounded: an invocation's records live while it or its scope
is live, productions' observed versions while they produce, and facts go to
the observer as they happen.
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import cast

from xdsl.dialects.builtin import FloatAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.ir import Block, BlockArgument, Operation, SSAValue

from library.iro.operation_contracts import (
    AccessMode,
    Admission,
    Boundary,
    FlowSemantics,
    LifecycleProtocol,
    RegionSemantics,
    ResourceSemantics,
    VersionSelection,
    path,
    sym_name,
    symbol,
    symbols,
)
from library.iro.prepare import HOST, Group, Prepared
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
from library.iro.seams import Device, Implementation, Owner, Use, Version
from library.iro.words.access import AccessOp, ReleaseOp
from library.iro.words.calls import CallOp, ConstantOp, YieldOp
from library.iro.words.data import BorrowOp, DisposeOp, ForkOp, StreamOp, TakeOp
from library.iro.words.lifetimes import HandlerOp, RequestOp, StopOp
from library.iro.words.model import RelationshipOp, ResourceOp
from library.iro.words.structure import StrategyOp
from library.iro.words.types import OptionalType

Observer = Callable[[Fact], None]


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
    ):
        self.prepared = prepared
        self.strategy: StrategyOp = prepared.loaded.normalized.strategy
        self.owners, self.installed, self.devices = dict(owners), dict(installed), dict(devices)
        if HOST.name not in self.devices:
            raise ValueError("the coordinator needs the host device")
        self.observe = observe or (lambda fact: None)
        self.named = {name: op for op in self.strategy.walk() if (name := sym_name(op)) is not None and op is not self.strategy}
        self.groups = {call: group for group in prepared.groups for call in group.calls}
        self.admission_begins = {a.begins: a for a in prepared.admissions}
        self._plans: dict[Block, list[tuple[str, object]]] = {}
        # Live runtime state.
        self.protection: dict[str, Protection] = defaultdict(Protection)
        self.leases: dict[int, Held] = {}
        self.products: dict[tuple[str, int], Product] = {}
        self.streams: dict[str, Stream] = {}
        self.handlers: dict[str, HandlerState] = defaultdict(HandlerState)
        self.outcomes: dict[Identity, Outcome] = {}
        self.request_of: dict[Identity, Request] = {}
        self.emissions: dict[Identity, list[tuple[RequestOp, Identity, tuple[object, ...]]]] = defaultdict(list)
        self.stops: set[tuple[Identity, str]] = set()
        self.call_counts: dict[str, int] = defaultdict(int)
        self.turns: dict[str, _Turn] = defaultdict(_Turn)
        self.admissions: dict[int, _Admission] = {}
        self._admission_of: dict[Identity, _Admission] = {}
        self._productions: dict[Identity, list[Version]] = {}
        self._ordinals: dict[Identity, dict[str, int]] = defaultdict(dict)
        self._started: dict[tuple[str, Identity], int] = {}
        self._live_in: dict[tuple[str, Identity], int] = defaultdict(int)
        self._next_token = 0
        self.cancelled = False
        self._failed = False
        self.errors: list[str] = []
        self._failures: list[BaseException] = []
        self._stall: RunFailure | None = None
        self._waiting = 0
        self._joining = 0
        self._working = 0
        self._tick = asyncio.Event()
        self._executor: ThreadPoolExecutor | None = None
        self._tasks: set[asyncio.Task] = set()

    # Facts, lifecycle and waiting.

    def _fact(self, owner: str, invocation: Identity, kind: str, *values: object) -> None:
        self.observe(Fact(owner, invocation, kind, values))

    @property
    def closing(self) -> bool:
        return self.cancelled or self._failed

    def _changed(self) -> None:
        tick, self._tick = self._tick, asyncio.Event()
        self._waiting = 0  # every waiter wakes and checks again
        tick.set()

    async def _until(self, ready: Callable[[], bool]) -> None:
        """Waits until ``ready``; when every live activity waits and no selected work runs, the run has stalled."""
        while True:
            tick = self._tick
            if ready():
                return
            if self._stall is not None:
                raise self._stall
            self._waiting += 1
            live = 1 + sum(1 for task in self._tasks if not task.done())
            if self._waiting + self._joining >= live and self._working == 0:
                self._stall = RunFailure("no progress: every activity waits for another and no selected work runs")
                self._note_failure(self._stall)
                raise self._stall
            await tick.wait()

    def _note_failure(self, failed: BaseException) -> None:
        if not isinstance(failed, Cancelled) and not any(f is failed for f in self._failures):
            self._failures.append(failed)
            self.errors.append(str(failed))
            self._failed = True
            self._changed()

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
        for scope in _scopes(identity):
            key = (owner, scope)
            self._started[key] = 1
            self._live_in[key] += 1

    def _state(self, identity: Identity, state: State) -> None:
        outcome = self.outcomes[identity]
        if state not in TRANSITIONS[outcome.state]:
            raise RunFailure(f"{outcome.owner}: {outcome.state} cannot become {state}")
        outcome.state = state
        self._fact(outcome.owner, identity, "state", str(state))
        if state == State.TERMINAL:
            self._fact(outcome.owner, identity, "outcome", outcome.status, outcome.error)
            self._retire(identity, outcome.owner)
        self._changed()

    def _retire(self, identity: Identity, owner: str) -> None:
        """Bookkeeping for a finished invocation, and for the scope it closed, is dropped (bounded bookkeeping)."""
        del self.outcomes[identity]
        for scope in _scopes(identity):
            key = (owner, scope)
            self._live_in[key] -= 1
            if not self._live_in[key]:
                del self._live_in[key]
        self._ordinals.pop(identity, None)
        for key in [k for k in self._started if k[1] == identity]:
            del self._started[key]
        self.stops = {(scope, target) for scope, target in self.stops if scope != identity}

    # Running.

    async def run(self) -> Outcome:
        """Runs the strategy to settlement and returns the root's outcome."""
        with ThreadPoolExecutor(max_workers=self.prepared.target.workers, thread_name_prefix="iro-work") as executor:
            self._executor = executor
            root = self._identity((), path(self.strategy))
            self._begin(root, path(self.strategy))
            outcome = self.outcomes[root]
            for state in (State.PENDING, State.ADMITTED, State.RUNNING):
                self._state(root, state)
            try:
                try:
                    await self._body(self.strategy.body.block, (), root)
                except RunFailure:
                    raise
                except Exception as error:
                    raise RunFailure(f"{path(self.strategy)}: {error!r}") from error
                outcome.status = "succeeded"
            except RunFailure as failed:
                self._note_failure(failed)
                outcome.status = "cancelled" if self.cancelled or isinstance(failed, Cancelled) else "failed"
                outcome.error = str(failed)
            finally:
                self._state(root, State.CLOSING)
                self._state(root, State.SETTLING)
                await self._until(lambda: self.closing or not self._pending_under(root))
                self._settle_requests(root)
                self._state(root, State.TERMINAL)
        return outcome

    def cancel(self) -> None:
        """Run cancellation: admission closes while entered work settles."""
        self.cancelled = True
        self._changed()

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(lambda t: (self._tasks.discard(t), self._changed()))
        return task

    # Lowering: each body's steps, resolved once.

    def _plan(self, block: Block) -> list[tuple[str, object]]:
        plan = self._plans.get(block)
        if plan is not None:
            return plan
        plan = []
        grouped: dict[int, list[CallOp]] = {}
        for op in block.ops:
            if isinstance(op, YieldOp):
                plan.append(("yield", op))
            elif isinstance(op, CallOp):
                group = self.groups[path(op)]
                if group.index not in grouped:
                    grouped[group.index] = []
                    plan.append(("group", (group, grouped[group.index])))
                grouped[group.index].append(op)
            elif isinstance(op, ConstantOp):
                plan.append(("constant", op))
            else:
                plan.append(("op", op))
        self._plans[block] = plan
        return plan

    async def _body(self, block: Block, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        if len(block.args) != len(args):
            raise RunFailure(f"{path(block.parent_op() or self.strategy)}: a body receives {len(block.args)} entries, given {len(args)}")
        env: dict[SSAValue, object] = dict(zip(block.args, args, strict=True))
        returned: tuple[object, ...] = ()
        failed: BaseException | None = None
        try:
            for kind, step in self._plan(block):
                if kind == "yield":
                    assert isinstance(step, YieldOp)
                    returned = tuple(env[v] for v in step.operands)
                    return returned
                if kind == "constant":
                    assert isinstance(step, ConstantOp)
                    env[step.result] = _literal(step)
                elif kind == "group":
                    group, calls = step  # ty: ignore[not-iterable]
                    await self._group(group, calls, env, identity)
                else:
                    assert isinstance(step, Operation)
                    result = await self._execute(step, tuple(env[v] for v in step.operands), identity)
                    if len(result) != len(step.results):
                        raise RunFailure(f"{path(step)}: gave {len(result)} results for {len(step.results)}")
                    env.update(zip(step.results, result, strict=True))
            return returned
        except BaseException as error:
            failed = error
            raise
        finally:
            given = {id(v) for v in returned}
            leaked: RunFailure | None = None
            made = [env[r] for op in block.ops for r in op.results if r in env]
            for value in reversed(made):
                if isinstance(value, Product) and value.active and id(value) not in given:
                    if failed is None and leaked is None:
                        leaked = RunFailure(f"{value.key[0]}: a retained result was never released")
                    self._release_product(value, "cancelled" if self.cancelled else "failed")
            for value in env.values():
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
            if leaked is not None:
                raise leaked

    async def _execute(self, op: Operation, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        if op.get_trait(RegionSemantics) is not None:
            return await self._invoke(op, args, identity)
        if isinstance(op, ResourceOp):
            if op.sym_name.data not in self.owners:
                raise RunFailure(f"{path(op)}: no owner holds @{op.sym_name.data}")
            return (op.sym_name.data,)
        if isinstance(op, RelationshipOp):
            return ((args[0], args[1]),)
        if isinstance(op, StreamOp):
            self.streams[op.sym_name.data] = Stream(op)
            return (op.sym_name.data,)
        self._admission_point()
        if isinstance(op, ReleaseOp):
            product = args[0]
            if not isinstance(product, Product):
                raise RunFailure(f"{path(op)}: releases something that is not a retained result")
            self._release_product(product, "released")
            return ()
        if isinstance(op, TakeOp):
            stream = self.streams[str(args[0])]
            while True:
                await self._until(lambda: self.closing or bool(stream.ready) or stream.closed)
                self._admission_point()
                ticket = self._take(op, stream, identity)
                if ticket is not None:
                    return (ticket,)
        if isinstance(op, BorrowOp):
            return (deepcopy(self._owned(op, args[0], identity).payload),)
        if isinstance(op, DisposeOp):
            self._settle(self._owned(op, args[0], identity), op.disposition.data, identity)
            return ()
        if isinstance(op, ForkOp):
            ticket = self._owned(op, args[0], identity)
            ticket.forks = [Ticket(ticket.stream, ticket.key, identity, ticket.payload, ticket.versions, ticket) for _ in op.forks]
            ticket.owner = None
            self._fact(ticket.stream, identity, "forked", ticket.key, len(ticket.forks))
            return tuple(ticket.forks)
        if isinstance(op, RequestOp):
            self._arm_request(op, identity, args)
            return ()
        if isinstance(op, StopOp):
            source, target = self.named[symbol(op.source)], self.named[symbol(op.target)]
            await self._until(lambda: self.closing or self._source_settled(source, op, identity))
            self._admission_point()
            self.stops.add((self._scope(op, target, identity), path(target)))
            self._fact(path(target), identity, "stop_requested", symbol(op.source))
            self._changed()
            return ()
        raise RunFailure(f"{path(op)}: the runtime does not realize {op.name}")

    def _owned(self, op: Operation, value: object, identity: Identity) -> Ticket:
        if not isinstance(value, Ticket) or value.owner != identity or value.disposition is not None or value.forks:
            raise RunFailure(f"{path(op)}: the claim is not held here")
        return value

    # Invocations.

    async def _invoke(
        self, op: Operation, args: tuple[object, ...], parent: Identity, *, body: tuple[object, ...] | None = None
    ) -> tuple[object, ...]:
        identity = self._identity(parent, path(op))
        self._begin(identity, path(op))
        outcome = self.outcomes[identity]
        self._state(identity, State.PENDING)
        if parent in self.request_of:
            self.request_of[identity] = self.request_of[parent]
        try:
            self._admission_point()
            self._state(identity, State.ADMITTED)
            self._state(identity, State.RUNNING)
            for ticket in (v for v in (body if body is not None else args) if isinstance(v, Ticket)):
                if ticket.owner != parent or ticket.disposition is not None:
                    raise RunFailure(f"{path(op)}: a captured claim is not held by the capturing body")
                ticket.owner = identity
            try:
                if body is not None:
                    result = await self._body(op.regions[0].block, body, identity)
                else:
                    result = await self._protocol(op, args, identity)
            except RunFailure:
                raise
            except Exception as error:
                raise RunFailure(f"{path(op)}: {error!r}") from error
            for value in result:
                if isinstance(value, Ticket):
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
            for stream in self.streams.values():
                for ticket in tuple(stream.outstanding.values()):
                    for held in _leaves(ticket):
                        if held.owner == identity and held.disposition is None:
                            self._settle(held, "cancelled" if self.cancelled else "failed", identity)
            for request_op, source, request_args in self.emissions.pop(identity, ()):
                await self._emit(request_op, source, request_args, outcome.status)
            self.request_of.pop(identity, None)
            self._state(identity, State.TERMINAL)

    def _entry(self, op: Operation, args: tuple[object, ...], supplied: tuple[object, ...] = (), *, role: int = 0) -> tuple[object, ...]:
        contract = _region(op).describe(op).roles[role]
        missing = object()
        values: list[object] = [missing] * len(contract.entries)
        for capture in contract.captures:
            values[capture.argument] = args[capture.operand]
        open_positions = [i for i, v in enumerate(values) if v is missing]
        if len(open_positions) != len(supplied):
            raise RunFailure(f"{path(op)}: the protocol supplies {len(supplied)} entries, the body takes {len(open_positions)}")
        for position, value in zip(open_positions, supplied, strict=True):
            values[position] = value
        return tuple(values)

    async def _protocol(self, op: Operation, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        protocol = _region(op).protocol
        if protocol == LifecycleProtocol.SEQUENCE:
            return await self._body(op.regions[0].block, self._entry(op, args), identity)
        if protocol == LifecycleProtocol.CHOICE:
            return await self._choice(op, args, identity)
        if protocol == LifecycleProtocol.PARALLEL:
            return await self._parallel(op, args, identity)
        if protocol == LifecycleProtocol.REPEAT:
            return await self._repeat(op, args, identity)
        if protocol == LifecycleProtocol.ACCESS:
            return await self._access(op, args, identity)
        if protocol == LifecycleProtocol.FILTER:
            return await self._filter(op, args, identity)
        if protocol == LifecycleProtocol.PRODUCER:
            return await self._producer(op, args, identity)
        if protocol == LifecycleProtocol.HANDLER:
            return await self._handler(op, args, identity)
        raise RunFailure(f"{path(op)}: the runtime does not realize {protocol}")

    async def _choice(self, op: Operation, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        chosen = args[0]
        if isinstance(op.operands[0].type, OptionalType):
            if chosen is not None:
                return await self._body(op.regions[0].block, self._entry(op, args, (chosen,)), identity)
            return await self._body(op.regions[1].block, self._entry(op, args, role=1), identity)
        if type(chosen) is not bool:
            raise RunFailure(f"{path(op)}: a choice chooses on a bool")
        role = 0 if chosen else 1
        return await self._body(op.regions[role].block, self._entry(op, args, role=role), identity)

    async def _parallel(self, op: Operation, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        branches = []
        for index, region in enumerate(op.regions):
            branch = (*identity, (f"{path(op)}#{index}", 0))
            if identity in self.request_of:
                self.request_of[branch] = self.request_of[identity]
            for position, value in enumerate(args):
                if isinstance(value, Ticket) and region.block.args[position].uses:
                    value.owner = branch
            branches.append(self._spawn(self._branch(region.block, self._entry(op, args, role=index), branch)))
        self._joining += 1
        try:
            results = await asyncio.gather(*branches, return_exceptions=True)
        finally:
            self._joining -= 1
        errors = [r for r in results if isinstance(r, BaseException)]
        if self.cancelled:
            raise Cancelled("the run was cancelled")
        if errors:
            raise next((e for e in errors if not isinstance(e, Cancelled)), errors[0])
        return ()

    async def _branch(self, block: Block, args: tuple[object, ...], branch: Identity) -> tuple[object, ...]:
        try:
            try:
                return await self._body(block, args, branch)
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

    async def _repeat(self, op: Operation, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        limit = op.properties.get("limit")
        if not isinstance(limit, IntegerAttr):
            raise RunFailure(f"{path(op)}: a repeat by policy is realized with training's words, not by the core runtime")
        carried = op.properties["carry_count"]
        assert isinstance(carried, IntegerAttr)
        split = len(args) - carried.value.data
        captures, carries = args[:split], args[split:]
        for iteration in range(limit.value.data):
            carries = await self._invoke(op, (), identity, body=self._entry(op, (*captures, *carries), (iteration, *carries)))
        return carries

    async def _access(self, op: Operation, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        assert isinstance(op, AccessOp)
        uses = ResourceSemantics().describe(op).uses
        admission = self.admission_begins.get(path(op))
        device = admission.device if admission is not None else None
        # Leases and the device turn are taken together: nothing waits for one while holding the other (D14).
        await self._turn_queue(device, identity, lambda: all(self._can_lease(op, use) for use in uses))
        leases = self._acquire(op, identity)
        if device is not None:
            self._grant_turn(device, identity, identity[:-1])
        try:
            return await self._body(op.regions[0].block, self._entry(op, args, leases), identity)
        finally:
            for lease in leases:
                lease.body_ended = True
            for lease in leases:
                self._release_lease_if_done(lease)
            if device is not None:
                self._work_ended(identity)

    async def _filter(self, op: Operation, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        ticket = self._owned(op, args[0], identity)
        disposition = "failed"
        try:
            kept = await self._body(op.regions[0].block, self._entry(op, args, (deepcopy(ticket.payload),)), identity)
            if len(kept) != 1 or type(kept[0]) is not bool:
                raise RunFailure(f"{path(op)}: a filter's body yields one bool")
            disposition = "forwarded" if kept[0] else "filtered_out"
            return (deepcopy(ticket.payload) if kept[0] else None,)
        except Cancelled:
            disposition = "cancelled"
            raise
        finally:
            self._settle(ticket, disposition, identity)

    async def _producer(self, op: Operation, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        name = str(args[0])
        stream = self.streams[name]
        last = op.regions[0].block.last_op
        optional = last is not None and bool(last.operands) and isinstance(last.operands[0].type, OptionalType)
        try:
            index = 0
            while not self._stopped(op, identity) and not self.closing:
                await self._until(
                    lambda: (
                        self.closing or self._stopped(op, identity) or len(stream.outstanding) < stream.declaration.outstanding.value.data
                    )
                )
                self._admission_point()
                if self._stopped(op, identity):
                    break
                ticket = Ticket(name, stream.next_key, identity)
                stream.next_key += 1
                stream.outstanding[ticket.key] = ticket
                production = (*identity, (path(op), index))
                self._productions[identity] = []
                try:
                    (value,) = await self._invoke(op, (), identity, body=self._entry(op, args, (index,)))
                except RunFailure as failed:
                    if ticket.disposition is None:
                        self._settle(ticket, "cancelled" if isinstance(failed, Cancelled) else "failed", production)
                    raise
                finally:
                    observed = self._productions.pop(identity, [])
                index += 1
                ticket.versions = tuple(dict.fromkeys(observed))
                if optional and value is None:
                    stream.outstanding.pop(ticket.key)
                    self._fact(name, identity, "nothing_produced", ticket.key)
                    self._changed()
                    continue
                try:
                    await self._until(
                        lambda: self.closing or self._stopped(op, identity) or len(stream.ready) < stream.declaration.ready.value.data
                    )
                    self._admission_point()
                except RunFailure as failed:
                    if ticket.disposition is None:
                        self._settle(ticket, "cancelled" if isinstance(failed, Cancelled) else "failed", identity)
                    raise
                ticket.payload, ticket.owner = deepcopy(value), None
                if self._stopped(op, identity):
                    self._settle(ticket, "cancelled", identity)
                else:
                    stream.ready.append(ticket)
                    self._fact(name, identity, "published", ticket.key, ticket.versions)
                self._changed()
        finally:
            stream.closed = True
            self._changed()
            # Wait for consumers to settle what they took; what the producer still holds settles at the drain.
            await self._until(
                lambda: all(
                    leaf.owner is None or leaf.owner == identity
                    for t in stream.outstanding.values()
                    if t not in stream.ready
                    for leaf in _leaves(t)
                )
            )
            for ticket in tuple(stream.outstanding.values()):
                if ticket in stream.ready or ticket.owner == identity:
                    self._settle(ticket, "cancelled", identity)
        return ()

    async def _handler(self, op: Operation, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        assert isinstance(op, HandlerOp)
        handler = self.handlers[path(op)]
        sources = [self.named[name] for name in symbols(op, "sources")]
        while True:
            await self._until(lambda: self.closing or bool(handler.pending) or all(self._source_settled(s, op, identity) for s in sources))
            if self.closing:
                while handler.pending:
                    request = handler.pending.popleft()
                    self._end_reservations(request)
                    self._fact(path(op), identity, "request_disposed", request.ordinal, "cancelled")
                self._changed()
                return ()
            if not handler.pending:
                return ()
            handler.active += 1
            request = handler.pending.popleft()
            self.request_of[identity] = request
            disposition = "failed"
            try:
                await self._invoke(op, (), identity, body=self._entry(op, args, (deepcopy(request.payload),)))
                disposition = "completed"
            except RunFailure as failed:
                disposition = "cancelled" if isinstance(failed, Cancelled) else "failed"
                raise
            finally:
                handler.active -= 1
                self._end_reservations(request)
                self._fact(path(op), identity, "request_settled", request.ordinal, disposition)
                del self.request_of[identity]
                self._changed()

    # Calls: a fused group is one submission to a worker thread on its device.

    async def _group(self, group: Group, calls: list[CallOp], env: dict[SSAValue, object], identity: Identity) -> None:
        self._admission_point()
        device = self.devices.get(group.device)
        if device is None:
            raise RunFailure(f"{group.calls[0]}: the coordinator has no device {group.device}")
        own_admission = group.device != HOST.name and self._admission_for(identity, group.device) is None
        if own_admission:
            await self._turn_queue(group.device, identity)
            self._grant_turn(group.device, identity, identity)
        borrowed: list[Product] = []
        try:
            produced = {r for c in calls for r in c.results}
            entries = []
            for call in calls:
                implementation = self.installed[symbol(call.implementation)]
                values = []
                for operand in call.operands:
                    if operand in produced:
                        values.append(operand)
                        continue
                    value = env[operand]
                    if isinstance(value, Held) and not value.active:
                        raise RunFailure(f"{path(call)}: a lease is used after its release")
                    if isinstance(value, Product):
                        if not value.active:
                            raise RunFailure(f"{path(call)}: a retained result is used after its release")
                        value.borrowers += 1
                        borrowed.append(value)
                    values.append(value)
                entries.append((call, implementation, tuple(values)))
            assert self._executor is not None
            self._working += 1
            try:
                reports = await asyncio.get_running_loop().run_in_executor(self._executor, _run_group, device, entries)
            finally:
                self._working -= 1
            for call, implementation, values in entries:
                report = reports.get(call)
                if report is None:
                    break
                ordinal = self.call_counts[path(call)]
                self.call_counts[path(call)] += 1
                self._fact(path(call), identity, "call_entered", ordinal)
                if isinstance(report, BaseException):
                    self._fact(path(call), identity, "call_failed", ordinal, str(report))
                    raise RunFailure(f"{path(call)}: {report}") from report
                results = cast(tuple[object, ...], report)
                if len(results) != len(call.results):
                    self._fact(path(call), identity, "call_invalid", ordinal)
                    raise RunFailure(f"{path(call)}: returned {len(results)} results for {len(call.results)}")
                resolved = tuple(env[v] if isinstance(v, SSAValue) and v in env else v for v in values)
                for position in implementation.declaration.updates:
                    self._updated(call, resolved[position], identity)
                shown = ("retained",) if implementation.declaration.retains else results
                request = self.request_of.get(identity)
                self._fact(path(call), identity, "call_completed", ordinal, shown, request.payload if request else None)
                if implementation.declaration.retains:
                    results = (self._retain(call, ordinal, results[0], resolved, implementation, identity),)
                env.update(zip(call.results, results, strict=True))
        finally:
            for product in borrowed:
                product.borrowers -= 1
            if own_admission:
                self._work_ended(identity)
            self._changed()

    def _updated(self, call: CallOp, held: object, identity: Identity) -> None:
        """Records an owner update, which only the current writer's lease can make.

        A call that updates owner state is never fused with another, so each
        update is observed after the one call that made it. The fact carries
        the version; the value stays with its owner.
        """
        if not isinstance(held, Held) or held.mode not in (AccessMode.OWNER_UPDATE, AccessMode.GRANT):
            raise RunFailure(f"{path(call)}: updates state it holds no owner_update lease for")
        if self.protection[held.resource].writer != held.token:
            raise RunFailure(f"{path(call)}: a stale owner update of @{held.resource}")
        now = self.owners[held.resource].current(held.resource)
        if now == held.version:
            return
        if (now.incarnation, now.lineage) != (held.version.incarnation, held.version.lineage) or now.ordinal <= held.version.ordinal:
            raise RunFailure(f"{path(call)}: @{held.resource} moved to an incomparable or older version")
        held.version = now
        self._fact(held.resource, identity, "updated", now)

    def _retain(
        self, call: CallOp, ordinal: int, value: object, args: tuple[object, ...], implementation: Implementation, identity: Identity
    ) -> Product:
        retained = [args[i] for i in implementation.declaration.retains]
        parents = tuple(v for v in retained if isinstance(v, Product))
        leases = {id(v): v for v in retained if isinstance(v, Held)}
        for parent in parents:
            leases.update((id(lease), lease) for lease in parent.leases)
        product = Product((path(call), ordinal), value, tuple(leases.values()), parents)
        self.products[product.key] = product
        device = self.groups[path(call)].device
        admission = self._admission_for(identity, device) if device != HOST.name else self._admission_for(identity)
        if admission is not None:
            admission.products.add(id(product))
        self._fact(path(call), identity, "product_retained", ordinal, tuple(lease.version for lease in product.leases))
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

    async def _turn_queue(self, device: str | None, identity: Identity, also: Callable[[], bool] = lambda: True) -> None:
        """Waits, in request order, until the device turn and ``also`` are both available."""
        if device is None:
            await self._until(lambda: self.closing or also())
            self._admission_point()
            return
        turn = self.turns[device]
        turn.waiting.append(identity)
        try:
            await self._until(lambda: self.closing or (self._turn_ready(device, identity) and also()))
        finally:
            turn.waiting.remove(identity)
            self._changed()
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
        self._changed()

    # Claims.

    def _settle(self, ticket: Ticket, disposition: str, identity: Identity) -> None:
        if ticket.disposition is not None:
            raise RunFailure(f"{ticket.stream}/{ticket.key}: a claim is settled twice")
        ticket.disposition = (disposition,)
        self._fact(ticket.stream, identity, "disposition", ticket.key, disposition)
        while ticket.parent is not None:
            parent = ticket.parent
            if any(fork.disposition is None for fork in parent.forks):
                self._changed()
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
        self._changed()

    def _take(self, op: Operation, stream: Stream, identity: Identity) -> Ticket | None:
        if not stream.ready:
            raise RunFailure(f"{path(op)}: @{stream.declaration.sym_name.data} closed before this take")
        ticket = stream.ready.popleft()
        admitted = self._admit(stream.declaration, ticket)
        self._fact(path(op), identity, "admission", ticket.key, ticket.versions, admitted)
        ticket.owner = identity
        self._changed()
        if not admitted:
            self._settle(ticket, "rejected", identity)
            return None
        return ticket

    def _admit(self, stream: StreamOp, ticket: Ticket) -> bool:
        policy = FlowSemantics().describe(stream).versions
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

    def _can_lease(self, op: Operation, use) -> bool:
        protection = self.protection[self._resource_name(op, use.subject)]
        if use.mode == AccessMode.SNAPSHOT and use.selection in (VersionSelection.PUBLISHED_INITIAL, VersionSelection.AT_REQUEST):
            return True
        if use.mode in (AccessMode.SNAPSHOT, AccessMode.PROTECTED_CURRENT):
            return protection.writer is None
        return protection.writer is None and not protection.readers and not protection.reservations

    def _acquire(self, op: Operation, identity: Identity) -> tuple[Held, ...]:
        held: list[Held] = []
        try:
            return self._acquire_each(op, identity, held)
        except BaseException:
            for lease in held:
                lease.body_ended = True
                self._release_lease_if_done(lease)
            raise

    def _acquire_each(self, op: Operation, identity: Identity, held: list[Held]) -> tuple[Held, ...]:
        request = self.request_of.get(identity)
        for use in ResourceSemantics().describe(op).uses:
            name = self._resource_name(op, use.subject)
            if use.mode == AccessMode.GRANT:
                raise RunFailure(f"{path(op)}: grants come with training's words")
            if use.selection == VersionSelection.AT_REQUEST and request is None:
                raise RunFailure(f"{path(op)}: at_request outside requested work")
            if use.mode == AccessMode.SNAPSHOT and use.selection == VersionSelection.AT_REQUEST:
                assert request is not None
                lease = request.snapshots.pop(name)
                lease.body_ended = False
            else:
                reserved = {name: request.reserved[name]} if request is not None and name in request.reserved else None
                (granted,) = self.owners[name].lease([Use(name, use.mode, use.selection, use.route, use.view)], reserved)
                if reserved is not None and granted.version != reserved[name]:
                    granted.release()
                    raise RunFailure(f"{path(op)}: @{name} moved past the version its request reserved")
                lease = self._hold(name, use.mode, granted)
            protection = self.protection[name]
            if use.mode == AccessMode.PROTECTED_CURRENT:
                protection.readers.add(lease.token)
            elif use.mode == AccessMode.OWNER_UPDATE:
                protection.writer = lease.token
            for production, observed in self._productions.items():
                if identity[: len(production)] == production:
                    observed.append(lease.version)
            self._fact(path(op), identity, "leased", name, str(use.mode), lease.version)
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
        self._changed()

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
        self._changed()

    # Requests.

    def _arm_request(self, op: RequestOp, identity: Identity, args: tuple[object, ...]) -> None:
        source = path(self.named[symbol(op.source)])
        running = [
            identity[:length]
            for length in range(1, len(identity) + 1)
            if identity[length - 1][0] == source
            and (outcome := self.outcomes.get(identity[:length])) is not None
            and outcome.state == State.RUNNING
        ]
        if not running:
            raise RunFailure(f"{path(op)}: a request is made inside its source")
        self.emissions[running[-1]].append((op, running[-1], args))

    async def _emit(self, op: RequestOp, source: Identity, args: tuple[object, ...], status: str) -> None:
        boundary = Boundary(op.boundary.data)
        if boundary == Boundary.COMPLETED:
            raise RunFailure(f"{path(op)}: requests at completion are not realized yet")
        if (boundary == Boundary.SETTLED_SUCCESS and status != "succeeded") or self.closing:
            self._fact(symbol(op.handler), source, "request_disposed", "unfulfilled")
            return
        target = self.named[symbol(op.handler)]
        assert isinstance(target, HandlerOp)
        handler = self.handlers[path(target)]
        bound = target.max_outstanding.value.data
        reads = self._at_request(target)
        await self._until(
            lambda: (
                self.closing
                or (not any(self.protection[name].writer is not None for name in reads) and len(handler.pending) + handler.active < bound)
            )
        )
        if self.closing:
            self._fact(symbol(op.handler), source, "request_disposed", "cancelled")
            return
        request = Request(path(target), handler.next_ordinal, source, deepcopy(args[0]))
        handler.next_ordinal += 1
        self._reserve(request, target)
        handler.pending.append(request)
        self._fact(path(target), source, "request_pending", request.ordinal, request.payload)
        self._changed()

    def _at_request(self, handler: Operation) -> list[str]:
        return [
            self._resource_name(access, use.subject)
            for access in handler.walk()
            if isinstance(access, AccessOp)
            for use in ResourceSemantics().describe(access).uses
            if use.selection == VersionSelection.AT_REQUEST
        ]

    def _reserve(self, request: Request, handler: Operation) -> None:
        """Holds the version at the request for each at_request use: a snapshot is taken now, a live read reserved."""
        for access in (op for op in handler.walk() if isinstance(op, AccessOp)):
            for use in ResourceSemantics().describe(access).uses:
                if use.selection != VersionSelection.AT_REQUEST:
                    continue
                name = self._resource_name(access, use.subject)
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
        request.reserved.clear()
        for lease in request.snapshots.values():
            lease.body_ended = True
            self._release_lease_if_done(lease)
        request.snapshots.clear()
        self._changed()

    def _pending_under(self, identity: Identity) -> list[Request]:
        return [r for h in self.handlers.values() for r in h.pending if r.cause[: len(identity)] == identity]

    def _settle_requests(self, identity: Identity) -> None:
        for request in self._pending_under(identity):
            self.handlers[request.handler].pending.remove(request)
            self._end_reservations(request)
            self._fact(request.handler, identity, "request_disposed", request.ordinal, "cancelled")

    def _resource_name(self, access: Operation, subject: SSAValue) -> str:
        value: SSAValue = subject
        while isinstance(value, BlockArgument):
            parent, region = value.owner.parent_op(), value.owner.parent
            if parent is None or region is None:
                break
            role = _region(parent).describe(parent).roles[parent.regions.index(region)]
            capture = next(c for c in role.captures if c.argument == value.index)
            value = parent.operands[capture.operand]
        owner = value.owner
        if not isinstance(owner, ResourceOp):
            raise RunFailure(f"{path(access)}: an access names something that is not a resource")
        return owner.sym_name.data

    # Scopes.

    def _stopped(self, op: Operation, identity: Identity) -> bool:
        return any(target == path(op) and identity[: len(scope)] == scope for scope, target in self.stops)

    @staticmethod
    def _common(waiting: Operation, source: Operation) -> Operation:
        ancestors = set()
        current: Operation | None = source
        while current is not None:
            ancestors.add(current)
            current = current.parent_op()
        common: Operation | None = waiting
        while common is not None and common not in ancestors:
            common = common.parent_op()
        if common is None:
            raise RunFailure(f"{path(waiting)}: refers outside its strategy")
        return common

    def _scope(self, waiting: Operation, source: Operation, identity: Identity) -> Identity:
        common = self._common(waiting, source)
        positions = [i for i, (owner, _) in enumerate(identity) if owner == path(common)]
        return identity[: max(positions) + 1] if positions else identity[:1]

    def _source_settled(self, source: Operation, waiting: Operation, identity: Identity) -> bool:
        """Whether the source, and every statement between it and the shared scope, has run and ended there."""
        common = self._common(waiting, source)
        scope = self._scope(waiting, source, identity)
        started = False
        current: Operation | None = source
        while current is not None and current is not common:
            if self._live_in.get((path(current), scope)):
                return False
            started = started or (path(current), scope) in self._started
            current = current.parent_op()
        return started


def _run_group(device: Device, entries: list[tuple[CallOp, Implementation, tuple[object, ...]]]) -> dict[CallOp, object]:
    """Runs a group's calls in order on a worker thread and waits for their device completion."""
    reports: dict[CallOp, object] = {}
    values: dict[SSAValue, object] = {}
    completions = []
    for call, implementation, args in entries:
        given = tuple(values[a] if isinstance(a, SSAValue) else a for a in args)
        unwrapped = tuple(a.handle if isinstance(a, Held) else a.value if isinstance(a, Product) else a for a in given)
        try:
            results, completion = implementation(device, *unwrapped)
            completions.append((call, completion))
        except Exception as failed:
            reports[call] = failed
            break
        values.update(zip(call.results, results, strict=False))
        reports[call] = tuple(results)
    for call, completion in completions:
        try:
            completion.wait()
        except Exception as failed:
            reports[call] = failed  # physical completion failed: the call failed
    return reports


def _scopes(identity: Identity) -> list[Identity]:
    """The invocations enclosing an identity, itself included: the scopes a source can settle in.

    A parallel branch is not an invocation, so it is never a scope.
    """
    return [identity[:length] for length in range(1, len(identity) + 1) if "#" not in identity[length - 1][0]]


def _leaves(ticket: Ticket) -> list[Ticket]:
    if not ticket.forks:
        return [ticket]
    return [leaf for fork in ticket.forks for leaf in _leaves(fork)]


def _region(op: Operation) -> RegionSemantics:
    semantics = op.get_trait(RegionSemantics)
    if semantics is None:
        raise RunFailure(f"{path(op)}: has no lifecycle protocol")
    return semantics


def _literal(op: ConstantOp) -> object:
    value = op.value
    if isinstance(value, IntegerAttr):
        number = value.value.data
        return bool(number) if isinstance(value.type, IntegerType) and value.type.width.data == 1 else number
    if isinstance(value, FloatAttr):
        return value.value.data
    if isinstance(value, StringAttr):
        return value.data
    raise RunFailure(f"{path(op)}: an unknown literal")


__all__ = ["Coordinator", "Observer"]
