"""The reference semantics: a deterministic lifecycle interpreter over normalized strategies (D14).

It executes each lifecycle protocol as a state machine over the strategy's
structure and records owner-visible facts: invocations and their outcomes,
leases and the versions they observed, admissions, dispositions, requests and
stops. Every runtime realization is checked against it.

Scheduling is explicit. Each invocation runs as a task that stops at every
point where the order of events could matter, as a command; :meth:`step`
performs one enabled command, chosen by the caller or round-robin, so tests
can explore orders. Physical completion of a call is a separate command, so
it can be held back. Selected computations are replaced by declared stand-ins
that honor their contracts and supply fixture values.

It imports neither Torch nor any model, optimizer or runtime code, and holds
its own state for one calling thread.
"""

import contextlib
from collections import defaultdict, deque
from collections.abc import Callable, Generator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum

from xdsl.dialects.builtin import FloatAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.ir import Block, BlockArgument, Operation, SSAValue

from library.iro.normalize import Normalized
from library.iro.operation_contracts import (
    AccessMode,
    Admission,
    Boundary,
    FlowSemantics,
    LifecycleProtocol,
    RealizationSemantics,
    RegionSemantics,
    ResourceSemantics,
    VersionSelection,
    path,
    sym_name,
    symbol,
    symbols,
)
from library.iro.reference.standins import CallResult, StandIn
from library.iro.words.access import AccessOp, ReleaseOp
from library.iro.words.calls import CallOp, ConstantOp, YieldOp
from library.iro.words.data import BorrowOp, DisposeOp, ForkOp, StreamOp, TakeOp
from library.iro.words.lifetimes import HandlerOp, RequestOp, StopOp
from library.iro.words.model import RelationshipOp, ResourceOp
from library.iro.words.structure import StrategyOp
from library.iro.words.types import OptionalType, written

Identity = tuple[tuple[str, int], ...]


class State(StrEnum):
    """Where an invocation is in its lifecycle."""

    CREATED = "created"
    PENDING = "pending"
    ADMITTED = "admitted"
    RUNNING = "running"
    CLOSING = "closing"
    SETTLING = "settling"
    TERMINAL = "terminal"


TRANSITIONS = {
    State.CREATED: (State.PENDING,),
    State.PENDING: (State.ADMITTED, State.CLOSING),
    State.ADMITTED: (State.RUNNING,),
    State.RUNNING: (State.CLOSING,),
    State.CLOSING: (State.SETTLING,),
    State.SETTLING: (State.TERMINAL,),
    State.TERMINAL: (),
}


class ReferenceFailure(ValueError):
    """The strategy or a stand-in broke a contract while running."""


class ReferenceLimit(ReferenceFailure):
    """A valid strategy uses something the reference semantics does not realize."""


class Cancelled(ReferenceFailure):
    """Run cancellation, or a sibling's failure, closed ordinary admission."""


@dataclass(frozen=True, order=True)
class Version:
    """An observed version of one resource."""

    owner: str
    incarnation: int
    lineage: str
    ordinal: int


@dataclass
class Resource:
    name: str
    value: object
    incarnation: int = 1
    lineage: str = "initial"
    ordinal: int = 0
    readers: set[int] = field(default_factory=set)
    writer: int | None = None
    reservations: set[int] = field(default_factory=set)

    @property
    def version(self) -> Version:
        return Version(self.name, self.incarnation, self.lineage, self.ordinal)


@dataclass(eq=False)
class Lease:
    """One resource's lease: the state a body's calls reach and the version it observed."""

    token: int
    resource: str
    mode: AccessMode
    version: Version
    value: object
    active: bool = True
    body_ended: bool = False


@dataclass(eq=False)
class Product:
    """A retained result: its value and the leases it keeps until its release."""

    key: tuple[str, int]
    value: object
    leases: tuple[Lease, ...]
    parents: tuple[Product, ...]
    borrowers: int = 0
    active: bool = True


@dataclass(eq=False)
class Ticket:
    """An admitted or produced item's claim, or one of its forks."""

    stream: str
    key: int
    owner: Identity | None
    payload: object = None
    versions: tuple[Version, ...] = ()
    parent: Ticket | None = None
    forks: list[Ticket] = field(default_factory=list)
    disposition: tuple[str, ...] | None = None


@dataclass
class Stream:
    declaration: StreamOp
    outstanding: dict[int, Ticket] = field(default_factory=dict)
    ready: deque[Ticket] = field(default_factory=deque)
    settled: dict[int, tuple[str, ...]] = field(default_factory=dict)
    next_key: int = 0
    closed: bool = False


@dataclass(eq=False)
class Request:
    handler: str
    ordinal: int
    cause: Identity
    payload: object
    snapshots: dict[str, tuple[Version, object]] = field(default_factory=dict)
    reserved: dict[str, Version] = field(default_factory=dict)


@dataclass
class Handler:
    pending: deque[Request] = field(default_factory=deque)
    active: int = 0
    next_ordinal: int = 0
    dispositions: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Fact:
    """One owner-visible lifecycle fact."""

    owner: str
    invocation: Identity
    kind: str
    values: tuple[object, ...] = ()


@dataclass
class Outcome:
    owner: str
    invocation: Identity
    state: State = State.CREATED
    status: str = "pending"
    values: tuple[object, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class Command:
    kind: str
    op: Operation
    invocation: Identity
    args: tuple[object, ...] = ()


@dataclass
class PendingCall:
    command: Command
    standin: StandIn
    ordinal: int
    inputs: tuple[object, ...]


Body = Generator[Command, object, tuple[object, ...]]


@dataclass
class Task:
    generator: Body
    command: Command | None = None
    done: bool = False
    result: tuple[object, ...] = ()
    error: ReferenceFailure | None = None


# Commands that still run while the run closes, so it can settle.
SETTLING_COMMANDS = frozenset({"call_complete", "release", "join", "drain", "next_request", "settle_requests"})


class Reference:
    """The reference semantics of one normalized strategy, with its resources' initial values and stand-ins."""

    def __init__(
        self,
        normalized: Normalized,
        resources: Mapping[str, object],
        standins: Mapping[str, StandIn],
        *,
        validators: Mapping[str, Callable[[object], bool]] | None = None,
    ):
        self.strategy: StrategyOp = normalized.strategy.clone()
        if self.strategy.normalized is None:
            raise ReferenceFailure("the reference semantics runs a normalized strategy")
        self.named = {name: op for op in self.strategy.walk() if (name := sym_name(op)) is not None and op is not self.strategy}
        self.standins = dict(standins)
        self.validators = dict(validators or {})
        self.resources = {name: Resource(name, deepcopy(value)) for name, value in resources.items()}
        self.initial = {name: (r.version, deepcopy(r.value)) for name, r in self.resources.items()}
        self.streams: dict[str, Stream] = {}
        self.handlers: dict[str, Handler] = defaultdict(Handler)
        self.outcomes: dict[Identity, Outcome] = {}
        self.facts: list[Fact] = []
        self.tasks: list[Task] = []
        self.errors: list[str] = []
        self.products: dict[tuple[str, int], Product] = {}
        self.leases: dict[int, Lease] = {}
        self.calls: dict[int, PendingCall] = {}
        self.call_counts: dict[str, int] = defaultdict(int)
        self.failures: dict[tuple[str, int], str] = {}
        self.held: set[tuple[str, int]] = set()
        self.ordinals: dict[tuple[Identity, str], int] = defaultdict(int)
        self.read_versions: dict[Identity, list[Version]] = defaultdict(list)
        self.request_of: dict[Identity, Request] = {}
        self.emissions: dict[Identity, list[Command]] = defaultdict(list)
        self.stops: set[tuple[Identity, str]] = set()
        self.cancelled = False
        self._failed = False
        self._reported: set[int] = set()
        self.cursor = 0
        self._discarding = False
        self._spawn(self._root())

    def __del__(self) -> None:
        # An unfinished run's tasks are closed without performing their settling commands.
        self._discarding = True
        for task in getattr(self, "tasks", ()):
            if not task.done:
                # An abandoned run reports nothing: its bodies' leak checks have no outcome to fail.
                with contextlib.suppress(ReferenceFailure):
                    task.generator.close()

    # Facts and lifecycle.

    def _fact(self, owner: str, invocation: Identity, kind: str, *values: object) -> None:
        self.facts.append(Fact(owner, invocation, kind, deepcopy(values)))

    def _identity(self, parent: Identity, owner: str) -> Identity:
        ordinal = self.ordinals[(parent, owner)]
        self.ordinals[(parent, owner)] += 1
        return (*parent, (owner, ordinal))

    def _state(self, identity: Identity, state: State) -> None:
        outcome = self.outcomes[identity]
        if state not in TRANSITIONS[outcome.state]:
            raise ReferenceFailure(f"{outcome.owner}: {outcome.state} cannot become {state}")
        outcome.state = state
        self._fact(outcome.owner, identity, "state", str(state))
        if state == State.TERMINAL:
            self._fact(outcome.owner, identity, "outcome", outcome.status, outcome.error)

    @property
    def closing(self) -> bool:
        return self.cancelled or self._failed

    def _note_failure(self, failed: ReferenceFailure) -> None:
        if not isinstance(failed, Cancelled) and id(failed) not in self._reported:
            self._reported.add(id(failed))
            self.errors.append(str(failed))
            self._failed = True

    # Tasks.

    def _spawn(self, generator: Body) -> int:
        task = Task(generator)
        self.tasks.append(task)
        self._resume(task)
        return len(self.tasks) - 1

    def _resume(self, task: Task, result: object = None, error: ReferenceFailure | None = None) -> None:
        try:
            task.command = task.generator.throw(error) if error is not None else task.generator.send(result)
        except StopIteration as completed:
            task.result, task.done, task.command = completed.value or (), True, None
        except ReferenceFailure as failed:
            task.error, task.done, task.command = failed, True, None
            self._note_failure(failed)

    # Bodies.

    def _root(self) -> Body:
        identity = self._identity((), path(self.strategy))
        self.outcomes[identity] = Outcome(path(self.strategy), identity)
        for state in (State.PENDING, State.ADMITTED, State.RUNNING):
            self._state(identity, state)
        try:
            yield from self._body(self.strategy.body.block, (), identity)
            self.outcomes[identity].status = "succeeded"
            return ()
        except ReferenceFailure as failed:
            self.outcomes[identity].status = "cancelled" if self.cancelled or isinstance(failed, Cancelled) else "failed"
            self.outcomes[identity].error = str(failed)
            raise
        finally:
            self._state(identity, State.CLOSING)
            self._state(identity, State.SETTLING)
            if not self._discarding:
                yield Command("settle_requests", self.strategy, identity)
            self._state(identity, State.TERMINAL)

    def _body(self, block: Block, args: tuple[object, ...], identity: Identity) -> Body:
        if len(block.args) != len(args):
            raise ReferenceFailure(
                f"{path(block.parent_op() or self.strategy)}: a body receives {len(block.args)} entries, given {len(args)}"
            )
        env: dict[SSAValue, object] = dict(zip(block.args, args, strict=True))
        returned: tuple[object, ...] = ()
        failed: ReferenceFailure | None = None
        try:
            for op in block.ops:
                if isinstance(op, YieldOp):
                    returned = tuple(env[v] for v in op.operands)
                    return returned
                values = tuple(env[v] for v in op.operands)
                result = yield from self._execute(op, values, identity)
                if len(result) != len(op.results):
                    raise ReferenceFailure(f"{path(op)}: gave {len(result)} results for {len(op.results)}")
                env.update(zip(op.results, result, strict=True))
            return returned
        except ReferenceFailure as error:
            failed = error
            raise
        finally:
            given = {id(v) for v in returned}
            leaked: ReferenceFailure | None = None
            made = [env[r] for op in block.ops for r in op.results if r in env]
            for value in reversed(made):
                if isinstance(value, Product) and value.active and id(value) not in given:
                    if failed is None and leaked is None:
                        leaked = ReferenceFailure(f"{value.key[0]}: a retained result was never released")
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
                        leaked = ReferenceFailure(f"{value.stream}/{value.key}: a claim was never settled")
                    self._settle(value, "cancelled" if self.cancelled or isinstance(failed, Cancelled) else "failed", identity)
            if leaked is not None:
                raise leaked

    def _execute(self, op: Operation, args: tuple[object, ...], identity: Identity) -> Body:
        if op.get_trait(RegionSemantics) is not None:
            return (yield from self._invoke(op, args, identity))
        if isinstance(op, ResourceOp):
            if op.sym_name.data not in self.resources:
                raise ReferenceFailure(f"{path(op)}: no initial value for @{op.sym_name.data}")
            return (op.sym_name.data,)
        if isinstance(op, RelationshipOp):
            return ((args[0], args[1]),)
        if isinstance(op, StreamOp):
            self.streams[op.sym_name.data] = Stream(op)
            return (op.sym_name.data,)
        if isinstance(op, ConstantOp):
            return (_literal(op),)
        yield Command("tick", op, identity)
        if isinstance(op, CallOp):
            entered = yield Command("call_enter", op, identity, args)
            result = yield Command("call_complete", op, identity, (entered,))
            assert isinstance(result, tuple)
            return result
        if isinstance(op, ReleaseOp):
            product = args[0]
            if not isinstance(product, Product):
                raise ReferenceFailure(f"{path(op)}: releases something that is not a retained result")
            self._release_product(product, "released")
            return ()
        if isinstance(op, TakeOp):
            ticket = None
            while ticket is None:
                ticket = yield Command("take", op, identity, args)
            return (ticket,)
        if isinstance(op, BorrowOp):
            ticket = self._owned(op, args[0], identity)
            return (deepcopy(ticket.payload),)
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
            yield Command("arm_request", op, identity, args)
            return ()
        if isinstance(op, StopOp):
            yield Command("stop_after", op, identity)
            return ()
        raise ReferenceLimit(f"{path(op)}: the reference semantics does not realize {op.name}")

    def _owned(self, op: Operation, value: object, identity: Identity) -> Ticket:
        if not isinstance(value, Ticket) or value.owner != identity or value.disposition is not None or value.forks:
            raise ReferenceFailure(f"{path(op)}: the claim is not held here")
        return value

    # Invocations.

    def _invoke(self, op: Operation, args: tuple[object, ...], parent: Identity, *, body: tuple[object, ...] | None = None) -> Body:
        identity = self._identity(parent, path(op))
        self.outcomes[identity] = Outcome(path(op), identity)
        self._state(identity, State.PENDING)
        if parent in self.request_of:
            self.request_of[identity] = self.request_of[parent]
        try:
            yield Command("admit", op, identity)
            self._state(identity, State.ADMITTED)
            self._state(identity, State.RUNNING)
            for ticket in (v for v in (body if body is not None else args) if isinstance(v, Ticket)):
                if ticket.owner != parent or ticket.disposition is not None:
                    raise ReferenceFailure(f"{path(op)}: a captured claim is not held by the capturing body")
                ticket.owner = identity
            if body is not None:
                result = yield from self._body(op.regions[0].block, body, identity)
            else:
                result = yield from self._protocol(op, args, identity)
            for value in result:
                if isinstance(value, Ticket):
                    value.owner = parent
            self.outcomes[identity].values = deepcopy(tuple(v for v in result if not isinstance(v, Ticket | Product | Lease)))
            self.outcomes[identity].status = "succeeded"
            return result
        except ReferenceFailure as failed:
            self.outcomes[identity].status = "cancelled" if isinstance(failed, Cancelled) else "failed"
            self.outcomes[identity].error = str(failed)
            raise
        finally:
            self._state(identity, State.CLOSING)
            self._state(identity, State.SETTLING)
            for stream in self.streams.values():
                for ticket in tuple(stream.outstanding.values()):
                    for held in self._leaves(ticket):
                        if held.owner == identity and held.disposition is None:
                            self._settle(held, "cancelled" if self.cancelled else "failed", identity)
            for command in self.emissions.pop(identity, ()):
                if not self._discarding:
                    yield from self._emit(command, self.outcomes[identity].status)
            self._state(identity, State.TERMINAL)

    def _emit(self, command: Command, status: str) -> Body:
        assert isinstance(command.op, RequestOp)
        boundary = Boundary(command.op.boundary.data)
        if boundary == Boundary.COMPLETED:
            raise ReferenceLimit(f"{path(command.op)}: the reference semantics requests at settlement only")
        if (boundary == Boundary.SETTLED_SUCCESS and status != "succeeded") or self.closing:
            self._fact(symbol(command.op.handler), command.invocation, "request_disposed", "unfulfilled")
            return ()
        try:
            yield command
        except Cancelled:
            self._fact(symbol(command.op.handler), command.invocation, "request_disposed", "cancelled")
        return ()

    def _entry(self, op: Operation, args: tuple[object, ...], supplied: tuple[object, ...] = (), *, role: int = 0) -> tuple[object, ...]:
        """A body's entry values from its contract's capture map, then what the protocol supplies."""
        contract = _region(op).describe(op).roles[role]
        missing = object()
        values: list[object] = [missing] * len(contract.entries)
        for capture in contract.captures:
            values[capture.argument] = args[capture.operand]
        open_positions = [i for i, v in enumerate(values) if v is missing]
        if len(open_positions) != len(supplied):
            raise ReferenceFailure(f"{path(op)}: the protocol supplies {len(supplied)} entries, the body takes {len(open_positions)}")
        for position, value in zip(open_positions, supplied, strict=True):
            values[position] = value
        return tuple(values)

    def _protocol(self, op: Operation, args: tuple[object, ...], identity: Identity) -> Body:
        protocol = _region(op).protocol
        if protocol == LifecycleProtocol.SEQUENCE:
            return (yield from self._body(op.regions[0].block, self._entry(op, args), identity))
        if protocol == LifecycleProtocol.CHOICE:
            return (yield from self._choice(op, args, identity))
        if protocol == LifecycleProtocol.PARALLEL:
            return (yield from self._parallel(op, args, identity))
        if protocol == LifecycleProtocol.REPEAT:
            return (yield from self._repeat(op, args, identity))
        if protocol == LifecycleProtocol.ACCESS:
            return (yield from self._access(op, args, identity))
        if protocol == LifecycleProtocol.FILTER:
            return (yield from self._filter(op, args, identity))
        if protocol == LifecycleProtocol.PRODUCER:
            return (yield from self._producer(op, args, identity))
        if protocol == LifecycleProtocol.HANDLER:
            return (yield from self._handler(op, args, identity))
        raise ReferenceLimit(f"{path(op)}: the reference semantics does not realize {protocol}")

    def _choice(self, op: Operation, args: tuple[object, ...], identity: Identity) -> Body:
        chosen = args[0]
        if isinstance(op.operands[0].type, OptionalType):
            if chosen is not None:
                return (yield from self._body(op.regions[0].block, self._entry(op, args, (chosen,)), identity))
            return (yield from self._body(op.regions[1].block, self._entry(op, args, role=1), identity))
        if type(chosen) is not bool:
            raise ReferenceFailure(f"{path(op)}: a choice chooses on a bool")
        role = 0 if chosen else 1
        return (yield from self._body(op.regions[role].block, self._entry(op, args, role=role), identity))

    def _parallel(self, op: Operation, args: tuple[object, ...], identity: Identity) -> Body:
        # A claim captured into a parallel goes to the one branch that uses it.
        branches = []
        for index, region in enumerate(op.regions):
            branch = (*identity, (f"{path(op)}#{index}", 0))
            if identity in self.request_of:
                self.request_of[branch] = self.request_of[identity]
            for position, value in enumerate(args):
                if isinstance(value, Ticket) and region.block.args[position].uses:
                    value.owner = branch
            branches.append(self._spawn(self._body(region.block, self._entry(op, args, role=index), branch)))
        yield Command("join", op, identity, tuple(branches))
        errors = [error for index in branches if (error := self.tasks[index].error) is not None]
        if self.cancelled:
            raise Cancelled("the run was cancelled")
        if errors:
            raise next((e for e in errors if not isinstance(e, Cancelled)), errors[0])
        return ()

    def _repeat(self, op: Operation, args: tuple[object, ...], identity: Identity) -> Body:
        if op.properties.get("limit") is None:
            raise ReferenceLimit(f"{path(op)}: the reference semantics repeats a number of times, not by policy")
        limit = op.properties["limit"]
        assert isinstance(limit, IntegerAttr)
        carried = op.properties["carry_count"]
        assert isinstance(carried, IntegerAttr)
        split = len(args) - carried.value.data
        captures, carries = args[:split], args[split:]
        for iteration in range(limit.value.data):
            carries = yield from self._invoke(op, (), identity, body=self._entry(op, (*captures, *carries), (iteration, *carries)))
        return carries

    def _access(self, op: Operation, args: tuple[object, ...], identity: Identity) -> Body:
        leases = yield Command("acquire", op, identity, args)
        assert isinstance(leases, tuple)
        try:
            return (yield from self._body(op.regions[0].block, self._entry(op, args, leases), identity))
        finally:
            for lease in leases:
                lease.body_ended = True
            if not self._discarding:
                yield Command("release", op, identity, leases)

    def _filter(self, op: Operation, args: tuple[object, ...], identity: Identity) -> Body:
        ticket = self._owned(op, args[0], identity)
        disposition = "failed"
        try:
            kept = yield from self._body(op.regions[0].block, self._entry(op, args, (deepcopy(ticket.payload),)), identity)
            if len(kept) != 1 or type(kept[0]) is not bool:
                raise ReferenceFailure(f"{path(op)}: a filter's body yields one bool")
            disposition = "forwarded" if kept[0] else "filtered_out"
            return (deepcopy(ticket.payload) if kept[0] else None,)
        except Cancelled:
            disposition = "cancelled"
            raise
        finally:
            self._settle(ticket, disposition, identity)

    def _producer(self, op: Operation, args: tuple[object, ...], identity: Identity) -> Body:
        name = str(args[0])
        last = op.regions[0].block.last_op
        optional = last is not None and bool(last.operands) and isinstance(last.operands[0].type, OptionalType)
        try:
            index = 0
            while not self._stopped(op, identity) and not self.closing:
                ticket = yield Command("reserve", op, identity, (name,))
                if ticket is None:
                    break
                assert isinstance(ticket, Ticket)
                production = (*identity, (path(op), index))
                start = len(self.read_versions[identity])
                try:
                    (value,) = yield from self._invoke(op, (), identity, body=self._entry(op, args, (index,)))
                except ReferenceFailure as failed:
                    if ticket.disposition is None:
                        self._settle(ticket, "cancelled" if isinstance(failed, Cancelled) else "failed", production)
                    raise
                index += 1
                ticket.versions = tuple(dict.fromkeys(self.read_versions[identity][start:]))
                if optional and value is None:
                    self.streams[name].outstanding.pop(ticket.key)
                    self._fact(name, identity, "nothing_produced", ticket.key)
                    continue
                try:
                    yield Command("publish", op, identity, (ticket, value))
                except ReferenceFailure as failed:
                    if ticket.disposition is None:
                        self._settle(ticket, "cancelled" if isinstance(failed, Cancelled) else "failed", identity)
                    raise
        finally:
            self.streams[name].closed = True
            if not self._discarding:
                yield Command("drain", op, identity, (name,))
        return ()

    def _handler(self, op: Operation, args: tuple[object, ...], identity: Identity) -> Body:
        handler = self.handlers[path(op)]
        while True:
            request = yield Command("next_request", op, identity)
            if request is None:
                return ()
            assert isinstance(request, Request)
            self.request_of[identity] = request
            try:
                yield from self._invoke(op, (), identity, body=self._entry(op, args, (deepcopy(request.payload),)))
                handler.dispositions[request.ordinal] = "completed"
            except ReferenceFailure as failed:
                handler.dispositions[request.ordinal] = "cancelled" if isinstance(failed, Cancelled) else "failed"
                raise
            finally:
                handler.active -= 1
                self._end_reservations(request)
                self._fact(path(op), identity, "request_settled", request.ordinal, handler.dispositions.get(request.ordinal))
                del self.request_of[identity]

    # Claims.

    @staticmethod
    def _leaves(ticket: Ticket) -> list[Ticket]:
        if not ticket.forks:
            return [ticket]
        return [leaf for fork in ticket.forks for leaf in Reference._leaves(fork)]

    def _settle(self, ticket: Ticket, disposition: str, identity: Identity) -> None:
        if ticket.disposition is not None:
            raise ReferenceFailure(f"{ticket.stream}/{ticket.key}: a claim is settled twice")
        ticket.disposition = (disposition,)
        self._fact(ticket.stream, identity, "disposition", ticket.key, disposition)
        while ticket.parent is not None:
            parent = ticket.parent
            if any(fork.disposition is None for fork in parent.forks):
                return
            parent.disposition = tuple(d for fork in parent.forks for d in fork.disposition or ())
            ticket = parent
        stream = self.streams[ticket.stream]
        if stream.outstanding.get(ticket.key) is not ticket:
            raise ReferenceFailure(f"{ticket.stream}/{ticket.key}: settled without its reservation")
        stream.outstanding.pop(ticket.key)
        if ticket in stream.ready:
            stream.ready.remove(ticket)
        stream.settled[ticket.key] = ticket.disposition or ()

    # Leases and retained results.

    def _release_product(self, product: Product, how: str) -> None:
        if self.products.get(product.key) is not product or not product.active:
            raise ReferenceFailure(f"{product.key[0]}: a retained result is released twice")
        if product.borrowers or any(any(p is product for p in other.parents) for other in self.products.values() if other.active):
            raise ReferenceFailure(f"{product.key[0]}: released while a dependent use is still running")
        product.active = False
        del self.products[product.key]
        self._fact(product.key[0], (), "product_released", product.key[1], how)
        for lease in product.leases:
            self._release_lease_if_done(lease)

    def _release_lease_if_done(self, lease: Lease) -> None:
        if not lease.active or not lease.body_ended:
            return
        if any(lease in product.leases for product in self.products.values()):
            return
        resource = self.resources[lease.resource]
        resource.readers.discard(lease.token)
        if resource.writer == lease.token:
            resource.writer = None
        lease.active = False
        self._fact(lease.resource, (), "lease_released", lease.version)

    # Requests.

    def _at_request(self, handler: Operation) -> list[str]:
        """The resources the handler's work reads at the version of its request."""
        return [
            self._resource_name(access, use.subject)
            for access in handler.walk()
            if isinstance(access, AccessOp)
            for use in ResourceSemantics().describe(access).uses
            if use.selection == VersionSelection.AT_REQUEST
        ]

    def _reserve(self, request: Request, handler: Operation) -> None:
        """Holds the version at the request for each at_request use the handler makes."""
        for access in (op for op in handler.walk() if isinstance(op, AccessOp)):
            for use in ResourceSemantics().describe(access).uses:
                if use.selection != VersionSelection.AT_REQUEST:
                    continue
                name = self._resource_name(access, use.subject)
                resource = self.resources[name]
                if use.mode == AccessMode.SNAPSHOT:
                    request.snapshots[name] = (resource.version, deepcopy(resource.value))
                else:
                    request.reserved[name] = resource.version
                    resource.reservations.add(id(request))
                self._fact(name, request.cause, "reserved", resource.version, str(use.mode))

    def _end_reservations(self, request: Request) -> None:
        for name in request.reserved:
            self.resources[name].reservations.discard(id(request))
        request.reserved.clear()

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
            raise ReferenceFailure(f"{path(access)}: an access names something that is not a resource")
        return owner.sym_name.data

    # Scheduling.

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
            raise ReferenceFailure(f"{path(waiting)}: refers outside its strategy")
        return common

    def _scope(self, waiting: Operation, source: Operation, identity: Identity) -> Identity:
        """The invocation of the nearest statement enclosing both, within the waiting invocation."""
        common = self._common(waiting, source)
        positions = [i for i, (owner, _) in enumerate(identity) if owner == path(common)]
        return identity[: max(positions) + 1] if positions else identity[:1]

    def _source_settled(self, source: Operation, waiting: Operation, identity: Identity) -> bool:
        """Whether the source has settled for good in the shared scope.

        The source and every statement between it and the shared scope must
        have ended, so a source inside a repeat waits for the repeat, and a
        source in a choice alternative that was not taken counts once the
        choice has ended. A source nothing has started yet has not settled.
        """
        common = self._common(waiting, source)
        scope = self._scope(waiting, source, identity)
        started = False
        current: Operation | None = source
        while current is not None and current is not common:
            found = [o for o in self.outcomes.values() if o.owner == path(current) and o.invocation[: len(scope)] == scope]
            if any(o.state != State.TERMINAL for o in found):
                return False
            started = started or bool(found)
            current = current.parent_op()
        return started

    def _pending_under(self, identity: Identity) -> list[Request]:
        return [r for h in self.handlers.values() for r in h.pending if r.cause[: len(identity)] == identity]

    def _ready(self, command: Command) -> bool:
        kind, op, args = command.kind, command.op, command.args
        if self.closing and kind not in SETTLING_COMMANDS:
            return True
        if kind == "call_complete":
            pending = self.calls[_number(args[0])]
            return (path(pending.command.op), pending.ordinal) not in self.held
        if kind == "acquire":
            return all(self._can_lease(op, use, args) for use in ResourceSemantics().describe(op).uses)
        if kind == "take":
            stream = self.streams[str(args[0])]
            return bool(stream.ready) or stream.closed
        if kind == "reserve":
            stream = self.streams[str(args[0])]
            return self._stopped(op, command.invocation) or len(stream.outstanding) < stream.declaration.outstanding.value.data
        if kind == "publish":
            ticket = args[0]
            assert isinstance(ticket, Ticket)
            stream = self.streams[ticket.stream]
            return self._stopped(op, command.invocation) or len(stream.ready) < stream.declaration.ready.value.data
        if kind == "drain":
            # Wait for consumers to settle what they took; what the producer still holds settles at the drain.
            stream = self.streams[str(args[0])]
            return all(
                leaf.owner is None or leaf.owner == command.invocation
                for t in stream.outstanding.values()
                if t not in stream.ready
                for leaf in self._leaves(t)
            )
        if kind == "join":
            return all(self.tasks[_number(index)].done for index in args)
        if kind == "settle_requests":
            return self.closing or not self._pending_under(command.invocation)
        if kind == "request":
            assert isinstance(op, RequestOp)
            target = self.named[symbol(op.handler)]
            handler = self.handlers[path(target)]
            assert isinstance(target, HandlerOp)
            if any(self.resources[name].writer is not None for name in self._at_request(target)):
                return False  # the version at the request is the one after the update in flight
            return len(handler.pending) + handler.active < target.max_outstanding.value.data
        if kind == "next_request":
            handler = self.handlers[path(op)]
            return (
                self.closing
                or bool(handler.pending)
                or all(self._source_settled(self.named[name], op, command.invocation) for name in symbols(op, "sources"))
            )
        if kind == "stop_after":
            assert isinstance(op, StopOp)
            return self._source_settled(self.named[symbol(op.source)], op, command.invocation)
        return True

    def _can_lease(self, op: Operation, use, args: tuple[object, ...]) -> bool:
        resource = self.resources[self._resource_name(op, use.subject)]
        if use.mode == AccessMode.SNAPSHOT and use.selection in (VersionSelection.PUBLISHED_INITIAL, VersionSelection.AT_REQUEST):
            return True
        if use.mode in (AccessMode.SNAPSHOT, AccessMode.PROTECTED_CURRENT):
            return resource.writer is None
        return resource.writer is None and not resource.readers and not resource.reservations

    def enabled(self) -> tuple[int, ...]:
        """The tasks whose next command can be performed now."""
        return tuple(i for i, task in enumerate(self.tasks) if task.command is not None and self._ready(task.command))

    def step(self, choice: int | None = None) -> bool:
        """Performs one enabled command, the chosen task's or the next in turn; False when none is enabled."""
        enabled = self.enabled()
        if not enabled:
            return False
        if choice is None:
            later = [i for i in enabled if i >= self.cursor]
            choice = later[0] if later else enabled[0]
        if choice not in enabled:
            raise ReferenceFailure(f"task {choice} has no enabled command")
        task = self.tasks[choice]
        assert task.command is not None
        self.cursor = (choice + 1) % len(self.tasks)
        try:
            result = self._perform(task.command)
        except ReferenceFailure as failed:
            self._note_failure(failed)
            self._resume(task, error=failed)
        else:
            self._resume(task, result)
        return True

    def run(self, *, max_steps: int = 100_000) -> None:
        """Steps until nothing is enabled."""
        for _ in range(max_steps):
            if not self.step():
                return
        raise ReferenceLimit("the step bound ran out")

    @property
    def settled(self) -> bool:
        return all(task.done for task in self.tasks)

    def cancel(self) -> None:
        """Run cancellation: admission closes while entered work settles."""
        self.cancelled = True

    def hold(self, statement: str, ordinal: int = 0) -> None:
        """Holds back a call's physical completion until :meth:`complete`."""
        self.held.add((statement, ordinal))

    def complete(self, statement: str, ordinal: int = 0) -> None:
        self.held.discard((statement, ordinal))

    def fail(self, statement: str, message: str, ordinal: int = 0) -> None:
        """Makes a call's physical completion fail."""
        self.failures[(statement, ordinal)] = message

    # Performing commands.

    def _perform(self, command: Command) -> object:
        kind, op, args, identity = command.kind, command.op, command.args, command.invocation
        if self.closing and kind not in SETTLING_COMMANDS:
            raise Cancelled("the run was cancelled" if self.cancelled else "a sibling failed, so admission closed")
        if kind in ("tick", "join", "admit"):
            return None
        if kind == "acquire":
            return self._acquire(op, args, identity)
        if kind == "release":
            for lease in args:
                assert isinstance(lease, Lease)
                self._release_lease_if_done(lease)
            return None
        if kind == "call_enter":
            return self._call_enter(command)
        if kind == "call_complete":
            return self._call_complete(command)
        if kind == "reserve":
            stream = self.streams[str(args[0])]
            if self._stopped(op, identity):
                return None
            ticket = Ticket(str(args[0]), stream.next_key, identity)
            stream.next_key += 1
            stream.outstanding[ticket.key] = ticket
            return ticket
        if kind == "publish":
            ticket, value = args
            assert isinstance(ticket, Ticket)
            ticket.payload, ticket.owner = deepcopy(value), None
            if self._stopped(op, identity):
                self._settle(ticket, "cancelled", identity)
            else:
                self.streams[ticket.stream].ready.append(ticket)
                self._fact(ticket.stream, identity, "published", ticket.key, ticket.versions)
            return None
        if kind == "take":
            return self._take(op, args, identity)
        if kind == "drain":
            stream = self.streams[str(args[0])]
            for ticket in tuple(stream.outstanding.values()):
                if ticket in stream.ready or ticket.owner == identity:
                    self._settle(ticket, "cancelled", identity)
            return None
        if kind == "arm_request":
            assert isinstance(op, RequestOp)
            source = self.named[symbol(op.source)]
            running = [
                k for k, o in self.outcomes.items() if o.owner == path(source) and o.state == State.RUNNING and identity[: len(k)] == k
            ]
            if not running:
                raise ReferenceFailure(f"{path(op)}: a request is made inside its source")
            self.emissions[max(running, key=len)].append(Command("request", op, max(running, key=len), args))
            return None
        if kind == "request":
            assert isinstance(op, RequestOp)
            target = self.named[symbol(op.handler)]
            handler = self.handlers[path(target)]
            request = Request(path(target), handler.next_ordinal, identity, deepcopy(args[0]))
            handler.next_ordinal += 1
            self._reserve(request, target)
            handler.pending.append(request)
            self._fact(path(target), identity, "request_pending", request.ordinal, request.payload)
            return None
        if kind == "next_request":
            handler = self.handlers[path(op)]
            if self.closing:
                while handler.pending:
                    request = handler.pending.popleft()
                    self._end_reservations(request)
                    handler.dispositions[request.ordinal] = "cancelled"
                    self._fact(path(op), identity, "request_disposed", request.ordinal, "cancelled")
                return None
            if not handler.pending:
                return None
            handler.active += 1
            return handler.pending.popleft()
        if kind == "stop_after":
            assert isinstance(op, StopOp)
            target = self.named[symbol(op.target)]
            self.stops.add((self._scope(op, target, identity), path(target)))
            self._fact(path(target), identity, "stop_requested", symbol(op.source))
            return None
        if kind == "settle_requests":
            for request in self._pending_under(identity):
                self.handlers[request.handler].pending.remove(request)
                self._end_reservations(request)
                self.handlers[request.handler].dispositions[request.ordinal] = "cancelled"
                self._fact(request.handler, identity, "request_disposed", request.ordinal, "cancelled")
            return None
        raise ReferenceFailure(f"unknown command {kind}")

    def _acquire(self, op: Operation, args: tuple[object, ...], identity: Identity) -> tuple[Lease, ...]:
        leases = []
        request = self.request_of.get(identity)
        for use in ResourceSemantics().describe(op).uses:
            name = self._resource_name(op, use.subject)
            resource = self.resources[name]
            version, value = resource.version, resource.value
            if use.mode == AccessMode.GRANT:
                raise ReferenceLimit(f"{path(op)}: the reference semantics does not realize grants")
            if use.selection == VersionSelection.PUBLISHED_INITIAL:
                version, value = self.initial[name]
            elif use.selection == VersionSelection.AT_REQUEST:
                if request is None:
                    raise ReferenceFailure(f"{path(op)}: at_request outside requested work")
                if use.mode == AccessMode.SNAPSHOT:
                    version, value = request.snapshots[name]
                elif request.reserved.get(name) != resource.version:
                    raise ReferenceFailure(f"{path(op)}: @{name} moved past the version its request reserved")
            if use.mode == AccessMode.SNAPSHOT:
                value = deepcopy(value)
            lease = Lease(len(self.leases), name, use.mode, version, value)
            self.leases[lease.token] = lease
            if use.mode == AccessMode.PROTECTED_CURRENT:
                resource.readers.add(lease.token)
            elif use.mode == AccessMode.OWNER_UPDATE:
                resource.writer = lease.token
            for length in range(1, len(identity) + 1):
                self.read_versions[identity[:length]].append(version)
            self._fact(path(op), identity, "leased", name, str(use.mode), version)
            leases.append(lease)
        return tuple(leases)

    def _take(self, op: Operation, args: tuple[object, ...], identity: Identity) -> Ticket | None:
        stream = self.streams[str(args[0])]
        if not stream.ready:
            raise ReferenceFailure(f"{path(op)}: @{stream.declaration.sym_name.data} closed before this take")
        ticket = stream.ready.popleft()
        admitted = self._admit(stream.declaration, ticket)
        self._fact(path(op), identity, "admission", ticket.key, ticket.versions, admitted)
        if not admitted:
            ticket.owner = identity
            self._settle(ticket, "rejected", identity)
            return None
        ticket.owner = identity
        return ticket

    def _admit(self, stream: StreamOp, ticket: Ticket) -> bool:
        """Admission by the policy's owners' versions, computed here rather than taken from the verifier."""
        policy = FlowSemantics().describe(stream).versions
        observed: dict[str, Version] = {}
        for version in ticket.versions:
            if observed.setdefault(version.owner, version) != version:
                return False  # production observed one owner at two versions: incomparable
        if policy.admission == Admission.ANY_WITH_PROVENANCE:
            return True
        allowed = policy.max_lag if policy.admission == Admission.BOUNDED_LAG else 0
        for owner in policy.owners:
            seen, now = observed.get(owner), self.resources[owner].version
            if seen is None or (seen.incarnation, seen.lineage) != (now.incarnation, now.lineage) or seen.ordinal > now.ordinal:
                return False
            if now.ordinal - seen.ordinal > allowed:
                return False
        return True

    def _call_enter(self, command: Command) -> int:
        op, args = command.op, command.args
        contract = RealizationSemantics().describe(op)
        standin = self.standins.get(contract.implementation)
        if standin is None or standin.version != contract.version:
            raise ReferenceLimit(f"{path(op)}: no stand-in for @{contract.implementation} v{contract.version}")
        inputs = tuple(written(v.type) for v in op.operands)
        outputs = tuple(written(r.type) for r in op.results)
        if (standin.inputs, standin.outputs, standin.retains) != (inputs, outputs, contract.retains):
            raise ReferenceFailure(f"{path(op)}: the stand-in for @{contract.implementation} does not match the call's contract")
        for value, name in zip(args, inputs, strict=True):
            if not self._valid(value, name):
                raise ReferenceFailure(f"{path(op)}: an input is not a {name}")
        ordinal = self.call_counts[path(op)]
        self.call_counts[path(op)] += 1
        for value in args:
            if isinstance(value, Product):
                value.borrowers += 1
        given = tuple(deepcopy(v.value) if isinstance(v, Lease | Product) else deepcopy(v) for v in args)
        self.calls[len(self.calls)] = PendingCall(command, standin, ordinal, given)
        self._fact(path(op), command.invocation, "call_entered", ordinal)
        return len(self.calls) - 1

    def _call_complete(self, command: Command) -> tuple[object, ...]:
        pending = self.calls[_number(command.args[0])]
        op, args, identity = pending.command.op, pending.command.args, pending.command.invocation
        try:
            failure = self.failures.get((path(op), pending.ordinal))
            if failure is not None:
                self._fact(path(op), identity, "call_failed", pending.ordinal, failure)
                raise ReferenceFailure(f"{path(op)}: {failure}")
            try:
                result = pending.standin.invoke(pending.inputs)
            except Exception as raised:
                self._fact(path(op), identity, "call_failed", pending.ordinal, str(raised))
                raise ReferenceFailure(f"{path(op)}: the stand-in raised {raised}") from raised
            if not isinstance(result, CallResult):
                raise ReferenceFailure(f"{path(op)}: the stand-in returned something other than a CallResult")
            for update in result.updates:
                lease = args[update.lease]
                if update.lease not in pending.standin.updates or not isinstance(lease, Lease) or lease.mode != AccessMode.OWNER_UPDATE:
                    raise ReferenceFailure(f"{path(op)}: updates state it holds no owner_update lease for")
                resource = self.resources[lease.resource]
                if resource.writer != lease.token or resource.version != lease.version:
                    raise ReferenceFailure(f"{path(op)}: a stale owner update of @{lease.resource}")
                resource.value, resource.ordinal = deepcopy(update.value), resource.ordinal + 1
                lease.value, lease.version = resource.value, resource.version
                self._fact(lease.resource, identity, "updated", resource.version, resource.value)
            outputs = pending.standin.outputs
            if len(result.values) != len(outputs) or not all(
                self._valid(v, name.removeprefix("retained ")) for v, name in zip(result.values, outputs, strict=True)
            ):
                self._fact(path(op), identity, "call_invalid", pending.ordinal)
                raise ReferenceFailure(f"{path(op)}: the stand-in's results do not match its outputs")
            shown = ("retained",) if pending.standin.retains else result.values
            self._fact(path(op), identity, "call_completed", pending.ordinal, shown, self._request_payload(identity))
            if not pending.standin.retains:
                return result.values
            retained = [args[i] for i in pending.standin.retains]
            parents = tuple(v for v in retained if isinstance(v, Product))
            leases = {id(v): v for v in retained if isinstance(v, Lease)}
            for parent in parents:
                leases.update((id(lease), lease) for lease in parent.leases)
            product = Product((path(op), pending.ordinal), result.values[0], tuple(leases.values()), parents)
            self.products[product.key] = product
            self._fact(path(op), identity, "product_retained", pending.ordinal, tuple(lease.version for lease in product.leases))
            return (product,)
        finally:
            for value in args:
                if isinstance(value, Product):
                    value.borrowers -= 1

    def _request_payload(self, identity: Identity) -> object:
        request = self.request_of.get(identity)
        return request.payload if request is not None else None

    def _valid(self, value: object, name: str) -> bool:
        if name.endswith(" lease"):
            return isinstance(value, Lease) and value.active
        if name.startswith("retained "):
            return isinstance(value, Product) and value.active and self._valid(value.value, name.removeprefix("retained "))
        if name.startswith("optional "):
            return value is None or self._valid(value, name.removeprefix("optional "))
        if isinstance(value, Ticket | Lease | Product):
            return False
        if name == "int":
            return type(value) is int
        if name == "float":
            return type(value) is float
        if name == "bool":
            return type(value) is bool
        if name == "str":
            return type(value) is str
        validator = self.validators.get(name)
        return validator(value) if validator is not None else True

    # Reading the outcome.

    def settlements(self, stream: str) -> dict[int, tuple[str, ...]]:
        """Each settled item's dispositions, one per fork for a forked item."""
        return dict(self.streams[stream].settled)

    def facts_of(self, kind: str) -> list[Fact]:
        return [f for f in self.facts if f.kind == kind]


def _region(op: Operation) -> RegionSemantics:
    semantics = op.get_trait(RegionSemantics)
    if semantics is None:
        raise ReferenceFailure(f"{path(op)}: has no lifecycle protocol")
    return semantics


def _number(value: object) -> int:
    assert isinstance(value, int)
    return value


def _literal(op: ConstantOp) -> object:
    value = op.value
    if isinstance(value, IntegerAttr):
        number = value.value.data
        return bool(number) if isinstance(value.type, IntegerType) and value.type.width.data == 1 else number
    if isinstance(value, FloatAttr):
        return value.value.data
    if isinstance(value, StringAttr):
        return value.data
    raise ReferenceFailure(f"{path(op)}: an unknown literal")


__all__ = [
    "Cancelled",
    "Fact",
    "Lease",
    "Outcome",
    "Product",
    "Reference",
    "ReferenceFailure",
    "ReferenceLimit",
    "Request",
    "State",
    "Ticket",
    "Version",
]
