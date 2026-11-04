"""Single-caller lifecycle oracle with explicit logical scheduling/completions.

This executes normalized IR, not prepared owner services. Physical completion
is an external model event; it is never evidence about an actual device.
"""

from __future__ import annotations

import json

from collections import defaultdict, deque
from collections.abc import Callable, Generator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Operation, SSAValue
from xdsl.traits import OpTrait

from .contracts import (
    AccessMode,
    Admission,
    FlowContract,
    FlowSemantics,
    Freshness,
    Protocol,
    RegionSemantics,
    ResourceSemantics,
    Version,
    number,
    origin,
    symbol,
    text,
)
from .dialects import (
    BorrowOp,
    CapabilityType,
    ClaimType,
    CallOp,
    ConstantOp,
    EndpointType,
    PayloadType,
    ProgramOp,
    RequestOp,
    ResourceOp,
    ResourceType,
    RetireOp,
    StopAfterOp,
    StreamOp,
    TakeOp,
    YieldOp,
)
from .normalize import Normalized, inspect_structure, source_index, symbol_index


Identity = tuple[tuple[str, int], ...]


class State(StrEnum):
    CREATED = "created"
    PENDING = "pending"
    ADMITTED = "admitted"
    RUNNING = "running"
    CLOSING = "closing"
    SETTLING = "settling"
    TERMINAL = "terminal"


class ReferenceError(ValueError):
    pass


class ReferenceLimit(ReferenceError):
    pass


class Cancelled(ReferenceError):
    pass


def checked[T](value: object, kind: type[T]) -> T:
    if not isinstance(value, kind):
        raise ReferenceError(f"reference command expected {kind.__name__}")
    return value


def require_trait[T: OpTrait](op: Operation, kind: type[T]) -> T:
    trait = op.get_trait(kind)
    if trait is None:
        raise ReferenceError(f"{origin(op)}: missing {kind.__name__}")
    return trait


@dataclass
class Resource:
    owner: str
    value: object
    incarnation: int = 1
    lineage: str = "initial"
    ordinal: int = 0
    readers: set[int] = field(default_factory=set)
    writer: int | None = None

    @property
    def version(self) -> Version:
        return Version(self.owner, self.incarnation, self.lineage, self.ordinal)


@dataclass
class Capability:
    resource: str
    mode: AccessMode
    version: Version
    value: object
    token: int
    active: bool = True


@dataclass(frozen=True)
class Update:
    capability: int
    value: object


@dataclass(frozen=True)
class CallResult:
    values: tuple[object, ...]
    updates: tuple[Update, ...] = ()


@dataclass(frozen=True)
class StandIn:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    invoke: Callable[[tuple[object, ...]], CallResult]
    updates: tuple[int, ...] = ()
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.version or any(
            type(index) is not int or not 0 <= index < len(self.inputs) or self.inputs[index] != "capability" for index in self.updates
        ):
            raise ReferenceError("invalid reference binding contract/update envelope")


@dataclass(frozen=True)
class Fact:
    owner: str
    invocation: Identity
    kind: str
    values: tuple[object, ...] = ()


@dataclass
class Outcome:
    owner: str
    invocation: Identity
    invocation_source: str = "parent_entry"
    state: State = State.CREATED
    status: str = "pending"
    values: tuple[object, ...] = ()
    error: str | None = None
    cleanup: bool = False
    cancel_requested: bool = False
    stop_requested: bool = False


@dataclass
class Ticket:
    stream: str
    key: int
    invocation: Identity
    payload: object = None
    versions: tuple[Version, ...] = ()
    owner: Identity | None = None
    disposition: str | None = None


@dataclass
class Stream:
    declaration: StreamOp
    outstanding: dict[int, Ticket] = field(default_factory=dict)
    ready: deque[Ticket] = field(default_factory=deque)
    terminal: dict[int, str] = field(default_factory=dict)
    next_key: int = 0
    high_water: int = 0
    closed: bool = False


@dataclass(frozen=True)
class Request:
    handler: str
    ordinal: int
    cause: Identity
    payload: object


@dataclass
class Handler:
    pending: deque[Request] = field(default_factory=deque)
    active: int = 0
    next_ordinal: int = 0
    dispositions: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Command:
    kind: str
    op: Operation
    invocation: Identity
    args: tuple[object, ...] = ()


@dataclass
class PendingCall:
    command: Command
    binding: StandIn
    ordinal: int
    snapshots: tuple[object, ...]


@dataclass
class Task:
    generator: Generator[Command, object, tuple[object, ...]]
    command: Command | None = None
    done: bool = False
    result: tuple[object, ...] = ()
    error: ReferenceError | None = None


def schema(value: SSAValue) -> str:
    kind = value.type.name
    if isinstance(value.type, PayloadType):
        return str(value.type.schema.data)
    return kind.removeprefix("lang.")


def valid(value: object, name: str, schemas: Mapping[str, Callable[[object], bool]]) -> bool:
    if name == "scalar":
        return type(value) in (float, int)
    if name == "index":
        return type(value) is int
    if name == "bool":
        return type(value) is bool
    if name == "capability":
        return isinstance(value, Capability) and value.active
    if name == "request":
        return isinstance(value, Request)
    validator = schemas.get(name)
    return validator(value) if validator is not None else False


class Reference:
    """Mutable oracle state belongs to this instance and one calling thread."""

    def __init__(
        self,
        program: Normalized,
        resources: Mapping[str, object],
        bindings: Mapping[str, StandIn],
        *,
        schemas: Mapping[str, Callable[[object], bool]] | None = None,
        history: bool = True,
    ):
        self.module: ModuleOp = program.module.clone()
        inspect_structure(self.module)
        self.index = source_index(self.module)
        self.symbols = symbol_index(self.module)
        self.root = next(iter(self.module.body.block.ops))
        if not isinstance(self.root, ProgramOp) or text(self.root, "stage") != "normalized":
            raise ReferenceError("reference requires normalized IR")
        if any(isinstance(op, ResourceOp | StreamOp) and op.parent_op() is not self.root for op in self.index.values()):
            raise ReferenceLimit("reference target requires root-owned persistent resource/endpoint declarations")
        for op in self.index.values():
            trait = op.get_trait(RegionSemantics)
            sources = trait.describe(op).request_sources if trait is not None and trait.protocol == Protocol.REQUESTS else ()
            if isinstance(op, StopAfterOp):
                sources = (symbol(op.properties["source"]),)
            if any(self.symbols[name].is_ancestor(op) for name in sources):
                raise ReferenceError(f"{origin(op)}: circular self-referential source settlement")
        self.bindings = dict(bindings)
        self.schemas = dict(schemas or {})
        self.resources = {key: Resource(key, deepcopy(value)) for key, value in resources.items()}
        self.initial_snapshots = {key: (value.version, deepcopy(value.value)) for key, value in self.resources.items()}
        self.streams: dict[str, Stream] = {}
        self.handlers: dict[str, Handler] = defaultdict(Handler)
        self.outcomes: dict[Identity, Outcome] = {}
        self.facts: list[Fact] = []
        self.history = history
        self.tasks: list[Task] = []
        self.stopped: set[str] = set()
        self._external_stops: set[str] = set()
        self._scoped_stops: set[tuple[Identity, str]] = set()
        self.cancelled = False
        self._failure_stop = False
        self.errors: list[str] = []
        self._reported_errors: set[int] = set()
        self.held: set[tuple[str, int]] = set()
        self.failures: dict[tuple[str, int], str] = {}
        self.calls: dict[int, PendingCall] = {}
        self.call_counts: dict[str, int] = defaultdict(int)
        self.ordinals: dict[tuple[Identity, str], int] = defaultdict(int)
        self.capabilities: dict[int, Capability] = {}
        self.read_versions: dict[Identity, list[Version]] = defaultdict(list)
        self.request_causes: dict[Identity, Request] = {}
        self.input_uses: dict[Identity, tuple[tuple[str, int], ...]] = {}
        self.product_uses: dict[Identity, tuple[tuple[str, int], ...]] = {}
        self.emissions: dict[Identity, list[Command]] = defaultdict(list)
        self.cursor = 0
        self._spawn(self._root())

    def _fact(self, owner: str, invocation: Identity, kind: str, *values: object) -> None:
        if self.history:
            self.facts.append(Fact(owner, invocation, kind, deepcopy(values)))

    def _identity(self, parent: Identity, owner: str) -> Identity:
        key = parent, owner
        ordinal = self.ordinals[key]
        self.ordinals[key] += 1
        return (*parent, (owner, ordinal))

    def _state(self, identity: Identity, state: State) -> None:
        outcome = self.outcomes[identity]
        allowed = {
            State.CREATED: (State.CREATED, State.PENDING),
            State.PENDING: (State.ADMITTED, State.CLOSING),
            State.ADMITTED: (State.RUNNING,),
            State.RUNNING: (State.CLOSING,),
            State.CLOSING: (State.SETTLING,),
            State.SETTLING: (State.TERMINAL,),
            State.TERMINAL: (),
        }
        if state not in allowed[outcome.state]:
            raise ReferenceError(f"{outcome.owner}: illegal lifecycle transition {outcome.state} -> {state}")
        outcome.state = state
        self._fact(outcome.owner, identity, "state", state)
        if state == State.TERMINAL:
            self._fact(
                outcome.owner,
                identity,
                "outcome",
                outcome.status,
                outcome.values,
                outcome.error,
                outcome.cleanup,
                outcome.invocation_source,
                outcome.cancel_requested,
                outcome.stop_requested,
            )

    def _spawn(self, generator: Generator[Command, object, tuple[object, ...]]) -> int:
        task = Task(generator)
        index = len(self.tasks)
        self.tasks.append(task)
        self._resume(task)
        return index

    def _resume(self, task: Task, result: object = None, error: ReferenceError | None = None) -> None:
        try:
            task.command = task.generator.throw(error) if error is not None else task.generator.send(result)
        except StopIteration as completed:
            task.result = completed.value or ()
            task.done = True
            task.command = None
        except ReferenceError as failed:
            task.error = failed
            task.done = True
            task.command = None
            self._note_failure(failed)

    @property
    def closing(self) -> bool:
        return self.cancelled or self._failure_stop

    def _note_failure(self, failed: ReferenceError) -> None:
        if not isinstance(failed, Cancelled) and id(failed) not in self._reported_errors:
            self._reported_errors.add(id(failed))
            self.errors.append(str(failed))
            self._failure_stop = True

    def _root(self) -> Generator[Command, object, tuple[object, ...]]:
        identity = self._identity((), origin(self.root))
        self.outcomes[identity] = Outcome(origin(self.root), identity)
        self._state(identity, State.CREATED)
        self._state(identity, State.PENDING)
        self._state(identity, State.ADMITTED)
        self._state(identity, State.RUNNING)
        try:
            result = yield from self._body(self.root.body.block, (), identity)
            self.outcomes[identity].status = "succeeded"
            return result
        except ReferenceError as failed:
            self.outcomes[identity].status = "cancelled" if self.cancelled or isinstance(failed, Cancelled) else "failed"
            self.outcomes[identity].error = str(failed)
            raise
        finally:
            self._state(identity, State.CLOSING)
            self._state(identity, State.SETTLING)
            yield Command("settle_requests", self.root, identity)
            self.outcomes[identity].cleanup = True
            self.outcomes[identity].cancel_requested = self.cancelled
            self._state(identity, State.TERMINAL)

    def _body(self, block: Block, args: tuple[object, ...], parent: Identity) -> Generator[Command, object, tuple[object, ...]]:
        returned: tuple[object, ...] = ()
        try:
            returned = yield from self._body_impl(block, args, parent)
            return returned
        finally:
            # Parallel branch bodies own locals without a separate region-op
            # invocation. Their claims still need disposition on failure/cancel.
            for stream in self.streams.values():
                for ticket in tuple(stream.outstanding.values()):
                    if ticket.owner == parent and not any(value is ticket for value in returned):
                        self._dispose(ticket, "cancelled" if self.cancelled else "unconsumed", parent)

    def _body_impl(self, block: Block, args: tuple[object, ...], parent: Identity) -> Generator[Command, object, tuple[object, ...]]:
        if len(block.args) != len(args):
            raise ReferenceError("reference entry argument mismatch")
        env: dict[SSAValue, object] = dict(zip(block.args, args, strict=True))
        for op in block.ops:
            if isinstance(op, YieldOp):
                if isinstance(block.parent_op(), ProgramOp) and any(
                    isinstance(value.type, CapabilityType | ClaimType | ResourceType | EndpointType) for value in op.operands
                ):
                    raise ReferenceError("program has no outgoing owner/claim transfer contract")
                return tuple(env[value] for value in op.operands)
            if isinstance(op, ResourceOp):
                name = text(op, "sym_name")
                if name not in self.resources:
                    raise ReferenceError(f"{origin(op)}: missing reference resource value")
                env[op.results[0]] = name
                continue
            if isinstance(op, StreamOp):
                name = text(op, "sym_name")
                self.streams[name] = Stream(op)
                env[op.results[0]] = name
                continue
            values = tuple(env[value] for value in op.operands)
            result = yield from self._execute(op, values, parent)
            if len(result) != len(op.results):
                raise ReferenceError(f"{origin(op)}: reference result arity mismatch")
            env.update(zip(op.results, result, strict=True))
        return ()

    def _execute(self, op: Operation, args: tuple[object, ...], parent: Identity) -> Generator[Command, object, tuple[object, ...]]:
        region = op.get_trait(RegionSemantics)
        if region is not None:
            return (yield from self._invoke(op, args, parent))
        command = Command("tick", op, parent)
        yield command
        if isinstance(op, ConstantOp):
            result = json.loads(text(op, "literal"))
            if not valid(result, schema(op.results[0]), self.schemas):
                raise ReferenceError(f"{origin(op)}: invalid typed constant")
            return (result,)
        if isinstance(op, CallOp):
            entered = yield Command("call_enter", op, parent, args)
            result = yield Command("call_complete", op, parent, (entered,))
            return checked(result, tuple)
        if isinstance(op, TakeOp):
            ticket = None
            while ticket is None:
                ticket = yield Command("take", op, parent, args)
            return (ticket,)
        if isinstance(op, BorrowOp):
            ticket = args[0]
            if not isinstance(ticket, Ticket) or ticket.disposition is not None or ticket.owner != parent:
                raise ReferenceError(f"{origin(op)}: borrowed claim is not owned")
            return (deepcopy(ticket.payload),)
        if isinstance(op, RetireOp):
            yield Command("retire", op, parent, args)
            return ()
        if isinstance(op, RequestOp):
            yield Command("arm_request", op, parent, args)
            return ()
        if isinstance(op, StopAfterOp):
            yield Command("stop_after", op, parent)
            return ()
        raise ReferenceLimit(f"{origin(op)}: no reference realization for {op.name}")

    def _invoke(
        self,
        op: Operation,
        args: tuple[object, ...],
        parent: Identity,
        *,
        body_args: tuple[object, ...] | None = None,
        selected_product: Ticket | None = None,
    ) -> Generator[Command, object, tuple[object, ...]]:
        identity = self._identity(parent, origin(op))
        self.outcomes[identity] = Outcome(origin(op), identity)
        if body_args is not None:
            protocol = require_trait(op, RegionSemantics).protocol
            self.outcomes[identity].invocation_source = {
                Protocol.REPEAT: "repeat_policy",
                Protocol.PRODUCER: "production_selection",
                Protocol.REQUESTS: "accepted_request",
            }[protocol]
        self._state(identity, State.CREATED)
        self._state(identity, State.PENDING)
        if parent in self.request_causes:
            self.request_causes[identity] = self.request_causes[parent]
        self.input_uses[identity] = self.input_uses.get(parent, ())
        self.product_uses[identity] = (
            ((selected_product.stream, selected_product.key),) if selected_product is not None else self.product_uses.get(parent, ())
        )
        try:
            admitted = yield Command("admit", op, identity)
            if admitted is False:
                self.outcomes[identity].status = "stopped_unadmitted"
                carries = number(op, "carry_count")
                return body_args[-carries:] if body_args is not None and carries else ()
            self._state(identity, State.ADMITTED)
            self._state(identity, State.RUNNING)
            incoming = [value for value in (body_args if body_args is not None else args) if isinstance(value, Ticket)]
            if len({id(value) for value in incoming}) != len(incoming):
                raise ReferenceError(f"{origin(op)}: duplicated claim capture")
            for ticket in incoming:
                if ticket.owner != parent or ticket.disposition is not None:
                    raise ReferenceError(f"{origin(op)}: captured claim is not owned by the invoking scope")
                ticket.owner = identity
            if body_args is not None:
                result = yield from self._body(op.regions[0].block, body_args, identity)
            else:
                result = yield from self._protocol(op, args, identity)
            for value in result:
                if isinstance(value, Ticket):
                    if value.owner != identity or value.disposition is not None:
                        raise ReferenceError(f"{origin(op)}: invalid claim result transfer")
                    value.owner = parent
            self.outcomes[identity].values = deepcopy(result)
            self.outcomes[identity].status = "succeeded"
            return result
        except ReferenceError as failed:
            self.outcomes[identity].status = "cancelled" if isinstance(failed, Cancelled) else "failed"
            self.outcomes[identity].error = str(failed)
            raise
        finally:
            self._state(identity, State.CLOSING)
            self._state(identity, State.SETTLING)
            if require_trait(op, RegionSemantics).protocol == Protocol.PARALLEL:
                yield Command("settle_requests", op, identity)
            for stream in self.streams.values():
                for ticket in tuple(stream.outstanding.values()):
                    if ticket.owner == identity:
                        self._dispose(ticket, "cancelled" if self.cancelled else "unconsumed", identity)
            self.outcomes[identity].cleanup = True
            self.outcomes[identity].cancel_requested = self.cancelled
            self.outcomes[identity].stop_requested = self._is_stopped(op, identity)
            for command in self.emissions.pop(identity, ()):
                if self.outcomes[identity].status != "succeeded":
                    self._fact(
                        origin(op),
                        identity,
                        "completion_request_disposed",
                        symbol(command.op.properties["handler"]),
                        self.outcomes[identity].status,
                    )
                elif self.closing:
                    self._emit_request(command, "cancelled_unfulfilled" if self.cancelled else "unfulfilled_on_failure")
                else:
                    try:
                        yield command
                    except Cancelled:
                        self._emit_request(command, "cancelled_unfulfilled" if self.cancelled else "unfulfilled_on_failure")
            self.outcomes[identity].cancel_requested = self.cancelled
            self._state(identity, State.TERMINAL)

    def _protocol(self, op: Operation, args: tuple[object, ...], identity: Identity) -> Generator[Command, object, tuple[object, ...]]:
        trait = op.get_trait(RegionSemantics)
        if trait is None:
            raise ReferenceError("missing invocation protocol")
        protocol = trait.protocol
        if protocol in (Protocol.ONCE, Protocol.SEQUENCE):
            return (yield from self._body(op.regions[0].block, self._entry(op, args), identity))
        if protocol == Protocol.CHOICE:
            if type(args[0]) is not bool:
                raise ReferenceError(f"{origin(op)}: choice predicate is not boolean")
            role = 0 if args[0] else 1
            return (yield from self._body(op.regions[role].block, self._entry(op, args, role=role), identity))
        if protocol == Protocol.PARALLEL:
            if op.results:
                raise ReferenceLimit("parallel result merging requires an explicit supported contract")
            if any(isinstance(value, Ticket) for value in args):
                raise ReferenceLimit("reference parallel claim transfer is not implemented")
            children = tuple(
                self._spawn(self._body(region.block, self._entry(op, args, role=index), (*identity, ("$region", index))))
                for index, region in enumerate(op.regions)
            )
            yield Command("join", op, identity, children)
            errors = [error for index in children if (error := self.tasks[index].error) is not None]
            if self.cancelled:
                raise Cancelled("run cancellation")
            if errors:
                raise next((error for error in errors if not isinstance(error, Cancelled)), errors[0])
            return ()
        if protocol == Protocol.REPEAT:
            if symbol(op.properties["policy"]) != "bounded_count":
                raise ReferenceLimit(f"{origin(op)}: reference policy is not installed")
            count = number(op, "carry_count")
            split = len(args) - count
            captures, carries = args[:split], args[split:]
            for index in range(number(op, "limit")):
                if self._is_stopped(op, identity):
                    break
                carries = yield from self._invoke(op, (), identity, body_args=self._entry(op, (*captures, *carries), (index, *carries)))
            return carries
        if protocol == Protocol.ACCESS:
            cap = checked((yield Command("acquire", op, identity, args)), Capability)
            try:
                return (yield from self._body(op.regions[0].block, self._entry(op, args, (cap,)), identity))
            finally:
                yield Command("release", op, identity, (cap,))
        if protocol == Protocol.PRODUCER:
            stream_name = str(args[0])
            try:
                while not self._is_stopped(op, identity) and not self.closing:
                    ticket = yield Command("reserve", op, identity, (stream_name,))
                    if ticket is None:
                        break
                    ticket = checked(ticket, Ticket)
                    start = len(self.read_versions[identity])
                    try:
                        values = yield from self._invoke(
                            op, (), identity, body_args=self._entry(op, args, (ticket.key,)), selected_product=ticket
                        )
                        ticket.versions = tuple(dict.fromkeys(self.read_versions[identity][start:]))
                        yield Command("publish", op, identity, (ticket, values[0]))
                    except ReferenceError as failed:
                        self._dispose(ticket, "cancelled" if isinstance(failed, Cancelled) else "failed", identity)
                        raise
            finally:
                self.streams[stream_name].closed = True
                yield Command("drain", op, identity, (stream_name,))
            return ()
        if protocol == Protocol.REQUESTS:
            handler = self.handlers[origin(op)]
            while True:
                request = yield Command("next_request", op, identity)
                if request is None:
                    break
                request = checked(request, Request)
                self.request_causes[identity] = request
                try:
                    yield from self._invoke(op, (), identity, body_args=self._entry(op, args, (request,)))
                    handler.dispositions[request.ordinal] = "completed"
                except ReferenceError as failed:
                    handler.dispositions[request.ordinal] = "cancelled" if isinstance(failed, Cancelled) else "failed"
                    raise
                finally:
                    handler.active -= 1
                    self._fact(origin(op), identity, "request_settled", request, handler.dispositions.get(request.ordinal))
            return ()
        raise ReferenceLimit(f"{origin(op)}: unsupported reference protocol {protocol}")

    def _entry(self, op: Operation, args: tuple[object, ...], extra: tuple[object, ...] = (), *, role: int = 0) -> tuple[object, ...]:
        """Interpret the registered capture map, rather than positional convention."""
        contract = require_trait(op, RegionSemantics).describe(op).roles[role]
        missing = object()
        values = [missing] * len(contract.entries)
        for capture in contract.captures:
            if not 0 <= capture.argument < len(values) or not 0 <= capture.operand < len(args) or values[capture.argument] is not missing:
                raise ReferenceError(f"{origin(op)}: malformed capture entry mapping")
            values[capture.argument] = args[capture.operand]
        unbound = [i for i, value in enumerate(values) if value is missing]
        if len(unbound) != len(extra):
            raise ReferenceError(f"{origin(op)}: invocation arguments do not fill the declared entry")
        for index, value in zip(unbound, extra, strict=True):
            values[index] = value
        return tuple(values)

    def _dispose(self, ticket: Ticket, disposition: str, identity: Identity) -> None:
        if ticket.disposition is not None:
            raise ReferenceError(f"{ticket.stream}/{ticket.key}: duplicate item disposition")
        stream = self.streams[ticket.stream]
        ticket.disposition = disposition
        stream.outstanding.pop(ticket.key)
        if ticket in stream.ready:
            stream.ready.remove(ticket)
        stream.terminal[ticket.key] = disposition
        self._fact(ticket.stream, identity, "disposition", ticket.key, disposition)

    def _ready(self, command: Command) -> bool:
        kind, op, args = command.kind, command.op, command.args
        if self.closing and kind not in ("call_complete", "release", "join", "drain", "next_request", "settle_requests"):
            return True
        if kind == "call_complete":
            call = self.calls[checked(args[0], int)]
            return (origin(call.command.op), call.ordinal) not in self.held
        if kind == "acquire":
            # Resolve/copy coherently; a snapshot installs no live read lock.
            resource = self.resources[str(args[0])]
            use = op.get_trait(ResourceSemantics)
            if use is None:
                return True
            contract = use.describe(op)
            mode = contract.mode
            if mode == AccessMode.SNAPSHOT and contract.freshness == Freshness.PUBLISHED and contract.version_selector == "initial":
                return True
            return resource.writer is None and (not resource.readers or mode in (AccessMode.SNAPSHOT, AccessMode.PROTECTED_CURRENT))
        if kind == "take":
            stream = self.streams[str(args[0])]
            return bool(stream.ready) or stream.closed
        if kind == "reserve":
            stream = self.streams[str(args[0])]
            return self._is_stopped(op, command.invocation) or len(stream.outstanding) < number(stream.declaration, "horizon")
        if kind == "publish":
            ticket = checked(args[0], Ticket)
            return self._is_stopped(op, command.invocation) or len(self.streams[ticket.stream].ready) < number(
                self.streams[ticket.stream].declaration, "ready_bound"
            )
        if kind == "drain":
            stream = self.streams[str(args[0])]
            return all(ticket.owner is None for ticket in stream.outstanding.values())
        if kind == "join":
            return all(self.tasks[checked(index, int)].done for index in args)
        if kind == "settle_requests":
            return self.closing or not self._pending_requests(command.invocation)
        if kind == "request":
            target = origin(self.symbols[symbol(op.properties["handler"])])
            handler = self.handlers[target]
            return len(handler.pending) + handler.active < number(self.index[target], "max_outstanding")
        if kind == "next_request":
            handler = self.handlers[origin(op)]
            return (
                self.closing
                or bool(handler.pending)
                or all(
                    self._source_settled(self.symbols[source], op, command.invocation)
                    for source in require_trait(op, RegionSemantics).describe(op).request_sources
                )
            )
        if kind == "stop_after":
            return self._source_settled(self.symbols[symbol(op.properties["source"])], op, command.invocation)
        return True

    def enabled(self) -> tuple[int, ...]:
        return tuple(index for index, task in enumerate(self.tasks) if task.command is not None and self._ready(task.command))

    def _perform(self, command: Command) -> object:
        kind, op, args, identity = command.kind, command.op, command.args, command.invocation
        if self.closing and kind not in ("call_complete", "release", "join", "drain", "next_request", "settle_requests"):
            raise Cancelled("run cancellation closes ordinary admission" if self.cancelled else "sibling failure closes ordinary admission")
        if kind in ("tick", "join"):
            return None
        if kind == "admit":
            return not (self.outcomes[identity].invocation_source == "repeat_policy" and self._is_stopped(op, identity))
        if kind == "acquire":
            resource = self.resources[str(args[0])]
            contract = require_trait(op, ResourceSemantics).describe(op)
            if contract.mode == AccessMode.GRANT:
                raise ReferenceLimit("grant profile reference realization is not installed")
            if (
                contract.mode == AccessMode.SNAPSHOT
                and contract.freshness == Freshness.PUBLISHED
                and contract.version_selector != "initial"
            ):
                raise ReferenceLimit("reference has no published snapshot for that selector")
            token = len(self.capabilities)
            cap = Capability(resource.owner, contract.mode, resource.version, deepcopy(resource.value), token)
            if contract.mode == AccessMode.SNAPSHOT and contract.freshness == Freshness.PUBLISHED:
                cap.version, cap.value = deepcopy(self.initial_snapshots[resource.owner])
            self.capabilities[token] = cap
            if contract.mode == AccessMode.PROTECTED_CURRENT:
                resource.readers.add(token)
            elif contract.mode == AccessMode.OWNER_UPDATE:
                resource.writer = token
            self._fact(origin(op), identity, "protected_version", cap.version, contract.mode, contract.freshness)
            for length in range(1, len(identity) + 1):
                self.read_versions[identity[:length]].append(cap.version)
            return cap
        if kind == "release":
            cap = checked(args[0], Capability)
            resource = self.resources[cap.resource]
            resource.readers.discard(cap.token)
            if resource.writer == cap.token:
                resource.writer = None
            cap.active = False
            self._fact(origin(op), identity, "released", cap.version)
            return None
        if kind == "call_enter":
            key = symbol(op.properties["binding"])
            binding = self.bindings.get(key)
            if binding is None or binding.version != text(op, "binding_version"):
                raise ReferenceLimit(f"{origin(op)}: missing reference binding/version @{key}")
            inputs = tuple(schema(value) for value in op.operands)
            outputs = tuple(schema(value) for value in op.results)
            if (
                binding.inputs != inputs
                or binding.outputs != outputs
                or not all(valid(value, name, self.schemas) for value, name in zip(args, inputs, strict=True))
            ):
                raise ReferenceError(f"{origin(op)}: stand-in signature/input mismatch")
            ordinal = self.call_counts[origin(op)]
            self.call_counts[origin(op)] += 1
            call_id = len(self.calls)
            self.calls[call_id] = PendingCall(command, binding, ordinal, deepcopy(args))
            self._fact(
                origin(op),
                identity,
                "physical_entered",
                ordinal,
                self.request_causes.get(identity),
                self.input_uses.get(identity, ()),
                self.product_uses.get(identity, ()),
            )
            return call_id
        if kind == "call_complete":
            pending = self.calls[checked(args[0], int)]
            original = pending.command
            failure = self.failures.get((origin(op), pending.ordinal))
            if failure is not None:
                self._fact(
                    origin(op),
                    identity,
                    "physical_failed",
                    pending.ordinal,
                    failure,
                    self.request_causes.get(identity),
                    self.input_uses.get(identity, ()),
                    self.product_uses.get(identity, ()),
                )
                raise ReferenceError(f"{origin(op)}: {failure}")
            try:
                result = pending.binding.invoke(pending.snapshots)
            except Exception as failed:
                self._fact(
                    origin(op),
                    identity,
                    "physical_failed",
                    pending.ordinal,
                    str(failed),
                    self.request_causes.get(identity),
                    self.input_uses.get(identity, ()),
                    self.product_uses.get(identity, ()),
                )
                raise ReferenceError(f"{origin(op)}: stand-in raised {failed}") from failed
            if not isinstance(result, CallResult):
                raise ReferenceError(f"{origin(op)}: stand-in returned an invalid completion envelope")
            for update in result.updates:
                if update.capability not in pending.binding.updates:
                    raise ReferenceError(f"{origin(op)}: undeclared stand-in update")
                cap = original.args[update.capability]
                if not isinstance(cap, Capability) or not cap.active or cap.mode != AccessMode.OWNER_UPDATE:
                    raise ReferenceError(f"{origin(op)}: update lacks owner-issued authority")
                resource = self.resources[cap.resource]
                if resource.version != cap.version or resource.writer != cap.token:
                    raise ReferenceError(f"{origin(op)}: stale owner update")
                resource.value = deepcopy(update.value)
                resource.ordinal += 1
                self._fact(cap.resource, identity, "updated", resource.version, resource.value)
            self._fact(
                origin(op),
                identity,
                "physical_completed",
                pending.ordinal,
                result.values,
                self.request_causes.get(identity),
                self.input_uses.get(identity, ()),
                self.product_uses.get(identity, ()),
            )
            if len(result.values) != len(pending.binding.outputs) or not all(
                valid(value, name, self.schemas) for value, name in zip(result.values, pending.binding.outputs, strict=True)
            ):
                self._fact(
                    origin(op), identity, "validation_rejected", self.input_uses.get(identity, ()), self.product_uses.get(identity, ())
                )
                raise ReferenceError(f"{origin(op)}: stand-in output contract mismatch")
            self._fact(origin(op), identity, "validated", self.input_uses.get(identity, ()), self.product_uses.get(identity, ()))
            return result.values
        if kind == "reserve":
            stream = self.streams[str(args[0])]
            if self._is_stopped(op, identity):
                return None
            ticket = Ticket(str(args[0]), stream.next_key, identity)
            stream.next_key += 1
            stream.outstanding[ticket.key] = ticket
            stream.high_water = max(stream.high_water, len(stream.outstanding))
            self._fact(ticket.stream, identity, "selected", ticket.key)
            return ticket
        if kind == "publish":
            ticket, value = checked(args[0], Ticket), args[1]
            ticket.payload = deepcopy(value)
            if self._is_stopped(op, identity):
                self._dispose(ticket, "discarded", identity)
            else:
                self.streams[ticket.stream].ready.append(ticket)
                self._fact(ticket.stream, identity, "published", ticket.key, ticket.versions)
            return None
        if kind == "take":
            stream = self.streams[str(args[0])]
            if not stream.ready:
                raise ReferenceError(f"{origin(op)}: input exhausted before required admission")
            contract = require_trait(stream.declaration, FlowSemantics).describe(stream.declaration)
            ticket = stream.ready.popleft()
            disposition = self._admission(contract, ticket)
            comparison = tuple(self.resources[owner].version for owner in contract.versions.owners)
            self._fact(
                origin(op),
                identity,
                "admission_checked",
                ticket.key,
                ticket.versions,
                ticket.stream,
                comparison,
                disposition,
                contract.versions.boundary,
            )
            if disposition != "accepted":
                self._fact(origin(op), identity, "item_rejected", ticket.key, ticket.versions, ticket.stream, disposition)
                self._dispose(ticket, disposition, identity)
                return None
            ticket.owner = identity
            self.input_uses[identity] = (*self.input_uses.get(identity, ()), (ticket.stream, ticket.key))
            self._fact(origin(op), identity, "item_admitted", ticket.key, ticket.versions, ticket.stream)
            return ticket
        if kind == "retire":
            ticket = args[0]
            if not isinstance(ticket, Ticket) or ticket.owner != identity:
                raise ReferenceError(f"{origin(op)}: claim is not owned by invocation")
            self._dispose(ticket, text(op, "disposition"), identity)
            return None
        if kind == "arm_request":
            if text(op, "boundary") != "settled_success":
                raise ReferenceLimit("reference supports settled-success completion requests")
            source_name = symbol(op.properties["source"])
            target = self.symbols[symbol(op.properties["handler"])]
            if source_name not in require_trait(target, RegionSemantics).describe(target).request_sources:
                raise ReferenceError(f"{origin(op)}: undeclared request source")
            source = origin(self.symbols[source_name])
            candidates = [
                key
                for key, value in self.outcomes.items()
                if value.owner == source and value.state == State.RUNNING and identity[: len(key)] == key
            ]
            if not candidates:
                raise ReferenceLimit(f"{origin(op)}: completion source must be an enclosing invocation")
            emitter = max(candidates, key=len)
            self.emissions[emitter].append(Command("request", op, emitter, args))
            return None
        if kind == "request":
            target = origin(self.symbols[symbol(op.properties["handler"])])
            self._emit_request(command)
            return None
        if kind == "next_request":
            handler = self.handlers[origin(op)]
            if self.closing:
                while handler.pending:
                    request = handler.pending.popleft()
                    handler.dispositions[request.ordinal] = "cancelled_unfulfilled" if self.cancelled else "unfulfilled_on_failure"
                    self._fact(origin(op), identity, "request_disposed", request, handler.dispositions[request.ordinal])
                return None
            if not handler.pending:
                return None
            handler.active += 1
            return handler.pending.popleft()
        if kind == "stop_after":
            target = origin(self.symbols[symbol(op.properties["target"])])
            self.stopped.add(target)
            scope, _ = self._common_scope(op, self.symbols[symbol(op.properties["target"])], identity)
            self._scoped_stops.add((scope, target))
            source = self.symbols[symbol(op.properties["source"])]
            owners = self._source_outcomes(source, op, identity)
            self._fact(target, identity, "stop_requested", origin(source), tuple(outcome.invocation for outcome in owners))
            return None
        if kind == "settle_requests":
            for request in self._pending_requests(identity):
                handler = self.handlers[request.handler]
                handler.pending.remove(request)
                disposition = "cancelled_unfulfilled" if self.cancelled else "unfulfilled_on_failure"
                handler.dispositions[request.ordinal] = disposition
                self._fact(request.handler, identity, "request_disposed", request, disposition)
            return None
        if kind == "drain":
            stream = self.streams[str(args[0])]
            for ticket in tuple(stream.outstanding.values()):
                self._dispose(ticket, "cancelled" if self.cancelled else "discarded", identity)
            return None
        raise ReferenceError(f"unknown reference command {kind}")

    def _admission(self, contract: FlowContract, ticket: Ticket) -> str:
        """Independent oracle implementation; does not import the verifier."""
        policy = contract.versions
        if policy.boundary != "item_admission" or policy.metric != "owner_update_ordinal_within_lineage":
            raise ReferenceLimit("reference admission boundary/metric is not installed")
        actual: dict[str, Version] = {}
        for observed in ticket.versions:
            if observed.owner in actual and actual[observed.owner] != observed:
                return "incomparable_provenance"
            actual[observed.owner] = observed
        if set(actual) != set(policy.owners):
            return "missing_or_extra_provenance"
        if policy.admission == Admission.ANY_WITH_PROVENANCE:
            return "accepted"
        allowed = policy.max_lag if policy.admission == Admission.BOUNDED_LAG else 0
        for owner, observed in actual.items():
            now = self.resources[owner].version
            if (observed.incarnation, observed.lineage) != (now.incarnation, now.lineage) or observed.ordinal > now.ordinal:
                return "incomparable_provenance"
            if now.ordinal - observed.ordinal > allowed:
                return "rejected_stale"
        return "accepted"

    def _is_stopped(self, op: Operation, identity: Identity) -> bool:
        return origin(op) in self._external_stops or any(
            target == origin(op) and identity[: len(scope)] == scope for scope, target in self._scoped_stops
        )

    def _common_scope(self, waiting: Operation, source: Operation, identity: Identity) -> tuple[Identity, Operation]:
        parents = []
        current: Operation | None = source
        while current is not None:
            parents.append(current)
            current = current.parent_op()
        current = waiting
        while current is not None and current not in parents:
            current = current.parent_op()
        if current is None:
            raise ReferenceError("reference relationship crosses programs")
        indices = [index for index, (owner, _) in enumerate(identity) if owner == origin(current)]
        if not indices:
            raise ReferenceError("reference relationship lacks an enclosing invocation")
        return identity[: max(indices) + 1], current

    def _source_outcomes(self, source: Operation, waiting: Operation, identity: Identity) -> list[Outcome]:
        scope, common = self._common_scope(waiting, source, identity)
        if common is source:
            raise ReferenceError(f"{origin(waiting)}: circular self-referential source settlement")
        ancestors = []
        current = source.parent_op()
        while current is not None and current is not common:
            ancestors.append(current)
            current = current.parent_op()
        barrier = source
        for ancestor in reversed(ancestors):
            trait = ancestor.get_trait(RegionSemantics)
            if trait is not None and trait.protocol in (Protocol.REPEAT, Protocol.REQUESTS, Protocol.CHOICE, Protocol.PRODUCER):
                barrier = ancestor
                break
        owners = [
            outcome
            for outcome in self.outcomes.values()
            if outcome.owner == origin(barrier)
            and outcome.invocation_source == "parent_entry"
            and outcome.invocation[: len(scope)] == scope
        ]
        return owners

    def _source_settled(self, source: Operation, waiting: Operation, identity: Identity) -> bool:
        owners = self._source_outcomes(source, waiting, identity)
        return bool(owners) and all(outcome.state == State.TERMINAL for outcome in owners)

    def _pending_requests(self, identity: Identity) -> tuple[Request, ...]:
        return tuple(
            request for handler in self.handlers.values() for request in handler.pending if request.cause[: len(identity)] == identity
        )

    def _emit_request(self, command: Command, disposition: str | None = None) -> None:
        target = origin(self.symbols[symbol(command.op.properties["handler"])])
        handler = self.handlers[target]
        request = Request(target, handler.next_ordinal, command.invocation, deepcopy(command.args[0]))
        handler.next_ordinal += 1
        if disposition is None:
            handler.pending.append(request)
            self._fact(target, command.invocation, "request_pending", request)
        else:
            handler.dispositions[request.ordinal] = disposition
            self._fact(target, command.invocation, "request_disposed", request, disposition)

    def step(self, choice: int | None = None) -> bool:
        enabled = self.enabled()
        if not enabled:
            return False
        if choice is None:
            candidates = [index for index in enabled if index >= self.cursor]
            choice = candidates[0] if candidates else enabled[0]
        if choice not in enabled:
            raise ReferenceError("external scheduler selected a non-enabled command")
        task = self.tasks[choice]
        command = task.command
        if command is None:
            raise ReferenceError("selected a terminal task")
        self.cursor = (choice + 1) % len(self.tasks)
        try:
            result = self._perform(command)
        except ReferenceError as failed:
            self._note_failure(failed)
            self._resume(task, error=failed)
        else:
            self._resume(task, result)
        return True

    def run(self, *, max_steps: int = 10000) -> None:
        for _ in range(max_steps):
            if not self.step():
                return
        raise ReferenceLimit("external exploration step bound exhausted")

    @property
    def settled(self) -> bool:
        return all(task.done for task in self.tasks)

    def request_stop(self, name: str) -> None:
        op = self.symbols.get(name)
        if op is None:
            raise ReferenceError(f"unknown stop symbol @{name}")
        if require_trait(op, RegionSemantics).protocol not in (Protocol.REPEAT, Protocol.PRODUCER):
            raise ReferenceLimit("this reference stop input targets a repeat or producer")
        self.stopped.add(origin(op))
        self._external_stops.add(origin(op))
