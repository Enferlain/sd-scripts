"""Derived fragments on the existing CPU owner engine, not another scheduler."""

from __future__ import annotations

import asyncio
import importlib
import json
import threading

from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from uuid import uuid4

import torch

from xdsl.ir import Block, Operation
from xdsl.traits import OpTrait

from .contracts import (
    AccessMode,
    FlowSemantics,
    Freshness,
    number,
    origin,
    Protocol,
    RegionSemantics,
    ResourceSemantics,
    symbol,
    text,
    Version,
)
from .dialects import BorrowOp, CallOp, ConstantOp, ProgramOp, RequestOp, ResourceOp, RetireOp, StopAfterOp, StreamOp, TakeOp, YieldOp
from .normalize import source_index, symbol_index
from .prepare import BindingSpec, Manifest, Prepared, ResourceSpec, _common, load, schema
from .reference import Fact, Identity, Outcome, Request, State, Ticket
from .verify import Analysis, Comparison, TargetLimit, evaluate_admission


_prefix = "docs_design.models-strategy-trainer.execution_candidate"
graph = importlib.import_module(_prefix + ".graph")
backend = importlib.import_module(_prefix + ".prepare")
owners = importlib.import_module(_prefix + ".state")
physical = importlib.import_module(_prefix + ".workers")


@dataclass
class Cell:
    value: object
    ordinal: int = 0


@dataclass
class Capability:
    resource: str
    mode: AccessMode
    version: Version
    value: object
    active: bool = True


@dataclass(frozen=True)
class Effect:
    capability: int
    action: str
    value: object


@dataclass(frozen=True)
class Result:
    values: tuple[object, ...]
    effects: tuple[Effect, ...] = ()


@dataclass(frozen=True)
class Binding:
    spec: BindingSpec
    invoke: Callable[[tuple[object, ...]], Result]


@dataclass(frozen=True)
class ResourceBinding:
    spec: ResourceSpec
    initialize: Callable[[object], object]
    validate: Callable[[object], None]


@dataclass(frozen=True)
class Installed:
    bindings: tuple[Binding, ...]
    resources: tuple[ResourceBinding, ...]
    schemas: Mapping[str, Callable[[object], bool]]

    @property
    def manifest(self) -> Manifest:
        return Manifest(
            tuple(item.spec for item in self.bindings), tuple(item.spec for item in self.resources), tuple(sorted(self.schemas))
        )


@dataclass
class Controls:
    """External choices only; neither oracle nor target reads the other's trace."""

    gates: dict[tuple[str, int], threading.Event] = field(default_factory=dict)
    entered: dict[tuple[str, int], threading.Event] = field(default_factory=dict)
    failures: dict[tuple[str, int], str] = field(default_factory=dict)
    before: dict[tuple[str, int], asyncio.Event] = field(default_factory=dict)
    reached: dict[tuple[str, int], asyncio.Event] = field(default_factory=dict)
    checkpoint: Callable[[Point], Awaitable[None]] | None = None


@dataclass(frozen=True)
class Point:
    source: str
    invocation: Identity
    phase: str
    ordinal: int = 0


@dataclass(frozen=True)
class SourceReceipt:
    source: str
    invocation: Identity
    status: str
    cancel_requested: bool


@dataclass
class Flow:
    declaration: StreamOp
    capacity: asyncio.BoundedSemaphore
    active: dict[int, Ticket] = field(default_factory=dict)
    terminal: dict[int, str] = field(default_factory=dict)
    next_key: int = 0
    high_water: int = 0
    changed: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class Handler:
    operation: Operation
    scope: Identity
    name: str
    capacity: asyncio.BoundedSemaphore = field(default_factory=lambda: asyncio.BoundedSemaphore(1))
    pending: dict[int, Request] = field(default_factory=dict)
    issued: dict[int, Request] = field(default_factory=dict)
    dispositions: dict[int, str] = field(default_factory=dict)
    next_ordinal: int = 0
    closed: set[str] = field(default_factory=set)
    current: Request | None = None


class ExecutionError(RuntimeError):
    pass


def trait[T: OpTrait](op: Operation, kind: type[T]) -> T:
    result = op.get_trait(kind)
    if result is None:
        raise ExecutionError(f"{origin(op)}: missing installed {kind.__name__}")
    return result


class Runtime:
    def __init__(
        self,
        prepared: Prepared,
        installed: Installed,
        initial: Mapping[str, object],
        *,
        controls: Controls | None = None,
        history: bool = True,
    ):
        # This target chooses early rejection; pure construction ordering is not
        # part of the portable correspondence contract.
        self.plan = load(prepared.artifact, installed.manifest, prepared.policy)
        self.module = self.plan.verification.normalized.module
        self.index, self.symbols = source_index(self.module), symbol_index(self.module)
        self.root = next(iter(self.module.body.block.ops))
        self.installed, self.controls, self.history = installed, controls or Controls(), history
        self.bindings = {item.spec.name: item for item in installed.bindings}
        self.resource_bindings = {item.spec.kind: item for item in installed.resources}
        self.declarations = {text(op, "sym_name"): op for op in self.index.values() if isinstance(op, ResourceOp)}
        if set(initial) != set(self.declarations):
            raise ExecutionError("initial resource set does not match root declarations")
        self.initial = deepcopy(dict(initial))
        for operation in self.index.values():
            if isinstance(operation, ConstantOp) and not self._valid(json.loads(text(operation, "literal")), schema(operation.results[0])):
                raise ExecutionError(f"{origin(operation)}: constant value disagrees with its declared schema")
        self.resource_names = {name: f"resource/{i}" for i, name in enumerate(self.declarations)}
        self.flow_names = {text(op, "sym_name"): f"flow/{i}" for i, op in enumerate(self.index.values()) if isinstance(op, StreamOp)}
        self.facts: list[Fact] = []
        self.outcomes: dict[Identity, Outcome] = {}
        self._live: dict[Identity, Outcome] = {}
        self.ordinals: dict[tuple[Identity, str], int] = defaultdict(int)
        self.call_counts: dict[str, int] = defaultdict(int)
        self.entry_counts: dict[str, int] = defaultdict(int)
        self.inputs: dict[Identity, tuple[tuple[str, int], ...]] = {}
        self.products: dict[Identity, tuple[tuple[str, int], ...]] = {}
        self.causes: dict[Identity, Request] = {}
        self.read_versions: dict[Identity, list[Version]] = defaultdict(list)
        self.emissions: dict[Identity, list[tuple[Operation, tuple[object, ...]]]] = defaultdict(list)
        self.handlers: dict[tuple[Identity, str], Handler] = {}
        self.requests_changed = asyncio.Event()
        self.events: dict[tuple[Identity, str], asyncio.Event] = defaultdict(asyncio.Event)
        self.closed_sources: dict[tuple[Identity, str], tuple[SourceReceipt, ...]] = {}
        self.production_stops: dict[Identity, asyncio.Event] = {}
        self.stopped: set[str] = set()
        self.cancelled = False
        self.failed = False
        self.failure: BaseException | None = None
        self.started = False
        self.workers = physical.Workers(prepared.policy.workers)
        self.analysis = Analysis(self.plan.verification.normalized)
        self.published_initial = {
            text(subject, "sym_name")
            for op, access in self.analysis.accesses.items()
            if access.mode == AccessMode.SNAPSHOT and access.freshness == Freshness.PUBLISHED
            for subject in self.analysis.value_sources(access.subject)
            if isinstance(subject, ResourceOp)
        }
        mutable = {
            text(subject, "sym_name")
            for op, access in self.analysis.accesses.items()
            if access.mode == AccessMode.OWNER_UPDATE
            for subject in self.analysis.value_sources(access.subject)
            if isinstance(subject, ResourceOp)
        }
        participants, states, units, channels = [], [graph.OwnedState("policies", lambda: None)], [], []
        for name, op in self.declarations.items():
            binding = self.resource_bindings[op.value.type.kind.data]
            internal = self.resource_names[name]
            if binding.spec.storage == "participant":
                participants.append(
                    graph.Participant(internal, lambda name=name, binding=binding: binding.initialize(self.initial[name]), binding.validate)
                )
                if name in mutable:
                    units.append(graph.Unit(internal, ((internal, binding.spec.parameter),), binding.spec.learning_rate))
            else:

                def initialize(name=name, binding=binding):
                    value = binding.initialize(self.initial[name])
                    binding.validate(value)
                    return Cell(value)

                states.append(graph.OwnedState(internal, initialize))
        for op in self.index.values():
            if isinstance(op, StreamOp):
                flow = trait(op, FlowSemantics).describe(op)
                name = text(op, "sym_name")
                channels.append(graph.Channel(name, flow.ready_bound))
                states.append(
                    graph.OwnedState(
                        self.flow_names[name], lambda op=op, flow=flow: Flow(op, asyncio.BoundedSemaphore(flow.max_outstanding))
                    )
                )

        async def entry(_context):
            with self.workers.lifetime():
                await self._invoke(self.root, (), ())

        filing = graph.Run(
            graph.Block("language-entry", (graph.Call("normalized-fragments", entry),)),
            tuple(participants),
            tuple(units),
            tuple(states),
            tuple(channels),
        )
        # Shared IR verification is the semantic authority. The old graph
        # Contract accepts a different language, so do not submit a fabricated
        # producer tree to it merely to allocate these owner services.
        authority = str(uuid4())
        accepted = graph.Accepted(
            authority,
            graph.Contract(),
            filing.root,
            MappingProxyType({item.name: item for item in participants}),
            MappingProxyType({item.name: item for item in units}),
            MappingProxyType({item.name: item for item in states}),
            MappingProxyType({item.name: item for item in channels}),
            MappingProxyType({item.name: authority + "/" + item.name for item in participants}),
            MappingProxyType({item.name: authority + "/" + item.name for item in units}),
            MappingProxyType({}),
        )
        attempt = backend.prepare(accepted)
        self.state = attempt.state
        self.ready = attempt.publish()
        self.engine = backend.Engine(self.ready)
        self.owner = backend.Lowering(accepted, self.state)
        self.flows = {name: self.state.owners[internal] for name, internal in self.flow_names.items()}
        self.initial_snapshots: dict[str, tuple[Version, object]] = {}
        self.compiled = {op: self._compile(op) for op in self.index.values() if not isinstance(op, YieldOp | ResourceOp | StreamOp)}

    async def run(self) -> None:
        if self.started:
            raise ExecutionError("owner runtime instances are single-use")
        self.started = True
        try:
            # Published initial versions are independent of all later writes.
            for name in self.declarations:
                if name not in self.published_initial:
                    continue
                async with self._access(name, AccessMode.SNAPSHOT, Freshness.CURRENT_AT_START) as cap:
                    self.initial_snapshots[name] = cap.version, cap.value
            await self.engine.run()
        except BaseException as failed:
            self.failure = failed
            raise

    def request_stop(self, symbol_name: str) -> None:
        if symbol_name not in self.symbols:
            raise ExecutionError(f"unknown stop symbol {symbol_name}")
        op = self.symbols[symbol_name]
        region = op.get_trait(RegionSemantics)
        if region is None or region.protocol not in (Protocol.REPEAT, Protocol.PRODUCER):
            raise TargetLimit(self.plan.policy.target, origin(op), "external stop requires a repeat or producer")
        self.stopped.add(symbol_name)
        if region.protocol == Protocol.PRODUCER:
            for scope, event in self.production_stops.items():
                if scope[-1][0] == origin(op):
                    event.set()

    def _is_stopped(self, op: Operation, identity: Identity) -> bool:
        return ("sym_name" in op.properties and text(op, "sym_name") in self.stopped) or any(
            origin(op) == scope[-1][0] and identity[: len(scope)] == scope and event.is_set()
            for scope, event in self.production_stops.items()
        )

    def cancel(self, task: asyncio.Task) -> None:
        self.cancelled = True
        task.cancel()

    def _fact(self, owner: str, identity: Identity, kind: str, *values: object) -> None:
        if self.history:
            self.facts.append(Fact(owner, identity, kind, deepcopy(values)))

    def _identity(self, parent: Identity, owner: str) -> Identity:
        ordinal = self.ordinals[parent, owner]
        self.ordinals[parent, owner] += 1
        return (*parent, (owner, ordinal))

    def _state(self, identity: Identity, state: State) -> None:
        outcome = self._live[identity]
        outcome.state = state
        self._fact(outcome.owner, identity, "state", state)

    async def _checkpoint(self, op: Operation, identity: Identity, phase: str, ordinal: int = 0) -> None:
        if self.controls.checkpoint is not None:
            await self.controls.checkpoint(Point(origin(op), identity, phase, ordinal))

    def _scope(self, waiting: Operation, source: Operation, identity: Identity) -> Identity:
        common = origin(_common(waiting, source))
        indices = [i for i, (name, _) in enumerate(identity) if name == common]
        if not indices:
            raise ExecutionError("completion relationship has no enclosing invocation")
        return identity[: max(indices) + 1]

    def _event(self, waiting: Operation, source: Operation, identity: Identity) -> asyncio.Event:
        return self.events[self._scope(waiting, source, identity), text(source, "sym_name")]

    def _handler(self, request: Operation, identity: Identity) -> Handler:
        target = self.symbols[symbol(request.properties["handler"])]
        return self._slot(target, self._handler_scope(target, identity))

    def _handler_scope(self, op: Operation, identity: Identity) -> Identity:
        common = op
        for source in trait(op, RegionSemantics).describe(op).request_sources:
            common = _common(common, self.symbols[source])
        indices = [i for i, (name, _) in enumerate(identity) if name == origin(common)]
        if not indices:
            raise ExecutionError("handler lifetime has no enclosing invocation")
        return identity[: max(indices) + 1]

    def _slot(self, op: Operation, scope: Identity) -> Handler:
        key = scope, text(op, "sym_name")
        if key not in self.handlers:
            name = "handler/" + repr(key)
            slot = Handler(op, scope, name)
            self.handlers[key] = slot
            self.owner.request_queues[name] = asyncio.Queue(maxsize=2)
        return self.handlers[key]

    def _entry(self, op: Operation, args: tuple[object, ...], extras: tuple[object, ...] = (), role: int = 0) -> tuple[object, ...]:
        contract = trait(op, RegionSemantics).describe(op)
        entry = contract.roles[role]
        result: list[object] = [None] * len(entry.entries)
        captured = set()
        for link in entry.captures:
            result[link.argument] = args[link.operand]
            captured.add(link.argument)
        free = [i for i in range(len(result)) if i not in captured]
        if len(free) != len(extras):
            raise ExecutionError(f"{origin(op)}: prepared entry arity mismatch")
        for position, value in zip(free, extras, strict=True):
            result[position] = value
        return tuple(result)

    async def _body(self, block: Block, values: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        env = dict(zip(block.args, values, strict=True))
        returned: tuple[object, ...] = ()
        try:
            for op in block.ops:
                if isinstance(op, YieldOp):
                    returned = tuple(env[value] for value in op.operands)
                    return returned
                if isinstance(op, ResourceOp | StreamOp):
                    env[op.results[0]] = text(op, "sym_name")
                    continue
                args = tuple(env[value] for value in op.operands)
                result = (
                    await self._invoke(op, args, identity) if op.get_trait(RegionSemantics) else await self.compiled[op](args, identity)
                )
                if len(result) != len(op.results):
                    raise ExecutionError(f"{origin(op)}: target result arity mismatch")
                env.update(zip(op.results, result, strict=True))
            return ()
        finally:
            for value in env.values():
                if isinstance(value, Ticket) and value.owner == identity and all(value is not output for output in returned):
                    self._dispose(value, "cancelled" if self.cancelled else "unconsumed", identity)

    async def _invoke(
        self,
        op: Operation,
        args: tuple[object, ...],
        parent: Identity,
        *,
        body_args: tuple[object, ...] | None = None,
        product: Ticket | None = None,
    ) -> tuple[object, ...]:
        identity = self._identity(parent, origin(op))
        outcome = Outcome(origin(op), identity)
        self.outcomes[identity] = outcome
        self._live[identity] = outcome
        protocol = Protocol.ONCE if isinstance(op, ProgramOp) else trait(op, RegionSemantics).protocol
        if body_args is not None:
            outcome.invocation_source = {
                Protocol.REPEAT: "repeat_policy",
                Protocol.PRODUCER: "production_selection",
                Protocol.REQUESTS: "accepted_request",
            }[protocol]
        self.inputs[identity] = self.inputs.get(parent, ())
        self.products[identity] = ((product.stream, product.key),) if product else self.products.get(parent, ())
        if parent in self.causes:
            self.causes[identity] = self.causes[parent]
        for phase in (State.CREATED, State.PENDING):
            self._state(identity, phase)
        incoming = body_args if body_args is not None else args
        succeeded = False
        try:
            await self._checkpoint(op, identity, "admit")
            if self.cancelled or self.failed:
                raise asyncio.CancelledError("run closing prevents new invocation admission")
            if body_args is not None and protocol == Protocol.REPEAT and self._is_stopped(op, identity):
                outcome.status = "stopped_unadmitted"
                count = number(op, "carry_count")
                return body_args[-count:] if count else ()
            for phase in (State.ADMITTED, State.RUNNING):
                self._state(identity, phase)
            tickets = [value for value in incoming if isinstance(value, Ticket)]
            if len({id(ticket) for ticket in tickets}) != len(tickets):
                raise ExecutionError("duplicated claim capture")
            if any(ticket.owner != parent or ticket.disposition is not None for ticket in tickets):
                raise ExecutionError("claim transfer lacks invoking ownership")
            for ticket in tickets:
                ticket.owner = identity
            entry_key = origin(op), self.entry_counts[origin(op)]
            self.entry_counts[origin(op)] += 1
            self.controls.reached.setdefault(entry_key, asyncio.Event()).set()
            if entry_key in self.controls.before:
                await self.controls.before[entry_key].wait()
            await asyncio.sleep(0)
            result = (
                await self._body(op.regions[0].block, body_args, identity)
                if body_args is not None
                else await self.compiled[op](args, identity)
            )
            for value in result:
                if isinstance(value, Ticket):
                    value.owner = parent
            outcome.values, outcome.status = deepcopy(result), "succeeded"
            succeeded = True
            return result
        except BaseException as failed:
            outcome.status = "cancelled" if isinstance(failed, asyncio.CancelledError) else "failed"
            outcome.error = str(failed)
            if not isinstance(failed, asyncio.CancelledError):
                self.failed = True
            raise
        finally:
            self._state(identity, State.CLOSING)
            self._state(identity, State.SETTLING)
            try:
                if body_args is None and (protocol == Protocol.PARALLEL or isinstance(op, ProgramOp)):
                    await self._settle_pending(identity)
                for request, payload in self.emissions.pop(identity, ()):
                    if succeeded:
                        await self._emit(request, payload, identity)
                    else:
                        self._fact(
                            origin(op), identity, "completion_request_disposed", symbol(request.properties["handler"]), outcome.status
                        )
                if body_args is None:
                    await self._close_source(op, identity, outcome)
            finally:
                outcome.cleanup, outcome.cancel_requested = True, self.cancelled
                outcome.stop_requested = self._is_stopped(op, identity)
                self._state(identity, State.TERMINAL)
                self._fact(
                    origin(op),
                    identity,
                    "outcome",
                    outcome.status,
                    outcome.values,
                    outcome.error,
                    True,
                    outcome.invocation_source,
                    outcome.cancel_requested,
                    outcome.stop_requested,
                )
                del self._live[identity]

    def _compile(self, op: Operation):
        if isinstance(op, ProgramOp):

            async def program(args, identity):
                return await self._body(op.regions[0].block, (), identity)

            return program
        trait = op.get_trait(RegionSemantics)
        if trait is not None:

            async def region(args, identity):
                return await self._protocol(op, trait.protocol, args, identity)

            return region

        async def leaf(args, identity):
            if isinstance(op, ConstantOp):
                return (json.loads(text(op, "literal")),)
            if isinstance(op, CallOp):
                return await self._call(op, args, identity)
            if isinstance(op, TakeOp):
                return (await self._take(op, str(args[0]), identity),)
            if isinstance(op, BorrowOp):
                ticket = args[0]
                if not isinstance(ticket, Ticket) or ticket.owner != identity or ticket.disposition is not None:
                    raise ExecutionError("borrow lacks an owned item claim")
                return (deepcopy(ticket.payload),)
            if isinstance(op, RetireOp):
                self._dispose(args[0], text(op, "disposition"), identity)
                return ()
            if isinstance(op, RequestOp):
                source = origin(self.symbols[symbol(op.properties["source"])])
                candidates = [identity[: i + 1] for i, (name, _) in enumerate(identity) if name == source]
                self.emissions[max(candidates, key=len)].append((op, args))
                return ()
            if isinstance(op, StopAfterOp):
                source = self.symbols[symbol(op.properties["source"])]
                target = self.symbols[symbol(op.properties["target"])]
                await self._event(op, source, identity).wait()
                await self._checkpoint(op, identity, "stop_after")
                if self.cancelled or self.failed:
                    raise asyncio.CancelledError("run closing supersedes completion-triggered stop")
                source_key = self._scope(op, source, identity), text(source, "sym_name")
                self._fact(
                    origin(target),
                    identity,
                    "stop_requested",
                    origin(source),
                    tuple(receipt.invocation for receipt in self.closed_sources[source_key]),
                )
                self.events[self._scope(op, target, identity), "stop/" + text(target, "sym_name")].set()
                return ()
            raise ExecutionError(f"unprepared leaf {op.name}")

        return leaf

    async def _protocol(self, op: Operation, protocol: Protocol, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        if protocol in (Protocol.ONCE, Protocol.SEQUENCE):
            return await self._body(op.regions[0].block, self._entry(op, args), identity)
        if protocol == Protocol.CHOICE:
            if type(args[0]) is not bool:
                raise ExecutionError("choice predicate is not boolean")
            role = 0 if args[0] else 1
            return await self._body(op.regions[role].block, self._entry(op, args, role=role), identity)
        if protocol == Protocol.PARALLEL:
            children = []
            for role, region in enumerate(op.regions):

                async def branch(_context, role=role, region=region):
                    await self._body(region.block, self._entry(op, args, role=role), (*identity, ("$region", role)))

                name = repr((identity, role))
                children.append(graph.Block(name, (graph.Call(name + "/body", branch),)))
            await self.owner.scope(graph.Together(repr(identity), tuple(children)))()
            return ()
        if protocol == Protocol.REPEAT:
            count = number(op, "carry_count")
            captures, carries = args[: len(args) - count], args[len(args) - count :]
            name = repr(identity)

            async def body(context):
                nonlocal carries
                index = context.choice.payload
                carries = await self._invoke(op, (), identity, body_args=self._entry(op, (*captures, *carries), (index, *carries)))

            block = graph.Block(name + "/iteration", (graph.Call(name + "/body", body),))

            def policy(counts, _state):
                index = sum(counts.values())
                return (
                    graph.Choice(block.name, index, index)
                    if index < number(op, "limit") and text(op, "sym_name") not in self.stopped and not self.cancelled and not self.failed
                    else None
                )

            await self.owner.scope(graph.Repeat(name, (block,), policy, "policies"))()
            return carries
        if protocol == Protocol.ACCESS:
            contract = trait(op, ResourceSemantics).describe(op)
            await self._checkpoint(op, identity, "acquire")
            async with self._access(str(args[0]), contract.mode, contract.freshness, contract.version_selector) as cap:
                self._fact(origin(op), identity, "protected_version", cap.version, contract.mode, contract.freshness)
                for length in range(1, len(identity) + 1):
                    self.read_versions[identity[:length]].append(cap.version)
                try:
                    return await self._body(op.regions[0].block, self._entry(op, args, (cap,)), identity)
                finally:
                    self._fact(origin(op), identity, "released", cap.version)
        if protocol == Protocol.PRODUCER:
            stream = str(args[0])
            flow = self.flows[stream]
            name = repr(identity)
            stop = next(
                control
                for control in self.index.values()
                if isinstance(control, StopAfterOp) and symbol(control.properties["target"]) == text(op, "sym_name")
            )
            stop_name = "stop/" + name
            stop_event = self.events[self._scope(stop, op, identity), "stop/" + text(op, "sym_name")]
            if text(op, "sym_name") in self.stopped:
                stop_event.set()
            self.production_stops[identity] = stop_event
            self.owner.finished[stop_name] = stop_event

            async def produce(context):
                while not stop_event.is_set() and text(op, "sym_name") not in self.stopped:
                    await flow.capacity.acquire()
                    if stop_event.is_set() or text(op, "sym_name") in self.stopped:
                        flow.capacity.release()
                        break
                    try:
                        await self._checkpoint(op, identity, "reserve", flow.next_key)
                    except BaseException:
                        flow.capacity.release()
                        raise
                    if stop_event.is_set() or text(op, "sym_name") in self.stopped:
                        flow.capacity.release()
                        break
                    key = flow.next_key
                    flow.next_key += 1
                    ticket = Ticket(stream, key, identity)
                    flow.active[key] = ticket
                    flow.high_water = max(flow.high_water, len(flow.active))
                    self._fact(stream, identity, "selected", key)
                    start = len(self.read_versions[identity])
                    try:
                        result = await self._invoke(op, (), identity, body_args=self._entry(op, args, (key,)), product=ticket)
                        ticket.payload = deepcopy(result[0])
                        ticket.versions = tuple(dict.fromkeys(self.read_versions[identity][start:]))
                        if stop_event.is_set() or text(op, "sym_name") in self.stopped:
                            self._dispose(ticket, "discarded", identity)
                        else:
                            await self._checkpoint(op, identity, "publish", ticket.key)
                            if stop_event.is_set() or text(op, "sym_name") in self.stopped:
                                self._dispose(ticket, "discarded", identity)
                            else:
                                await context.publish(stream, owners.Product(key, ticket, ()))
                                self._fact(stream, identity, "published", key, ticket.versions)
                    except BaseException as failed:
                        interrupted = isinstance(failed, asyncio.CancelledError)
                        disposition = (
                            "cancelled"
                            if interrupted and self.cancelled
                            else "discarded"
                            if interrupted and stop_event.is_set()
                            else "failed"
                        )
                        self._dispose(ticket, disposition, identity)
                        raise

            async def cleanup(_context):
                for ticket in tuple(flow.active.values()):
                    if ticket.owner is None:
                        self._dispose(ticket, "cancelled" if self.cancelled else "discarded", identity)
                while flow.active:
                    flow.changed.clear()
                    if flow.active:
                        await flow.changed.wait()

            primitive = graph.Producer(name, (stream,), produce, stop_when=stop_name, remaining="discard", cleanup=cleanup)
            await self.owner.scope(primitive)()
            return ()
        if protocol == Protocol.REQUESTS:
            sources = trait(op, RegionSemantics).describe(op).request_sources
            scope = self._handler_scope(op, identity)
            slot = self._slot(op, scope)
            if not sources:
                self.owner.request_queues[slot.name].put_nowait(None)

            def request(completion):
                registered = slot.issued.get(completion.key)
                pending = slot.pending.get(completion.key)
                if pending is not None and registered != pending:
                    raise ExecutionError("request packet differs from the owned registration")
                current = slot.pending.pop(completion.key, None)
                if current is None:
                    if completion.key not in slot.dispositions:
                        raise ExecutionError("request packet has no owned registration or disposition")
                    return None
                self.requests_changed.set()
                slot.current = current
                self.causes[identity] = current
                return graph.Choice(slot.name + "/body", current.ordinal, current)

            async def body(context):
                current = context.choice.payload
                try:
                    await self._invoke(op, (), identity, body_args=self._entry(op, args, (current,)))
                    slot.dispositions[current.ordinal] = "completed"
                except BaseException as failed:
                    slot.dispositions[current.ordinal] = "cancelled" if isinstance(failed, asyncio.CancelledError) else "failed"
                    raise
                finally:
                    slot.current = None
                    slot.capacity.release()
                    self._fact(origin(op), identity, "request_settled", current, slot.dispositions.get(current.ordinal))

            block = graph.Block(slot.name + "/body", (graph.Call(slot.name + "/execute", body),))
            primitive = graph.Requested(slot.name, "explicit-completion-registration", block, request)
            try:
                await self.owner.scope(primitive)()
            finally:
                for ordinal, pending in tuple(slot.pending.items()):
                    slot.dispositions[ordinal] = "cancelled_unfulfilled" if self.cancelled else "unfulfilled_on_failure"
                    self._fact(origin(op), identity, "request_disposed", pending, slot.dispositions[ordinal])
                    del slot.pending[ordinal]
                    slot.capacity.release()
                    self.requests_changed.set()
            return ()
        raise ExecutionError(f"unprepared protocol {protocol}")

    async def _emit(self, op: Operation, payload: tuple[object, ...], cause: Identity) -> None:
        slot = self._handler(op, cause)
        ordinal = slot.next_ordinal
        slot.next_ordinal += 1
        request = Request(origin(slot.operation), ordinal, cause, deepcopy(payload[0]))
        if self.cancelled or self.failed:
            self._unfulfilled(slot, request, cause)
            return
        try:
            await slot.capacity.acquire()
        except asyncio.CancelledError:
            self._unfulfilled(slot, request, cause)
            return
        if self.cancelled or self.failed:
            slot.capacity.release()
            self._unfulfilled(slot, request, cause)
            return
        slot.pending[ordinal] = request
        slot.issued[ordinal] = request
        self.requests_changed.set()
        self._fact(origin(op), cause, "request_registered", request)
        self._fact(origin(slot.operation), cause, "request_pending", request)
        packet = graph.Completion("source", "source", ordinal, 0, ordinal)
        self.owner.request_queues[slot.name].put_nowait((packet, None))

    def _unfulfilled(self, slot: Handler, request: Request, identity: Identity) -> None:
        slot.issued[request.ordinal] = request
        disposition = "cancelled_unfulfilled" if self.cancelled else "unfulfilled_on_failure"
        slot.dispositions[request.ordinal] = disposition
        self._fact(origin(slot.operation), identity, "request_disposed", request, disposition)

    async def _settle_pending(self, identity: Identity) -> None:
        while not self.cancelled and not self.failed:
            self.requests_changed.clear()
            if not any(request.cause[: len(identity)] == identity for slot in self.handlers.values() for request in slot.pending.values()):
                return
            await self.requests_changed.wait()
        for slot in self.handlers.values():
            for ordinal, request in tuple(slot.pending.items()):
                if request.cause[: len(identity)] == identity:
                    self._unfulfilled(slot, request, identity)
                    del slot.pending[ordinal]
                    slot.capacity.release()
                    self.requests_changed.set()

    async def _close_source(self, op: Operation, identity: Identity, outcome: Outcome) -> None:
        if "sym_name" not in op.properties:
            return
        name = text(op, "sym_name")
        for waiting in self.index.values():
            if isinstance(waiting, StopAfterOp) and symbol(waiting.properties["source"]) == name:
                self.closed_sources[self._scope(waiting, op, identity), name] = (
                    SourceReceipt(origin(op), identity, outcome.status, self.cancelled),
                )
                self._event(waiting, op, identity).set()
            region = waiting.get_trait(RegionSemantics)
            if region is not None and region.protocol == Protocol.REQUESTS and name in region.describe(waiting).request_sources:
                slot = self._slot(waiting, self._handler_scope(waiting, identity))
                newly_closed = name not in slot.closed
                slot.closed.add(name)
                required = set(trait(slot.operation, RegionSemantics).describe(slot.operation).request_sources)
                if newly_closed and slot.closed == required:
                    self.owner.request_queues[slot.name].put_nowait(None)

    def _version(self, name: str) -> Version:
        internal = self.resource_names[name]
        binding = self.resource_bindings[self.declarations[name].value.type.kind.data]
        ordinal = self.state.numerical_epochs[internal] if binding.spec.storage == "participant" else self.state.owners[internal].ordinal
        return Version(name, 1, "initial", ordinal)

    def value(self, name: str) -> object:
        internal = self.resource_names[name]
        binding = self.resource_bindings[self.declarations[name].value.type.kind.data]
        if binding.spec.storage == "participant":
            return float(self.state.image.bindings[internal].model.get_parameter(binding.spec.parameter).detach())
        return deepcopy(self.state.owners[internal].value)

    @asynccontextmanager
    async def _access(self, name: str, mode: AccessMode, freshness: Freshness, selector: str = "current"):
        binding = self.resource_bindings[self.declarations[name].value.type.kind.data]
        internal = self.resource_names[name]
        if mode == AccessMode.SNAPSHOT and freshness == Freshness.PUBLISHED:
            version, value = self.initial_snapshots[name]
            cap = Capability(name, mode, version, value)
            try:
                yield cap
            finally:
                cap.active = False
            return
        if binding.spec.storage == "participant":
            if mode == AccessMode.SNAPSHOT:
                region = object()
                retained = await self.state.retain(graph.Retain("snapshot", (internal,), (), "snapshot", isolated=True), region)
                stamp = retained.projection.stamps[internal]
                cap = Capability(
                    name, mode, Version(name, stamp.binding, "initial", stamp.numerical_epoch), retained.projection.models[internal]
                )
                try:
                    yield cap
                finally:
                    cap.active = False
                    await self.state.release(retained, region, aborted=self.failed)
            else:
                async with self.state.use((graph.Use(internal, "write" if mode == AccessMode.OWNER_UPDATE else "read"),)) as projection:
                    cap = Capability(name, mode, self._version(name), projection.models[internal])
                    try:
                        yield cap
                    except BaseException as failed:
                        if not isinstance(failed, asyncio.CancelledError):
                            self.state.unusable.add(internal)
                        raise
                    finally:
                        cap.active = False
        elif mode == AccessMode.SNAPSHOT:
            async with self.state.acquire((), (internal,)):
                cap = Capability(name, mode, self._version(name), deepcopy(self.state.owners[internal].value))
            try:
                yield cap
            finally:
                cap.active = False
        else:
            async with self.state.acquire((), (internal,)):
                cap = Capability(name, mode, self._version(name), self.state.owners[internal].value)
                try:
                    yield cap
                except BaseException as failed:
                    if not isinstance(failed, asyncio.CancelledError):
                        self.state.unusable_states.add(internal)
                    raise
                finally:
                    cap.active = False

    def _valid(self, value: object, name: str) -> bool:
        if name == "capability":
            return isinstance(value, Capability) and value.active
        if name == "scalar":
            return type(value) in (int, float)
        if name == "index":
            return type(value) is int
        if name == "bool":
            return type(value) is bool
        validator = self.installed.schemas.get(name)
        return validator(value) if validator else False

    async def _call(self, op: Operation, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        if self.cancelled or self.failed:
            raise asyncio.CancelledError("run closing prevents selected-call entry")
        binding = self.bindings[symbol(op.properties["binding"])]
        if not all(self._valid(value, kind) for value, kind in zip(args, binding.spec.inputs, strict=True)):
            raise ExecutionError(f"{origin(op)}: binding input contract mismatch")
        ordinal = self.call_counts[origin(op)]
        self.call_counts[origin(op)] += 1
        key = origin(op), ordinal
        await self._checkpoint(op, identity, "call_enter", ordinal)
        if self.cancelled or self.failed:
            raise asyncio.CancelledError("run closing prevents selected-call entry")
        self.controls.reached.setdefault(key, asyncio.Event()).set()
        if key in self.controls.before:
            await self.controls.before[key].wait()
        entered = self.controls.entered.setdefault(key, threading.Event())
        source = origin(op)
        loop = asyncio.get_running_loop()
        gate, failure = self.controls.gates.get(key), self.controls.failures.get(key)
        metadata = ordinal, self.causes.get(identity), self.inputs.get(identity, ()), self.products.get(identity, ())

        def invoke():
            entered.set()
            loop.call_soon_threadsafe(self._fact, source, identity, "physical_entered", *metadata)
            if gate is not None:
                gate.wait()
            if failure is not None:
                raise ExecutionError(failure)
            return binding.invoke(args)

        try:
            finished = await self.workers.execute(invoke, threading.Event())
        except physical.NotEntered as failed:
            self._fact(origin(op), identity, "physical_not_entered", ordinal, str(failed.cause))
            raise failed.cause from failed
        except BaseException as failed:
            self._fact(
                origin(op),
                identity,
                "physical_failed",
                ordinal,
                str(failed),
                self.causes.get(identity),
                self.inputs.get(identity, ()),
                self.products.get(identity, ()),
            )
            raise
        result = finished.value
        interruption = finished.cancellation
        try:
            await self._checkpoint(op, identity, "call_complete", ordinal)
        except asyncio.CancelledError as cancelled:
            interruption = interruption or cancelled
        if not isinstance(result, Result):
            raise ExecutionError("binding returned an invalid completion envelope")
        for effect in result.effects:
            if effect.capability not in binding.spec.updates:
                raise ExecutionError("binding attempted an undeclared owner update")
            cap = args[effect.capability]
            if (
                not isinstance(cap, Capability)
                or cap.mode != AccessMode.OWNER_UPDATE
                or not cap.active
                or self._version(cap.resource) != cap.version
            ):
                raise ExecutionError("update lacks current owner authority")
            internal = self.resource_names[cap.resource]
            if effect.action == "gradient":
                self.state.invocation += 1
                invocation = self.state.invocation
                with torch.enable_grad():
                    contribution = self.state.contribute(
                        cap.value.get_parameter(self.resource_bindings[self.declarations[cap.resource].value.type.kind.data].spec.parameter)
                        * float(effect.value),
                        (internal,),
                        invocation,
                    )
                self.state.advance(contribution, (internal,), invocation)
            elif effect.action == "assign" and internal in self.state.owners:
                cell = self.state.owners[internal]
                cell.value, cell.ordinal = deepcopy(effect.value), cell.ordinal + 1
            else:
                raise ExecutionError("unsupported selected update realization")
            self._fact(cap.resource, identity, "updated", self._version(cap.resource), self.value(cap.resource))
        self._fact(
            origin(op),
            identity,
            "physical_completed",
            ordinal,
            result.values,
            self.causes.get(identity),
            self.inputs.get(identity, ()),
            self.products.get(identity, ()),
        )
        if len(result.values) != len(binding.spec.outputs) or not all(
            self._valid(value, kind) for value, kind in zip(result.values, binding.spec.outputs, strict=True)
        ):
            self._fact(origin(op), identity, "validation_rejected", self.inputs.get(identity, ()), self.products.get(identity, ()))
            raise ExecutionError(f"{origin(op)}: binding output contract mismatch")
        self._fact(origin(op), identity, "validated", self.inputs.get(identity, ()), self.products.get(identity, ()))
        if interruption is not None:
            orderly_production = (
                not self.cancelled
                and not self.failed
                and any(identity[: len(scope)] == scope and event.is_set() for scope, event in self.production_stops.items())
            )
            if not orderly_production:
                raise interruption
        return deepcopy(result.values)

    async def _take(self, op: Operation, name: str, identity: Identity) -> Ticket:
        entry_key = origin(op), self.entry_counts[origin(op)]
        self.entry_counts[origin(op)] += 1
        self.controls.reached.setdefault(entry_key, asyncio.Event()).set()
        if entry_key in self.controls.before:
            await self.controls.before[entry_key].wait()
        flow, feed = self.flows[name], self.state.feeds[name]
        contract = flow.declaration.get_trait(FlowSemantics).describe(flow.declaration)
        while True:
            await self._checkpoint(op, identity, "take", entry_key[1])
            async with feed.changed:
                await feed.changed.wait_for(lambda: bool(feed.ready) or feed.closed)
                if self.cancelled or self.failed:
                    raise asyncio.CancelledError("run closing prevents input admission")
                if feed.failure is not None:
                    raise graph.NotReady("input producer failed") from feed.failure
                if not feed.ready:
                    raise ExecutionError("input exhausted before required admission")
                key = min(feed.ready)
                ticket = feed.ready.pop(key).value
                if ticket.disposition is not None or key in flow.terminal:
                    feed.changed.notify_all()
                    continue
                comparison = tuple(self._version(owner) for owner in contract.versions.owners)
                admitted = evaluate_admission(contract.versions, ticket.versions, Comparison(contract.versions.boundary, comparison))
                disposition = admitted.disposition
                self._fact(
                    origin(op),
                    identity,
                    "admission_checked",
                    ticket.key,
                    ticket.versions,
                    name,
                    comparison,
                    disposition,
                    contract.versions.boundary,
                )
                feed.changed.notify_all()
                if admitted.accepted:
                    feed.handed_off.append((key, self.state.invocation, ()))
                    ticket.owner = identity
                    self.inputs[identity] = (*self.inputs.get(identity, ()), (name, key))
                    self._fact(origin(op), identity, "item_admitted", key, ticket.versions, name)
                    return ticket
                self._fact(origin(op), identity, "item_rejected", key, ticket.versions, name, disposition)
                self._dispose(ticket, disposition, identity)
            await asyncio.sleep(0)

    def _dispose(self, ticket: Ticket, disposition: str, identity: Identity) -> None:
        if ticket.disposition is not None:
            return
        flow = self.flows[ticket.stream]
        ticket.disposition = disposition
        self.state.feeds[ticket.stream].ready.pop(ticket.key, None)
        flow.terminal[ticket.key] = disposition
        del flow.active[ticket.key]
        flow.capacity.release()
        flow.changed.set()
        self._fact(ticket.stream, identity, "disposition", ticket.key, disposition)
