"""Lowering: the prepared run as the coordinator executes it (D15, D16).

Everything the strategy fixes is resolved once, here, so executing a body
reads its values from numbered slots and dispatches on nothing but each
step's kind:

- **slots**: a body's entries take the first slots and each statement's
  results the next ones, in order. Constants are filled in when lowering;
- **steps**: each statement is bound to what it needs. A fused group of calls
  to its implementations, its argument and result slots, and which of its
  arguments are leases or retained results; an activity to its protocol, its
  bodies, how their entries are built and its protocol's fixed data; an
  access to the names of the resources it leases;
- **end checks**: when a body ends, the runtime settles the claims and
  releases the retained results it still holds, and fails the body that left
  them (D16). The types say where those can be, so the check reads only
  claim-typed values and retained-typed results, and only an activity whose
  bodies can hold a claim looks for claims it still owns when it settles;
- **scope records**: a wait for a source to settle needs to know which
  invocations started and ended in a scope. Only the statements something
  waits on, the sources of handlers and stops and the statements between
  them and the scope they share with the waiter, keep those records.

Lowering is derived, like the prepared run, and never an authority. A
coordinator in checked mode runs every check in full beside the lowered
one and fails the run if the full check finds what lowering excluded. A step
lowering cannot realize carries the reason, and fails when it executes.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from xdsl.dialects.builtin import FloatAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.ir import Block, Operation, SSAValue

from library.iro.operation_contracts import (
    Boundary,
    FlowContract,
    FlowSemantics,
    LifecycleProtocol,
    RegionSemantics,
    ResourceSemantics,
    ResourceUse,
    VersionSelection,
    path,
    provider,
    sym_name,
    symbol,
    symbols,
)
from library.iro.prepare.prepared import Group, Prepared
from library.iro.prepare.target import HOST
from library.iro.seams import Implementation
from library.iro.words.access import AccessOp, ReleaseOp
from library.iro.words.calls import CallOp, ConstantOp, YieldOp
from library.iro.words.data import BorrowOp, DisposeOp, ForkOp, StreamOp, TakeOp
from library.iro.words.lifetimes import HandlerOp, RequestOp, StopOp
from library.iro.words.model import RelationshipOp, ResourceOp
from library.iro.words.structure import StrategyOp
from library.iro.words.types import ClaimType, LeaseType, OptionalType, RetainedType

# Step kinds.
CALLS = "calls"
INVOKE = "invoke"
RESOURCE = "resource"
RELATIONSHIP = "relationship"
STREAM = "stream"
RELEASE = "release"
TAKE = "take"
BORROW = "borrow"
DISPOSE = "dispose"
FORK = "fork"
REQUEST = "request"
STOP = "stop"
UNREALIZED = "unrealized"


@dataclass(frozen=True, eq=False)
class Step:
    """One statement of a body: what it reads and where its results go."""

    kind: str
    op: Operation
    path: str
    operands: tuple[int, ...]
    results: tuple[int, ...]
    problem: str | None = None  # why the runtime cannot realize it; executing it fails with this


@dataclass(frozen=True, eq=False)
class Body:
    """One body: its slots, its steps, what it yields and what its end checks read."""

    path: str
    entries: int
    template: tuple[object, ...]  # the slots after the entries: constants filled in, the rest empty
    steps: tuple[Step, ...]
    returns: tuple[int, ...]
    retained: tuple[int, ...]  # results that can hold a retained result, in the order they are made
    claims: tuple[int, ...]  # entries and results that can hold a claim
    results: frozenset[int] = frozenset()  # every result slot, which checked mode scans


@dataclass(frozen=True, eq=False)
class Call:
    """One call of a group: its implementation and where its arguments come from."""

    path: str
    implementation: Implementation
    slots: tuple[int, ...]  # an argument's slot in the body, or -1 for an earlier call's result in the group
    locals: tuple[tuple[int, int, int], ...]  # (argument, earlier call, its result) for results passed within the group
    leases: tuple[int, ...]  # arguments that are leases
    retained: tuple[int, ...]  # arguments that are retained results from the body
    results: tuple[int, ...]
    updates: tuple[int, ...]  # the declaration's updated and retained arguments
    retains: tuple[int, ...]


@dataclass(frozen=True, eq=False, kw_only=True)
class Calls(Step):
    """A fused group: one submission to a worker thread on its device."""

    group: Group
    calls: tuple[Call, ...]
    host: bool
    wait: bool  # the worker waits for the work's completion: host work, or results that leave device work
    safe: bool  # every call is declared safe for concurrent use


@dataclass(frozen=True, eq=False)
class Entry:
    """How one body's entries are built: captured operands, and the positions the protocol supplies."""

    size: int
    captures: tuple[tuple[int, int], ...]  # (entry, operand)
    supplied: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class Settling:
    """A wait for a source to settle: the scope it shares with the waiter and the statements up to it."""

    source: str
    common: str
    chain: tuple[str, ...]  # the source and every statement between it and the shared scope
    keys: tuple[tuple, ...]  # what wakes the wait: each of those statements settling


@dataclass(frozen=True, eq=False, kw_only=True)
class Invoke(Step):
    """An activity: its protocol, its bodies and the protocol's fixed data."""

    protocol: LifecycleProtocol
    bodies: tuple[Body, ...]
    entries: tuple[Entry, ...]
    captures_claims: bool  # claims can be passed into its bodies
    owns_claims: bool  # it can still own a claim when it settles
    optional: bool = False  # a choice on an optional, or a producer that may produce nothing
    limit: int | None = None
    carried: int = 0
    branches: tuple[str, ...] = ()
    branch_claims: tuple[frozenset[int], ...] = ()  # per branch, the operands its body uses
    uses: tuple[tuple[str, ResourceUse], ...] = ()
    keys: tuple[tuple, ...] = ()
    device: str | None = None  # the device admission an access begins
    sources: tuple[Settling, ...] = ()


@dataclass(frozen=True, eq=False, kw_only=True)
class Named(Step):
    """A statement whose fixed data is one word: a resource's name, or the disposition a claim settles with."""

    word: str


@dataclass(frozen=True, eq=False, kw_only=True)
class StreamStep(Step):
    name: str
    flow: FlowContract


@dataclass(frozen=True, eq=False, kw_only=True)
class StopStep(Step):
    target: str
    source: str  # the source's symbol, as the stop's fact names it
    settling: Settling
    scope: str  # the statement whose invocation scopes the stop: the one shared with the target


@dataclass(frozen=True, eq=False)
class HandlerInfo:
    """What a request needs of its handler: its bound and the uses it reserves at the request."""

    path: str
    bound: int
    at_request: tuple[tuple[str, ResourceUse], ...]


@dataclass(frozen=True, eq=False, kw_only=True)
class RequestStep(Step):
    handler: HandlerInfo
    handler_name: str
    source: str  # the path of the activity whose settlement emits the request
    boundary: Boundary


@dataclass(frozen=True, eq=False)
class Lowered:
    """The lowered run: its root body and what scope records it keeps."""

    path: str
    root: Body
    waited: Mapping[str, frozenset[str]]  # statement -> the scopes something waits for it to settle in


def lower(prepared: Prepared, installed: Mapping[str, Implementation]) -> Lowered:
    """The prepared run's lowered form."""
    return _Lowering(prepared, installed).lowered()


class _Lowering:
    def __init__(self, prepared: Prepared, installed: Mapping[str, Implementation]):
        self.prepared, self.installed = prepared, installed
        self.strategy: StrategyOp = prepared.loaded.normalized.strategy
        self.named = {name: op for op in self.strategy.walk() if (name := sym_name(op)) is not None and op is not self.strategy}
        self.groups = {call: group for group in prepared.groups for call in group.calls}
        self.admissions = {a.begins: a.device for a in prepared.admissions}
        self.handlers: dict[str, HandlerInfo] = {}
        self.waited: dict[str, set[str]] = {}

    def lowered(self) -> Lowered:
        root = self._body(self.strategy.body.block)
        return Lowered(path(self.strategy), root, {name: frozenset(scopes) for name, scopes in self.waited.items()})

    # Bodies.

    def _body(self, block: Block) -> Body:
        slots: dict[SSAValue, int] = {arg: i for i, arg in enumerate(block.args)}
        template: list[object] = []
        retained: list[int] = []
        claims = [i for i, arg in enumerate(block.args) if isinstance(arg.type, ClaimType)]
        for op in block.ops:
            for result in op.results:
                slot = slots[result] = len(slots)
                template.append(None)
                if isinstance(result.type, RetainedType):
                    retained.append(slot)
                if isinstance(result.type, ClaimType):
                    claims.append(slot)
        steps: list[Step] = []
        returns: tuple[int, ...] = ()
        pending: list[CallOp] = []
        entries = len(block.args)
        for op in block.ops:
            if isinstance(op, CallOp) and pending and self.groups[path(op)] is not self.groups[path(pending[0])]:
                steps.append(self._calls(pending, slots))
                pending = []
            if isinstance(op, CallOp):
                pending.append(op)
                continue
            if isinstance(op, ConstantOp):
                literal = _literal(op)
                if literal is _UNKNOWN:
                    if pending:  # the calls before it run first, as they are written
                        steps.append(self._calls(pending, slots))
                        pending = []
                    steps.append(Step(UNREALIZED, op, path(op), (), (), f"{path(op)}: an unknown literal"))
                else:
                    template[slots[op.result] - entries] = literal
                continue
            if pending:
                steps.append(self._calls(pending, slots))
                pending = []
            if isinstance(op, YieldOp):
                returns = tuple(slots[v] for v in op.operands)
                break
            steps.append(self._step(op, slots))
        if pending:
            steps.append(self._calls(pending, slots))
        parent = block.parent_op()
        return Body(
            path(parent) if parent is not None else "",
            entries,
            tuple(template),
            tuple(steps),
            returns,
            tuple(retained),
            tuple(claims),
            frozenset(range(entries, len(slots))),
        )

    def _calls(self, ops: list[CallOp], slots: dict[SSAValue, int]) -> Calls:
        group = self.groups[path(ops[0])]
        problem = None
        local: dict[SSAValue, tuple[int, int]] = {result: (j, r) for j, op in enumerate(ops) for r, result in enumerate(op.results)}
        calls = []
        for op in ops:
            arguments, locals_, leases, retained = [], [], [], []
            for position, operand in enumerate(op.operands):
                if operand in local:
                    j, r = local[operand]
                    arguments.append(-1)
                    locals_.append((position, j, r))
                    continue
                arguments.append(slots[operand])
                if isinstance(operand.type, LeaseType):
                    leases.append(position)
                elif isinstance(operand.type, RetainedType):
                    retained.append(position)
            implementation = self.installed[symbol(op.implementation)]
            declaration = implementation.declaration
            if declaration.retains and (len(op.results) != 1 or not isinstance(op.results[0].type, RetainedType)):
                # The retained result is what keeps its inputs protected; the runtime, like the reference, makes it of the one result.
                problem = (
                    problem or f"{path(op)}: @{declaration.name} retains its inputs, so it returns exactly one result, the retained one"
                )
            calls.append(
                Call(
                    path(op),
                    implementation,
                    tuple(arguments),
                    tuple(locals_),
                    tuple(leases),
                    tuple(retained),
                    tuple(slots[r] for r in op.results),
                    tuple(declaration.updates),
                    tuple(declaration.retains),
                )
            )
        host = group.device == HOST.name
        return Calls(
            CALLS,
            ops[0],
            group.calls[0],
            (),
            tuple(slot for call in calls for slot in call.results),
            problem,
            group=group,
            calls=tuple(calls),
            host=host,
            wait=group.sync or host,
            safe=all(call.implementation.declaration.concurrency_safe for call in calls),
        )

    # Statements.

    def _step(self, op: Operation, slots: dict[SSAValue, int]) -> Step:
        operands = tuple(slots[v] for v in op.operands)
        results = tuple(slots[r] for r in op.results)
        where = path(op)
        if op.get_trait(RegionSemantics) is not None:
            return self._invoke(op, operands, results)
        if isinstance(op, ResourceOp):
            return Named(RESOURCE, op, where, operands, results, word=op.sym_name.data)
        if isinstance(op, RelationshipOp):
            return Step(RELATIONSHIP, op, where, operands, results)
        if isinstance(op, StreamOp):
            return StreamStep(STREAM, op, where, operands, results, name=op.sym_name.data, flow=FlowSemantics().describe(op))
        if isinstance(op, ReleaseOp):
            return Step(RELEASE, op, where, operands, results)
        if isinstance(op, TakeOp):
            return Step(TAKE, op, where, operands, results)
        if isinstance(op, BorrowOp):
            return Step(BORROW, op, where, operands, results)
        if isinstance(op, DisposeOp):
            return Named(DISPOSE, op, where, operands, results, word=op.disposition.data)
        if isinstance(op, ForkOp):
            return Step(FORK, op, where, operands, results)
        if isinstance(op, RequestOp):
            target = self.named[symbol(op.handler)]
            assert isinstance(target, HandlerOp)
            return RequestStep(
                REQUEST,
                op,
                where,
                operands,
                results,
                handler=self._handler_info(target),
                handler_name=symbol(op.handler),
                source=path(self.named[symbol(op.source)]),
                boundary=Boundary(op.boundary.data),
            )
        if isinstance(op, StopOp):
            source, target = self.named[symbol(op.source)], self.named[symbol(op.target)]
            return StopStep(
                STOP,
                op,
                where,
                operands,
                results,
                target=path(target),
                source=symbol(op.source),
                settling=self._settling(source, op),
                scope=path(_common(op, target)),
            )
        return Step(UNREALIZED, op, where, operands, results, f"{where}: the runtime does not realize {op.name}")

    def _invoke(self, op: Operation, operands: tuple[int, ...], results: tuple[int, ...]) -> Invoke:
        semantics = op.get_trait(RegionSemantics)
        assert semantics is not None
        contract = semantics.describe(op)
        protocol = contract.protocol
        where = path(op)
        bodies = tuple(self._body(region.block) for region in op.regions)
        entries = []
        for role in contract.roles:
            captured = tuple((c.argument, c.operand) for c in role.captures)
            taken = {argument for argument, _ in captured}
            entries.append(Entry(len(role.entries), captured, tuple(i for i in range(len(role.entries)) if i not in taken)))
        captures_claims = any(isinstance(v.type, ClaimType) for v in op.operands)
        owns_claims = (
            protocol == LifecycleProtocol.PRODUCER
            or captures_claims
            or any(isinstance(v.type, ClaimType) for region in op.regions for v in _values(region.block))
        )
        problem = None
        known = {
            LifecycleProtocol.SEQUENCE,
            LifecycleProtocol.CHOICE,
            LifecycleProtocol.PARALLEL,
            LifecycleProtocol.REPEAT,
            LifecycleProtocol.ACCESS,
            LifecycleProtocol.FILTER,
            LifecycleProtocol.PRODUCER,
            LifecycleProtocol.HANDLER,
        }
        if protocol not in known:
            problem = f"{where}: the runtime does not realize {protocol}"
        extra: dict = {}
        if protocol == LifecycleProtocol.CHOICE:
            extra["optional"] = isinstance(op.operands[0].type, OptionalType)
        elif protocol == LifecycleProtocol.PARALLEL:
            extra["branches"] = tuple(f"{where}#{index}" for index in range(len(op.regions)))
            extra["branch_claims"] = tuple(
                frozenset(position for position, arg in enumerate(region.block.args) if arg.uses) for region in op.regions
            )
        elif protocol == LifecycleProtocol.REPEAT:
            limit = op.properties.get("limit")
            if not isinstance(limit, IntegerAttr):
                problem = f"{where}: a repeat by policy is realized with training's words, not by the core runtime"
            else:
                extra["limit"] = limit.value.data
            extra["carried"] = contract.roles[0].invocation_arguments.count("carried_value")
        elif protocol == LifecycleProtocol.ACCESS:
            uses = []
            for use in ResourceSemantics().describe(op).uses:
                name = _resource(use.subject)
                if name is None:
                    problem = f"{where}: an access names something that is not a resource"
                    break
                uses.append((name, use))
            extra["uses"] = tuple(uses)
            extra["keys"] = tuple(dict.fromkeys(("resource", name) for name, _ in uses))
            extra["device"] = self.admissions.get(where)
        elif protocol == LifecycleProtocol.PRODUCER:
            last = op.regions[0].block.last_op
            extra["optional"] = last is not None and bool(last.operands) and isinstance(last.operands[0].type, OptionalType)
        elif protocol == LifecycleProtocol.HANDLER:
            assert isinstance(op, HandlerOp)
            self._handler_info(op)
            sources = tuple(self._settling(self.named[name], op) for name in symbols(op, "sources"))
            extra["sources"] = sources
            extra["keys"] = (("handler", where), *(key for s in sources for key in s.keys))
        return Invoke(
            INVOKE,
            op,
            where,
            operands,
            results,
            problem,
            protocol=protocol,
            bodies=bodies,
            entries=tuple(entries),
            captures_claims=captures_claims,
            owns_claims=owns_claims,
            **extra,
        )

    def _handler_info(self, handler: HandlerOp) -> HandlerInfo:
        where = path(handler)
        info = self.handlers.get(where)
        if info is None:
            at_request = []
            for access in handler.walk():
                if isinstance(access, AccessOp):
                    for use in ResourceSemantics().describe(access).uses:
                        name = _resource(use.subject)
                        if use.selection == VersionSelection.AT_REQUEST and name is not None:
                            at_request.append((name, use))
            info = self.handlers[where] = HandlerInfo(where, handler.max_outstanding.value.data, tuple(at_request))
        return info

    # Scopes.

    def _settling(self, source: Operation, waiting: Operation) -> Settling:
        """A wait for ``source`` to settle from ``waiting``; the statements on the way keep scope records."""
        common = _common(waiting, source)
        chain = []
        current: Operation | None = source
        while current is not None and current is not common:
            chain.append(path(current))
            current = current.parent_op()
        for statement in chain:
            self.waited.setdefault(statement, set()).add(path(common))
        return Settling(path(source), path(common), tuple(chain), tuple(("settled", statement) for statement in chain))


def _common(waiting: Operation, source: Operation) -> Operation:
    """The innermost statement enclosing both, which scopes a wait or a stop."""
    ancestors = set()
    current: Operation | None = source
    while current is not None:
        ancestors.add(current)
        current = current.parent_op()
    common: Operation | None = waiting
    while common is not None and common not in ancestors:
        common = common.parent_op()
    if common is None:
        raise ValueError(f"{path(waiting)}: refers outside its strategy")
    return common


_UNKNOWN = object()


def _literal(op: ConstantOp) -> object:
    value = op.value
    if isinstance(value, IntegerAttr):
        number = value.value.data
        return bool(number) if isinstance(value.type, IntegerType) and value.type.width.data == 1 else number
    if isinstance(value, FloatAttr):
        return value.value.data
    if isinstance(value, StringAttr):
        return value.data
    return _UNKNOWN


def _resource(subject: SSAValue) -> str | None:
    """The resource an access names, following captures outward."""
    owner = provider(subject).owner
    return owner.sym_name.data if isinstance(owner, ResourceOp) else None


def _values(block: Block) -> list[SSAValue]:
    return [*block.args, *(r for op in block.ops for r in op.results)]


__all__ = [
    "BORROW",
    "CALLS",
    "DISPOSE",
    "FORK",
    "INVOKE",
    "RELATIONSHIP",
    "RELEASE",
    "REQUEST",
    "RESOURCE",
    "STOP",
    "STREAM",
    "TAKE",
    "UNREALIZED",
    "Body",
    "Call",
    "Calls",
    "Entry",
    "HandlerInfo",
    "Invoke",
    "Lowered",
    "Named",
    "RequestStep",
    "Settling",
    "Step",
    "StopStep",
    "StreamStep",
    "lower",
]
