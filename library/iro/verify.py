"""Verification: whether a normalized strategy's lifetimes, ownership, access and admission hold (D14, D16, D18).

Verification reads a normalized strategy and its contracts and runs nothing.
It reports every independent problem it finds as a source-attributed
diagnostic, or it derives the strategy's obligations: what the run must
discharge, each attached to the readiness checkpoint at which it must first be
fulfilled (``training-contract``). It also reports the strategy's
reproducibility class with what contributes to it.

Its analyses query contracts through the traits each word carries and never
select behavior by word name, except to recognize the core structure words
the contracts are built from.
"""

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from xdsl.ir import Block, BlockArgument, Operation, OpResult, SSAValue
from xdsl.traits import IsTerminator

from library.iro.diagnostics import Problems, Stage
from library.iro.normalize import Normalized, dependents
from library.iro.operation_contracts import (
    AccessMode,
    Admission,
    CompletionSemantics,
    Decision,
    DecisionSemantics,
    FlowSemantics,
    ItemSemantics,
    LifecycleProtocol,
    ProductSemantics,
    RankRule,
    RealizationSemantics,
    RegionSemantics,
    Reproducibility,
    ResourceSemantics,
    ResourceUse,
    VersionSelection,
    origin,
    path,
    provider,
    sym_name,
    symbol,
    symbols,
)
from library.iro.words.access import AccessOp, ReleaseOp
from library.iro.words.calls import CallOp, retained_positions
from library.iro.words.data import StreamOp, TakeOp
from library.iro.words.lifetimes import HandlerOp, ProducerOp, RequestOp, StopOp
from library.iro.words.model import ResourceOp
from library.iro.words.structure import ParallelOp, RepeatOp, StrategyOp
from library.iro.words.types import ClaimType, LeaseType, ResourceType, RetainedType, StreamType

# Bodies that belong to the activity around them rather than being activities of their own.
LOCAL_PROTOCOLS = frozenset({LifecycleProtocol.ACCESS, LifecycleProtocol.CHOICE})
LIVE_MODES = frozenset({AccessMode.PROTECTED_CURRENT, AccessMode.OWNER_UPDATE, AccessMode.GRANT})
UPDATE_MODES = frozenset({AccessMode.OWNER_UPDATE, AccessMode.GRANT})


# What verification derives.


@dataclass(frozen=True, order=True)
class Checkpoint:
    """A readiness checkpoint: core execution, or a handler's first accepted request."""

    handler: str = ""

    def __str__(self) -> str:
        return f"request to @{self.handler}" if self.handler else "core execution"


CORE_EXECUTION = Checkpoint()


@dataclass(frozen=True, order=True)
class Obligation:
    """Something the run must discharge, with where it came from and who judges it.

    ``subject`` is the statement it is about, by its structural path;
    ``evidence`` what must be shown; ``evaluator`` the owner that judges it:
    the run authority, a resource owner or the IR runtime.
    """

    checkpoint: Checkpoint
    subject: str
    evidence: str
    evaluator: str
    names: tuple[str, ...] = ()
    source: str = ""


@dataclass(frozen=True)
class Obligations:
    """The obligation set at its revision."""

    entries: tuple[Obligation, ...]

    @property
    def revision(self) -> str:
        listing = "\n".join(f"{o.checkpoint}|{o.subject}|{o.evidence}|{o.evaluator}|{','.join(o.names)}" for o in self.entries)
        return hashlib.blake2s(listing.encode(), digest_size=8).hexdigest()

    @property
    def checkpoints(self) -> tuple[Checkpoint, ...]:
        return tuple(sorted({o.checkpoint for o in self.entries}))

    def at(self, checkpoint: Checkpoint) -> tuple[Obligation, ...]:
        return tuple(o for o in self.entries if o.checkpoint == checkpoint)

    def __iter__(self) -> Iterator[Obligation]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class Contributor:
    """One thing that contributes to the strategy's reproducibility class, and why."""

    subject: str
    reproducibility: Reproducibility
    reason: str
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Verification:
    """A verified strategy: its obligations and its reproducibility class."""

    normalized: Normalized
    obligations: Obligations
    reproducibility: Reproducibility
    contributors: tuple[Contributor, ...]


def verify(normalized: Normalized, *, replicated: bool = False) -> Verification:
    """Verifies a normalized strategy, or rejects it with every independent problem found.

    ``replicated`` says the run executes on several ranks, so every decision
    must agree across them.
    """
    analysis = Analysis(normalized.strategy)
    analysis.claims()
    analysis.leases_and_retained_results()
    analysis.accesses()
    analysis.calls()
    analysis.lifecycles()
    analysis.flows()
    if replicated:
        analysis.rank_agreement()
    analysis.problems.reject_if_any()
    reproducibility, contributors = analysis.reproducibility()
    return Verification(normalized, Obligations(tuple(sorted(set(analysis.obligations)))), reproducibility, contributors)


def uses_of(access: Operation) -> tuple[ResourceUse, ...]:
    return ResourceSemantics().describe(access).uses


# Waits: which lifetime points must come before which.

Event = tuple[Operation, str]


@dataclass
class Waits:
    """Lifetime points, an activity's entry and its settlement, with what each waits for."""

    edges: dict[Event, set[Event]] = field(default_factory=lambda: defaultdict(set))

    def add(self, waiting: Event, required: Event) -> None:
        self.edges[waiting].add(required)

    def requires(self, waiting: Event, required: Event) -> bool:
        pending, seen = [waiting], set()
        while pending:
            node = pending.pop()
            if node == required:
                return True
            if node not in seen:
                seen.add(node)
                pending.extend(self.edges.get(node, ()))
        return False

    def cycle(self) -> list[Event] | None:
        """One cycle, when the points wait for each other in a circle."""
        state: dict[Event, int] = {}
        for start in list(self.edges):
            if start in state:
                continue
            stack: list[tuple[Event, Iterator[Event]]] = [(start, iter(self.edges.get(start, ())))]
            trail = [start]
            state[start] = 1
            while stack:
                node, successors = stack[-1]
                following = next(successors, None)
                if following is None:
                    state[node] = 2
                    stack.pop()
                    trail.pop()
                elif state.get(following) == 1:
                    return [*trail[trail.index(following) :], following]
                elif following not in state:
                    state[following] = 1
                    stack.append((following, iter(self.edges.get(following, ()))))
                    trail.append(following)
        return None


# The analysis.


class Analysis:
    def __init__(self, strategy: StrategyOp):
        if strategy.normalized is None:
            raise ValueError("verification reads a normalized strategy; normalize it first")
        self.strategy = strategy
        self.problems = Problems(Stage.VERIFY)
        self.obligations: list[Obligation] = []
        self.named = {name: op for op in strategy.walk() if (name := sym_name(op)) is not None and op is not strategy}
        self.waits = self._waits()
        self.retained = [r for op in strategy.walk() for r in op.results if isinstance(r.type, RetainedType)]

    # Reporting.

    def problem(self, code: str, op: Operation, message: str, *, names: Iterable[str] = (), fix: str | None = None) -> None:
        self.problems.add(code, message, names=tuple(names), source=origin(op), fix=fix, path=path(op))

    def owe(self, op: Operation, evidence: str, evaluator: str, *, names: Iterable[str] = (), checkpoint: Checkpoint | None = None) -> None:
        self.obligations.append(
            Obligation(checkpoint or self.checkpoint(op), path(op), evidence, evaluator, tuple(names), origin(op) or "")
        )

    def checkpoint(self, op: Operation) -> Checkpoint:
        """Where a statement's obligations first apply: core execution, or its handler's first request."""
        current: Operation | None = op
        while current is not None:
            if isinstance(current, HandlerOp):
                return Checkpoint(current.sym_name.data if current.sym_name is not None else path(current))
            current = current.parent_op()
        return CORE_EXECUTION

    # Following values.

    def provider(self, value: SSAValue) -> SSAValue:
        return provider(value)

    def resource(self, value: SSAValue) -> ResourceOp | None:
        found = self.provider(value)
        return found.owner if isinstance(found, OpResult) and isinstance(found.owner, ResourceOp) else None

    def lease_of(self, value: SSAValue) -> tuple[AccessOp, int] | None:
        """The access that issued a lease, and which of its resources the lease is for."""
        found = self.provider(value)
        if isinstance(found, BlockArgument):
            access = found.owner.parent_op()
            if isinstance(access, AccessOp) and found.index < len(access.uses.data):
                return access, found.index
        return None

    def held(self, value: SSAValue, seen: frozenset[SSAValue] = frozenset()) -> set[str]:
        """The resources a retained value keeps protected: those of its call's live leases and retained inputs."""
        if value in seen:
            return set()
        seen = seen | {value}
        found = self.provider(value)
        if not isinstance(found, OpResult):
            return set()
        owner = found.owner
        resources: set[str] = set()
        if isinstance(owner, CallOp):
            for position in retained_positions(owner):
                given = owner.inputs[position]
                if isinstance(given.type, LeaseType):
                    issued = self.lease_of(given)
                    if issued is not None and AccessMode(given.type.mode.data) in LIVE_MODES:
                        subject = self.resource(issued[0].operands[issued[1]])
                        if subject is not None:
                            resources.add(subject.sym_name.data)
                else:
                    resources |= self.held(given, seen)
            return resources
        for region in owner.regions:
            last = region.block.last_op
            if last is not None and last.has_trait(IsTerminator) and found.index < len(last.operands):
                resources |= self.held(last.operands[found.index], seen)
        return resources

    # Claims (D14, D16): affine, settled exactly once on every path.

    def claims(self) -> None:
        for op in self.strategy.walk():
            semantics = op.get_trait(RegionSemantics)
            if semantics is not None:
                self._captured_claims(op, semantics.describe(op).protocol)
            for region in op.regions:
                self._settle_in(region.block, op)
            if isinstance(op, ParallelOp):
                self._parallel_claims(op)

    def _captured_claims(self, op: Operation, protocol: LifecycleProtocol) -> None:
        claims = [i for i, value in enumerate(op.operands) if isinstance(value.type, ClaimType)]
        item = op.get_trait(ItemSemantics)
        consumed = set(item.describe(op).consumes) if item is not None else set()
        for index in claims:
            if index in consumed:
                continue
            once = protocol in (LifecycleProtocol.SEQUENCE, LifecycleProtocol.CHOICE, LifecycleProtocol.ACCESS, LifecycleProtocol.PARALLEL)
            if protocol == LifecycleProtocol.REPEAT:
                assert isinstance(op, RepeatOp)
                carried = index >= len(op.operands) - op.carry_count.value.data
                once = carried or (op.limit is not None and op.limit.value.data == 1)
            if not once:
                self.problem(
                    "iro-claim-004",
                    op,
                    f"a claim is captured into {op.name}, whose body does not run exactly once, so it would be settled zero or several times",
                    fix="fork the claim, carry it, or settle it before the activity",
                )

    def _settle_in(self, block: Block, owner: Operation) -> None:
        positions = {child: i for i, child in enumerate(block.ops)}
        values = [*block.args, *(r for child in block.ops for r in child.results)]
        parallel_branch = isinstance(owner, ParallelOp)
        for value in values:
            if not isinstance(value.type, ClaimType):
                continue
            settles, borrows = [], []
            for use in value.uses:
                user = use.operation
                item = user.get_trait(ItemSemantics)
                described = item.describe(user) if item is not None else None
                if described is not None and use.index in described.consumes:
                    settles.append(user)
                elif described is not None and use.index in described.borrows:
                    borrows.append(user)
                elif user.has_trait(IsTerminator) or user.get_trait(RegionSemantics) is not None:
                    settles.append(user)  # transferred by a yield or a capture
                else:
                    self.problem(
                        "iro-claim-006",
                        user,
                        f"{user.name} uses a claim without settling or borrowing it",
                        fix="borrow the claim and pass its value",
                    )
            if parallel_branch and isinstance(value, BlockArgument) and not settles and not borrows:
                continue  # another branch holds it; checked at the parallel
            where = value.owner if isinstance(value, OpResult) else owner
            if not settles:
                self.problem(
                    "iro-claim-001",
                    where,
                    f"a claim {'entering' if isinstance(value, BlockArgument) else 'made'} here is never settled",
                    fix="dispose it, filter it, or hand it on with a capture or a yield",
                )
            elif len(settles) > 1:
                self.problem(
                    "iro-claim-002",
                    settles[1],
                    f"a claim is settled {len(settles)} times; a claim is settled exactly once on each path",
                    fix="fork it first when two parts each settle it",
                )
            else:
                late = [b for b in borrows if positions.get(b, -1) > positions.get(settles[0], len(positions))]
                if late:
                    self.problem("iro-claim-003", late[0], "a claim is borrowed after it was settled or handed on")

    def _parallel_claims(self, op: ParallelOp) -> None:
        for index, value in enumerate(op.operands):
            if not isinstance(value.type, ClaimType):
                continue
            users = [region for region in op.regions if region.block.args[index].uses]
            if len(users) != 1:
                self.problem(
                    "iro-claim-005",
                    op,
                    f"a claim captured into @{op.sym_name.data if op.sym_name else path(op)} is used by {len(users)} branches; exactly one branch settles it",
                    fix="fork the claim and capture one fork into each branch",
                )

    # Leases and retained results (D16).

    def leases_and_retained_results(self) -> None:
        for op in self.strategy.walk():
            for value in op.operands:
                if isinstance(value.type, LeaseType) and op.has_trait(IsTerminator):
                    self.problem(
                        "iro-lease-001",
                        op,
                        "a lease is yielded out of its body; a lease never leaves its access",
                        fix="yield what the call computes with the lease, not the lease",
                    )
            for result in op.results:
                if isinstance(result.type, RetainedType):
                    self._retained(result)
        for op in self.strategy.walk():
            for region in op.regions:
                self._updates_after_release(region.block, {})

    def _retained(self, value: OpResult) -> None:
        block = value.owner.parent_block()
        if block is None:
            return
        positions = {child: i for i, child in enumerate(block.ops)}
        releases, others = [], []
        for use in value.uses:
            user = use.operation
            if user.has_trait(IsTerminator):
                parent = user.parent_op()
                semantics = parent.get_trait(RegionSemantics) if parent is not None else None
                if semantics is None or semantics.protocol not in LOCAL_PROTOCOLS:
                    self.problem(
                        "iro-lease-002",
                        user,
                        "a retained result leaves its activity; it may leave its access body, never its activity",
                        fix="use the result inside the activity, or return a detached value",
                    )
                return  # handed to the enclosing result
            semantics = user.get_trait(RegionSemantics)
            if semantics is not None and semantics.describe(user).protocol not in LOCAL_PROTOCOLS:
                self.problem(
                    "iro-lease-002",
                    user,
                    f"a retained result is captured by {user.name}; it is never captured by another activity",
                    fix="use the result where it was made, or return a detached value",
                )
            if user.has_trait(ProductSemantics):
                releases.append(user)
            else:
                others.append(user)
        if len(releases) != 1:
            self.problem(
                "iro-lease-003",
                value.owner,
                f"a retained result has {len(releases)} releases; normalization places exactly one after its last use",
            )
            return
        release = positions[releases[0]]
        last = max((positions[self._statement_in(user, block)] for user in others), default=-1)
        kept_by = [
            use.operation
            for dependent in dependents(value, self.retained)
            for use in dependent.uses
            if use.operation.has_trait(ProductSemantics) and use.operation.parent_block() is block
        ]
        if last > release:
            self.problem("iro-lease-003", releases[0], "a retained result is used after its release")
        elif any(positions[other] > release for other in kept_by):
            self.problem(
                "iro-lease-003",
                releases[0],
                "a retained result is released before a result that keeps it",
                fix="normalize the accepted form again instead of editing the normalized one",
            )
        else:
            self.owe(
                releases[0],
                "the leased state stays protected until this release",
                "the IR runtime",
                names=tuple(sorted(f"@{name}" for name in self.held(value))),
            )

    @staticmethod
    def _statement_in(op: Operation, block: Block) -> Operation:
        current: Operation | None = op
        while current is not None and current.parent_block() is not block:
            current = current.parent_op()
        return current if current is not None else op

    def _updates_after_release(self, block: Block, inherited: dict[SSAValue, set[str]]) -> None:
        """An owner update of leased state waits for every retained result that keeps it protected."""
        live = dict(inherited)
        for child in block.ops:
            if isinstance(child, ReleaseOp):
                live.pop(child.product, None)
                continue
            for inner in child.walk():
                if not isinstance(inner, AccessOp):
                    continue
                for use in uses_of(inner):
                    subject = self.resource(use.subject)
                    if use.mode not in UPDATE_MODES or subject is None:
                        continue
                    blocking = [v for v, held in live.items() if subject.sym_name.data in held]
                    if blocking:
                        self.problem(
                            "iro-lease-004",
                            inner,
                            f"@{subject.sym_name.data} is updated while a retained result still keeps it protected",
                            names=(f"@{subject.sym_name.data}",),
                            fix="use the retained result up before the update, so its release comes first",
                        )
            for result in child.results:
                if isinstance(result.type, RetainedType) and any(u.operation.has_trait(ProductSemantics) for u in result.uses):
                    live[result] = self.held(result)

    # Accesses (D14, D16).

    def accesses(self) -> None:
        for op in self.strategy.walk():
            semantics = op.get_trait(ResourceSemantics)
            if semantics is None:
                continue
            contract = semantics.describe(op)
            for use in contract.uses:
                subject = self.resource(use.subject)
                if subject is None:
                    self.problem("iro-access-003", op, "an access names something that is not a declared resource")
                    continue
                name = f"@{subject.sym_name.data}"
                if use.selection == VersionSelection.AT_REQUEST and self.checkpoint(op) == CORE_EXECUTION:
                    self.problem(
                        "iro-access-001",
                        op,
                        f"{name} is read at_request outside a handler; only requested work has a request version",
                        names=(name,),
                        fix="read current_at_start here, or move the access into a handler",
                    )
                self._nested(op, use.mode, subject)
                via = f" through route @{use.route}" if use.route else f" as its {use.view} view" if use.view else ""
                self.owe(
                    op,
                    f"{use.mode} lease of {name}{via} at {use.selection}",
                    "the resource's owner",
                    names=(name,),
                )

    def _nested(self, op: Operation, mode: AccessMode, subject: ResourceOp) -> None:
        current = op.parent_op()
        while current is not None:
            if isinstance(current, AccessOp):
                for use in uses_of(current):
                    if self.resource(use.subject) is not subject:
                        continue
                    conflicts = (use.mode == AccessMode.PROTECTED_CURRENT and mode in UPDATE_MODES) or (
                        use.mode in UPDATE_MODES and mode in LIVE_MODES
                    )
                    if conflicts:
                        name = f"@{subject.sym_name.data}"
                        self.problem(
                            "iro-access-002",
                            op,
                            f"this {mode} access to {name} waits for the enclosing {use.mode} access to end, which waits for this one",
                            names=(name,),
                            fix="take both uses in one access, or end the outer access first",
                        )
            current = current.parent_op()

    # Calls (D16).

    def calls(self) -> None:
        pinned = set(symbols(self.strategy, "uses"))
        for op in self.strategy.walk():
            if op.get_trait(RealizationSemantics) is not None:
                implementation = symbol(op.properties["implementation"])
                if implementation not in pinned:
                    self.problem(
                        "iro-call-001",
                        op,
                        f"@{implementation} is not pinned in the strategy's uses",
                        names=(f"@{implementation}",),
                        fix=f"add @{implementation} with its version to uses(...)",
                    )
                for value in op.operands:
                    if isinstance(value.type, ResourceType):
                        self.problem(
                            "iro-call-002",
                            op,
                            "a call receives a resource directly; it reaches a resource through a lease",
                            fix="call it inside an access and pass the lease",
                        )
                for value in op.results:
                    if isinstance(value.type, ClaimType | StreamType | LeaseType | ResourceType):
                        self.problem("iro-call-003", op, "a call returns values, not claims, streams, leases or resources")
            if isinstance(op, ResourceOp):
                if op.materialize is not None and symbol(op.materialize) not in pinned:
                    implementation = symbol(op.materialize)
                    self.problem(
                        "iro-call-001",
                        op,
                        f"@{implementation} materializes @{op.sym_name.data} and is not pinned in the strategy's uses",
                        names=(f"@{implementation}",),
                        fix=f"add @{implementation} with its version to uses(...)",
                    )
                kind = op.handle.type.kind.data if isinstance(op.handle.type, ResourceType) else "resource"
                how = f"materialized by @{symbol(op.materialize)}" if op.materialize is not None else "supplied by its owner"
                self.owe(
                    op,
                    f"{kind} @{op.sym_name.data} bound, {how}",
                    "the run authority",
                    names=(f"@{op.sym_name.data}",),
                    checkpoint=CORE_EXECUTION,
                )

    # Lifecycles: requests, stops and settlement (D14).

    def _waits(self) -> Waits:
        waits = Waits()

        def visit(op: Operation) -> None:
            waits.add((op, "settled"), (op, "entry"))
            for region in op.regions:
                previous: Operation | None = None
                for child in region.block.ops:
                    waits.add((child, "entry"), (op, "entry"))
                    waits.add((op, "settled"), (child, "settled"))
                    if previous is not None and not isinstance(op, ParallelOp):
                        waits.add((child, "entry"), (previous, "settled"))
                    previous = child
                    visit(child)

        visit(self.strategy)
        for op in self.strategy.walk():
            if isinstance(op, HandlerOp):
                for name in symbols(op, "sources"):
                    source = self.named.get(name)
                    if source is not None:
                        waits.add((op, "settled"), (source, "settled"))
            if isinstance(op, StopOp):
                source, target = self.named.get(symbol(op.source)), self.named.get(symbol(op.target))
                if source is not None:
                    waits.add((op, "settled"), (source, "settled"))
                if target is not None:
                    waits.add((target, "settled"), (op, "settled"))
        return waits

    def lifecycles(self) -> None:
        cycle = self.waits.cycle()
        if cycle is not None:
            steps = " -> ".join(f"{self._label(op)} {point}" for op, point in cycle)
            self.problem(
                "iro-life-005",
                cycle[0][0],
                f"these points wait for each other: {steps}",
                fix="stop or request after an activity outside this circle",
            )
        producers = [op for op in self.strategy.walk() if isinstance(op, ProducerOp)]
        stops: dict[Operation, list[StopOp]] = defaultdict(list)
        for op in self.strategy.walk():
            control = op.get_trait(CompletionSemantics)
            if control is None:
                continue
            described = control.describe(op)
            source, target = self.named.get(described.source), self.named.get(described.target)
            if source is None or target is None:
                continue
            if isinstance(op, RequestOp):
                if not source.is_ancestor(op):
                    self.problem(
                        "iro-life-001",
                        op,
                        f"a request from @{described.source} is made outside @{described.source}",
                        names=(f"@{described.source}",),
                        fix=f"make the request inside @{described.source}",
                    )
                self.owe(
                    op,
                    f"registered before @{described.source} reaches {described.boundary}; kept through an orderly stop",
                    "the IR runtime",
                    names=(f"@{described.target}",),
                )
            elif isinstance(op, StopOp):
                if not isinstance(target, ProducerOp):
                    self.problem(
                        "iro-life-002",
                        op,
                        f"a stop closes a producer's admission; @{described.target} is {target.name}",
                        names=(f"@{described.target}",),
                    )
                    continue
                stops[target].append(op)
                self.owe(
                    op,
                    f"@{described.target} stops admitting once @{described.source} settles; its admitted items still settle",
                    "the IR runtime",
                    names=(f"@{described.target}", f"@{described.source}"),
                )
        for handler in (op for op in self.strategy.walk() if isinstance(op, HandlerOp)):
            named = symbols(handler, "sources")
            sources = ", ".join(f"@{name}" for name in named) + (" settle" if len(named) > 1 else " settles")
            self.owe(
                handler,
                f"settles after {sources}, with every accepted request disposed",
                "the IR runtime",
                names=(f"@{handler.sym_name.data}",) if handler.sym_name is not None else (),
                checkpoint=CORE_EXECUTION,
            )
        for producer in producers:
            name = f"@{producer.sym_name.data}" if producer.sym_name is not None else path(producer)
            if not stops[producer]:
                self.problem(
                    "iro-life-003",
                    producer,
                    f"nothing stops {name}, so it never settles and neither does the strategy",
                    names=(name,),
                    fix=f"add iro.stop {name} after the activity that consumes its items",
                )
            elif not any(self._always_runs(stop, producer) for stop in stops[producer]):
                self.problem(
                    "iro-life-003",
                    producer,
                    f"every stop of {name} may not run, inside a choice, a handler, a producer or a repeat that may not run once,"
                    f" so {name} may never settle",
                    names=(name,),
                    fix=f"stop {name} from a statement that always runs",
                )
            for stop in stops[producer]:
                self._stop_coverage(stop, producer)

    def _always_runs(self, stop: StopOp, producer: ProducerOp) -> bool:
        """Whether the stop runs whenever the producer does: nothing between them chooses or waits for requests."""
        for ancestor in self._ancestors(stop):
            if ancestor.is_ancestor(producer):
                return True
            semantics = ancestor.get_trait(RegionSemantics)
            if semantics is None:
                continue
            protocol = semantics.protocol
            if protocol in (LifecycleProtocol.CHOICE, LifecycleProtocol.HANDLER, LifecycleProtocol.PRODUCER, LifecycleProtocol.FILTER):
                return False
            if isinstance(ancestor, RepeatOp) and (ancestor.limit is None or ancestor.limit.value.data < 1):
                return False
        return True

    def _stop_coverage(self, stop: StopOp, producer: ProducerOp) -> None:
        source = self.named.get(symbol(stop.source))
        if source is None:
            return
        stream = self.provider(producer.operands[0])
        for take in (op for op in self.strategy.walk() if isinstance(op, TakeOp)):
            if self.provider(take.source) is not stream or self.waits.requires((source, "settled"), (take, "settled")):
                continue
            taker = self._activity(take)
            handler = next((op for op in self._ancestors(take) if isinstance(op, HandlerOp)), None)
            stream_name = (
                f"@{stream.owner.sym_name.data}" if isinstance(stream, OpResult) and isinstance(stream.owner, StreamOp) else "its stream"
            )
            pending = f"{taker}'s pending request" if handler is not None else taker
            self.problem(
                "iro-life-004",
                stop,
                f"stop @{symbol(stop.target)} after @{symbol(stop.source)} does not cover {pending}; {taker} still takes from {stream_name}.",
                names=(f"@{symbol(stop.target)}", taker),
                fix=f"stop @{symbol(stop.target)} after an activity that settles after {taker}, or give {taker} its own stream.",
            )

    @staticmethod
    def _ancestors(op: Operation) -> Iterator[Operation]:
        current = op.parent_op()
        while current is not None:
            yield current
            current = current.parent_op()

    def _activity(self, op: Operation) -> str:
        for ancestor in self._ancestors(op):
            name = sym_name(ancestor)
            if name is not None and not isinstance(ancestor, StrategyOp):
                return f"@{name}"
        return path(op)

    def _label(self, op: Operation) -> str:
        name = sym_name(op)
        return f"@{name}" if name is not None else path(op)

    # Streams (D14).

    def flows(self) -> None:
        for stream in (op for op in self.strategy.walk() if isinstance(op, StreamOp)):
            value = stream.endpoint
            producers = [op for op in self.strategy.walk() if isinstance(op, ProducerOp) and self.provider(op.operands[0]) is value]
            takes = [op for op in self.strategy.walk() if isinstance(op, TakeOp) and self.provider(op.source) is value]
            name = f"@{stream.sym_name.data}"
            if len(producers) > 1 or (takes and not producers):
                self.problem(
                    "iro-flow-001",
                    stream,
                    f"{name} has {len(producers)} producers; a stream has exactly one",
                    names=(name,),
                    fix="give each producer its own stream" if producers else f"add a producer into {name}",
                )
                continue
            contract = FlowSemantics().describe(stream)
            for producer in producers:
                read = {
                    resource.sym_name.data
                    for access in producer.walk()
                    if isinstance(access, AccessOp)
                    for use in uses_of(access)
                    if (resource := self.resource(use.subject)) is not None
                }
                for owner in contract.versions.owners:
                    if owner not in read:
                        self.problem(
                            "iro-flow-002",
                            stream,
                            f"{name} admits by @{owner}'s version, and its producer never reads @{owner}",
                            names=(name, f"@{owner}"),
                            fix="admit by an owner the producer reads, or admit any_with_provenance",
                        )
            if takes:
                self.owe(
                    stream,
                    f"every item admitted from {name} is settled, and admission follows {contract.versions.admission}",
                    "the IR runtime",
                    names=(name,),
                    checkpoint=CORE_EXECUTION,
                )

    # Decisions (D14).

    def _decisions(self) -> Iterator[tuple[Operation, Decision]]:
        """Each decision: calls, requests and stops, stream admission and access version choice."""
        for op in self.strategy.walk():
            if op.has_trait(DecisionSemantics):
                yield op, DecisionSemantics().describe(op)
            elif op.has_trait(FlowSemantics):
                yield op, FlowSemantics().describe(op).decision
            elif op.has_trait(ResourceSemantics):
                yield op, ResourceSemantics().describe(op).decision

    def rank_agreement(self) -> None:
        for op, decision in self._decisions():
            rank = decision.rank
            if rank == RankRule.DISTRIBUTED:
                self.owe(op, "decided once and distributed before any dependent collective work", "the IR runtime")
            elif rank != RankRule.REPLICATED:
                self.problem(
                    "iro-decide-001",
                    op,
                    f"on several ranks this decision must agree, and it is declared {rank}",
                    fix="declare rank_rule(replicated_deterministic) or rank_rule(decided_once_and_distributed)",
                )

    def overlap(self, first: Operation, second: Operation) -> bool:
        """Whether two statements can run at the same time."""
        if first is second or first.is_ancestor(second) or second.is_ancestor(first):
            return False
        return not (
            self.waits.requires((first, "entry"), (second, "settled")) or self.waits.requires((second, "entry"), (first, "settled"))
        )

    def reproducibility(self) -> tuple[Reproducibility, tuple[Contributor, ...]]:
        contributors = [
            Contributor(
                path(op), d.reproducibility, "declared" if d.reproducibility != Reproducibility.UNKNOWN else "not declared", d.assumptions
            )
            for op, d in self._decisions()
        ]
        updates = [
            (op, self.resource(use.subject))
            for op in self.strategy.walk()
            if isinstance(op, AccessOp)
            for use in uses_of(op)
            if use.mode in UPDATE_MODES
        ]
        for op in self.strategy.walk():
            if not isinstance(op, AccessOp):
                continue
            for use in uses_of(op):
                if use.mode in UPDATE_MODES or use.selection == VersionSelection.PUBLISHED_INITIAL:
                    continue
                subject = self.resource(use.subject)
                if any(target is subject and self.overlap(op, writer) for writer, target in updates):
                    contributors.append(
                        Contributor(
                            path(op),
                            Reproducibility.MAY_DEPEND,
                            f"which version of @{subject.sym_name.data if subject else '?'} it reads can depend on when an update runs",
                        )
                    )
        takes: dict[SSAValue, list[TakeOp]] = defaultdict(list)
        for op in self.strategy.walk():
            if isinstance(op, TakeOp):
                takes[self.provider(op.source)].append(op)
        for stream, found in takes.items():
            if any(self.overlap(a, b) for i, a in enumerate(found) for b in found[i + 1 :]):
                owner = stream.owner if isinstance(stream, OpResult) else None
                contributors.append(
                    Contributor(
                        path(owner) if owner is not None else "?",
                        Reproducibility.MAY_DEPEND,
                        "which activity takes which item can depend on scheduling",
                    )
                )
        for stream in (op for op in self.strategy.walk() if isinstance(op, StreamOp)):
            policy = FlowSemantics().describe(stream).versions
            if policy.admission == Admission.ANY_WITH_PROVENANCE:
                continue
            owners = {self.named.get(owner) for owner in policy.owners}
            racing = [
                owner
                for take in takes.get(stream.endpoint, [])
                for writer, owner in updates
                if owner is not None and owner in owners and self.overlap(take, writer)
            ]
            if racing:
                contributors.append(
                    Contributor(
                        path(stream),
                        Reproducibility.MAY_DEPEND,
                        f"whether an item is admitted can depend on when @{racing[0].sym_name.data} is updated",
                    )
                )
        classes = {c.reproducibility for c in contributors}
        overall = (
            Reproducibility.UNKNOWN
            if Reproducibility.UNKNOWN in classes
            else Reproducibility.MAY_DEPEND
            if Reproducibility.MAY_DEPEND in classes
            else Reproducibility.INDEPENDENT
        )
        return overall, tuple(contributors)


__all__ = ["CORE_EXECUTION", "Checkpoint", "Contributor", "Obligation", "Obligations", "Verification", "verify"]
