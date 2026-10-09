"""An isolated whole-run language candidate, not the production authoring API.

The tree gives work its lifetime. Value references and ``after`` links give
the work inside a block its relationships. Preparation, not the author,
chooses an executable ordering and binds the implementation calls.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal
from uuid import uuid4

from torch import nn


class Rejected(ValueError):
    """Known authored incompatibility before realization/invocation."""


class OutOfBounds(RuntimeError):
    """Invoked selected behavior exceeded its accepted runtime authority."""


class NotReady(RuntimeError):
    """Accepted work lacks current concrete evidence or usable state."""


@dataclass(frozen=True)
class Use:
    participant: str
    access: Literal["read", "write"] = "read"
    view: str | None = None


@dataclass(frozen=True)
class View:
    """A prepared invocation role, not another participant or optimizer target."""

    name: str
    participant: str
    gradients: bool = True
    evaluation: bool = False


@dataclass(frozen=True)
class Participant:
    name: str
    load: Callable[[], nn.Module]
    verify: Callable[[nn.Module], None]


@dataclass(frozen=True)
class Unit:
    name: str
    members: tuple[tuple[str, str], ...]
    learning_rate: float = 0.05
    clip_norm: float | None = None


@dataclass(frozen=True)
class OwnedState:
    name: str
    initialize: Callable[[], object]
    dependencies: tuple[str, ...] = ()
    on_replace: Literal["preserve", "reset", "reject"] = "reject"


@dataclass(frozen=True)
class Output:
    name: str
    type: type = object


@dataclass(frozen=True)
class Call:
    name: str
    implementation: Callable[..., object]
    inputs: tuple[str, ...] = ()
    outputs: tuple[Output, ...] = ()
    uses: tuple[Use, ...] = ()
    states: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    evaluation: bool = False
    retained: str | None = None


@dataclass(frozen=True)
class Join:
    name: str
    channels: tuple[str, ...]
    outputs: tuple[Output, ...]
    admit: Callable[..., bool]
    uses: tuple[Use, ...] = ()
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class Differentiate:
    name: str
    loss: str
    units: tuple[str, ...]
    output: str
    after: tuple[str, ...] = ()
    retained: str | None = None


@dataclass(frozen=True)
class Advance:
    name: str
    contribution: str
    units: tuple[str, ...]
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class TwoPass:
    name: str
    implementation: Callable[..., object]
    inputs: tuple[str, ...]
    unit: str
    output: str
    uses: tuple[Use, ...] = ()
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class Replace:
    name: str
    proposal: str
    participant: str
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class Retain:
    """Last-use protection or an explicit isolated CPU state copy."""

    name: str
    participants: tuple[str, ...]
    units: tuple[str, ...]
    output: str
    isolated: bool = False
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class Release:
    name: str
    retained: str
    after: tuple[str, ...] = ()


type Work = Call | Join | Differentiate | Advance | TwoPass | Replace | Retain | Release


@dataclass(frozen=True)
class Block:
    name: str
    work: tuple[Work, ...]


@dataclass(frozen=True)
class Choice:
    block: str
    key: Hashable = ""
    payload: object = None


@dataclass(frozen=True)
class Completion:
    scope: str
    block: str
    key: Hashable
    invocation: int
    position: int


@dataclass(frozen=True)
class Repeat:
    name: str
    blocks: tuple[Block, ...]
    policy: Callable[[Mapping[str, int], object], Choice | None]
    state: str


@dataclass(frozen=True)
class Producer:
    name: str
    channels: tuple[str, ...]
    implementation: Callable[..., Awaitable[None]]
    uses: tuple[Use, ...] = ()
    states: tuple[str, ...] = ()
    stop_when: str | None = None
    remaining: Literal["discard"] | None = None
    cleanup: Callable[..., Awaitable[None]] | None = None


@dataclass(frozen=True)
class Requested:
    name: str
    source: str
    block: Block
    request: Callable[[Completion], Choice | None]
    enabled: bool = True


@dataclass(frozen=True)
class Together:
    name: str
    children: tuple[Scope, ...]


@dataclass(frozen=True)
class Sequence:
    """Ordered blocks with explicit value links crossing their boundaries.

    This target supports one first-order window per unit in the sequence.
    It does not prescribe a universal training action or optimizer cadence.
    """

    name: str
    blocks: tuple[Block, ...]


type Scope = Block | Repeat | Producer | Requested | Together | Sequence


@dataclass(frozen=True)
class Channel:
    name: str
    capacity: int


@dataclass(frozen=True)
class Run:
    root: Scope
    participants: tuple[Participant, ...] = ()
    units: tuple[Unit, ...] = ()
    states: tuple[OwnedState, ...] = ()
    channels: tuple[Channel, ...] = ()
    views: tuple[View, ...] = ()


@dataclass(frozen=True)
class Accepted:
    """Retained meaning and identities; no live models or readiness claim."""

    authority: str
    contract: Contract
    root: Scope
    participants: Mapping[str, Participant]
    units: Mapping[str, Unit]
    states: Mapping[str, OwnedState]
    channels: Mapping[str, Channel]
    participant_refs: Mapping[str, str]
    unit_refs: Mapping[str, str]
    views: Mapping[str, View]


def inputs(work: Work) -> tuple[str, ...]:
    if isinstance(work, (Call, TwoPass)):
        return (*work.inputs, work.retained) if isinstance(work, Call) and work.retained is not None else work.inputs
    if isinstance(work, Differentiate):
        return (work.loss, work.retained) if work.retained is not None else (work.loss,)
    if isinstance(work, Advance):
        return (work.contribution,)
    if isinstance(work, Replace):
        return (work.proposal,)
    if isinstance(work, Release):
        return (work.retained,)
    if isinstance(work, (Join, Retain)):
        return ()
    raise Rejected(f"Unknown work kind {type(work).__name__}")


def outputs(work: Work) -> tuple[Output, ...]:
    if isinstance(work, (Call, Join)):
        return work.outputs
    if isinstance(work, (Differentiate, TwoPass, Retain)):
        return (Output(work.output),)
    if isinstance(work, (Advance, Replace, Release)):
        return ()
    raise Rejected(f"Unknown work kind {type(work).__name__}")


def ordered(block: Block, available: tuple[str, ...] = ()) -> tuple[Work, ...]:
    """Shared static wiring check and stable topological order, not a loop."""
    by_name = {work.name: work for work in block.work}
    if len(by_name) != len(block.work):
        raise Rejected(f"{block.name}: duplicate work address")
    producers: dict[str, str] = {}
    for work in block.work:
        for output in outputs(work):
            if output.name in producers or output.name in available or output.name == "choice":
                raise Rejected(f"{block.name}: duplicate value {output.name}")
            producers[output.name] = work.name
    dependencies: dict[str, set[str]] = {}
    for work in block.work:
        deps = set(work.after)
        for value in inputs(work):
            if value != "choice" and value not in available:
                if value not in producers:
                    raise Rejected(f"{work.name}: missing value {value}")
                deps.add(producers[value])
        if deps - by_name.keys():
            raise Rejected(f"{work.name}: missing ordering dependency")
        dependencies[work.name] = deps
    result: list[Work] = []
    done: set[str] = set()
    while len(done) < len(by_name):
        ready = [work for work in block.work if work.name not in done and dependencies[work.name] <= done]
        if not ready:
            raise Rejected(f"{block.name}: cyclic work dependencies")
        result.extend(ready)
        done.update(work.name for work in ready)
    return tuple(result)


def sequence_order(scope: Sequence) -> tuple[tuple[Block, tuple[Work, ...]], ...]:
    available: list[str] = []
    result = []
    for block in scope.blocks:
        work = ordered(block, tuple(available))
        result.append((block, work))
        available.extend(output.name for item in work for output in outputs(item))
    return tuple(result)


def block_uses(block: Block, units: Mapping[str, Unit]) -> tuple[Use, ...]:
    """One source for this target's conservative whole-block protection."""
    uses: list[Use] = []
    for work in block.work:
        uses.extend(getattr(work, "uses", ()))
        destinations = work.units if isinstance(work, (Differentiate, Advance)) else (work.unit,) if isinstance(work, TwoPass) else ()
        for unit in destinations:
            uses.extend(Use(member[0], "write") for member in units[unit].members)
        if isinstance(work, Call) and work.evaluation:
            uses.extend(Use(use.participant, "write") for use in work.uses)
    return tuple(uses)


def check_target(scopes: Mapping[str, Scope], units: Mapping[str, Unit], active: set[str]) -> None:
    """Known CPU/asyncio target limits, NOT universal language constraints.

    Prefix readiness holds no lease. Later joins do, so conservatively reject
    direct/indirect declared lease/source cycles. Private Python scheduling is
    not analyzed and a clean result is not a global liveness guarantee.
    """
    producers = {channel: scope for scope in scopes.values() if isinstance(scope, Producer) for channel in scope.channels}
    accesses: dict[str, dict[str, str]] = {}
    edges: dict[str, set[str]] = {}
    across = {block.name: work for scope in scopes.values() if isinstance(scope, Sequence) for block, work in sequence_order(scope)}
    for block in scopes.values():
        if not isinstance(block, Block) or block.name not in active:
            continue
        uses = block_uses(block, units)
        states = {name for item in block.work if isinstance(item, Call) for name in item.states}
        if any(isinstance(item, Replace) for item in block.work) and (uses or states):
            raise Rejected("This target publishes replacement only outside protected numerical/state work")
        access = dict.fromkeys((use.participant for use in uses), "read")
        access.update(dict.fromkeys((use.participant for use in uses if use.access == "write"), "write"))
        accesses[block.name] = access
        edges[block.name] = set()
        prefix = True
        for item in across.get(block.name, ()) or ordered(block):
            if not isinstance(item, Join):
                prefix = False
            elif not prefix:
                edges[block.name].update(producers[channel].name for channel in item.channels)
    for producer in producers.values():
        edges[producer.name] = {
            block
            for block, access in accesses.items()
            if any(use.participant in access and (use.access == "write" or access[use.participant] == "write") for use in producer.uses)
        }

    def acyclic(dependencies: Mapping[str, set[str]], description: str) -> None:
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise Rejected(f"Potential {description} in this target: {node}")
            if node in done:
                return
            visiting.add(node)
            for dependency in dependencies[node]:
                visit(dependency)
            visiting.remove(node)
            done.add(node)

        for node in dependencies:
            visit(node)

    acyclic(edges, "lease/source wait cycle")
    # Full lifetime completion is NOT input readiness: mixing these with the
    # lease edges would falsely reject ordinary producer/consumer relationships.
    completion = {name: set() for name in active}
    for name in active:
        scope = scopes[name]
        if isinstance(scope, Together):
            completion[name].update(child.name for child in scope.children if child.name in active)
        elif isinstance(scope, Sequence):
            completion[name].update(block.name for block in scope.blocks if block.name in active)
        elif isinstance(scope, Producer) and scope.stop_when is not None:
            completion[name].add(scope.stop_when)
        elif isinstance(scope, Requested) and scope.enabled:
            completion[name].add(scope.source)
    acyclic(completion, "scope-completion wait cycle")


@dataclass(frozen=True)
class Contract:
    """Probe-sized schema supplied by these mechanics before authoring.

    This is not the final training contract representation. It deliberately
    rejects unimplemented meanings instead of pretending that metadata makes
    them executable. Arbitrary Python bodies remain trusted implementations.
    """

    version: str = "candidate-cpu-v1"
    two_pass: bool = False

    def accept(self, filing: Run) -> Accepted:
        if not isinstance(filing, Run):
            raise Rejected("Expected an authored run description")

        def unique(items):
            values = {item.name: item for item in items}
            if len(values) != len(items):
                raise Rejected("Duplicate authored address")
            return MappingProxyType(values)

        participants, units = unique(filing.participants), unique(filing.units)
        states, channels, views = unique(filing.states), unique(filing.channels), unique(filing.views)
        if any(view.participant not in participants for view in views.values()):
            raise Rejected("Unknown participant for execution view")

        def valid_use(use: Use) -> bool:
            return (
                use.participant in participants
                and use.access in ("read", "write")
                and (use.view is None or (use.view in views and views[use.view].participant == use.participant))
            )

        for unit in units.values():
            if not unit.members or any(member[0] not in participants for member in unit.members):
                raise Rejected(f"{unit.name}: missing training subject")
        if any(channel.capacity <= 0 for channel in channels.values()):
            raise Rejected("Handoff capacity must be bounded and positive")
        for state in states.values():
            if set(state.dependencies) - participants.keys():
                raise Rejected(f"{state.name}: unknown state dependency")

        scopes: dict[str, Scope] = {}
        paths: dict[str, tuple[str, ...]] = {}
        lifetimes: set[str] = set()
        active: set[str] = set()
        producers: dict[str, str] = {}
        joins: set[str] = set()

        def block_check(block: Block | Sequence):
            spanning = isinstance(block, Sequence)
            blocks = block.blocks if spanning else (block,)
            if any(
                not isinstance(work, (Call, Join, Differentiate, Advance, TwoPass, Replace, Retain, Release))
                for child in blocks
                for work in child.work
            ):
                raise Rejected(f"{block.name}: unknown work kind")
            groups = sequence_order(block) if spanning else ((block, ordered(block)),)
            work_items = tuple(work for _, group in groups for work in group)
            contributions: dict[str, tuple[str, ...]] = {}
            phase_owners: dict[str, str] = {}
            advancements: set[str] = set()
            retentions: dict[str, Retain] = {}
            released: set[str] = set()
            borrowed: dict[str, set[str]] = {}
            for work in work_items:
                uses = getattr(work, "uses", ())
                if any(not valid_use(use) for use in uses):
                    raise Rejected(f"{work.name}: unknown participant use")
                if not isinstance(work, Call) and any(use.view is not None for use in uses):
                    raise Rejected("This target exposes named execution views only to selected calls")
                dependencies = set().union(*(borrowed.get(value, set()) for value in inputs(work)))
                if dependencies & released:
                    raise Rejected(f"{work.name}: borrowed values outlive released old state")
                retained = work.retained if isinstance(work, (Call, Differentiate, Release)) else None
                if spanning and isinstance(work, TwoPass):
                    raise Rejected("This target's two-pass grant remains block-local, not a Sequence handback")
                if retained is not None and isinstance(work, Call) and work.evaluation:
                    raise Rejected("Retained calls use named synchronous view modes, not the broad evaluation flag")
                if retained is not None:
                    if retained not in retentions or retained in released:
                        raise Rejected(f"{work.name}: retained state is absent or already released")
                    declaration = retentions[retained]
                    if isinstance(work, Call) and any(
                        use.participant not in declaration.participants or use.access != "read" for use in uses
                    ):
                        raise Rejected(f"{work.name}: retained calls require covered read-only views")
                    if isinstance(work, Differentiate) and set(work.units) - set(declaration.units):
                        raise Rejected(f"{work.name}: retained derivative has no accepted member correspondence")
                if isinstance(work, Differentiate) and (
                    (spanning and (retained is None or dependencies != {retained})) or (dependencies and dependencies != {retained})
                ):
                    raise Rejected(f"{work.name}: derivative source needs matching retained correspondence")
                if isinstance(work, Call):
                    if retained is not None:
                        dependencies.add(retained)
                    for output in work.outputs:
                        borrowed[output.name] = set(dependencies)
                if isinstance(work, (Retain, Release)) and not spanning:
                    raise Rejected("This target supports retained lifetimes inside a Sequence")
                if isinstance(work, Retain):
                    if (
                        not work.participants
                        or len(set(work.participants)) != len(work.participants)
                        or len(set(work.units)) != len(work.units)
                        or set(work.participants) - participants.keys()
                        or set(work.units) - units.keys()
                        or any(member[0] not in work.participants for name in work.units for member in units[name].members)
                    ):
                        raise Rejected(f"{work.name}: invalid retained participant/unit coverage")
                    retentions[work.output] = work
                if isinstance(work, Release):
                    released.add(work.retained)
                if isinstance(work, Call) and (set(work.inputs) & (retentions.keys() | contributions.keys())):
                    raise Rejected("Ordinary Python calls cannot receive retained/window coordination handles")
                live = {
                    participant
                    for name, declaration in retentions.items()
                    if name not in released and not declaration.isolated
                    for participant in declaration.participants
                }
                writes = {use.participant for use in uses if use.access == "write"}
                if isinstance(work, (Advance, TwoPass)):
                    names = work.units if isinstance(work, Advance) else (work.unit,)
                    writes.update(member[0] for name in names if name in units for member in units[name].members)
                if writes & live or (isinstance(work, Replace) and live):
                    raise Rejected(f"{work.name}: update/publication precedes protected old-state last use")
                if isinstance(work, Call) and set(work.states) - states.keys():
                    raise Rejected(f"{work.name}: unknown owned state")
                if isinstance(work, Join):
                    if spanning and retentions:
                        raise Rejected("This target does not wait for producer inputs inside a retained sequence")
                    joins.update(work.channels)
                    if not work.channels or len(work.channels) != len(work.outputs) or len(set(work.channels)) != len(work.channels):
                        raise Rejected(f"{work.name}: invalid input join")
                if isinstance(work, Differentiate):
                    contributions[work.output] = work.units
                if isinstance(work, TwoPass):
                    if not self.two_pass:
                        raise Rejected(f"{work.name}: two-pass authority is not offered")
                    contributions[work.output] = (work.unit,)
                if isinstance(work, (Differentiate, Advance)) and (
                    not work.units or len(set(work.units)) != len(work.units) or set(work.units) - units.keys()
                ):
                    raise Rejected(f"{work.name}: invalid unit destinations")
                if isinstance(work, TwoPass) and work.unit not in units:
                    raise Rejected(f"{work.name}: unknown granted unit")
                destinations = work.units if isinstance(work, Differentiate) else (work.unit,) if isinstance(work, TwoPass) else ()
                for destination in destinations:
                    if destination in phase_owners:
                        raise Rejected(f"{work.name}: competing backward ownership for {destination}")
                    phase_owners[destination] = work.name
                if isinstance(work, Advance) and contributions.get(work.contribution) != work.units:
                    raise Rejected(f"{work.name}: unmatched contribution/handback")
                if isinstance(work, Advance):
                    if set(work.units) & advancements:
                        raise Rejected(f"{work.name}: repeated final advancement")
                    advancements.update(work.units)
                if isinstance(work, Replace) and work.participant not in participants:
                    raise Rejected(f"{work.name}: undeclared replacement")
            if phase_owners.keys() != advancements:
                raise Rejected(f"{block.name}: this target requires each local window to reach one final advancement")
            if retentions.keys() != released:
                raise Rejected(f"{block.name}: retained state needs one explicit last-use release")

        def visit(scope: Scope, parents: tuple[str, ...] = (), *, invoked: bool = False, enabled: bool = True):
            if not isinstance(scope, (Block, Repeat, Producer, Requested, Together, Sequence)):
                raise Rejected(f"Unknown scope kind {type(scope).__name__}")
            if scope.name in scopes:
                raise Rejected(f"Duplicate scope {scope.name}")
            scopes[scope.name] = scope
            paths[scope.name] = (*parents, scope.name)
            if not invoked:
                lifetimes.add(scope.name)
            if enabled and not (isinstance(scope, Requested) and not scope.enabled):
                active.add(scope.name)
            if isinstance(scope, Together):
                for child in scope.children:
                    visit(child, paths[scope.name], enabled=enabled)
            elif isinstance(scope, Sequence):
                if not scope.blocks or any(not isinstance(block, Block) for block in scope.blocks):
                    raise Rejected("This candidate requires nonempty Block sequences")
                block_check(scope)
                for block in scope.blocks:
                    visit(block, paths[scope.name], invoked=True, enabled=enabled)
            elif isinstance(scope, Block):
                if not parents or not isinstance(scopes[parents[-1]], Sequence):
                    block_check(scope)
            elif isinstance(scope, Repeat):
                if scope.state not in states or not scope.blocks:
                    raise Rejected(f"{scope.name}: missing policy state or alternatives")
                if any(not isinstance(block, Block) for block in scope.blocks):
                    raise Rejected(f"{scope.name}: this candidate requires Block alternatives")
                for block in scope.blocks:
                    visit(block, paths[scope.name], invoked=True, enabled=enabled)
            elif isinstance(scope, Requested):
                if not isinstance(scope.block, Block):
                    raise Rejected(f"{scope.name}: this candidate requires a Block capability body")
                visit(scope.block, paths[scope.name], invoked=True, enabled=enabled and scope.enabled)
            elif isinstance(scope, Producer):
                if set(scope.states) - states.keys() or any(not valid_use(use) or use.view is not None for use in scope.uses):
                    raise Rejected(f"{scope.name}: invalid provider projection")
                for channel in scope.channels:
                    if channel not in channels or channel in producers:
                        raise Rejected(f"{channel}: missing or competing producer")
                    producers[channel] = scope.name
            else:
                raise Rejected(f"Unsupported scope {type(scope).__name__}")

        visit(filing.root)
        if joins - producers.keys():
            raise Rejected(f"No producer for {sorted(joins - producers.keys())}")
        if channels.keys() - producers.keys():
            raise Rejected("Declared channel has no owning producer")
        for scope in scopes.values():
            if isinstance(scope, Requested) and not isinstance(scopes.get(scope.source), Repeat):
                raise Rejected(f"{scope.name}: unsupported request source {scope.source}")
            if isinstance(scope, Producer):
                if scope.stop_when is None:
                    if scope.remaining is not None:
                        raise Rejected("Remaining-work policy requires an explicit demand lifetime")
                    continue
                target = scope.stop_when
                if scope.remaining != "discard":
                    raise Rejected("This target requires an explicit discard policy at demand end")
                if target not in lifetimes or isinstance(scopes[target], Producer) or target not in active:
                    raise Rejected("Producer stop target must be an active whole-scope lifetime")
                if target in paths[scope.name] or scope.name in paths[target]:
                    raise Rejected("Producer stop target must progress independently")
                for block in scopes.values():
                    if (
                        isinstance(block, Block)
                        and block.name in active
                        and target not in paths[block.name]
                        and any(isinstance(work, Join) and set(work.channels) & set(scope.channels) for work in block.work)
                    ):
                        raise Rejected("End-of-demand policy cannot discard another live consumer's work")
        # Private provider state has one activity owner in this candidate. It
        # is not quietly shared with policy/call owners without a handoff.
        provider_states = [name for scope in scopes.values() if isinstance(scope, Producer) for name in scope.states]
        other_states = {
            name
            for scope in scopes.values()
            if isinstance(scope, Block)
            for work in scope.work
            if isinstance(work, Call)
            for name in work.states
        } | {scope.state for scope in scopes.values() if isinstance(scope, Repeat)}
        if len(set(provider_states)) != len(provider_states) or set(provider_states) & other_states:
            raise Rejected("This target requires separately owned provider continuation state")
        check_target(scopes, units, active)
        authority = uuid4().hex
        return Accepted(
            authority,
            self,
            filing.root,
            participants,
            units,
            states,
            channels,
            MappingProxyType({key: f"{authority}/participant/{uuid4().hex}" for key in participants}),
            MappingProxyType({key: f"{authority}/unit/{uuid4().hex}" for key in units}),
            views,
        )
