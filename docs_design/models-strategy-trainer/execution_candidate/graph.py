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


type Work = Call | Join | Differentiate | Advance | TwoPass | Replace


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


type Scope = Block | Repeat | Producer | Requested | Together


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


def inputs(work: Work) -> tuple[str, ...]:
    if isinstance(work, (Call, TwoPass)):
        return work.inputs
    if isinstance(work, Differentiate):
        return (work.loss,)
    if isinstance(work, Advance):
        return (work.contribution,)
    if isinstance(work, Replace):
        return (work.proposal,)
    if isinstance(work, Join):
        return ()
    raise Rejected(f"Unknown work kind {type(work).__name__}")


def outputs(work: Work) -> tuple[Output, ...]:
    if isinstance(work, (Call, Join)):
        return work.outputs
    if isinstance(work, (Differentiate, TwoPass)):
        return (Output(work.output),)
    if isinstance(work, (Advance, Replace)):
        return ()
    raise Rejected(f"Unknown work kind {type(work).__name__}")


def ordered(block: Block) -> tuple[Work, ...]:
    """Shared static wiring check and stable topological order, not a loop."""
    by_name = {work.name: work for work in block.work}
    if len(by_name) != len(block.work):
        raise Rejected(f"{block.name}: duplicate work address")
    producers: dict[str, str] = {}
    for work in block.work:
        for output in outputs(work):
            if output.name in producers or output.name == "choice":
                raise Rejected(f"{block.name}: duplicate value {output.name}")
            producers[output.name] = work.name
    dependencies: dict[str, set[str]] = {}
    for work in block.work:
        deps = set(work.after)
        for value in inputs(work):
            if value != "choice":
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
        states, channels = unique(filing.states), unique(filing.channels)
        for unit in units.values():
            if not unit.members or any(member[0] not in participants for member in unit.members):
                raise Rejected(f"{unit.name}: missing training subject")
        if any(channel.capacity <= 0 for channel in channels.values()):
            raise Rejected("Handoff capacity must be bounded and positive")
        for state in states.values():
            if set(state.dependencies) - participants.keys():
                raise Rejected(f"{state.name}: unknown state dependency")

        scopes: dict[str, Scope] = {}
        producers: dict[str, str] = {}
        joins: set[str] = set()

        def block_check(block: Block):
            if any(not isinstance(work, (Call, Join, Differentiate, Advance, TwoPass, Replace)) for work in block.work):
                raise Rejected(f"{block.name}: unknown work kind")
            contributions: dict[str, tuple[str, ...]] = {}
            phase_owners: dict[str, str] = {}
            advancements: set[str] = set()
            for work in ordered(block):
                uses = getattr(work, "uses", ())
                if any(use.participant not in participants for use in uses):
                    raise Rejected(f"{work.name}: unknown participant use")
                if isinstance(work, Call) and set(work.states) - states.keys():
                    raise Rejected(f"{work.name}: unknown owned state")
                if isinstance(work, Join):
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

        def visit(scope: Scope):
            if not isinstance(scope, (Block, Repeat, Producer, Requested, Together)):
                raise Rejected(f"Unknown scope kind {type(scope).__name__}")
            if scope.name in scopes:
                raise Rejected(f"Duplicate scope {scope.name}")
            scopes[scope.name] = scope
            if isinstance(scope, Together):
                for child in scope.children:
                    visit(child)
            elif isinstance(scope, Block):
                block_check(scope)
            elif isinstance(scope, Repeat):
                if scope.state not in states or not scope.blocks:
                    raise Rejected(f"{scope.name}: missing policy state or alternatives")
                if any(not isinstance(block, Block) for block in scope.blocks):
                    raise Rejected(f"{scope.name}: this candidate requires Block alternatives")
                for block in scope.blocks:
                    visit(block)
            elif isinstance(scope, Requested):
                if not isinstance(scope.block, Block):
                    raise Rejected(f"{scope.name}: this candidate requires a Block capability body")
                visit(scope.block)
            elif isinstance(scope, Producer):
                if set(scope.states) - states.keys() or any(use.participant not in participants for use in scope.uses):
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
        )
