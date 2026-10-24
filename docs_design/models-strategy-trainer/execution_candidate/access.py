"""Shared access boundaries and cold wait analysis for the candidate CPU target.

This is derived evidence, not another authored run or a runtime scheduler.
Shared use helpers keep the analysis aligned with actual acquisition boundaries.
Private Python waits, queue protocols and backend internals are not inferred.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .graph import (
    Accepted,
    Advance,
    Block,
    Call,
    Differentiate,
    Join,
    NotReady,
    Producer,
    Release,
    Repeat,
    Replace,
    Requested,
    Retain,
    Scope,
    Sequence,
    Together,
    TwoPass,
    Use,
    Work,
    block_uses,
    ordered,
    sequence_order,
)


def mode_participants(accepted: Accepted, *, selected_workers: bool) -> frozenset[str]:
    """Bind physical mode protection from every source of native overlap.

    Selected-call workers retain this target's existing conservative policy.
    A provider may also own native workers, independently of that pool. Its
    declared source participants therefore need mode protection even when all
    selected calls execute cooperatively. No private worker schedule is inferred.
    """
    if selected_workers:
        return frozenset(accepted.participants)
    result: set[str] = set()

    def visit(scope: Scope):
        if isinstance(scope, Together):
            for child in scope.children:
                visit(child)
        elif isinstance(scope, (Repeat, Sequence)):
            for child in scope.blocks:
                visit(child)
        elif isinstance(scope, Requested) and scope.enabled:
            visit(scope.block)
        elif isinstance(scope, Producer):
            result.update(use.participant for use in scope.uses)

    visit(accepted.root)
    return frozenset(result)


def mode_uses(work: tuple[Work, ...], accepted: Accepted, *, physical_modes: frozenset[str]) -> tuple[Use, ...]:
    """Physical exclusivity, never a numerical-write grant or epoch change.

    Whole-call evaluation may span an await, so always needs exclusion. A
    synchronous named view additionally needs it on native-overlap participants.
    """
    return tuple(
        Use(use.participant, "write")
        for item in work
        if isinstance(item, Call)
        for use in item.uses
        if item.evaluation or (use.participant in physical_modes and use.view is not None and accepted.views[use.view].evaluation)
    )


def retained_exclusions(work: tuple[Work, ...], accepted: Accepted, *, physical_modes: frozenset[str]) -> dict[str, tuple[str, ...]]:
    """Live sources acquire mode exclusion up front; copies own separate views."""
    return {
        item.output: tuple(
            dict.fromkeys(
                use.participant
                for call in work
                if isinstance(call, Call) and call.retained == item.output
                for use in mode_uses((call,), accepted, physical_modes=physical_modes)
            )
        )
        for item in work
        if isinstance(item, Retain) and not item.isolated
    }


def retained_uses(source: Retain, exclusive: tuple[str, ...]) -> tuple[Use, ...]:
    return tuple(Use(name, "write" if name in exclusive else "read") for name in source.participants)


def sequence_uses(item: Work, accepted: Accepted, *, physical_modes: frozenset[str]) -> tuple[Use, ...]:
    """The per-item reservation acquired by sequence execution, before reuse."""
    uses = getattr(item, "uses", ())
    if isinstance(item, Call) and item.retained is not None:
        return ()  # Its source already owns the lifetime and view representation.
    if isinstance(item, (Differentiate, Advance)):
        access = "write" if isinstance(item, Advance) else "read"
        uses = tuple(Use(member[0], access) for unit in item.units for member in accepted.units[unit].members)
    if isinstance(item, TwoPass):
        uses = (*uses, *(Use(member[0], "write") for member in accepted.units[item.unit].members))
    if isinstance(item, Call):
        uses = (*uses, *mode_uses((item,), accepted, physical_modes=physical_modes))
    return uses


def block_access(block: Block, accepted: Accepted, *, physical_modes: frozenset[str]) -> tuple[tuple[Use, ...], tuple[str, ...]]:
    """The ordinary block's one reservation, shared by lowering and analysis."""
    uses = block_uses(block, accepted.units)
    uses += mode_uses(block.work, accepted, physical_modes=physical_modes)
    states = tuple(dict.fromkeys(name for item in block.work if isinstance(item, Call) for name in item.states))
    return uses, states


def prefix_joins(work: tuple[Work, ...]) -> tuple[Join, ...]:
    result = []
    for item in work:
        if not isinstance(item, Join):
            break
        result.append(item)
    return tuple(result)


def participant_access(uses: tuple[Use, ...]) -> dict[str, Literal["read", "write"]]:
    result: dict[str, Literal["read", "write"]] = {}
    for use in uses:
        if result.get(use.participant) != "write":
            result[use.participant] = use.access
    return result


def remaining_uses(uses: tuple[Use, ...], held: Mapping[str, str], *, reuse: bool) -> tuple[Use, ...]:
    """Resolve own-region access without changing another owner's lifetime.

    A work-sized read may borrow existing protection. New retentions instead
    acquire independent lifetimes, so even a read conflicts with an own writer.
    Writes never upgrade/reuse retained state. This rule is also used before
    execution when deriving waits; it neither copies nor reschedules work.
    """
    for name, mode in participant_access(uses).items():
        if name in held and (mode == "write" or (not reuse and held[name] == "write")):
            raise NotReady("Current use cannot upgrade or nest conflicting own retained reservations")
    return tuple(use for use in uses if not reuse or use.participant not in held)


type Access = dict[tuple[str, str], Literal["read", "write"]]


def _access(uses: tuple[Use, ...], states: tuple[str, ...] = ()) -> Access:
    result: Access = {("state", name): "write" for name in states}
    result.update({("participant", name): mode for name, mode in participant_access(uses).items()})
    return result


def _merge(*groups: Access) -> Access:
    result: Access = {}
    for group in groups:
        for key, mode in group.items():
            if result.get(key) != "write":
                result[key] = mode
    return result


@dataclass(frozen=True)
class _Wait:
    activity: str
    where: str
    held: Access
    requested: Access
    sources: tuple[str, ...] = ()


def _waits(accepted: Accepted, *, physical_modes: frozenset[str]) -> tuple[_Wait, ...]:
    points: list[_Wait] = []

    def add(activity, where, held, requested, sources=()):
        points.append(_Wait(activity, where, held, requested, sources))

    def block(scope: Block, activity: str):
        work = ordered(scope)
        held = _access(*block_access(scope, accepted, physical_modes=physical_modes))
        prefix = {item.name for item in prefix_joins(work)}
        for item in work:
            if isinstance(item, Join):
                # Ordinary block prefix readiness precedes its single lease.
                add(activity, f"{scope.name}/{item.name}", {} if item.name in prefix else held, {}, item.channels)
            if isinstance(item, Replace):
                if held:
                    raise NotReady("This target publishes replacement only outside protected numerical/state work")
                add(activity, f"{scope.name}/{item.name}", held, quiescence())
        add(activity, scope.name, {}, held)

    def quiescence() -> Access:
        return _access(tuple(Use(name, "write") for name in accepted.participants), tuple(accepted.states))

    def sequence(scope: Sequence, activity: str):
        work = tuple(item for _, group in sequence_order(scope) for item in group)
        exclusions = retained_exclusions(work, accepted, physical_modes=physical_modes)
        live: dict[str, Access] = {}
        retained: set[str] = set()
        for item in work:
            held = _merge(*live.values())
            participants = {name: mode for (kind, name), mode in held.items() if kind == "participant"}
            where = f"{scope.name}/{item.name}"
            if isinstance(item, Retain):
                uses = retained_uses(item, exclusions.get(item.output, ()))
                remaining_uses(uses, participants, reuse=False)
                requested = _access(uses)
                add(activity, where, held, requested)
                if not item.isolated:
                    live[item.output] = requested
                retained.add(item.output)
            elif isinstance(item, Release):
                live.pop(item.retained, None)
                retained.remove(item.retained)
            else:
                uses = remaining_uses(sequence_uses(item, accepted, physical_modes=physical_modes), participants, reuse=True)
                requested = _access(uses, item.states if isinstance(item, Call) else ())
                if isinstance(item, Replace):
                    if retained:
                        raise NotReady("A sequence cannot wait on its own protected old-state last use")
                    requested = quiescence()
                add(activity, where, held, requested)
                if isinstance(item, Join):
                    # Sequence Join acquires its own uses BEFORE input waiting,
                    # unlike the ordinary block's prefix-readiness path.
                    add(activity, where, _merge(held, requested), {}, item.channels)

    def visit(scope: Scope):
        if isinstance(scope, Together):
            for child in scope.children:
                visit(child)
        elif isinstance(scope, Sequence):
            sequence(scope, scope.name)
        elif isinstance(scope, Block):
            block(scope, scope.name)
        elif isinstance(scope, Repeat):
            add(scope.name, f"{scope.name}/policy", {}, _access((), (scope.state,)))
            for child in scope.blocks:
                if isinstance(child, Sequence):
                    sequence(child, scope.name)
                else:
                    block(child, scope.name)  # Alternatives do not run concurrently.
        elif isinstance(scope, Requested) and scope.enabled:
            block(scope.block, scope.name)
        elif isinstance(scope, Producer):
            # Source access precedes publication when used. Its internal queue
            # and acquisition protocol remain private, not invented phases.
            add(scope.name, scope.name, {}, _access(scope.uses))

    visit(accepted.root)
    return tuple(points)


def check_waits(accepted: Accepted, *, physical_modes: frozenset[str]) -> None:
    """Reject potential declared wait cycles in this realization before loading.

    Different activities here are leaves of Together; ordered sequence blocks
    and repeated alternatives deliberately share one activity. A cycle must
    have at most one simultaneous wait point from each activity. No execution
    order is changed, no selected policy is evaluated, and no check runs per step.
    """
    points = _waits(accepted, physical_modes=physical_modes)
    producers: dict[str, str] = {}

    def sources(scope: Scope):
        if isinstance(scope, Together):
            for child in scope.children:
                sources(child)
        elif isinstance(scope, Producer):
            producers.update(dict.fromkeys(scope.channels, scope.name))

    sources(accepted.root)
    edges: list[list[int]] = [[] for _ in points]
    for index, point in enumerate(points):
        owners = {producers[name] for name in point.sources}
        for other_index, other in enumerate(points):
            if point.activity == other.activity:
                continue
            conflict = any(key in other.held and (mode == "write" or other.held[key] == "write") for key, mode in point.requested.items())
            if conflict or other.activity in owners:
                edges[index].append(other_index)

    # Remove acyclic tails first. A routine acyclic run should not enumerate
    # every path through its possible activity phases just to prove no cycle.
    incoming: list[list[int]] = [[] for _ in points]
    for index, following in enumerate(edges):
        for target in following:
            incoming[target].append(index)
    degrees = [len(following) for following in edges]
    terminal = [index for index, degree in enumerate(degrees) if not degree]
    candidates = set(range(len(points)))
    while terminal:
        index = terminal.pop()
        candidates.remove(index)
        for previous in incoming[index]:
            degrees[previous] -= 1
            if not degrees[previous]:
                terminal.append(previous)

    def visit(index: int, path: tuple[int, ...], activities: frozenset[str]):
        for following in edges[index]:
            if following not in candidates:
                continue
            if following == path[0]:
                chain = " -> ".join(points[node].where for node in (*path, following))
                raise NotReady(f"Potential reservation wait cycle in CPU target: {chain}")
            owner = points[following].activity
            if owner not in activities:
                visit(following, (*path, following), activities | {owner})

    for start in sorted(candidates):
        visit(start, (start,), frozenset((points[start].activity,)))
