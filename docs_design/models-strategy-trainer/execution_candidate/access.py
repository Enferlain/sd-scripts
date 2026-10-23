"""Preparation-only access analysis for the candidate's CPU worker target.

This is derived evidence, not another authored run or a runtime scheduler.
Shared use helpers keep the analysis aligned with actual acquisition boundaries.
Private Python waits, queue protocols and backend internals are not inferred.
"""

from __future__ import annotations

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


def mode_uses(work: tuple[Work, ...], accepted: Accepted) -> tuple[Use, ...]:
    """Physical exclusivity, never a numerical-write grant or epoch change."""
    return tuple(
        Use(use.participant, "write")
        for item in work
        if isinstance(item, Call)
        for use in item.uses
        if item.evaluation or (use.view is not None and accepted.views[use.view].evaluation)
    )


def retained_exclusions(work: tuple[Work, ...], accepted: Accepted) -> dict[str, tuple[str, ...]]:
    """Live sources acquire mode exclusion up front; copies own separate views."""
    return {
        item.output: tuple(
            dict.fromkeys(
                use.participant
                for call in work
                if isinstance(call, Call) and call.retained == item.output
                for use in mode_uses((call,), accepted)
            )
        )
        for item in work
        if isinstance(item, Retain) and not item.isolated
    }


def retained_uses(source: Retain, exclusive: tuple[str, ...]) -> tuple[Use, ...]:
    return tuple(Use(name, "write" if name in exclusive else "read") for name in source.participants)


def sequence_uses(item: Work, accepted: Accepted, *, physical_modes: bool) -> tuple[Use, ...]:
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
        if physical_modes:
            uses = (*uses, *mode_uses((item,), accepted))
        elif item.evaluation:
            uses = (*uses, *(Use(use.participant, "write") for use in item.uses))
    return uses


type Access = dict[tuple[str, str], Literal["read", "write"]]


def _access(uses: tuple[Use, ...], states: tuple[str, ...] = ()) -> Access:
    result: Access = {("state", name): "write" for name in states}
    for use in uses:
        key = ("participant", use.participant)
        if result.get(key) != "write":
            result[key] = use.access
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


def _waits(accepted: Accepted) -> tuple[_Wait, ...]:
    points: list[_Wait] = []

    def add(activity, where, held, requested, sources=()):
        points.append(_Wait(activity, where, held, requested, sources))

    def block(scope: Block, activity: str):
        work = ordered(scope)
        held = _access(
            (*block_uses(scope, accepted.units), *mode_uses(scope.work, accepted)),
            tuple(name for item in work if isinstance(item, Call) for name in item.states),
        )
        prefix = True
        for item in work:
            if isinstance(item, Join):
                # Ordinary block prefix readiness precedes its single lease.
                add(activity, f"{scope.name}/{item.name}", {} if prefix else held, {}, item.channels)
            else:
                prefix = False
            if isinstance(item, Replace):
                add(activity, f"{scope.name}/{item.name}", held, quiescence())
        add(activity, scope.name, {}, held)

    def quiescence() -> Access:
        return _access(tuple(Use(name, "write") for name in accepted.participants), tuple(accepted.states))

    def sequence(scope: Sequence, activity: str):
        work = tuple(item for _, group in sequence_order(scope) for item in group)
        exclusions = retained_exclusions(work, accepted)
        live: dict[str, Access] = {}
        for item in work:
            held = _merge(*live.values())
            where = f"{scope.name}/{item.name}"
            if isinstance(item, Retain):
                requested = _access(retained_uses(item, exclusions.get(item.output, ())))
                add(activity, where, held, requested)
                if not item.isolated:
                    live[item.output] = requested
            elif isinstance(item, Release):
                live.pop(item.retained, None)
            else:
                requested = _access(sequence_uses(item, accepted, physical_modes=True), item.states if isinstance(item, Call) else ())
                # Runtime reuses covered reads; it cannot reacquire its own writer.
                requested = {key: mode for key, mode in requested.items() if key not in held}
                if isinstance(item, Replace):
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


def check_waits(accepted: Accepted) -> None:
    """Reject potential declared wait cycles in this realization before loading.

    Different activities here are leaves of Together; ordered sequence blocks
    and repeated alternatives deliberately share one activity. A cycle must
    have at most one simultaneous wait point from each activity. No execution
    order is changed, no selected policy is evaluated, and no check runs per step.
    """
    points = _waits(accepted)
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
                raise NotReady(f"Potential reservation wait cycle in CPU worker target: {chain}")
            owner = points[following].activity
            if owner not in activities:
                visit(following, (*path, following), activities | {owner})

    for start in sorted(candidates):
        visit(start, (start,), frozenset((points[start].activity,)))
