"""Observable comparison keys and required causal edges, without clock equality."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from .reference import Fact, Identity, ReferenceError, Request, checked


OBSERVABLE = frozenset(
    {
        "outcome",
        "selected",
        "published",
        "item_admitted",
        "disposition",
        "request_pending",
        "request_settled",
        "request_disposed",
        "protected_version",
        "released",
        "physical_entered",
        "physical_completed",
        "physical_failed",
        "validated",
        "validation_rejected",
        "updated",
        "stop_requested",
        "completion_request_disposed",
    }
)


@dataclass(frozen=True)
class FactKey:
    owner: str
    invocation: Identity
    kind: str
    occurrence: int


@dataclass(frozen=True)
class Observation:
    values: dict[FactKey, tuple[object, ...]]
    positions: dict[FactKey, int]
    required_before: frozenset[tuple[FactKey, FactKey]]


def observe(facts: Sequence[Fact]) -> Observation:
    values, positions = {}, {}
    occurrences: dict[tuple[str, Identity, str], int] = defaultdict(int)
    invocations: dict[Identity, list[FactKey]] = defaultdict(list)
    tickets: dict[tuple[str, int], dict[str, FactKey]] = defaultdict(dict)
    outcomes: dict[Identity, FactKey] = {}
    requests: list[tuple[Request, FactKey]] = []
    entered: list[tuple[Request, FactKey]] = []
    stops: list[tuple[tuple[Identity, ...], FactKey]] = []
    physical_uses: list[tuple[tuple[str, int], FactKey]] = []
    for fact in facts:
        if fact.kind not in OBSERVABLE:
            continue
        base = fact.owner, fact.invocation, fact.kind
        occurrence = occurrences[base]
        if fact.kind in ("selected", "published", "disposition"):
            occurrence = checked(fact.values[0], int)
        elif fact.kind in ("request_pending", "request_settled", "request_disposed"):
            occurrence = checked(fact.values[0], Request).ordinal
        key = FactKey(*base, occurrence)
        if key in values:
            raise ReferenceError("duplicate owner-visible fact key")
        occurrences[base] += 1
        # Global call counters are diagnostic; invocation identity owns the use.
        payload = fact.values[1:] if fact.kind.startswith("physical_") else fact.values
        values[key] = payload
        positions[key] = len(positions)
        if fact.kind not in ("disposition", "request_disposed", "completion_request_disposed"):
            invocations[fact.invocation].append(key)
        if fact.kind == "outcome":
            outcomes[fact.invocation] = key
        if fact.kind in ("selected", "published", "disposition"):
            tickets[(fact.owner, checked(fact.values[0], int))][fact.kind] = key
        elif fact.kind == "item_admitted":
            tickets[(str(fact.values[2]), checked(fact.values[0], int))][fact.kind] = key
        if fact.kind == "request_pending" and isinstance(fact.values[0], Request):
            requests.append((fact.values[0], key))
        if fact.kind == "physical_entered" and isinstance(fact.values[1], Request):
            entered.append((fact.values[1], key))
        if fact.kind == "stop_requested":
            stops.append((checked(fact.values[1], tuple), key))
        if fact.kind in ("physical_completed", "physical_failed"):
            for associations in fact.values[3:5]:
                for stream, item in checked(associations, tuple):
                    physical_uses.append(((str(stream), checked(item, int)), key))
        if fact.kind in ("validated", "validation_rejected"):
            for associations in fact.values:
                for stream, item in checked(associations, tuple):
                    physical_uses.append(((str(stream), checked(item, int)), key))
    edges: set[tuple[FactKey, FactKey]] = set()
    for keys in invocations.values():
        edges.update(pairwise(keys))
    for identity, key in outcomes.items():
        parents = [parent for parent in outcomes if parent != identity and identity[: len(parent)] == parent]
        if parents:
            edges.add((key, outcomes[max(parents, key=len)]))
    for phases in tickets.values():
        order = [phases[kind] for kind in ("selected", "published", "item_admitted", "disposition") if kind in phases]
        edges.update(pairwise(order))
        disposition = phases.get("disposition")
        if disposition is not None and disposition.invocation in outcomes:
            edges.add((disposition, outcomes[disposition.invocation]))
    for item, completed in physical_uses:
        disposition = tickets.get(item, {}).get("disposition")
        if disposition is not None:
            edges.add((completed, disposition))
    for request, pending in requests:
        edges.update((pending, key) for cause, key in entered if cause == request)
        if request.cause in outcomes:
            edges.update((outcomes[request.cause], key) for cause, key in entered if cause == request)
    for sources, stopped in stops:
        for source in sources:
            if source not in outcomes:
                raise ReferenceError("stop claim has no completed source invocation")
            edges.add((outcomes[source], stopped))
    return Observation(values, positions, frozenset(edges))


def compare(expected: Observation, actual: Observation) -> None:
    if expected.values != actual.values:
        raise ReferenceError("owner-visible facts differ (identity, association, outcome, version or disposition)")
    for before, after in expected.required_before:
        if actual.positions[before] >= actual.positions[after]:
            raise ReferenceError(f"required causal relationship violated: {before.kind} before {after.kind}")
