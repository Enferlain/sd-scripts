"""Running one strategy through both the reference semantics and the runtime, and comparing their facts.

The runtime reaches state through owners and runs implementations through
the seams. Here an in-memory owner holds each resource, and each declared
stand-in becomes an implementation that writes its updates through its
owner_update lease, so both realizations run the same fixtures.
"""

import asyncio
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field

from library.iro.accepted import cold_load
from library.iro.devices import Host, Returned
from library.iro.frontend import Declared
from library.iro.normalize import normalize
from library.iro.operation_contracts import AccessMode, VersionSelection
from library.iro.prepare import Target, prepare
from library.iro.reference import Reference
from library.iro.reference.standins import StandIn
from library.iro.runtime import Coordinator, Fact
from library.iro.seams import Declaration, Use
from library.iro.text import parse
from tests.unit.iro.accepted_runs import accepted_run


@dataclass(frozen=True)
class Seen:
    owner: str
    incarnation: int
    lineage: str
    ordinal: int


@dataclass
class Cell:
    """What a live lease reaches: the resource's value, updated in place by an owner update."""

    owner: Memory
    name: str

    @property
    def value(self) -> object:
        return self.owner.values[self.name]

    def update(self, value: object) -> None:
        self.owner.values[self.name] = deepcopy(value)
        self.owner.ordinals[self.name] += 1


@dataclass(frozen=True)
class Frozen:
    value: object


@dataclass
class Granted:
    use: Use
    version: Seen
    handle: object
    released: bool = False

    def release(self) -> None:
        self.released = True


@dataclass
class Memory:
    """An owner of plain values, versioned by update ordinal within one incarnation."""

    values: dict[str, object]
    ordinals: dict[str, int] = field(default_factory=dict)
    initial: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = deepcopy(dict(self.values))
        self.initial = deepcopy(self.values)
        self.ordinals = dict.fromkeys(self.values, 0)

    def resources(self):
        return tuple(self.values)

    def current(self, resource: str) -> Seen:
        return Seen(resource, 1, "initial", self.ordinals[resource])

    def lease(self, uses, reserved=None):
        granted = []
        for use in uses:
            if use.mode == AccessMode.SNAPSHOT and use.selection == VersionSelection.PUBLISHED_INITIAL:
                granted.append(Granted(use, Seen(use.resource, 1, "initial", 0), Frozen(deepcopy(self.initial[use.resource]))))
            elif use.mode == AccessMode.SNAPSHOT:
                granted.append(Granted(use, self.current(use.resource), Frozen(deepcopy(self.values[use.resource]))))
            else:
                granted.append(Granted(use, self.current(use.resource), Cell(self, use.resource)))
        return granted


def implementation(name: str, standin: StandIn) -> Declared:
    declaration = Declaration(name, standin.version, standin.inputs, standin.outputs)

    def run(*args):
        result = standin.invoke(tuple(deepcopy(a.value) if isinstance(a, Cell | Frozen) else deepcopy(a) for a in args))
        for update in result.updates:
            handle = args[update.lease]
            assert isinstance(handle, Cell), "an update goes through an owner_update lease"
            handle.update(update.value)
        if len(standin.outputs) == 1:
            return result.values[0]
        return result.values or None

    return Declared(declaration, run)


def comparable(fact) -> tuple:
    values = tuple(_plain(v) for v in fact.values)
    if fact.kind == "updated":
        values = values[:1]
    return (fact.owner, fact.invocation, fact.kind, values)


def _plain(value):
    if hasattr(value, "ordinal") and hasattr(value, "lineage"):
        return ("version", value.owner, value.incarnation, value.lineage, value.ordinal)
    if isinstance(value, tuple):
        return tuple(_plain(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _plain(v)) for k, v in value.items()))
    return value


RUNTIME_ONLY = frozenset({"settled"})


@dataclass
class Both:
    reference: Reference
    facts: list[Fact]
    coordinator: Coordinator
    owner: Memory

    def reference_facts(self) -> Counter:
        return Counter(comparable(f) for f in self.reference.facts)

    def runtime_facts(self) -> Counter:
        return Counter(comparable(f) for f in self.facts if f.kind not in RUNTIME_ONLY)


def run_runtime(text: str, resources: Mapping[str, object], installed: dict, *, target=None):
    """Only the runtime, with the given implementations: its outcome, facts and coordinator."""
    run = accepted_run(text, installed=installed)
    prepared = prepare(cold_load(run, installed=installed), installed, target or Target())
    owner = Memory(dict(resources))
    facts: list[Fact] = []
    coordinator = Coordinator(prepared, dict.fromkeys(resources, owner), installed, {"host": Host()}, observe=facts.append)
    outcome = asyncio.run(coordinator.run())
    return outcome, facts, coordinator


def run_both(text: str, resources: Mapping[str, object], standins: Mapping[str, StandIn], *, validators=None, target=None) -> Both:
    reference = Reference(normalize(parse(text)), resources, standins, validators=validators)
    reference.run()
    installed = {name: implementation(name, s) for name, s in standins.items()}
    run = accepted_run(text, installed=installed)
    prepared = prepare(cold_load(run, installed=installed), installed, target or Target())
    owner = Memory(dict(resources))
    facts: list[Fact] = []
    coordinator = Coordinator(prepared, dict.fromkeys(resources, owner), installed, {"host": Host()}, observe=facts.append)
    asyncio.run(coordinator.run())
    return Both(reference, facts, coordinator, owner)


__all__ = ["Both", "Memory", "Returned", "comparable", "implementation", "run_both", "run_runtime"]
