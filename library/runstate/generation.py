"""Generations and their publication (D19).

A generation is the coherent published state: one section per owner, under
a generation number. Each owner builds and checks its next section off to
the side; one publication then installs every changed section by swapping
the current generation, a single store that does no fallible work and calls
no observers (D8). Code holds a generation without locking, published
sections never change, and each section has one writer, so correctness does
not depend on the GIL.

Owners register with the run's generations, and each may refuse a
publication of its next section, such as the run authority over a lease it
still has out. A publication of in-place work is checked by every owner
with a section, since the work may affect any of them, whether or not it
publishes a next section for that owner (D8). A publication whose basis is not the current generation is
stale and never publishes; its dependencies are the whole of every section,
which is the conservative default (D6). Rank agreement before the swap
comes with multi-process runs (D15).

An executable run exists only after publication, once every owner that
judges obligations finds its readiness checkpoint met (D17). It reads
through the generation it was realized on and runs once. When its work has
ended, it asks each judge that checks late to settle, and a reason fails it.
"""

import asyncio
import threading
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from library.iro.prepare import Prepared
from library.iro.runtime import Coordinator, Fact, Observer, Outcome
from library.iro.seams import Device, Implementation, Owner
from library.iro.verify import CORE_EXECUTION, Checkpoint
from library.iro.words.model import ResourceOp
from library.runstate.owners import Attempt, AttemptRef, Basis, Mint, Mutation, Refused, Stale


class Section(Protocol):
    """An owner's part of a generation: built off to the side, checked, then published and never changed (D19)."""

    @property
    def owner(self) -> str: ...

    @property
    def revision(self) -> int: ...

    def revision_of(self, key: str) -> int | None:
        """The revision of one thing the section holds, or None when it holds no such thing."""
        ...


class Publishes(Protocol):
    """An owner registered with the generations, which checks every publication of its next section."""

    def admit(self, publication: Publication, current: Section | None, next: Section) -> AbstractContextManager[None]:
        """Raises :class:`Refused` on entry when ``next`` may not replace ``current`` now.

        For an owner whose section a publication of in-place work does not
        change, ``next`` is ``current``.

        On entry it checks, and it holds what the check relied on until the
        swap is done, when it is left. It changes no state and does no
        fallible external work: another owner may still refuse the
        publication.
        """
        ...


@dataclass(frozen=True)
class Generation:
    """The coherent published state: one section per owner, under a generation number."""

    number: int
    sections: Mapping[str, Section]

    def section(self, owner: str) -> Section:
        found = self.sections.get(owner)
        if found is None:
            raise KeyError(f"generation {self.number} has no section for {owner}")
        return found


@dataclass(frozen=True)
class Publication:
    """One publication: the next sections of the owners it changes, with the basis they were built from."""

    basis: Basis
    sections: tuple[Section, ...]
    attempt: Attempt | None = None


class Generations:
    """One run's generations: the current one, the owners registered with it, and the publications that install the next.

    Reading :attr:`current` takes no lock. Publications are serialized with
    each other, and each owner's admission check sees the generation the
    swap replaces.
    """

    def __init__(self, run: str):
        self.run = run
        self._current = Generation(0, MappingProxyType({}))
        self._owners: dict[str, Publishes] = {}
        self._lock = threading.Lock()
        self._attempts = Mint(AttemptRef, run)

    @property
    def current(self) -> Generation:
        return self._current

    def register(self, owner: str, publishes: Publishes) -> None:
        """Registers an owner, whose sections may then be published."""
        with self._lock:
            if owner in self._owners:
                raise ValueError(f"{owner} is already registered")
            self._owners[owner] = publishes

    def basis(self, obligations: str, generation: Generation | None = None) -> Basis:
        """The revisions work starts from: a generation, the current one by default, and the obligation revision."""
        generation = generation or self._current
        sections = tuple(sorted((owner, section.revision) for owner, section in generation.sections.items()))
        return Basis(generation.number, sections, obligations)

    def attempt(self, obligations: str, mutation: Mutation = Mutation.REPLACES, generation: Generation | None = None) -> Attempt:
        """A new attempt, based on a generation the caller read, the current one by default."""
        return Attempt(self._attempts(), self.basis(obligations, generation), mutation)

    def publish(self, publication: Publication) -> Generation:
        """Installs the publication's sections as the next generation, or raises :class:`Refused` and changes nothing."""
        with self._lock:
            current = self._current
            if publication.basis.generation != current.number:
                raise Stale(f"built from generation {publication.basis.generation}, and generation {current.number} is current")
            changed = _changed(publication.sections, current)
            unregistered = [owner for owner in changed if owner not in self._owners]
            if unregistered:
                raise Refused(f"{unregistered[0]} is not registered with the run's generations")
            admitting = dict(changed)
            if publication.attempt is not None and publication.attempt.mutation == Mutation.IN_PLACE:
                # In-place work may affect any owner's guarantees, so every owner with a section checks it.
                admitting |= {owner: section for owner, section in current.sections.items() if owner not in changed}
            with ExitStack() as admitted:
                for owner, section in admitting.items():
                    admitted.enter_context(self._owners[owner].admit(publication, current.sections.get(owner), section))
                installed = Generation(current.number + 1, MappingProxyType({**current.sections, **changed}))
                self._current = installed
            return installed


class Judges(Protocol):
    """An owner that judges the obligations on what it holds at a readiness checkpoint (D19)."""

    def readiness(self, checkpoint: Checkpoint) -> tuple[str, ...]:
        """Why its obligations at the checkpoint do not hold now; empty once they all do."""
        ...


@runtime_checkable
class Settles(Protocol):
    """A judge whose checks on the run's work finish after that work, such as a gradient norm read late (D17)."""

    def settle(self) -> tuple[str, ...]:
        """Once the run's work has ended: why it was unsound; empty when it was sound."""
        ...


@dataclass(frozen=True)
class Readiness:
    """The evidence an executable run was accepted with: who judged which checkpoint, at which generation."""

    checkpoint: Checkpoint
    generation: int
    judges: tuple[str, ...]


class ExecutableRun:
    """A prepared run ready to execute against the published state it was realized on (D17).

    It holds the prepared strategy with its lowered runtime instance, the
    owners and device adapters it runs against, the generation and
    obligation revision its state came from, the preparation attempts that
    published it, and the readiness evidence it was accepted with. It is
    single-use: a governed transition or a restoration realizes a new one.
    """

    def __init__(
        self,
        generations: Generations,
        prepared: Prepared,
        owners: Mapping[str, Owner],
        installed: Mapping[str, Implementation],
        devices: Mapping[str, Device],
        judges: Mapping[str, Judges],
        *,
        attempts: tuple[AttemptRef, ...] = (),
        observe: Observer | None = None,
    ):
        """Checks every judge's core-execution checkpoint, then builds the run's runtime; raises :class:`Refused` with every reason."""
        declared = {op.sym_name.data for op in prepared.loaded.normalized.strategy.walk() if isinstance(op, ResourceOp)}
        reasons = [f"@{name} has no owner to serve it" for name in sorted(declared - set(owners))]
        generation = generations.current
        for name, judge in judges.items():
            reasons += [f"{name}: {reason}" for reason in judge.readiness(CORE_EXECUTION)]
        if reasons:
            raise Refused("the run is not ready to execute:\n" + "\n".join(f"  {reason}" for reason in reasons))
        self.generations = generations
        self.prepared = prepared
        self.obligations = prepared.loaded.run.obligations.revision
        self.attempts = attempts
        self.devices = dict(devices)
        self.readiness = Readiness(CORE_EXECUTION, generation.number, tuple(judges))
        self._judges = dict(judges)
        self.facts: list[Fact] = []
        self._observe = observe or self.facts.append
        self.coordinator = Coordinator(prepared, owners, installed, self.devices, observe=self._observe)
        self._executed = False
        self._lock = threading.Lock()

    @property
    def generation(self) -> int:
        """The generation the run was realized on."""
        return self.readiness.generation

    def execute(self) -> Outcome:
        """Runs the strategy once, while the generation it was realized on is still current.

        Once its work has ended, each judge that checks late settles. A reason
        is reported as a ``settle_failed`` fact after the run's own facts and
        fails a run that succeeded; a run that failed or was cancelled keeps
        that outcome, with the reasons added to its error. A judge whose
        settling raises gives that as its reason.
        """
        with self._lock:
            if self._executed:
                raise Refused("an executable run is single-use; realize a new one")
            self._executed = True
        current = self.generations.current.number
        if current != self.generation:
            raise Stale(f"realized on generation {self.generation}, and generation {current} is current")
        outcome = asyncio.run(self.coordinator.run())
        reasons: list[str] = []
        for name, judge in self._judges.items():
            if not isinstance(judge, Settles):
                continue
            try:
                found = [f"{name}: {reason}" for reason in judge.settle()]
            except Exception as failed:
                found = [f"{name}: settling raised {type(failed).__name__}: {failed}"]
            if found:
                self._observe(Fact(name, (), "settle_failed", tuple(found)))
            reasons += found
        if reasons:
            if outcome.status == "succeeded":
                outcome.status = "failed"
            outcome.error = "\n".join([*([outcome.error] if outcome.error else []), *reasons])
        return outcome


def _changed(sections: Iterable[Section], current: Generation) -> dict[str, Section]:
    changed: dict[str, Section] = {}
    for section in sections:
        if section.owner in changed:
            raise Refused(f"the publication carries two sections for {section.owner}")
        previous = current.sections.get(section.owner)
        if previous is not None and section.revision <= previous.revision:
            raise Refused(f"{section.owner}'s next section has revision {section.revision}, not past {previous.revision}")
        changed[section.owner] = section
    return changed


__all__ = ["ExecutableRun", "Generation", "Generations", "Judges", "Publication", "Publishes", "Readiness", "Section", "Settles"]
