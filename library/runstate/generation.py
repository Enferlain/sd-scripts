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
still has out. A publication whose basis is not the current generation is
stale and never publishes; its dependencies are the whole of every section,
which is the conservative default (D6). Rank agreement before the swap
comes with multi-process runs (D15).
"""

import threading
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

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
            if publication.attempt is not None and publication.attempt.mutation == Mutation.IN_PLACE:
                raise Refused(
                    "in-place work first withdraws the guarantees it affects; withdrawal comes with the second pass of section 15"
                )
            changed = _changed(publication.sections, current)
            with ExitStack() as admitted:
                for owner, section in changed.items():
                    registered = self._owners.get(owner)
                    if registered is None:
                        raise Refused(f"{owner} is not registered with the run's generations")
                    admitted.enter_context(registered.admit(publication, current.sections.get(owner), section))
                installed = Generation(current.number + 1, MappingProxyType({**current.sections, **changed}))
                self._current = installed
            return installed


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


__all__ = ["Generation", "Generations", "Publication", "Publishes", "Section"]
