"""The owner pattern: references, versions, dependencies, evidence and attempts (D19).

Every owner of run state uses these types: the run authority, Trainer
optimization, and later owners such as a capability's implementation. A new
owner fills them in and adds no machinery to the run-state layer (D15).
Each owner's part of a cut comes with the cut protocol's first contributor.
"""

import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Protocol


class Revisioned(Protocol):
    """What dependencies are judged against: an owner's section, by its revision and the revisions of what it holds."""

    @property
    def revision(self) -> int: ...

    def revision_of(self, key: str) -> int | None:
        """The revision of one thing the section holds, or None when it holds no such thing."""
        ...


# References.


@dataclass(frozen=True)
class Reference:
    """One incarnation of something an owner manages: the run identity, a kind and a serial number (D19).

    Only its owner creates one, by minting it or by restoring it from a cut;
    other code only compares references, and none is ever reused. Each owner
    declares its kinds as subclasses, each with its own ``kind``. Cuts and
    facts store a reference as the string of its three parts, which
    :meth:`parse` reads back exactly.
    """

    run: str
    serial: int
    kind: ClassVar[str] = ""
    _kinds: ClassVar[dict[str, type[Reference]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.kind or "/" in cls.kind:
            raise TypeError(f"{cls.__name__} names a kind without '/'")
        if cls.kind in Reference._kinds:
            raise TypeError(f"{cls.__name__} and {Reference._kinds[cls.kind].__name__} share the kind {cls.kind!r}")
        Reference._kinds[cls.kind] = cls

    def __post_init__(self) -> None:
        if type(self) is Reference:
            raise TypeError("a reference has an owner's kind; mint one of its subclasses")
        if not self.run or "/" in self.run:
            raise ValueError(f"{self.run!r} is not a run identity")
        if self.serial < 1:
            raise ValueError(f"a reference's serial number starts at 1, not {self.serial}")

    def __str__(self) -> str:
        return f"{self.run}/{self.kind}/{self.serial}"

    @staticmethod
    def parse(text: str) -> Reference:
        """The reference a string of its three parts names, read back exactly."""
        parts = text.split("/")
        kind = Reference._kinds.get(parts[1]) if len(parts) == 3 else None
        if kind is None or not parts[2].isdigit() or str(int(parts[2])) != parts[2]:
            raise ValueError(f"{text!r} is not a reference")
        return kind(parts[0], int(parts[2]))


class Mint[R: Reference]:
    """Mints one owner's references of one kind, in order: none is ever reused (D19).

    ``after`` continues past the serial numbers a cut restored.
    """

    def __init__(self, kind: type[R], run: str, after: int = 0):
        self._kind, self._run, self._last = kind, run, after
        self._lock = threading.Lock()

    def __call__(self) -> R:
        with self._lock:
            self._last += 1
            return self._kind(self._run, self._last)


def new_run() -> str:
    """A run identity that no other run shares."""
    return uuid.uuid4().hex


# Versions and dependencies.


@dataclass(frozen=True)
class Version:
    """Which state a use saw: a reference, the revision of what is bound and a numeric version (D19).

    At the owner seam (:class:`library.iro.seams.Version`) the reference's
    serial number is the incarnation, the bound revision the version lineage
    and the numeric version the ordinal, with the resource's name as owner.
    The numeric version is never reused within an incarnation, so versions
    from different incarnations, or from either side of a bound-state
    replacement, are incomparable, and admission gives them its explicit
    disposition instead of comparing them (D14).
    """

    resource: str
    reference: Reference
    bound: int
    numeric: int

    @property
    def owner(self) -> str:
        return self.resource

    @property
    def incarnation(self) -> int:
        return self.reference.serial

    @property
    def lineage(self) -> str:
        return str(self.bound)

    @property
    def ordinal(self) -> int:
        return self.numeric


@dataclass(frozen=True)
class Dependencies:
    """What a projection or prepared realization relies on (D6, D19).

    ``whole`` lists owners' sections at one revision each: any change in such
    a section makes it stale, which is the default. ``exact`` is the narrower
    alternative: named revisions within a section, ``(owner, key,
    revision)``, which stay fresh while each still holds.
    """

    whole: tuple[tuple[str, int], ...] = ()
    exact: tuple[tuple[str, str, int], ...] = ()

    @classmethod
    def on(cls, sections: Mapping[str, Revisioned]) -> Dependencies:
        """The default: the whole of each section, at its current revision."""
        return cls(whole=tuple(sorted((owner, section.revision) for owner, section in sections.items())))

    def stale(self, sections: Mapping[str, Revisioned]) -> tuple[str, ...]:
        """What no longer holds, as ``owner`` or ``owner key``; empty while every dependency holds."""
        changed = [owner for owner, revision in self.whole if owner not in sections or sections[owner].revision != revision]
        changed += [
            f"{owner} {key}" for owner, key, revision in self.exact if owner not in sections or sections[owner].revision_of(key) != revision
        ]
        return tuple(changed)


# Attempts and evidence.


class AttemptRef(Reference):
    """One governed job's identity, distinct from the state it started from (D8)."""

    kind = "attempt"


class Mutation(StrEnum):
    """Whether an attempt replaces state or mutates it in place; in-place work first withdraws the guarantees it affects (D8)."""

    REPLACES = "replaces"
    IN_PLACE = "in place"


@dataclass(frozen=True)
class Basis:
    """The revisions an attempt started from: its generation, each owner's section revision and the obligation revision."""

    generation: int
    sections: tuple[tuple[str, int], ...]
    obligations: str


@dataclass(frozen=True)
class Attempt:
    """A governed job: its identity, its basis and whether it replaces state or mutates it in place (D19).

    Its result is bound to it, and publishes only while its basis is current.
    """

    reference: AttemptRef
    basis: Basis
    mutation: Mutation = Mutation.REPLACES


@dataclass(frozen=True)
class Evidence:
    """A typed value that one kind of obligation needs, declared beside the evaluator that judges it (D18, D19).

    Each value records the attempt that produced it and the revisions that
    attempt started from. A producer's own claim, or a Boolean, is never
    evidence: the evaluator judges the value.
    """

    attempt: AttemptRef
    basis: Basis


# Refusals.


class Refused(Exception):
    """A governed change the run state refuses, with why; nothing changed."""


class Stale(Refused):
    """A result whose basis is no longer current, which never publishes (D8)."""


__all__ = [
    "Attempt",
    "AttemptRef",
    "Basis",
    "Dependencies",
    "Evidence",
    "Mint",
    "Mutation",
    "Reference",
    "Refused",
    "Revisioned",
    "Stale",
    "Version",
    "new_run",
]
