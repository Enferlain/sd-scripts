"""The owner, implementation and device seams (D16).

The IR system's boundary has four seams. The first, definitions, is the
contract that :mod:`library.iro.composition` composes. The other three are
the interfaces here, which a domain fills in and the IR runtime calls:

- an **owner** holds one kind of state and realizes the access modes for it,
  with versions and evidence: the run authority for participants, Trainer
  optimization for weight updates, a producer for its cursor;
- an **implementation** is selected code that declares its inputs, outputs,
  effects, concurrency safety and completion;
- a **device** supplies the execution context and physical completion for
  one backend, and what it knows of the device's capacity.

The IR never holds live objects (D15). It refers to state by resource name
and observes versions, which access and admission compare. The run state
imports the IR system, never the reverse, so its owners satisfy these
interfaces structurally. Registration through every seam is explicit
composition (D2).
"""

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from library.iro.operation_contracts import AccessMode, VersionSelection

# Owners.


@runtime_checkable
class Version(Protocol):
    """An observed version of an owner's state: the IR compares versions, it never makes them.

    Versions of one owner compare by ``ordinal`` within one ``lineage`` and
    ``incarnation``; any other pair is incomparable, and admission rejects
    rather than guesses (D14).
    """

    @property
    def owner(self) -> str: ...
    @property
    def incarnation(self) -> int: ...
    @property
    def lineage(self) -> str: ...
    @property
    def ordinal(self) -> int: ...


@dataclass(frozen=True)
class Use:
    """One resource an access names, as the runtime asks its owner for it."""

    resource: str
    mode: AccessMode
    selection: VersionSelection
    route: str | None = None
    view: str | None = None


@runtime_checkable
class Lease(Protocol):
    """An owner's grant of one use for an access's body and every retained result that keeps it.

    ``handle`` is what the body's calls receive: a route's callable, a view or
    the owner's update interface. It never leaves the access (D16).
    """

    @property
    def use(self) -> Use: ...
    @property
    def version(self) -> Version: ...
    @property
    def handle(self) -> object: ...
    def release(self) -> None: ...


@runtime_checkable
class Owner(Protocol):
    """Holds one kind of state and realizes access to it (D16, D19)."""

    def resources(self) -> Sequence[str]:
        """The resources this owner holds."""
        ...

    def current(self, resource: str) -> Version:
        """The resource's current version, as admission and an at_request reservation observe it."""
        ...

    def lease(self, uses: Sequence[Use], reserved: Mapping[str, Version] | None = None) -> Sequence[Lease]:
        """Leases for the uses, taken together, in the same order.

        ``reserved`` holds the versions a request reserved for its
        ``at_request`` uses; the owner keeps each reserved version until the
        lease is released.
        """
        ...


# Implementations.


@dataclass(frozen=True)
class Effects:
    """What an implementation does to state beyond its results, by input position.

    ``reads`` and ``updates`` name lease inputs; ``unknown`` says there may be
    effects it does not declare, which blocks local transformations.
    """

    reads: tuple[int, ...] = ()
    updates: tuple[int, ...] = ()
    unknown: bool = True


@runtime_checkable
class Completion(Protocol):
    """Physical completion of work submitted to a device: results are usable once it is done."""

    def done(self) -> bool: ...
    def wait(self) -> None: ...


@runtime_checkable
class Implementation(Protocol):
    """Selected code that a call names, with what it declares about itself.

    ``retains`` are the input positions whose leased state its result keeps
    protected until the result's last use (D16). ``concurrency_safe`` says it
    may run in several invocations at once.
    """

    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def inputs(self) -> tuple[str, ...]: ...
    @property
    def outputs(self) -> tuple[str, ...]: ...
    @property
    def effects(self) -> Effects: ...
    @property
    def retains(self) -> tuple[int, ...]: ...
    @property
    def concurrency_safe(self) -> bool: ...

    def __call__(self, device: Device, *arguments: object) -> tuple[tuple[object, ...], Completion]:
        """Submits the work; its results are usable once the completion is done."""
        ...


# Devices.


@runtime_checkable
class Device(Protocol):
    """Execution context and physical completion for one backend."""

    @property
    def name(self) -> str: ...

    def context(self) -> AbstractContextManager[object]:
        """The context work runs in, such as a CUDA stream."""
        ...

    def completion(self) -> Completion:
        """A completion for the work submitted so far in the current context."""
        ...

    def capacity(self) -> Mapping[str, int]:
        """What the device knows of its capacity, such as free memory in bytes; empty when it knows nothing."""
        ...


__all__ = ["Completion", "Device", "Effects", "Implementation", "Lease", "Owner", "Use", "Version"]
