"""The owner, implementation and device seams (D16).

The IR system's boundary has four seams. The first, definitions, is the
contract that :mod:`library.iro.composition` composes. The other three are
the interfaces here, which a domain fills in and the IR runtime calls:

- an **owner** holds one kind of state and realizes the access modes for it,
  with versions and evidence: the run authority for participants, Trainer
  optimization for its units, a producer for its cursor;
- an **implementation** is selected code with a declaration: its inputs,
  outputs, effects, concurrency safety and completion at one implementation
  version;
- a **device** supplies the execution context and physical completion for
  one backend, and what it knows of the device's capacity.

The IR never holds live objects (D15). It refers to state by resource name
and observes versions, which access and admission compare. The run state
imports the IR system, never the reverse, so its owners satisfy these
interfaces structurally. Registration through every seam is explicit
composition (D2).
"""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
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

    def updated(self) -> Version:
        """Told by the runtime after each call whose declaration says it updates through this lease.

        The owner advances the version and returns it, so a change made in
        place, such as an optimizer step, never goes unrecorded; a call that
        changed nothing still advances it (D16). Only an ``owner_update`` or
        ``grant`` lease is told.
        """
        ...


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


@dataclass(frozen=True)
class Capture:
    """A snapshot lease's handle when the snapshot is of device storage: what to copy and how to build the handle (D15).

    The owner names the storage the version lives in and how its ordinary
    handle is built over copies of it; the runtime queues the copy on the
    device's stream, after the work that produced the version and before
    later updates, and gives the access's calls ``over(copies)``. The copies
    hold the storage's values at the lease's version, which must be the
    current one when the lease is taken: a capture copies live storage. The
    owner never copies device storage itself, and ``over`` only wraps.

    For a participant the run never updates, the live storage already is the
    version, so the runtime calls ``over(storage)`` on the coordinator thread
    instead of copying: ``over`` must neither copy nor change what it is given,
    and must build the same handle over the live storage as over copies of
    it. Such a participant's storage objects are the same for the whole run:
    a route may move their contents within a use, never rebind them.
    """

    storage: Sequence[object]
    over: Callable[[Sequence[object]], object]


@runtime_checkable
class HoldsDeviceStorage(Protocol):
    """An owner whose participant may leave the device: it says what its device storage is (D15).

    The storage is opaque to the runtime and known to the device adapter
    (tensors, for a PyTorch backend). It is complete before the run, as
    materialization publishes it. The owner never copies or frees it itself;
    the runtime queues those copies on the device's stream. After the
    participant's last use the storage stays released (D15): its exact copy
    belongs to the adapter of that run.
    """

    def storage(self, resource: str) -> Sequence[object]:
        """The participant's own device storage."""
        ...


@runtime_checkable
class Establishes(Protocol):
    """A lease that establishes what its state needs for each call that uses it (D15, D16).

    The runtime calls :meth:`establish` on the worker thread, inside the
    device's context, before the calls of each group that uses the lease,
    with the group's context and the handle its calls receive. What it
    establishes is the owner's business, such as the module mode a
    participant's route sets from what the call's derivatives update. A
    group that uses such a lease uses its resource alone, even when its calls
    are declared safe for concurrent use, since what it establishes is state
    other calls on the resource would see.
    """

    def establish(self, call: CallContext, handle: object) -> None: ...


# Implementations.


class Effect(StrEnum):
    """What an implementation does beyond its results that its signature cannot show (D18)."""

    DEVICE = "device"  # submits work to a device, so it needs a device admission (D15)
    RANDOMNESS = "randomness"  # draws random numbers; its call sets the random state it draws from
    EXTERNAL = "external"  # reads or writes outside the run, such as files or the network
    COLLECTIVES = "collectives"  # issues collectives, which every rank orders the same way (D15)
    PRIVATE_STATE = "private_state"  # keeps state of its own beyond the call


UPDATING_LEASES = (f"{AccessMode.OWNER_UPDATE} lease", f"{AccessMode.GRANT} lease")


@dataclass(frozen=True)
class Declaration:
    """What an implementation declares about itself at one implementation version (D16, D18).

    ``inputs`` and ``outputs`` are types as the written form prints them, such
    as ``latents``, ``protected_current lease`` or ``retained loss``. Which
    leased state it reads or updates follows from its lease types, and a
    retained output keeps every lease and retained input, as a call's does,
    so neither is stated a second time. ``completion`` says whether the
    device's completion of the call's work covers it (``device``) or the
    implementation returns a completion of its own (``own``), as one that
    uses streams of its own does.

    ``differentiates`` and ``differentiable`` are input positions, from which
    preparation derives each call's grad mode (D15). ``differentiates`` are
    the inputs whose derivatives the implementation computes, such as the
    loss a contribution takes; ``differentiable`` the inputs its results are
    differentiable with respect to, such as a loss's prediction but not its
    target. Both name values, never leases.
    """

    name: str
    version: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    effects: frozenset[Effect] = frozenset()
    concurrency_safe: bool = False
    completion: str = "device"
    differentiates: tuple[int, ...] = ()
    differentiable: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("an implementation declares its name and its implementation version")
        if self.completion not in ("device", "own"):
            raise ValueError(f"{self.name}: completion is 'device' or 'own', not {self.completion!r}")
        object.__setattr__(self, "effects", frozenset(Effect(effect) for effect in self.effects))
        for field_name in ("differentiates", "differentiable"):
            positions = tuple(getattr(self, field_name))
            object.__setattr__(self, field_name, positions)
            if len(set(positions)) != len(positions):
                raise ValueError(f"{self.name}: {field_name} names an input twice")
            for position in positions:
                if not 0 <= position < len(self.inputs):
                    raise ValueError(f"{self.name}: {field_name} names input {position}, and it declares {len(self.inputs)} inputs")
                if self.inputs[position].endswith(" lease"):
                    raise ValueError(f"{self.name}: {field_name} names input {position}, a lease; only values are differentiated")

    @property
    def reads(self) -> tuple[int, ...]:
        """Input positions whose leased state it only reads."""
        return tuple(i for i, t in enumerate(self.inputs) if t.endswith(" lease") and t not in UPDATING_LEASES)

    @property
    def updates(self) -> tuple[int, ...]:
        """Input positions whose leased state it updates."""
        return tuple(i for i, t in enumerate(self.inputs) if t in UPDATING_LEASES)

    @property
    def retains(self) -> tuple[int, ...]:
        """Input positions a retained output keeps protected until its last use (D16)."""
        if not any(t.startswith("retained ") for t in self.outputs):
            return ()
        return tuple(i for i, t in enumerate(self.inputs) if t.endswith(" lease") or t.startswith("retained "))

    @property
    def digest(self) -> str:
        """A digest of the declaration, which a pin records beside the implementation version (D16).

        It covers everything declared except the version. Value types are
        covered by name: they carry no declared structure yet.
        """
        listing = "\n".join(
            (
                f"name {self.name}",
                f"inputs {', '.join(self.inputs)}",
                f"outputs {', '.join(self.outputs)}",
                f"effects {', '.join(sorted(self.effects))}",
                f"concurrency_safe {self.concurrency_safe}",
                f"completion {self.completion}",
                f"differentiates {', '.join(map(str, self.differentiates))}",
                f"differentiable {', '.join(map(str, self.differentiable))}",
            )
        )
        return hashlib.blake2s(listing.encode(), digest_size=8).hexdigest()


@runtime_checkable
class Completion(Protocol):
    """Physical completion of work submitted to a device: results are usable once it is done."""

    def done(self) -> bool: ...
    def wait(self) -> None: ...


@runtime_checkable
class Implementation(Protocol):
    """Selected code that a call names, with its declaration."""

    @property
    def declaration(self) -> Declaration: ...

    def __call__(self, device: Device, *arguments: object) -> tuple[tuple[object, ...], Completion]:
        """Submits the work inside the device's context; its results are usable once the completion is done."""
        ...


# Devices.


@dataclass(frozen=True)
class CallContext:
    """What a prepared call establishes on entry, so it never inherits a thread's ambient state (D15).

    ``grad`` and ``derivatives_to`` are derived by preparation from the
    dataflow: ``grad`` says a result of the call reaches a differentiated
    input through inputs declared differentiable, and ``derivatives_to``
    names where those derivatives go, the resources the differentiating
    calls update. The IR gives them no further meaning; a lease that
    establishes something per call reads them (:class:`Establishes`).
    ``precision`` names a reduced-precision dtype, such as ``bfloat16``, or
    is None for full precision. ``random`` names whose random state the call
    draws from, or is None for a call that draws none.
    """

    grad: bool = False
    derivatives_to: tuple[str, ...] = ()
    precision: str | None = None
    random: str | None = None

    def __post_init__(self) -> None:
        if self.derivatives_to and not self.grad:
            raise ValueError("a call's derivatives go somewhere only with grad mode")


@runtime_checkable
class Device(Protocol):
    """Execution context and physical completion for one backend."""

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> str:
        """``host``, or the PyTorch device type of the backend, such as ``cuda`` (D15)."""
        ...

    @property
    def offers(self) -> frozenset[str]:
        """What the adapter realizes (D15); the target records it, and a run's needs will be checked against it."""
        ...

    def context(self, call: CallContext, after: Sequence[Completion]) -> AbstractContextManager[object]:
        """The context a call runs in: its settings, the device's stream, and the work it must follow.

        ``after`` is work that must complete before the call's work runs: the
        device orders it on the device where it can, and waits on the CPU
        where it cannot.
        """
        ...

    def completion(self) -> Completion:
        """A completion for the work submitted so far in the current context."""
        ...

    def capacity(self) -> Mapping[str, int]:
        """What the device knows of its capacity, such as free memory in bytes; empty when it knows nothing."""
        ...


@runtime_checkable
class CopiesStorage(Protocol):
    """A device adapter that offers ``copies``: it copies an owner's device storage for a snapshot, and moves it off the device and back (D15)."""

    def empty(self, storage: Sequence[object]) -> None:
        """Frees the storage's device memory after the work already queued, keeping an exact host copy."""
        ...

    def fill(self, storage: Sequence[object]) -> Completion:
        """Brings the storage back from its host copy, queued before any later work; the completion says when it is there."""
        ...

    def capture(self, storage: Sequence[object], after: Sequence[Completion]) -> tuple[Sequence[object], Completion]:
        """Exact copies of the storage, one per item, queued after ``after`` and the work already queued; the completion says when they are made."""
        ...


__all__ = [
    "CallContext",
    "Capture",
    "Completion",
    "CopiesStorage",
    "Declaration",
    "Device",
    "Effect",
    "Establishes",
    "HoldsDeviceStorage",
    "Implementation",
    "Lease",
    "Owner",
    "Use",
    "Version",
]
