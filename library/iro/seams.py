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

Setups establish what calls need beyond the core's own context: a domain
attaches call setups and annotations to calls at preparation, and an owner
attaches a held setup to the leases it grants (D16).

The IR never holds live objects (D15). It refers to state by resource name
and observes versions, which access and admission compare. The run state
imports the IR system, never the reverse, so its owners satisfy these
interfaces structurally. Registration through every seam is explicit
composition (D2).
"""

import hashlib
from collections.abc import Callable, Hashable, Mapping, Sequence
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


# Setups and annotations (D16).

Annotations = tuple[tuple[str, tuple[str, ...]], ...]
"""Values a domain attaches to calls at preparation, each a key with written values, sorted by key (D16).

They are written data, such as unit names, references and module paths, never
live objects. The core carries them to the calls' setups unread and compares
them only for equality.
"""


def annotate(values: Mapping[str, Sequence[str]]) -> Annotations:
    """Annotations in their one written order: keys sorted, each key's values sorted and without repeats."""
    return tuple((key, tuple(sorted(set(values[key])))) for key in sorted(values) if values[key])


@runtime_checkable
class CallSetup(Protocol):
    """What calls need beyond the core's own context that lasts only while they run (D16).

    The runtime enters it on the calls' worker, inside the device's context,
    with the calls' annotations and the handles they receive, and leaves it
    when they return, however they end. One a domain derives covers its
    group's calls; one a lease carries covers each call the lease is passed
    to, entered with that lease's handle (:class:`Enters`). It changes only its worker's own
    context, such as PyTorch's grad mode or autocast, both kept per thread;
    a switch the whole process shares can never be a call setup. Its code is
    pinned by name and version, as an implementation's is.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def entered(self, annotations: Annotations, handles: Sequence[object]) -> AbstractContextManager[object]: ...


@runtime_checkable
class HeldSetup(Protocol):
    """What calls need of a resource's realization that later work relies on, held while their leases are out (D16).

    The runtime applies it on the calls' worker, inside the device's context,
    with the calls' annotations and the handle the lease gives, and it
    changes only state reachable through that handle, such as a
    participant's module mode. :meth:`key` says, for a call's annotations,
    which realization state the setup needs, as an opaque value the core
    only compares, together with the setup's name and version: calls whose
    setups need the same state share the hold, whichever setup object each
    lease carries, and a call that needs another
    waits until the hold ends, when the last lease of the calls it covered is
    released, including those a retained result keeps. It changes no version.

    What D16 also gives a held setup is not realized yet: undoing its change
    when the hold ends, and excluding the resource's other work while it
    holds a change of storage. They come with D20's evaluation-point swap,
    their first use. Until then a held setup changes only state the next
    setup sets again, as module mode does, and the runtime refuses one that
    declares ``changes_storage`` or an ``undo``.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def key(self, annotations: Annotations) -> Hashable: ...

    def hold(self, annotations: Annotations, handle: object) -> None: ...


@runtime_checkable
class Holds(Protocol):
    """A lease its owner attached a held setup to, as a route's realization does (D16); ``setup`` is None where it has none."""

    @property
    def setup(self) -> HeldSetup | None: ...


@runtime_checkable
class Enters(Protocol):
    """A lease its owner attached call setups to, as a route's realization does, such as autocast for its mixed precision (D15, D16).

    They cover only the calls the lease is passed to: the runtime enters them
    around each such call, never around the other calls of its group, as
    today's trainer autocasts the model's forward and not its loss.
    """

    @property
    def call_setups(self) -> tuple[CallSetup, ...]: ...


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

    ``relations`` says, for each named relation, which inputs propagate into
    which outputs, as ``(input, output)`` positions, such as a denoiser's
    weights and latents into its prediction for ``derivative``. The core
    compares relation names and never interprets them; the domain that asks
    gives a name its meaning (D16). An input with no edge for a relation
    propagates nothing, as a loss's target does for ``derivative``.
    """

    name: str
    version: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    effects: frozenset[Effect] = frozenset()
    concurrency_safe: bool = False
    completion: str = "device"
    relations: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("an implementation declares its name and its implementation version")
        if self.completion not in ("device", "own"):
            raise ValueError(f"{self.name}: completion is 'device' or 'own', not {self.completion!r}")
        object.__setattr__(self, "effects", frozenset(Effect(effect) for effect in self.effects))
        given = self.relations.items() if isinstance(self.relations, Mapping) else self.relations
        relations: dict[str, tuple[tuple[int, int], ...]] = {}
        for relation, edges in given:
            if not relation:
                raise ValueError(f"{self.name}: a relation has a name")
            if relation in relations:
                raise ValueError(f"{self.name}: relation {relation!r} is declared twice")
            normalized = tuple(sorted({(int(source), int(target)) for source, target in edges}))
            for source, target in normalized:
                if not 0 <= source < len(self.inputs):
                    raise ValueError(f"{self.name}: relation {relation!r} names input {source}, and it declares {len(self.inputs)} inputs")
                if not 0 <= target < len(self.outputs):
                    raise ValueError(
                        f"{self.name}: relation {relation!r} names output {target}, and it declares {len(self.outputs)} outputs"
                    )
            if normalized:
                relations[relation] = normalized
        object.__setattr__(self, "relations", tuple(sorted(relations.items())))

    def edges(self, relation: str) -> tuple[tuple[int, int], ...]:
        """The ``(input, output)`` positions through which the relation propagates; none when it is not declared."""
        return dict(self.relations).get(relation, ())

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
                *(f"relation {relation} {', '.join(f'{s}>{t}' for s, t in edges)}" for relation, edges in self.relations),
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
    """The core's own part of a call's context (D15, D16).

    The device establishes it on entry with the device's stream and the work
    the call follows, so a call never inherits a thread's ambient state.
    ``random`` names whose random state the call draws from, or is None for a
    call that draws none. Whatever a domain or an owner decides for the
    call, such as grad mode, a route's precision or a participant's module
    mode, the setups that cover it establish (:class:`CallSetup`,
    :class:`HeldSetup`).
    """

    random: str | None = None


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
    "Annotations",
    "CallContext",
    "CallSetup",
    "Capture",
    "Completion",
    "CopiesStorage",
    "Declaration",
    "Device",
    "Effect",
    "Enters",
    "HeldSetup",
    "Holds",
    "HoldsDeviceStorage",
    "Implementation",
    "Lease",
    "Owner",
    "Use",
    "Version",
    "annotate",
]
