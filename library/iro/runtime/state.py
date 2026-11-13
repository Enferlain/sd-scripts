"""The runtime's control state: facts, invocation outcomes, leases, retained results, claims and requests (D15).

This is runtime state, never semantic authority. Owners hold the state that
leases reach; the coordinator holds who is using what, for how long, and in
what lifecycle state each invocation is. Facts are reported to an observer as
they happen and are not kept here, so a long run's bookkeeping stays bounded.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from library.iro.operation_contracts import AccessMode
from library.iro.seams import Lease, Version
from library.iro.words.data import StreamOp

Identity = tuple[tuple[str, int], ...]


class State(StrEnum):
    """Where an invocation is in its lifecycle; the same states as the reference semantics."""

    CREATED = "created"
    PENDING = "pending"
    ADMITTED = "admitted"
    RUNNING = "running"
    CLOSING = "closing"
    SETTLING = "settling"
    TERMINAL = "terminal"


TRANSITIONS = {
    State.CREATED: (State.PENDING,),
    State.PENDING: (State.ADMITTED, State.CLOSING),
    State.ADMITTED: (State.RUNNING,),
    State.RUNNING: (State.CLOSING,),
    State.CLOSING: (State.SETTLING,),
    State.SETTLING: (State.TERMINAL,),
    State.TERMINAL: (),
}


class RunFailure(Exception):
    """The run broke a contract while executing, or selected code failed."""


class Cancelled(RunFailure):
    """Admission closed: the run was cancelled, or a sibling failed."""


@dataclass(frozen=True)
class Fact:
    """One owner-visible lifecycle fact, in the reference semantics' vocabulary."""

    owner: str
    invocation: Identity
    kind: str
    values: tuple[object, ...] = ()


@dataclass
class Outcome:
    owner: str
    invocation: Identity
    state: State = State.CREATED
    status: str = "pending"
    error: str | None = None


@dataclass(eq=False)
class Held:
    """One lease as the coordinator tracks it: the owner's lease and its protection."""

    token: int
    resource: str
    mode: AccessMode
    lease: Lease
    version: Version
    active: bool = True
    body_ended: bool = False

    @property
    def handle(self) -> object:
        return self.lease.handle


@dataclass(eq=False)
class Product:
    """A retained result: its value and the leases it keeps until its release (D16)."""

    key: tuple[str, int]
    value: object
    leases: tuple[Held, ...]
    parents: tuple[Product, ...]
    borrowers: int = 0
    active: bool = True


@dataclass(eq=False)
class Ticket:
    """An admitted or produced item's claim, or one of its forks."""

    stream: str
    key: int
    owner: Identity | None
    payload: object = None
    versions: tuple[Version, ...] = ()
    parent: Ticket | None = None
    forks: list[Ticket] = field(default_factory=list)
    disposition: tuple[str, ...] | None = None


@dataclass
class Stream:
    declaration: StreamOp
    outstanding: dict[int, Ticket] = field(default_factory=dict)
    ready: deque[Ticket] = field(default_factory=deque)
    next_key: int = 0
    closed: bool = False


@dataclass(eq=False)
class Request:
    handler: str
    ordinal: int
    cause: Identity
    payload: object
    snapshots: dict[str, Held] = field(default_factory=dict)
    reserved: dict[str, Version] = field(default_factory=dict)


@dataclass
class HandlerState:
    pending: deque[Request] = field(default_factory=deque)
    active: int = 0
    next_ordinal: int = 0


@dataclass
class Protection:
    """Who is using one resource: protected readers, the one writer and request reservations."""

    readers: set[int] = field(default_factory=set)
    writer: int | None = None
    reservations: set[int] = field(default_factory=set)


__all__ = [
    "TRANSITIONS",
    "Cancelled",
    "Fact",
    "HandlerState",
    "Held",
    "Identity",
    "Outcome",
    "Product",
    "Protection",
    "Request",
    "RunFailure",
    "State",
    "Stream",
    "Ticket",
]
