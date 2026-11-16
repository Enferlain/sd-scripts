"""The run authority's records (D19).

A participant's record keeps four layers apart: its reference (identity),
its declaration (meaning), its binding with the routes and views prepared
from it (realization), and, outside the record, its numeric version (live
state), which changes every step without changing a record.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from library.iro.accepted import DeclaredParticipant
from library.iro.seams import CallContext
from library.runstate.owners import Dependencies, Evidence, Reference

NORMAL = "normal"
"""The route an access goes through unless it names another (D16)."""


class ParticipantRef(Reference):
    """One participant incarnation, minted by the run authority."""

    kind = "participant"


class Lifecycle(StrEnum):
    """A participant's lifecycle: declared and unbound, bound, or retired (D6, D7)."""

    DECLARED = "declared"
    BOUND = "bound"
    RETIRED = "retired"


@dataclass(frozen=True)
class Binding:
    """A participant's bound state: the loaded object, its binding revision and the evidence it was accepted with."""

    value: object
    revision: int
    evidence: Evidence


Establish = Callable[[CallContext, object], None]
"""What a route establishes for each call that uses it, given the call's context and the handle the call receives (D15)."""


@dataclass(frozen=True)
class Route:
    """A named way to call a participant: its current prepared callable, route revision and dependencies.

    The dependencies include the binding it was prepared from; a route whose
    dependencies no longer hold is withdrawn until preparation publishes a
    replacement. ``establish``, where preparation gives one, sets up what
    the participant needs for each call through the route, such as its
    modules' mode; a lease through the route runs it (D15, D20).
    """

    name: str
    callable: object
    revision: int
    dependencies: Dependencies
    evidence: Evidence
    establish: Establish | None = None


@dataclass(frozen=True)
class View:
    """A typed, revision-pinned look at a participant that is not a route, such as its inspection view."""

    name: str
    value: object
    revision: int
    dependencies: Dependencies
    evidence: Evidence


@dataclass(frozen=True)
class Participant:
    """The authority's record of one participant incarnation.

    ``address`` locates its declaration in the accepted strategy; it never
    establishes identity. A state-only participant may have no route.
    """

    reference: ParticipantRef
    address: str
    declaration: DeclaredParticipant
    lifecycle: Lifecycle = Lifecycle.DECLARED
    binding: Binding | None = None
    routes: Mapping[str, Route] = field(default_factory=lambda: MappingProxyType({}))
    views: Mapping[str, View] = field(default_factory=lambda: MappingProxyType({}))


__all__ = ["NORMAL", "Binding", "Establish", "Lifecycle", "Participant", "ParticipantRef", "Route", "View"]
