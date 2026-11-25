"""The run authority's records (D19).

A participant's record keeps four layers apart: its reference (identity),
its declaration (meaning), its binding with the routes and views prepared
from it (realization), and, outside the record, its numeric version (live
state), which changes every step without changing a record.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from library.iro.accepted import DeclaredParticipant
from library.iro.evidence import CodeEvidence
from library.iro.seams import CallSetup, HeldSetup
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


@dataclass(frozen=True)
class Route:
    """A named way to call a participant: its current prepared callable, route revision and dependencies.

    The dependencies include the binding it was prepared from; a route whose
    dependencies no longer hold is withdrawn until preparation publishes a
    replacement. ``setup``, where preparation gives one, is the route's held
    setup: what the participant's realization needs for the calls through
    the route, such as its modules' mode; ``call_setups`` are what the calls
    through it need while they run, such as autocast for its mixed
    precision. Every lease through the route carries them, and
    ``setup_records`` records each by its name and the evidence of its
    source; what the realization configures them with is recorded with its
    settings in the strategy (D16, D20).
    """

    name: str
    callable: object
    revision: int
    dependencies: Dependencies
    evidence: Evidence
    setup: HeldSetup | None = None
    call_setups: tuple[CallSetup, ...] = ()
    setup_records: tuple[CodeEvidence, ...] = ()


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


__all__ = ["NORMAL", "Binding", "Lifecycle", "Participant", "ParticipantRef", "Route", "View"]
