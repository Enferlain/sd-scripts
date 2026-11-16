"""Transitions, the evidence they carry, and the evaluators that judge it (D7, D19).

Producers never write records. They propose a transition: its target, the
attempt that produced it, the revisions it expects, its candidate and its
evidence. The authority builds the prospective section, checks the expected
revisions, runs the evaluator of every obligation the change affects, and
publishes the whole section or nothing.

The kinds are a closed set. This pass realizes declaration, materialization
and route rebinding, which an initial realization needs; the others come
with the second pass of section 15, and nothing can propose them before.
Evidence kinds are declared here beside the evaluators that judge them,
one per kind of obligation the authority judges.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from library.iro.operation_contracts import AccessMode
from library.iro.verify import Evaluator, Obligation, ObligationKind, Obligations
from library.runstate.authority.records import NORMAL, Establish, Participant, ParticipantRef
from library.runstate.authority.section import AUTHORITY, AuthoritySection
from library.runstate.owners import Evidence


class TransitionKind(StrEnum):
    """The closed set of changes to the authority's records: D6's operations and D8's withdrawal."""

    DECLARATION = "declaration"
    MATERIALIZATION = "materialization"
    BOUND_STATE_REPLACEMENT = "bound-state replacement"
    ROUTE_REBINDING = "route rebinding"
    RELATIONSHIP_TRANSITION = "relationship transition"
    ARRANGEMENT_AMENDMENT = "arrangement amendment"
    MERGE_OR_FOLD = "merge or fold"
    RETIREMENT = "retirement"
    WITHDRAWAL = "withdrawal"


# Evidence, beside its evaluators.


@dataclass(frozen=True)
class Materialized(Evidence):
    """That a binding came from the participant's declared materializer, at a version and declaration digest (bound)."""

    materializer: str
    version: str
    digest: str


@dataclass(frozen=True)
class Supplied(Evidence):
    """That a participant declared without a materializer was bound to what its owner supplied (bound)."""

    supplier: str


@dataclass(frozen=True)
class Prepared(Evidence):
    """That a route was prepared from the participant's binding at one binding revision (lease)."""

    binding: int


def judge_bound(participant: Participant, pins: Mapping[str, tuple[str, str]]) -> str | None:
    """The evaluator of a participant's ``bound`` obligation: why it does not hold, or None.

    A participant declared with a materializer is bound by that
    materializer at the implementation version and declaration digest the
    run pins; one declared without is bound to what its owner supplied.
    """
    name = f"@{participant.address}"
    binding = participant.binding
    if binding is None:
        return f"{name} is not bound"
    declared = participant.declaration.materializer
    evidence = binding.evidence
    if declared is None:
        if not isinstance(evidence, Supplied):
            return f"{name} is declared without a materializer, so its owner supplies it"
        return None
    if not isinstance(evidence, Materialized) or evidence.materializer != declared:
        return f"{name} is materialized by @{declared}"
    pinned = pins.get(declared)
    if pinned is None:
        return f"@{declared} materializes {name} and the run does not pin it"
    if (evidence.version, evidence.digest) != pinned:
        return f"@{declared} materialized {name} at v{evidence.version}, and the run pins v{pinned[0]} with its declaration"
    return None


def judge_lease(obligation: Obligation, participant: Participant, section: AuthoritySection, *, updated: bool) -> str | None:
    """The evaluator of a ``lease`` obligation on a participant: why the authority cannot realize the use, or None.

    The participant is bound, and the use's route is prepared from its
    current binding. ``updated`` says some use in the run updates the
    participant, so a snapshot of it needs a copy of its device storage,
    which the model definitions supply in section 18. Grants come with the
    second pass of section 16 and views with the second pass of section 15. Each
    fails here, before the run starts, instead of at the use.
    """
    use = obligation.use
    assert use is not None, "a lease obligation carries its use"
    name = f"@{participant.address}"
    binding = participant.binding
    if binding is None:
        return f"{name} is not bound"
    if use.mode == AccessMode.GRANT:
        return f"{name} is used through a grant; grants come with the second pass of section 16"
    if use.view is not None:
        return f"{name} is used as its {use.view} view; views come with the second pass of section 15"
    if use.mode == AccessMode.SNAPSHOT and updated:
        return f"a snapshot of {name}, which the run updates, needs a copy of its device storage, which comes with the model definitions (section 18)"
    route_name = use.route or NORMAL
    route = participant.routes.get(route_name)
    if route is None:
        return f"{name} has no prepared {route_name} route"
    if (
        route.dependencies.stale({AUTHORITY: section})
        or not isinstance(route.evidence, Prepared)
        or route.evidence.binding != binding.revision
    ):
        return f"{name}'s {route_name} route was prepared from an earlier binding"
    return None


def affected(obligations: Obligations, kind: ObligationKind, address: str) -> Iterator[Obligation]:
    """The authority's obligations of one kind about one participant."""
    name = f"@{address}"
    for obligation in obligations:
        if obligation.kind != kind:
            continue
        if (kind == ObligationKind.BOUND and obligation.evaluator == Evaluator.RUN_AUTHORITY and obligation.names == (name,)) or (
            kind == ObligationKind.LEASE and obligation.use is not None and obligation.use.resource == address
        ):
            yield obligation


# Transitions.


@dataclass(frozen=True)
class Declaration:
    """Creates a declared, unbound incarnation at an address the accepted strategy declares."""

    address: str
    kind: ClassVar[TransitionKind] = TransitionKind.DECLARATION


@dataclass(frozen=True)
class Materialization:
    """Gives a declared participant its first binding."""

    target: ParticipantRef
    value: object
    evidence: Materialized | Supplied
    kind: ClassVar[TransitionKind] = TransitionKind.MATERIALIZATION


@dataclass(frozen=True)
class RouteRebinding:
    """Installs a route's new prepared callable, with what it establishes per call; ``expected`` is the route's current revision, or None for a new route."""

    target: ParticipantRef
    route: str
    callable: object
    expected: int | None
    evidence: Prepared
    establish: Establish | None = None
    kind: ClassVar[TransitionKind] = TransitionKind.ROUTE_REBINDING


type Transition = Declaration | Materialization | RouteRebinding


# Results.


@dataclass(frozen=True)
class Published:
    """An accepted change: the generation it installed, the authority's new revision and what it invalidated."""

    generation: int
    revision: int
    invalidated: tuple[str, ...] = ()


@dataclass(frozen=True)
class Rejected:
    """A refused change, with every reason; nothing changed.

    ``stale`` says it was refused only because it was derived from state
    that is no longer current, so deriving it again may succeed.
    """

    reasons: tuple[str, ...]
    stale: bool = False

    def __str__(self) -> str:
        return "; ".join(self.reasons)


__all__ = [
    "Declaration",
    "Materialization",
    "Materialized",
    "Prepared",
    "Published",
    "Rejected",
    "RouteRebinding",
    "Supplied",
    "Transition",
    "TransitionKind",
    "affected",
    "judge_bound",
    "judge_lease",
]
