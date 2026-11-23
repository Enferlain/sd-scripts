"""Preparation attempts: one coherent job derived from one generation (D8, D19).

The run-state layer defines the generic attempt, which a workflow without
training also runs. Its request names the routes to prepare, every
participant the job needs, including frozen ones that still need movement,
casting, wrapping or compilation, what it may change, and the other owners
that contribute their next sections; in a training run, training's
preparation coordination contributes prepared units and backend state
(D20).

A request that names a participant which is not bound, or that asks for
nothing to publish, is rejected before any work. Every contributor the
request names must supply its next section. The job reads its participants
through the preparation projection,
pinned to the generation the attempt was derived from, does its fallible
work off to the side, and returns a result whose parts stay apart by owner.
The result publishes in one publication, the authority's routes with every
contributor's section, only while that generation is still current;
otherwise it is stale and never publishes. A failed or refused result
publishes nothing, and the routes it would have replaced stay usable.

Publishing a result does not establish readiness: the host checks the
readiness checkpoint once every attempt it needs has published (D17).

In-place preparation changes the participants it requires in place, such as
setting trainability or turning on gradient checkpointing. Before a
participant's first route or view is published, and while no lease on it is
out, it has nothing to withdraw, and only then is it offered: in-place work
that first withdraws routes, views and leases comes with the second pass of
section 15 (D19). The authority checks at publication that no lease was
taken on those participants while the work ran. An attempt that fails after
changing a participant in place leaves the change, which the next attempt
makes again; nothing published relied on the participant yet.
"""

from dataclasses import dataclass
from types import MappingProxyType

from library.iro.seams import CallSetup, HeldSetup
from library.runstate.authority.authority import RunAuthority
from library.runstate.authority.records import ParticipantRef, SetupPin
from library.runstate.authority.section import AUTHORITY, AuthoritySection
from library.runstate.authority.transitions import Prepared, Published, Rejected, RouteRebinding
from library.runstate.generation import Publication, Section
from library.runstate.owners import Attempt, AttemptRef, Dependencies, Mutation, Refused, Stale
from library.runstate.projections import Preparation


@dataclass(frozen=True)
class PreparationRequest:
    """What one attempt is asked to prepare.

    ``routes`` names each route to prepare as ``(participant, route)``;
    ``participants`` names the other participants the job needs, such as
    frozen ones it moves or casts without preparing a route; ``contributors``
    names the owners whose next sections join the publication.
    """

    routes: tuple[tuple[ParticipantRef, str], ...] = ()
    participants: tuple[ParticipantRef, ...] = ()
    mutation: Mutation = Mutation.REPLACES
    contributors: tuple[str, ...] = ()
    pinned: tuple[SetupPin, ...] = ()  # setups the run already pins elsewhere, such as the prepared form's (D16)

    @property
    def required(self) -> tuple[ParticipantRef, ...]:
        """Every participant the job needs, in the order the request names them."""
        return tuple(dict.fromkeys((*self.participants, *(participant for participant, _ in self.routes))))


@dataclass(frozen=True)
class PreparedRoute:
    """A route's prepared callable, proposed for the authority, with the setups its realization carries (D16)."""

    participant: ParticipantRef
    route: str
    callable: object
    setup: HeldSetup | None = None
    call_setups: tuple[CallSetup, ...] = ()


@dataclass(frozen=True)
class PreparationResult:
    """A preparation attempt's result, its parts kept apart by owner (D8).

    ``routes`` go to the authority; ``contributed`` holds each contributing
    owner's next section; ``failures`` says why the job failed, in which
    case nothing publishes.
    """

    attempt: AttemptRef
    routes: tuple[PreparedRoute, ...] = ()
    contributed: tuple[Section, ...] = ()
    failures: tuple[str, ...] = ()


class PreparationAttempt:
    """One coherent preparation job, derived from one generation; :meth:`derive` makes one."""

    def __init__(self, authority: RunAuthority, request: PreparationRequest, attempt: Attempt, projection: Preparation):
        self.authority, self.request, self.attempt, self.projection = authority, request, attempt, projection

    @classmethod
    def derive(cls, authority: RunAuthority, request: PreparationRequest) -> PreparationAttempt | Rejected:
        """An attempt for the request on the current generation, or why current facts already rule it out."""
        if AUTHORITY in request.contributors:
            return Rejected(("the authority's routes are the attempt's own part, not a contribution",))
        if not request.routes and not request.contributors:
            return Rejected(("the request asks for no route and no contribution, so nothing would publish",))
        generation = authority.generations.current
        section = generation.sections.get(AUTHORITY)
        assert isinstance(section, AuthoritySection), "the authority publishes its section when it is seeded"
        reasons = []
        for reference in request.required:
            participant = section.participants.get(reference)
            if participant is None:
                reasons.append(f"{reference} is not a participant of this run")
            elif participant.binding is None:
                reasons.append(f"@{participant.address} is {participant.lifecycle}, not bound, so it cannot be prepared")
        if reasons:
            return Rejected(tuple(reasons))
        attempt = authority.generations.attempt(authority.obligations.revision, request.mutation, generation)
        if request.mutation == Mutation.IN_PLACE:
            refused = authority.begin_in_place(attempt.reference, request.required)
            if refused:
                return Rejected(refused)
        projection = Preparation(
            generation.number,
            MappingProxyType({reference: section.participants[reference] for reference in request.required}),
            Dependencies(whole=((AUTHORITY, section.revision),)),
        )
        return cls(authority, request, attempt, projection)

    def publish(self, result: PreparationResult) -> Published | Rejected:
        """Publishes the result's parts together, or nothing, with why; an attempt publishes at most once."""
        try:
            return self._publish(result)
        finally:
            if self.attempt.mutation == Mutation.IN_PLACE:
                self.authority.end_in_place(self.attempt.reference)

    def _conflicting_pins(self, result: PreparationResult) -> list[str]:
        """Setups that claim one name and version with different code (D16).

        It compares the routes this result prepares, the routes it keeps, and
        the setups the run pins elsewhere, such as the prepared form's. Every
        route in the record is live: withdrawal, which takes one out, comes
        with the second pass of section 15.
        """
        replaced = {(route.participant, route.route) for route in result.routes}
        codes: dict[tuple[str, str], set[str]] = {}
        for pin in self.request.pinned:
            codes.setdefault((pin.name, pin.version), set()).add(pin.code)
        for reference, participant in self.authority.section.participants.items():
            for name, route in participant.routes.items():
                if (reference, name) not in replaced:
                    for pin in route.pins:
                        codes.setdefault((pin.name, pin.version), set()).add(pin.code)
        for route in result.routes:
            for setup in (*((route.setup,) if route.setup else ()), *route.call_setups):
                pin = SetupPin.of(setup)
                codes.setdefault((pin.name, pin.version), set()).add(pin.code)
        return [
            f"@{name} v{version} is claimed by setups with different code; give changed code a new version"
            for (name, version), found in sorted(codes.items())
            if len(found) > 1
        ]

    def _publish(self, result: PreparationResult) -> Published | Rejected:
        if result.attempt != self.attempt.reference:
            return Rejected(("the result belongs to another attempt",))
        if result.failures:
            return Rejected(result.failures)
        reasons = []
        requested = set(self.request.routes)
        prepared = [(route.participant, route.route) for route in result.routes]
        reasons += [
            f"{participant} {route}: the request did not ask for this route"
            for participant, route in prepared
            if (participant, route) not in requested
        ]
        reasons += [
            f"{participant} {route}: prepared twice"
            for participant, route in sorted(set(prepared), key=lambda pair: (str(pair[0]), pair[1]))
            if prepared.count((participant, route)) > 1
        ]
        reasons += [
            f"{participant} {route}: the result does not prepare it"
            for participant, route in self.request.routes
            if (participant, route) not in prepared
        ]
        reasons += [
            f"{section.owner} is not a contributor to this attempt"
            for section in result.contributed
            if section.owner not in self.request.contributors
        ]
        contributed = {section.owner for section in result.contributed}
        reasons += [f"{owner} did not contribute its section" for owner in self.request.contributors if owner not in contributed]
        reasons += self._conflicting_pins(result)
        if reasons:
            return Rejected(tuple(reasons))
        transitions = []
        for route in result.routes:
            participant = self.projection[route.participant]
            assert participant.binding is not None
            existing = participant.routes.get(route.route)
            evidence = Prepared(self.attempt.reference, self.attempt.basis, participant.binding.revision)
            transitions.append(
                RouteRebinding(
                    route.participant,
                    route.route,
                    route.callable,
                    existing.revision if existing else None,
                    evidence,
                    route.setup,
                    tuple(route.call_setups),
                )
            )
        sections: tuple[Section, ...] = tuple(result.contributed)
        revision, invalidated = self.authority.section.revision, ()
        if transitions:
            prospective = self.authority.prospective(transitions, self.attempt)
            if isinstance(prospective, Rejected):
                return prospective
            authority_section, invalidated = prospective
            sections, revision = (authority_section, *sections), authority_section.revision
        try:
            generation = self.authority.generations.publish(Publication(self.attempt.basis, sections, self.attempt))
        except Stale as stale:
            return Rejected((str(stale),), stale=True)
        except Refused as refused:
            return Rejected((str(refused),))
        return Published(generation.number, revision, invalidated)


__all__ = ["PreparationAttempt", "PreparationRequest", "PreparationResult", "PreparedRoute"]
