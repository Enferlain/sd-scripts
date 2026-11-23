"""The run authority: the single writer of participant records, and their owner at the seam (D6, D16, D19).

The Trainer creates it from an accepted run's seed at the start of
realization; it mints the run identity and publishes the declared, unbound
participants as the first generation (D17). From then on producers propose
transitions, which the authority judges against the accepted obligations
and publishes whole or not at all.

At the owner seam it realizes access to participants (D16): a lease goes
through a route, the normal one unless another is named, taken from the
current generation. A participant's numeric version is live state outside
the records: it advances when the runtime tells an update lease that a
call updated through it, and never otherwise. While a lease is out, the
authority refuses to publish a change to that participant's records; a
publication while the run executes goes through the runtime, which waits
for uses and reservations (D19).

A snapshot is the live route while nothing has updated the participant,
and then the participant may no longer be updated for the run. A snapshot
of a participant the run updates needs a copy of its device storage, which
the model definitions supply with section 18; until then a strategy that
both snapshots and updates a participant is refused at route rebinding and
readiness, since its lease obligations show both uses, and the check at the
use remains as a backstop. A grant, which comes with the second pass of
section 16, is refused the same way.
"""

import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from types import MappingProxyType

from library.iro.accepted import AcceptedRun
from library.iro.operation_contracts import AccessMode
from library.iro import seams
from library.iro.seams import HeldSetup, Use
from library.iro.verify import UPDATE_MODES, Checkpoint, ObligationKind, Obligations
from library.runstate.authority.records import NORMAL, Binding, Lifecycle, Participant, ParticipantRef, Route
from library.runstate.authority.section import AUTHORITY, AuthoritySection
from library.runstate.authority.transitions import (
    Declaration,
    Materialization,
    Published,
    Rejected,
    RouteRebinding,
    Transition,
    affected,
    judge_bound,
    judge_lease,
)
from library.runstate.generation import Generations, Publication, Section
from library.runstate.owners import Attempt, AttemptRef, Dependencies, Mint, Mutation, Refused, Stale, Version, new_run


class Unavailable(Exception):
    """A use of a participant the authority cannot realize now, with why."""


class AuthorityLease:
    """One use of a participant, granted from one generation; it never leaves its access (D16)."""

    __slots__ = ("_active", "_authority", "handle", "reference", "use", "version")

    def __init__(self, authority: RunAuthority, reference: ParticipantRef, use: Use, version: Version, handle: object):
        self._authority, self.reference, self.use, self.version, self.handle = authority, reference, use, version, handle
        self._active = True

    def release(self) -> None:
        self._authority._release(self)

    def updated(self) -> Version:
        """The runtime's notice that a call updated through this lease: the numeric version advances (D16)."""
        return self._authority._advance(self)


class HoldingLease(AuthorityLease):
    """A lease through a route with a held setup, which the runtime holds for the calls that use the lease (D16)."""

    __slots__ = ("_setup",)

    def __init__(self, authority: RunAuthority, reference: ParticipantRef, use: Use, version: Version, handle: object, setup: HeldSetup):
        super().__init__(authority, reference, use, version, handle)
        self._setup = setup

    @property
    def setup(self) -> HeldSetup:
        return self._setup


class RunAuthority:
    """The run authority of one run.

    Consumers never receive it: the coordinator gets the execution
    projection, which realizes the owner seam, and other consumers get
    their own projections (:mod:`library.runstate.projections`).
    """

    def __init__(self, run: AcceptedRun, generations: Generations):
        self.accepted = run
        self.generations = generations
        self.obligations: Obligations = run.obligations
        self._pins = {name: (version, digest) for name, version, digest in run.implementations}
        self._declared = {participant.name: participant for participant in run.seed.participants}
        self._updated = frozenset(
            o.use.resource for o in run.obligations if o.kind == ObligationKind.LEASE and o.use is not None and o.use.mode in UPDATE_MODES
        )
        self._participants = Mint(ParticipantRef, generations.run)
        self._lock = threading.Lock()
        self._outstanding: dict[ParticipantRef, int] = {}
        self._writers: dict[ParticipantRef, AuthorityLease] = {}
        self._numeric: dict[ParticipantRef, int] = {}
        self._pinned: set[ParticipantRef] = set()
        self._granted: dict[ParticipantRef, int] = {}  # leases ever granted on each participant
        self._in_place: dict[AttemptRef, dict[ParticipantRef, int]] = {}  # in-place attempts, with the grants each began after

    @classmethod
    def seed(cls, run: AcceptedRun, *, identity: str | None = None) -> RunAuthority:
        """Creates the run authority from an accepted run, minting the run identity, and publishes its declared participants.

        Relationships come with the second pass of section 15; until then a
        run that declares one is refused rather than seeded without it.
        """
        if run.seed.relationships:
            names = ", ".join(f"@{relationship.name}" for relationship in run.seed.relationships)
            raise Refused(f"the run declares relationships ({names}); they come with the second pass of section 15")
        if run.seed.obligations != run.obligations.revision:
            raise Refused("the run's authority seed was derived under other obligations")
        generations = Generations(identity or new_run())
        authority = cls(run, generations)
        generations.register(AUTHORITY, authority)
        declared = authority.submit(tuple(Declaration(participant.name) for participant in run.seed.participants))
        if isinstance(declared, Rejected):
            raise Refused(str(declared))
        return authority

    # Reading.

    @property
    def run(self) -> str:
        return self.generations.run

    @property
    def section(self) -> AuthoritySection:
        section = self.generations.current.sections.get(AUTHORITY)
        if not isinstance(section, AuthoritySection):
            raise RuntimeError("the run authority has published nothing yet")
        return section

    def incarnation(self, address: str) -> ParticipantRef:
        """The current incarnation of the participant declared at an address.

        Addresses only locate declarations (D19): an owner deriving what the
        accepted strategy defines over participants, such as an optimization
        unit's members, uses it to name them by reference.
        """
        return self.section.at(address).reference

    # Changing the records.

    def attempt(self) -> Attempt:
        """A new attempt based on the current generation and the authority's obligation revision."""
        return self.generations.attempt(self.obligations.revision)

    def submit(self, transitions: Sequence[Transition], attempt: Attempt | None = None) -> Published | Rejected:
        """Judges the transitions together and publishes the authority's next section alone, or rejects them all."""
        attempt = attempt or self.attempt()
        prospective = self.prospective(transitions, attempt)
        if isinstance(prospective, Rejected):
            return prospective
        section, invalidated = prospective
        try:
            generation = self.generations.publish(Publication(attempt.basis, (section,), attempt))
        except Stale as stale:
            return Rejected((str(stale),), stale=True)
        except Refused as refused:
            return Rejected((str(refused),))
        return Published(generation.number, section.revision, invalidated)

    def prospective(self, transitions: Sequence[Transition], attempt: Attempt) -> tuple[AuthoritySection, tuple[str, ...]] | Rejected:
        """The section the transitions would make, with what they invalidate, judged but not published.

        Preparation publishes it together with the other owners' sections.
        """
        current = self.generations.current.sections.get(AUTHORITY)
        assert current is None or isinstance(current, AuthoritySection)
        reasons: list[str] = []
        if attempt.basis.obligations != self.obligations.revision:
            reasons.append("the attempt was derived under another obligation revision")
        based = dict(attempt.basis.sections).get(AUTHORITY)
        if based != (current.revision if current is not None else None):
            reasons.append(f"the attempt was based on authority revision {based}, and {current.revision if current else None} is current")
        if reasons:
            return Rejected(tuple(reasons), stale=True)
        participants = dict(current.participants) if current is not None else {}
        addresses = dict(current.addresses) if current is not None else {}
        for transition in transitions:
            match transition:
                case Declaration():
                    self._declare(transition, participants, addresses, reasons)
                case Materialization():
                    self._materialize(transition, attempt, participants, reasons)
                case RouteRebinding():
                    self._rebind(transition, attempt, participants, reasons)
        section = AuthoritySection(
            current.revision + 1 if current is not None else 1,
            self.run,
            self.accepted.identity,
            self.obligations,
            MappingProxyType(participants),
            MappingProxyType(addresses),
        )
        for transition in transitions:
            if isinstance(transition, Materialization):
                participant = participants.get(transition.target)
                if participant is not None:
                    reasons += self._judge(ObligationKind.BOUND, participant, section)
            elif isinstance(transition, RouteRebinding):
                participant = participants.get(transition.target)
                if participant is not None:
                    reasons += self._judge(ObligationKind.LEASE, participant, section, route=transition.route)
        if reasons:
            return Rejected(tuple(reasons))
        return section, _invalidated(section)

    def _declare(self, transition: Declaration, participants: dict, addresses: dict, reasons: list[str]) -> None:
        declared = self._declared.get(transition.address)
        if declared is None:
            reasons.append(f"@{transition.address} is not a participant the accepted strategy declares")
        elif transition.address in addresses:
            reasons.append(f"@{transition.address} is already declared; a successor comes with the second pass of section 15")
        else:
            reference = self._participants()
            participants[reference] = Participant(reference, transition.address, declared)
            addresses[transition.address] = reference

    def _materialize(self, transition: Materialization, attempt: Attempt, participants: dict, reasons: list[str]) -> None:
        participant = participants.get(transition.target)
        if participant is None:
            reasons.append(f"{transition.target} is not a participant of this run")
        elif participant.lifecycle != Lifecycle.DECLARED:
            reasons.append(
                f"@{participant.address} is {participant.lifecycle}; bound-state replacement comes with the second pass of section 15"
            )
        elif transition.evidence.attempt != attempt.reference or transition.evidence.basis != attempt.basis:
            reasons.append(f"@{participant.address}'s evidence comes from another attempt")
        else:
            binding = Binding(transition.value, 1, transition.evidence)
            participants[participant.reference] = replace(participant, lifecycle=Lifecycle.BOUND, binding=binding)

    def _rebind(self, transition: RouteRebinding, attempt: Attempt, participants: dict, reasons: list[str]) -> None:
        participant = participants.get(transition.target)
        if participant is None:
            reasons.append(f"{transition.target} is not a participant of this run")
            return
        name = f"@{participant.address}"
        if participant.binding is None:
            reasons.append(f"{name} is not bound, so it has no routes")
            return
        existing = participant.routes.get(transition.route)
        if (existing.revision if existing is not None else None) != transition.expected:
            reasons.append(
                f"{name}'s {transition.route} route is at revision {existing.revision if existing else None}, not {transition.expected}"
            )
            return
        evidence = transition.evidence
        if evidence.attempt != attempt.reference or evidence.basis != attempt.basis:
            reasons.append(f"{name}'s {transition.route} route has evidence from another attempt")
            return
        if evidence.binding != participant.binding.revision:
            reasons.append(
                f"{name}'s {transition.route} route was prepared from binding {evidence.binding}, and {participant.binding.revision} is current"
            )
            return
        route = Route(
            transition.route,
            transition.callable,
            existing.revision + 1 if existing is not None else 1,
            Dependencies(exact=((AUTHORITY, str(participant.reference), evidence.binding),)),
            evidence,
            transition.setup,
        )
        routes = {**participant.routes, transition.route: route}
        participants[participant.reference] = replace(participant, routes=MappingProxyType(routes))

    def _judge(self, kind: ObligationKind, participant: Participant, section: AuthoritySection, *, route: str | None = None) -> list[str]:
        reasons = []
        for obligation in affected(self.obligations, kind, participant.address):
            if kind == ObligationKind.BOUND:
                reason = judge_bound(participant, self._pins)
            else:
                assert obligation.use is not None
                if route is not None and (obligation.use.view is not None or (obligation.use.route or NORMAL) != route):
                    continue
                reason = judge_lease(obligation, participant, section, updated=participant.address in self._updated)
            if reason is not None:
                reasons.append(f"{reason} ({obligation.source or obligation.subject})")
        return reasons

    # Readiness.

    def readiness(self, checkpoint: Checkpoint) -> tuple[str, ...]:
        """Why the participants' obligations at a readiness checkpoint do not hold now; empty once they all do.

        An executable run exists only after its core-execution checkpoint
        holds (D17); a handler's checkpoint holds before its first request.
        """
        section = self.section
        reasons: list[str] = []
        for participant in section.participants.values():
            for kind in (ObligationKind.BOUND, ObligationKind.LEASE):
                for obligation in affected(self.obligations, kind, participant.address):
                    if obligation.checkpoint != checkpoint:
                        continue
                    if kind == ObligationKind.BOUND:
                        reason = judge_bound(participant, self._pins)
                    else:
                        reason = judge_lease(obligation, participant, section, updated=participant.address in self._updated)
                    if reason is not None:
                        reasons.append(f"{reason} ({obligation.source or obligation.subject})")
        return tuple(reasons)

    # In-place work.

    def begin_in_place(self, attempt: AttemptRef, participants: Sequence[ParticipantRef]) -> tuple[str, ...]:
        """Registers in-place work on the participants, or says why there is something to withdraw first (D8, D19).

        Before a participant's first route or view is published, and while no
        lease on it is out, in-place work has nothing to withdraw, such as
        setting trainability or turning on gradient checkpointing. Withdrawal
        comes with the second pass of section 15, so in-place work on a
        participant that has a route or view is refused. When the work
        publishes, :meth:`admit` checks that no lease was taken on these
        participants meanwhile.
        """
        with self._lock:
            section = self.section
            reasons = []
            for reference in participants:
                participant = section.participants.get(reference)
                if participant is None:
                    reasons.append(f"{reference} is not a participant of this run")
                    continue
                name = f"@{participant.address}"
                if participant.routes or participant.views:
                    reasons.append(
                        f"{name} has a published route or view, which in-place work first withdraws; "
                        "withdrawal comes with the second pass of section 15"
                    )
                if self._outstanding.get(reference):
                    reasons.append(f"{name} has a lease out, which in-place work first withdraws")
            if not reasons:
                self._in_place[attempt] = {reference: self._granted.get(reference, 0) for reference in participants}
            return tuple(reasons)

    def end_in_place(self, attempt: AttemptRef) -> None:
        """Forgets an in-place attempt that will not publish."""
        with self._lock:
            self._in_place.pop(attempt, None)

    # Publication checks.

    @contextmanager
    def admit(self, publication: Publication, current: Section | None, next: Section) -> Generator[None]:
        """Refuses a publication that changes a participant's records while a lease on it is out (D19).

        A publication of in-place work is refused unless the attempt
        registered its participants, none has a lease out, no lease was taken
        on any of them while the work ran, and none has gained a route or view.
        The authority's lock is held until the swap is done, so no lease is
        taken from the generation the swap replaces after this check.
        """
        assert isinstance(next, AuthoritySection)
        assert current is None or isinstance(current, AuthoritySection)
        with self._lock:
            if current is not None:
                for reference, participant in next.participants.items():
                    if current.participants.get(reference) is not participant and self._outstanding.get(reference):
                        raise Refused(
                            f"@{participant.address} has a lease out; a publication while the run executes goes through the runtime (D19)"
                        )
            attempt = publication.attempt
            if attempt is not None and attempt.mutation == Mutation.IN_PLACE:
                held = self._in_place.pop(attempt.reference, None)
                if held is None:
                    raise Refused("in-place work publishes only from an attempt that registered its participants with the authority")
                for reference, granted in held.items():
                    before = current.participants.get(reference) if current is not None else None
                    name = f"@{before.address}" if before is not None else str(reference)
                    if self._outstanding.get(reference) or self._granted.get(reference, 0) != granted:
                        raise Refused(f"{name} was leased while in-place work on it ran")
                    if before is not None and (before.routes or before.views):
                        raise Refused(f"{name} gained a route or view while in-place work on it ran")
            yield

    # The owner seam.

    def resources(self) -> tuple[str, ...]:
        return tuple(self._declared)

    def current(self, resource: str) -> Version:
        participant = self._bound(self.section, resource)
        assert participant.binding is not None
        return Version(resource, participant.reference, participant.binding.revision, self._numeric.get(participant.reference, 0))

    def lease(self, uses: Sequence[Use], reserved: Mapping[str, seams.Version] | None = None) -> list[AuthorityLease]:
        """Leases for the uses, taken together from the current generation; if one cannot be taken, none is held.

        A reserved version stays current while the runtime holds its
        reservation, because updates wait for it, so the lease observes it.
        """
        del reserved
        granted: list[AuthorityLease] = []
        pinning: set[ParticipantRef] = set()
        with self._lock:
            section = self.section
            try:
                for use in uses:
                    granted.append(self._grant(section, use, pinning))
            except BaseException:
                for lease in granted:
                    self._release_held(lease)
                raise
            self._pinned |= pinning
        return granted

    def _grant(self, section: AuthoritySection, use: Use, pinning: set[ParticipantRef]) -> AuthorityLease:
        """One lease of a call that takes several together; ``pinning`` collects the snapshots it pins once all are granted."""
        participant = self._bound(section, use.resource)
        binding = participant.binding
        assert binding is not None
        reference = participant.reference
        name = f"@{participant.address}"
        numeric = self._numeric.get(reference, 0)
        if use.mode == AccessMode.GRANT:
            raise Unavailable(f"{name} is used through a grant; grants come with the second pass of section 16")
        if use.view is not None:
            raise Unavailable(f"{name} is used as its {use.view} view; views come with the second pass of section 15")
        setup = None  # an update runs nothing through a route, so no route setup covers it
        if use.mode == AccessMode.OWNER_UPDATE:
            if reference in self._pinned or reference in pinning:
                raise Unavailable(
                    f"{name} was read through a snapshot of its live state, so the run may not update it; a snapshot of a participant "
                    "the run updates needs a copy of its device storage, which comes with the model definitions (section 18)"
                )
            if reference in self._writers:
                raise Unavailable(f"{name} already has an update lease out")
            handle = binding.value
        else:
            route = self._route(section, participant, use.route or NORMAL)
            if use.mode == AccessMode.SNAPSHOT:
                if numeric or binding.revision != 1 or reference in self._writers:
                    raise Unavailable(
                        f"a snapshot of {name}, which the run has updated, needs a copy of its device storage, "
                        "which comes with the model definitions (section 18)"
                    )
                pinning.add(reference)
            handle, setup = route.callable, route.setup
        version = Version(use.resource, reference, binding.revision, numeric)
        if setup is None:
            lease = AuthorityLease(self, reference, use, version, handle)
        else:
            lease = HoldingLease(self, reference, use, version, handle, setup)
        if use.mode == AccessMode.OWNER_UPDATE:
            self._writers[reference] = lease
        self._outstanding[reference] = self._outstanding.get(reference, 0) + 1
        self._granted[reference] = self._granted.get(reference, 0) + 1
        return lease

    @staticmethod
    def _bound(section: AuthoritySection, resource: str) -> Participant:
        reference = section.addresses.get(resource)
        if reference is None:
            raise Unavailable(f"@{resource} is not a participant the accepted strategy declares")
        participant = section.participants[reference]
        if participant.binding is None:
            raise Unavailable(f"@{resource} is {participant.lifecycle}, not bound")
        return participant

    @staticmethod
    def _route(section: AuthoritySection, participant: Participant, name: str) -> Route:
        route = participant.routes.get(name)
        if route is None:
            raise Unavailable(f"@{participant.address} has no prepared {name} route")
        if route.dependencies.stale({AUTHORITY: section}):
            raise Unavailable(f"@{participant.address}'s {name} route was prepared from an earlier binding")
        return route

    def _advance(self, lease: AuthorityLease) -> Version:
        with self._lock:
            if not lease._active or self._writers.get(lease.reference) is not lease:
                raise Unavailable(f"@{lease.use.resource}: only its current update lease is told of updates")
            numeric = self._numeric.get(lease.reference, 0) + 1
            self._numeric[lease.reference] = numeric
            lease.version = Version(lease.version.resource, lease.reference, lease.version.bound, numeric)
            return lease.version

    def _release(self, lease: AuthorityLease) -> None:
        with self._lock:
            self._release_held(lease)

    def _release_held(self, lease: AuthorityLease) -> None:
        if not lease._active:
            raise Unavailable(f"@{lease.use.resource}: a lease is released twice")
        lease._active = False
        remaining = self._outstanding[lease.reference] - 1
        if remaining:
            self._outstanding[lease.reference] = remaining
        else:
            del self._outstanding[lease.reference]
        if self._writers.get(lease.reference) is lease:
            del self._writers[lease.reference]


def _invalidated(section: AuthoritySection) -> tuple[str, ...]:
    stale = []
    for participant in section.participants.values():
        for name, route in participant.routes.items():
            if route.dependencies.stale({AUTHORITY: section}):
                stale.append(f"@{participant.address} route {name}")
    return tuple(stale)


__all__ = ["AuthorityLease", "HoldingLease", "RunAuthority", "Unavailable"]
