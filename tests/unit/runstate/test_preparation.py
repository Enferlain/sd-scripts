"""Preparation attempts: one coherent job, parts kept apart by owner, published together or not at all (D8, D19)."""

import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

from library.iro.operation_contracts import AccessMode, VersionSelection
from library.iro.seams import Use
from library.iro.verify import CORE_EXECUTION
from library.runstate.authority import NORMAL, AuthoritySection, Prepared, Published, Rejected, RouteRebinding
from library.runstate.authority.records import ParticipantRef
from library.runstate.generation import Publication
from library.runstate.owners import Mutation, Refused
from library.runstate.preparation import PreparationAttempt, PreparationRequest, PreparationResult, PreparedRoute
from tests.unit.runstate.test_authority import Box, materialized, routed, seeded

OPTIMIZATION = "trainer optimization"


@dataclass(frozen=True)
class Units:
    """A stand-in for a contributing owner's next section, such as Trainer optimization's prepared units."""

    revision: int
    units: tuple[str, ...] = ()
    owner: str = OPTIMIZATION

    def revision_of(self, key: str) -> int | None:
        return self.revision if key in self.units else None


@dataclass
class Contributor:
    refuse: str | None = None
    admitted: list = field(default_factory=list)

    @contextmanager
    def admit(self, publication, current, next) -> Generator[None]:
        if self.refuse:
            raise Refused(self.refuse)
        self.admitted.append(next)
        yield


def refs(authority, *names: str) -> tuple[ParticipantRef, ...]:
    return tuple(authority.section.at(name).reference for name in names)


def every_normal_route(authority) -> PreparationRequest:
    return PreparationRequest(routes=tuple((reference, NORMAL) for reference in refs(authority, "m", "k", "n")))


def derived(authority, request: PreparationRequest) -> PreparationAttempt:
    attempt = PreparationAttempt.derive(authority, request)
    assert isinstance(attempt, PreparationAttempt), attempt
    return attempt


def prepared_routes(attempt: PreparationAttempt, wrap=lambda value: ("prepared", value)) -> tuple[PreparedRoute, ...]:
    routes = []
    for reference, route in attempt.request.routes:
        binding = attempt.projection[reference].binding
        assert binding is not None
        routes.append(PreparedRoute(reference, route, wrap(binding.value)))
    return tuple(routes)


def test_a_request_that_current_facts_rule_out_is_rejected_before_any_work() -> None:
    authority = seeded()
    result = PreparationAttempt.derive(authority, PreparationRequest(routes=((refs(authority, "m")[0], NORMAL),)))
    assert isinstance(result, Rejected) and "@m is declared, not bound, so it cannot be prepared" in str(result)
    result = PreparationAttempt.derive(materialized(authority), PreparationRequest(mutation=Mutation.IN_PLACE))
    assert isinstance(result, Rejected) and "asks for no route and no contribution" in str(result)


def test_one_publication_installs_every_prepared_route_and_makes_the_run_ready() -> None:
    authority = materialized()
    attempt = derived(authority, every_normal_route(authority))
    before = authority.generations.current.number
    result = attempt.publish(PreparationResult(attempt.attempt.reference, prepared_routes(attempt)))
    assert isinstance(result, Published) and result.generation == before + 1, "one generation for the whole result"
    m = authority.section.at("m")
    assert m.binding is not None and m.routes[NORMAL].callable == ("prepared", m.binding.value)
    assert authority.readiness(CORE_EXECUTION) == ()


def test_a_frozen_participant_is_part_of_the_job_without_a_route() -> None:
    """The job includes a participant it moves or casts even though it prepares no route for it and no unit trains it."""
    authority = materialized()
    m, n = refs(authority, "m", "n")
    attempt = derived(authority, PreparationRequest(routes=((m, NORMAL),), participants=(n,)))
    assert set(attempt.projection.participants) == {n, m}
    assert isinstance(attempt.publish(PreparationResult(attempt.attempt.reference, prepared_routes(attempt))), Published)


def test_contributed_sections_publish_in_the_same_generation_as_the_routes() -> None:
    authority = materialized()
    contributor = Contributor()
    authority.generations.register(OPTIMIZATION, contributor)
    request = PreparationRequest(routes=every_normal_route(authority).routes, contributors=(OPTIMIZATION,))
    attempt = derived(authority, request)
    result = attempt.publish(PreparationResult(attempt.attempt.reference, prepared_routes(attempt), (Units(1, ("main",)),)))
    assert isinstance(result, Published)
    generation = authority.generations.current
    assert generation.section(OPTIMIZATION) == Units(1, ("main",)) and isinstance(generation.section("run authority"), AuthoritySection)
    assert generation.section(OPTIMIZATION) is not generation.section("run authority"), "joint creation keeps ownership apart"


def test_a_contributor_that_refuses_its_section_stops_the_routes_too() -> None:
    """Final publication installs only complete, validated state: no owner publishes a partial result."""
    authority = materialized()
    authority.generations.register(OPTIMIZATION, Contributor(refuse="the optimizer targets parameters the wrapper replaced"))
    request = PreparationRequest(routes=every_normal_route(authority).routes, contributors=(OPTIMIZATION,))
    attempt = derived(authority, request)
    result = attempt.publish(PreparationResult(attempt.attempt.reference, prepared_routes(attempt), (Units(1),)))
    assert isinstance(result, Rejected) and "wrapper replaced" in str(result)
    assert all(not authority.section.at(name).routes for name in ("m", "k", "n"))


def test_a_failed_job_publishes_nothing_and_the_old_routes_stay_usable() -> None:
    authority = routed()
    before = authority.section.at("m").routes[NORMAL]
    attempt = derived(authority, every_normal_route(authority))
    result = attempt.publish(PreparationResult(attempt.attempt.reference, failures=("wrapper creation failed",)))
    assert isinstance(result, Rejected) and str(result) == "wrapper creation failed"
    assert authority.section.at("m").routes[NORMAL] is before
    assert authority.readiness(CORE_EXECUTION) == ()


def test_a_result_derived_before_a_later_change_never_publishes() -> None:
    authority = routed()
    attempt = derived(authority, every_normal_route(authority))
    other = authority.attempt()
    m = authority.section.at("m")
    assert m.binding is not None
    change = RouteRebinding(m.reference, NORMAL, "newer", 1, Prepared(other.reference, other.basis, m.binding.revision))
    assert isinstance(authority.submit([change], other), Published)
    result = attempt.publish(PreparationResult(attempt.attempt.reference, prepared_routes(attempt)))
    assert isinstance(result, Rejected) and result.stale
    assert authority.section.at("m").routes[NORMAL].callable == "newer"
    assert attempt.projection[m.reference].routes[NORMAL] is m.routes[NORMAL], "the attempt's projection stays pinned to its generation"
    assert attempt.projection.dependencies.stale({"run authority": authority.section}) == ("run authority",)


def test_a_result_must_answer_exactly_its_request() -> None:
    authority = materialized()
    m, k, n = refs(authority, "m", "k", "n")
    attempt = derived(authority, PreparationRequest(routes=((m, NORMAL), (k, NORMAL))))
    reference = attempt.attempt.reference
    missing = attempt.publish(PreparationResult(reference, (PreparedRoute(m, NORMAL, "x"),)))
    assert isinstance(missing, Rejected) and "does not prepare it" in str(missing)
    extra = attempt.publish(
        PreparationResult(reference, (PreparedRoute(m, NORMAL, "x"), PreparedRoute(k, NORMAL, "y"), PreparedRoute(n, NORMAL, "z")))
    )
    assert isinstance(extra, Rejected) and "did not ask for this route" in str(extra)
    foreign = attempt.publish(
        PreparationResult(authority.attempt().reference, (PreparedRoute(m, NORMAL, "x"), PreparedRoute(k, NORMAL, "y")))
    )
    assert isinstance(foreign, Rejected) and "another attempt" in str(foreign)
    uninvited = attempt.publish(PreparationResult(reference, (PreparedRoute(m, NORMAL, "x"), PreparedRoute(k, NORMAL, "y")), (Units(1),)))
    assert isinstance(uninvited, Rejected) and "not a contributor" in str(uninvited)
    assert all(not authority.section.at(name).routes for name in ("m", "k", "n")), "nothing published"


def test_a_request_must_ask_for_something_the_attempt_can_publish() -> None:
    authority = materialized()
    assert "asks for no route and no contribution" in str(
        PreparationAttempt.derive(authority, PreparationRequest(participants=refs(authority, "n")))
    )
    named = PreparationAttempt.derive(
        authority, PreparationRequest(routes=every_normal_route(authority).routes, contributors=("run authority",))
    )
    assert isinstance(named, Rejected) and "not a contribution" in str(named)
    foreign = PreparationAttempt.derive(authority, PreparationRequest(routes=((ParticipantRef("another run", 1), NORMAL),)))
    assert isinstance(foreign, Rejected) and "not a participant of this run" in str(foreign)


def test_every_named_contributor_must_supply_its_section() -> None:
    """The routes never publish without the sections that belong with them, such as the prepared units."""
    authority = materialized()
    authority.generations.register(OPTIMIZATION, Contributor())
    attempt = derived(authority, PreparationRequest(routes=every_normal_route(authority).routes, contributors=(OPTIMIZATION,)))
    result = attempt.publish(PreparationResult(attempt.attempt.reference, prepared_routes(attempt)))
    assert isinstance(result, Rejected) and f"{OPTIMIZATION} did not contribute its section" in str(result)
    assert all(not authority.section.at(name).routes for name in ("m", "k", "n"))


def test_a_route_prepared_twice_is_refused() -> None:
    authority = materialized()
    (m,) = refs(authority, "m")
    attempt = derived(authority, PreparationRequest(routes=((m, NORMAL),)))
    result = attempt.publish(PreparationResult(attempt.attempt.reference, (PreparedRoute(m, NORMAL, "x"), PreparedRoute(m, NORMAL, "y"))))
    assert isinstance(result, Rejected) and "prepared twice" in str(result)


def test_a_contributor_s_own_publication_makes_the_result_stale_too() -> None:
    """The basis is the whole generation: a change in another owner's section, with the authority's unchanged, still makes the result stale."""
    authority = materialized()
    authority.generations.register(OPTIMIZATION, Contributor())
    attempt = derived(authority, PreparationRequest(routes=every_normal_route(authority).routes, contributors=(OPTIMIZATION,)))
    authority.generations.publish(Publication(authority.generations.basis(authority.obligations.revision), (Units(1, ("main",)),)))
    result = attempt.publish(PreparationResult(attempt.attempt.reference, prepared_routes(attempt), (Units(2, ("main",)),)))
    assert isinstance(result, Rejected) and result.stale
    assert authority.generations.current.section(OPTIMIZATION) == Units(1, ("main",))


def test_a_result_that_rebinds_a_participant_with_a_lease_out_is_refused() -> None:
    authority = routed()
    (lease,) = authority.lease([Use("m", AccessMode.PROTECTED_CURRENT, VersionSelection.CURRENT_AT_START)])
    attempt = derived(authority, every_normal_route(authority))
    result = attempt.publish(PreparationResult(attempt.attempt.reference, prepared_routes(attempt)))
    assert isinstance(result, Rejected) and not result.stale and "has a lease out" in str(result)
    assert authority.section.at("m").routes[NORMAL].callable is lease.handle
    lease.release()


def test_a_route_never_changes_under_a_lease_held_on_another_thread() -> None:
    """Without the GIL: leases taken on other threads and publications that rebind the route never overlap."""
    authority = routed()
    (m,) = refs(authority, "m")
    use = Use("m", AccessMode.PROTECTED_CURRENT, VersionSelection.CURRENT_AT_START)
    mismatches: list[str] = []
    errors: list[BaseException] = []
    held: list[int] = []
    done = threading.Event()

    def hold() -> None:
        count = 0
        try:
            # Lease until the publications are done, so the two always overlap however late either thread starts.
            while not done.is_set() or count < 300:
                (lease,) = authority.lease([use])
                count += 1
                for _ in range(3):
                    if authority.section.at("m").routes[NORMAL].callable is not lease.handle:
                        mismatches.append("the route changed under a held lease")
                lease.release()
        except BaseException as failed:
            errors.append(failed)
        held.append(count)

    workers = [threading.Thread(target=hold) for _ in range(4)]
    for worker in workers:
        worker.start()
    published = refused = 0
    for _ in range(300):
        attempt = PreparationAttempt.derive(authority, PreparationRequest(routes=((m, NORMAL),)))
        if not isinstance(attempt, PreparationAttempt):
            continue
        result = attempt.publish(PreparationResult(attempt.attempt.reference, (PreparedRoute(m, NORMAL, object()),)))
        if isinstance(result, Published):
            published += 1
        else:
            refused += 1
    done.set()
    for worker in workers:
        worker.join()
    assert errors == [] and mismatches == []
    assert published + refused > 0 and sum(held) >= 1200
    assert not authority._outstanding


# In-place preparation before a participant's first route (D19).


def in_place(authority, *names: str) -> PreparationAttempt:
    return derived(authority, PreparationRequest(routes=tuple((r, NORMAL) for r in refs(authority, *names)), mutation=Mutation.IN_PLACE))


def test_in_place_work_before_the_first_route_has_nothing_to_withdraw_and_publishes() -> None:
    authority = materialized()
    attempt = in_place(authority, "m")
    binding = attempt.projection[refs(authority, "m")[0]].binding
    assert binding is not None and isinstance(binding.value, Box)
    binding.value.value += 1  # the work changes the bound object in place, as setting trainability does
    result = attempt.publish(PreparationResult(attempt.attempt.reference, prepared_routes(attempt)))
    assert isinstance(result, Published), result
    assert authority.section.at("m").routes[NORMAL].callable == ("prepared", binding.value)


def test_in_place_work_after_a_route_is_published_waits_for_withdrawal() -> None:
    authority = materialized()
    attempt = derived(authority, every_normal_route(authority))
    assert isinstance(attempt.publish(PreparationResult(attempt.attempt.reference, prepared_routes(attempt))), Published)
    result = PreparationAttempt.derive(
        authority, PreparationRequest(routes=((refs(authority, "m")[0], NORMAL),), mutation=Mutation.IN_PLACE)
    )
    assert isinstance(result, Rejected) and "withdrawal comes with the second pass of section 15" in str(result)


def test_in_place_work_on_a_participant_leased_while_it_ran_never_publishes() -> None:
    authority = materialized()
    attempt = in_place(authority, "m")
    (lease,) = authority.lease([Use("m", AccessMode.OWNER_UPDATE, VersionSelection.OWNER_PROTOCOL)])
    lease.release()
    result = attempt.publish(PreparationResult(attempt.attempt.reference, prepared_routes(attempt)))
    assert isinstance(result, Rejected) and "@m was leased while in-place work on it ran" in str(result)
    assert not authority.section.at("m").routes, "nothing published"


def test_in_place_work_on_a_participant_with_a_lease_out_is_refused_before_it_starts() -> None:
    authority = materialized()
    (lease,) = authority.lease([Use("m", AccessMode.OWNER_UPDATE, VersionSelection.OWNER_PROTOCOL)])
    result = PreparationAttempt.derive(
        authority, PreparationRequest(routes=((refs(authority, "m")[0], NORMAL),), mutation=Mutation.IN_PLACE)
    )
    assert isinstance(result, Rejected) and "@m has a lease out" in str(result)
    lease.release()
