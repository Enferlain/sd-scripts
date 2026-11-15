"""The run authority: seeding, materialization, route rebinding, readiness and the owner seam (D6, D7, D16, D19)."""

import asyncio
import dataclasses
from pathlib import Path

import pytest

from library.iro.accepted import cold_load
from library.iro.devices import Host
from library.iro.frontend import implementation
from library.iro.operation_contracts import AccessMode, VersionSelection
from library.iro.prepare import Target, prepare
from library.iro.runtime import Coordinator, Fact
from library.iro.seams import Owner, Use
from library.iro.verify import CORE_EXECUTION
from library.runstate.authority import (
    NORMAL,
    Lifecycle,
    MaterializationAttempt,
    Materialization,
    Materialized,
    Prepared,
    Published,
    Rejected,
    RouteRebinding,
    RunAuthority,
    Supplied,
    Unavailable,
)
from library.runstate.owners import Refused, Version
from library.runstate.projections import Execution
from tests.unit.iro.accepted_runs import accepted_run

STRATEGY = """iro.strategy @s uses(@load v1, @look v1, @step v1, @peek v1) {
  %m = iro.resource @m : participant materialize(@load)
  %k = iro.resource @k : participant materialize(@load)
  %n = iro.resource @n : participant
  iro.sequence @once capture(%m, %k, %n) {
    iro.access %m protected_current {
    ^(%l):
      iro.call @look(%l)
    }
    iro.access %m owner_update {
    ^(%u):
      iro.call @step(%u)
    }
    iro.access %n snapshot {
    ^(%l):
      iro.call @peek(%l)
    }
    iro.access %k protected_current {
    ^(%l):
      iro.call @look(%l)
    }
  }
}
"""


class Box:
    """A participant's loaded object in these tests: a value changed in place."""

    def __init__(self, value: int):
        self.value = value


LOADS: list[int] = []


@implementation(version=1, outputs=("weights", "weights"), name="load")
def load():
    LOADS.append(1)
    return Box(1), Box(2)


@implementation(version=1, inputs=("protected_current lease",), name="look")
def look(box):
    return None


@implementation(version=1, inputs=("owner_update lease",), name="step")
def step(box):
    box.value += 1


@implementation(version=1, inputs=("snapshot lease",), name="peek")
def peek(box):
    return None


INSTALLED = {"load": load, "look": look, "step": step, "peek": peek}


def seeded() -> RunAuthority:
    return RunAuthority.seed(accepted_run(STRATEGY, installed=INSTALLED))


def materialized(authority: RunAuthority | None = None) -> RunAuthority:
    authority = authority or seeded()
    result = MaterializationAttempt(authority, ["m", "k", "n"], installed=INSTALLED, device=Host(), supplied={"n": Box(3)}).run()
    assert isinstance(result, Published), result
    return authority


def routed(authority: RunAuthority | None = None, names=("m", "k", "n")) -> RunAuthority:
    authority = authority or materialized()
    attempt = authority.attempt()
    section = authority.section
    transitions = []
    for name in names:
        participant = section.at(name)
        assert participant.binding is not None
        transitions.append(
            RouteRebinding(participant.reference, NORMAL, participant.binding.value, None, Prepared(attempt.reference, attempt.basis, 1))
        )
    result = authority.submit(transitions, attempt)
    assert isinstance(result, Published), result
    return authority


# Seeding.


def test_seeding_declares_every_participant_unbound_under_its_own_reference() -> None:
    authority = seeded()
    section = authority.section
    assert (authority.generations.current.number, section.revision) == (1, 1)
    references = [section.at(name).reference for name in ("m", "k", "n")]
    assert len(set(references)) == 3 and all(reference.run == authority.run for reference in references)
    assert all(section.at(name).lifecycle == Lifecycle.DECLARED and section.at(name).binding is None for name in ("m", "k", "n"))
    assert seeded().run != authority.run, "another run of the same strategy is another run, with its own references"


def test_a_run_that_declares_relationships_is_refused_until_they_come() -> None:
    wide = accepted_run((Path(__file__).parents[1] / "iro" / "wide.iro").read_text())
    with pytest.raises(Refused, match="relationships"):
        RunAuthority.seed(wide)


# Materialization.


def test_one_materializer_supplies_its_participants_in_one_call_under_distinct_references() -> None:
    """Two participants from one source keep distinct identities; the source is lineage, not identity."""
    LOADS.clear()
    authority = materialized()
    section = authority.section
    m, k, n = (section.at(name) for name in ("m", "k", "n"))
    assert LOADS == [1], "participants naming the same materializer are materialized by one call"
    assert m.reference != k.reference
    values = [participant.binding.value if participant.binding is not None else None for participant in (m, k, n)]
    assert [box.value if isinstance(box, Box) else None for box in values] == [1, 2, 3]
    assert m.binding is not None and n.binding is not None
    assert isinstance(m.binding.evidence, Materialized) and m.binding.evidence.digest == load.declaration.digest
    assert isinstance(n.binding.evidence, Supplied)
    assert all(participant.lifecycle == Lifecycle.BOUND for participant in (m, k, n))


def test_a_binding_counts_only_from_the_declared_materializer_at_its_pinned_version() -> None:
    authority = seeded()
    other = implementation(version=2, outputs=("weights", "weights"), name="load")(lambda: (Box(1), Box(2)))
    result = MaterializationAttempt(authority, ["m", "k"], installed={"load": other}, device=Host()).run()
    assert isinstance(result, Rejected) and "the run pins v1" in str(result)
    assert authority.section.revision == 1 and authority.section.at("m").binding is None, "nothing changed"


def test_a_failing_materializer_changes_nothing() -> None:
    authority = seeded()

    def broken():
        raise OSError("checkpoint not found")

    failing = implementation(version=1, outputs=("weights", "weights"), name="load")(broken)
    result = MaterializationAttempt(authority, ["m", "k"], installed={"load": failing}, device=Host()).run()
    assert isinstance(result, Rejected) and "checkpoint not found" in str(result)
    assert authority.section.revision == 1


def test_one_bad_binding_rejects_the_whole_change() -> None:
    """Transitions are atomic: no participant from a refused change becomes current."""
    authority = seeded()
    attempt = authority.attempt()
    section = authority.section
    good = Materialization(section.at("n").reference, Box(3), Supplied(attempt.reference, attempt.basis, "the test"))
    bad = Materialization(section.at("m").reference, Box(1), Supplied(attempt.reference, attempt.basis, "the test"))
    result = authority.submit([good, bad], attempt)
    assert isinstance(result, Rejected) and "@m is materialized by @load" in str(result)
    assert authority.section.at("n").binding is None


def test_a_proposal_from_an_attempt_on_older_state_is_rejected() -> None:
    authority = seeded()
    stale = authority.attempt()
    materialized(authority)
    section = authority.section
    rebinding = RouteRebinding(section.at("m").reference, NORMAL, object(), None, Prepared(stale.reference, stale.basis, 1))
    result = authority.submit([rebinding], stale)
    assert isinstance(result, Rejected) and "based on authority revision 1" in str(result)
    assert result.stale, "deriving it again may succeed"


def test_a_candidate_never_becomes_current_by_assignment() -> None:
    section = materialized().section
    participant = section.at("m")
    with pytest.raises(TypeError):
        section.participants[participant.reference] = participant  # ty: ignore[invalid-assignment]
    with pytest.raises(dataclasses.FrozenInstanceError):
        participant.binding = None  # ty: ignore[invalid-assignment]


# Routes and readiness.


def test_a_route_needs_a_binding_its_expected_revision_and_evidence_from_its_attempt() -> None:
    authority = seeded()
    attempt = authority.attempt()
    unbound = authority.section.at("m").reference
    result = authority.submit([RouteRebinding(unbound, NORMAL, object(), None, Prepared(attempt.reference, attempt.basis, 1))], attempt)
    assert isinstance(result, Rejected) and "not bound" in str(result)

    routed(materialized(authority))
    route = authority.section.at("m").routes[NORMAL]
    assert route.revision == 1
    attempt = authority.attempt()
    reference = authority.section.at("m").reference
    wrong = RouteRebinding(reference, NORMAL, object(), None, Prepared(attempt.reference, attempt.basis, 1))
    assert "at revision 1, not None" in str(authority.submit([wrong], attempt))
    other = authority.attempt()
    foreign = RouteRebinding(reference, NORMAL, object(), 1, Prepared(other.reference, other.basis, 1))
    assert "another attempt" in str(authority.submit([foreign], attempt))
    rebound = authority.submit([RouteRebinding(reference, NORMAL, "compiled", 1, Prepared(attempt.reference, attempt.basis, 1))], attempt)
    assert isinstance(rebound, Published)
    after = authority.section.at("m")
    assert (after.reference, after.routes[NORMAL].revision, after.routes[NORMAL].callable) == (reference, 2, "compiled"), (
        "route rebinding keeps the participant's reference"
    )


def test_a_route_prepared_from_another_binding_is_refused_even_where_no_use_names_it() -> None:
    authority = materialized()
    attempt = authority.attempt()
    reference = authority.section.at("m").reference
    sampling = RouteRebinding(reference, "sampling", "compiled", None, Prepared(attempt.reference, attempt.basis, 2))
    result = authority.submit([sampling], attempt)
    assert isinstance(result, Rejected) and "prepared from binding 2, and 1 is current" in str(result)


def test_readiness_names_what_core_execution_still_lacks() -> None:
    authority = seeded()
    assert any("@m is not bound" in reason for reason in authority.readiness(CORE_EXECUTION))
    materialized(authority)
    lacking = authority.readiness(CORE_EXECUTION)
    assert sorted({reason.split(" has")[0] for reason in lacking}) == ["@k", "@m", "@n"]
    assert all("has no prepared normal route" in reason for reason in lacking)
    routed(authority)
    assert authority.readiness(CORE_EXECUTION) == ()


# The owner seam.


def test_the_execution_projection_realizes_the_owner_seam_and_nothing_else() -> None:
    execution = Execution(routed())
    assert isinstance(execution, Owner)
    assert not hasattr(execution, "submit") and not hasattr(execution, "section")


def test_a_lease_goes_through_the_route_and_reports_the_version_the_seam_compares() -> None:
    authority = routed()
    m = authority.section.at("m")
    assert authority.current("m") == Version("m", m.reference, 1, 0)
    (lease,) = authority.lease([Use("m", AccessMode.PROTECTED_CURRENT, VersionSelection.CURRENT_AT_START)])
    assert lease.handle is m.routes[NORMAL].callable
    lease.release()
    with pytest.raises(Unavailable, match="released twice"):
        lease.release()


def test_only_the_runtime_s_notice_advances_the_numeric_version() -> None:
    authority = routed()
    (update,) = authority.lease([Use("m", AccessMode.OWNER_UPDATE, VersionSelection.CURRENT_AT_START)])
    m = authority.section.at("m")
    assert m.binding is not None and update.handle is m.binding.value
    box = update.handle
    assert isinstance(box, Box)
    box.value += 5  # a change in place, which the authority never sees
    assert authority.current("m").ordinal == 0
    assert update.updated().ordinal == 1 and update.updated().ordinal == 2, "every notice advances, even one for no change"
    assert authority.current("m") == Version("m", m.reference, 1, 2)
    with pytest.raises(Unavailable, match="already has an update lease"):
        authority.lease([Use("m", AccessMode.OWNER_UPDATE, VersionSelection.CURRENT_AT_START)])
    update.release()
    with pytest.raises(Unavailable, match="current update lease"):
        update.updated()


def test_a_snapshot_of_live_state_pins_the_participant_against_later_updates() -> None:
    authority = routed()
    (snapshot,) = authority.lease([Use("n", AccessMode.SNAPSHOT, VersionSelection.PUBLISHED_INITIAL)])
    n = authority.section.at("n")
    assert snapshot.handle is n.routes[NORMAL].callable and snapshot.version.ordinal == 0
    snapshot.release()
    with pytest.raises(Unavailable, match="may not update it"):
        authority.lease([Use("n", AccessMode.OWNER_UPDATE, VersionSelection.CURRENT_AT_START)])


def test_a_snapshot_of_an_updated_participant_needs_a_copy_and_fails_with_the_reason() -> None:
    authority = routed()
    (update,) = authority.lease([Use("m", AccessMode.OWNER_UPDATE, VersionSelection.CURRENT_AT_START)])
    update.updated()
    update.release()
    with pytest.raises(Unavailable, match="needs a copy of its device storage"):
        authority.lease([Use("m", AccessMode.SNAPSHOT, VersionSelection.CURRENT_AT_START)])


def test_leases_taken_together_are_all_released_when_one_cannot_be_taken() -> None:
    authority = routed()
    uses = [
        Use("m", AccessMode.PROTECTED_CURRENT, VersionSelection.CURRENT_AT_START),
        Use("m", AccessMode.GRANT, VersionSelection.CURRENT_AT_START),
    ]
    with pytest.raises(Unavailable, match="grants come with training's words"):
        authority.lease(uses)
    rebinding = authority.attempt()
    reference = authority.section.at("m").reference
    change = RouteRebinding(reference, NORMAL, "other", 1, Prepared(rebinding.reference, rebinding.basis, 1))
    assert isinstance(authority.submit([change], rebinding), Published), "the first lease was released"


def test_a_snapshot_taken_with_a_lease_that_fails_pins_nothing() -> None:
    """Leases taken together are all or none: a snapshot granted before a refused use leaves no pin behind."""
    authority = routed()
    snapshot = Use("n", AccessMode.SNAPSHOT, VersionSelection.CURRENT_AT_START)
    with pytest.raises(Unavailable, match="grants"):
        authority.lease([snapshot, Use("m", AccessMode.GRANT, VersionSelection.CURRENT_AT_START)])
    with pytest.raises(Unavailable, match="may not update it"):
        authority.lease([snapshot, Use("n", AccessMode.OWNER_UPDATE, VersionSelection.CURRENT_AT_START)])
    (update,) = authority.lease([Use("n", AccessMode.OWNER_UPDATE, VersionSelection.CURRENT_AT_START)])
    update.release()
    assert not authority._outstanding


def test_a_change_to_a_participant_with_a_lease_out_is_refused() -> None:
    authority = routed()
    (lease,) = authority.lease([Use("m", AccessMode.PROTECTED_CURRENT, VersionSelection.CURRENT_AT_START)])
    attempt = authority.attempt()
    reference = authority.section.at("m").reference
    change = RouteRebinding(reference, NORMAL, "other", 1, Prepared(attempt.reference, attempt.basis, 1))
    refused = authority.submit([change], attempt)
    assert isinstance(refused, Rejected) and "has a lease out" in str(refused)
    assert not refused.stale, "a conflict with a use is not staleness"
    assert authority.section.at("m").routes[NORMAL].revision == 1
    lease.release()
    assert isinstance(authority.submit([change], attempt), Published)


def test_a_participant_that_is_not_bound_cannot_be_used() -> None:
    authority = seeded()
    with pytest.raises(Unavailable, match="declared, not bound"):
        authority.lease([Use("m", AccessMode.PROTECTED_CURRENT, VersionSelection.CURRENT_AT_START)])
    with pytest.raises(Unavailable, match="declared, not bound"):
        authority.current("m")
    with pytest.raises(Unavailable, match="not a participant"):
        authority.current("x")


# Running a strategy against the authority.


def test_the_runtime_runs_a_strategy_against_the_authority() -> None:
    run = accepted_run(STRATEGY, installed=INSTALLED)
    authority = routed(materialized(RunAuthority.seed(run)))
    assert authority.readiness(CORE_EXECUTION) == ()
    prepared = prepare(cold_load(run, installed=INSTALLED), INSTALLED, Target())
    execution = Execution(authority)
    facts: list[Fact] = []
    coordinator = Coordinator(prepared, dict.fromkeys(execution.resources(), execution), INSTALLED, {"host": Host()}, observe=facts.append)
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "succeeded", outcome.error
    m = authority.section.at("m")
    (updated,) = [fact.values[0] for fact in facts if fact.kind == "updated"]
    assert updated == Version("m", m.reference, 1, 1)
    assert m.binding is not None and isinstance(m.binding.value, Box) and m.binding.value.value == 2, (
        "the update changed the bound object in place"
    )
    assert authority.current("m").ordinal == 1
    assert not authority._outstanding, "every lease came back"


UPDATE_THEN_SNAPSHOT = """iro.strategy @s uses(@load v1, @step v1, @peek v1) {
  %m = iro.resource @m : participant materialize(@load)
  %k = iro.resource @k : participant materialize(@load)
  iro.sequence @once capture(%m) {
    iro.access %m owner_update {
    ^(%u):
      iro.call @step(%u)
    }
    iro.access %m snapshot {
    ^(%l):
      iro.call @peek(%l)
    }
  }
}
"""


def test_a_strategy_that_snapshots_a_participant_it_updates_fails_before_it_runs() -> None:
    """The obligations show both uses, so the refusal comes at preparation and readiness, not mid-run."""
    installed = {"load": load, "step": step, "peek": peek}
    authority = RunAuthority.seed(accepted_run(UPDATE_THEN_SNAPSHOT, installed=installed))
    assert isinstance(MaterializationAttempt(authority, ["m", "k"], installed=installed, device=Host()).run(), Published)
    attempt = authority.attempt()
    m = authority.section.at("m")
    result = authority.submit(
        [RouteRebinding(m.reference, NORMAL, "callable", None, Prepared(attempt.reference, attempt.basis, 1))], attempt
    )
    assert isinstance(result, Rejected) and "which the run updates, needs a copy of its device storage" in str(result)
    assert any("needs a copy" in reason for reason in authority.readiness(CORE_EXECUTION))


def test_a_use_the_authority_refuses_fails_the_run_with_its_reason() -> None:
    """Readiness was skipped here, so the unbound participant fails at its use, and the run reports why."""
    run = accepted_run(STRATEGY, installed=INSTALLED)
    authority = RunAuthority.seed(run)
    assert isinstance(MaterializationAttempt(authority, ["m", "k"], installed=INSTALLED, device=Host()).run(), Published)
    routed(authority, names=("m", "k"))
    prepared = prepare(cold_load(run, installed=INSTALLED), INSTALLED, Target())
    execution = Execution(authority)
    coordinator = Coordinator(prepared, dict.fromkeys(execution.resources(), execution), INSTALLED, {"host": Host()})
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "failed" and outcome.error is not None and "@n is declared, not bound" in outcome.error
    assert not authority._outstanding, "the refused access holds nothing"
