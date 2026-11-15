"""The IR's admission, freshness and protected-use cases, re-run against the real run authority (task 15.2).

Each strategy's participants are bound to what the test supplies, their
normal routes come from a real preparation attempt, and the coordinator
reaches them through the execution projection; other resources stay with an
in-memory owner. Where the outcome does not depend on the schedule, the
runtime reports exactly the reference semantics' owner-visible facts, with
versions compared by owner and ordinal: the authority's incarnation and
version lineage are its own (a reference's serial and a binding revision),
where the reference's are fixed.
"""

import asyncio
from collections import Counter
from collections.abc import Mapping

import pytest

from library.iro.accepted import cold_load
from library.iro.devices import Host
from library.iro.normalize import normalize
from library.iro.prepare import Target, prepare
from library.iro.reference import Reference, StandIn, returning
from library.iro.runtime import Coordinator, Fact
from library.iro.seams import Owner
from library.iro.seams import Version as SeamVersion
from library.iro.text import parse
from library.iro.verify import CORE_EXECUTION
from library.runstate.authority import NORMAL, MaterializationAttempt, Published, RunAuthority
from library.runstate.preparation import PreparationAttempt, PreparationRequest, PreparationResult, PreparedRoute
from library.runstate.projections import Execution
from tests.unit.iro.accepted_runs import accepted_run
from tests.unit.iro.conformance import standins
from tests.unit.iro.conformance.test_conformance import CASES, fixture
from tests.unit.iro.runtime.harness import RUNTIME_ONLY, Cell, Memory, implementation
from tests.unit.iro.test_reference import REQUESTED, increment


def realized(text: str, resources: Mapping[str, object], made: Mapping[str, StandIn]) -> tuple[RunAuthority, Coordinator, list[Fact]]:
    """The strategy realized against a real authority: seeded, bound, prepared, ready, and wired into a coordinator."""
    installed = {name: implementation(name, standin) for name, standin in made.items()}
    run = accepted_run(text, installed=installed)
    authority = RunAuthority.seed(run)
    participants = [participant.name for participant in run.seed.participants]
    store = Memory({name: resources[name] for name in participants})
    supplied = {name: Cell(store, name) for name in participants}
    bound = MaterializationAttempt(authority, participants, supplied=supplied, supplier="the test").run()
    assert isinstance(bound, Published), bound
    references = [authority.section.at(name).reference for name in participants]
    attempt = PreparationAttempt.derive(authority, PreparationRequest(routes=tuple((reference, NORMAL) for reference in references)))
    assert isinstance(attempt, PreparationAttempt), attempt
    routes = []
    for reference in references:
        binding = attempt.projection[reference].binding
        assert binding is not None
        routes.append(PreparedRoute(reference, NORMAL, binding.value))
    assert isinstance(attempt.publish(PreparationResult(attempt.attempt.reference, tuple(routes))), Published)
    assert authority.readiness(CORE_EXECUTION) == ()
    execution = Execution(authority)
    others = Memory({name: value for name, value in resources.items() if name not in participants})
    owners: Mapping[str, Owner] = {**dict.fromkeys(participants, execution), **dict.fromkeys(others.values, others)}
    prepared = prepare(cold_load(run, installed=installed), installed, Target())
    facts: list[Fact] = []
    coordinator = Coordinator(prepared, owners, installed, {"host": Host()}, observe=facts.append, checked=True)
    return authority, coordinator, facts


def ran(text: str, resources: Mapping[str, object], made: Mapping[str, StandIn]) -> tuple[RunAuthority, Coordinator, list[Fact]]:
    authority, coordinator, facts = realized(text, resources, made)
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "succeeded", outcome.error
    assert coordinator.errors == []
    assert not authority._outstanding, "every lease came back to the authority"
    return authority, coordinator, facts


def plain(value: object) -> object:
    if isinstance(value, SeamVersion):
        return ("version", value.owner, value.ordinal)
    if isinstance(value, tuple):
        return tuple(plain(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, plain(v)) for k, v in value.items()))
    return value


def comparable(fact) -> tuple:
    values = tuple(plain(v) for v in fact.values)
    if fact.kind == "updated":
        values = values[:1]
    return (fact.owner, fact.invocation, fact.kind, values)


def ordinal(value: object) -> int:
    assert isinstance(value, SeamVersion)
    return value.ordinal


@pytest.mark.parametrize("name", ["lifetime", "lifetime_scoped"])
def test_against_the_authority_the_runtime_reports_exactly_the_reference_facts(name: str) -> None:
    """Protected reads against updates, snapshots of a frozen participant, and admission with provenance."""
    resources, made = CASES[name]
    reference = Reference(normalize(parse(fixture(name))), resources, made(), validators=standins.validators())
    reference.run()
    assert reference.errors == []
    authority, _, facts = ran(fixture(name), resources, made())
    runtime = Counter(comparable(f) for f in facts if f.kind not in RUNTIME_ONLY)
    assert runtime == Counter(comparable(f) for f in reference.facts)
    model = authority.section.at("model")
    updates = [f.values[0] for f in facts if f.kind == "updated" and f.owner == "model"]
    assert updates and all(isinstance(version, SeamVersion) for version in updates)
    assert all(
        (version.incarnation, version.lineage) == (model.reference.serial, "1") for version in updates if isinstance(version, SeamVersion)
    ), "the versions the run observed are the authority's"
    assert authority.current("model").ordinal == len(updates)


def data_only(admission: str) -> str:
    text = fixture("data_only").replace("admit any_with_provenance(@policy)", f"admit {admission}")
    return text.replace("%policy  = iro.resource @policy\n", "%policy  = iro.resource @policy : participant\n")


def test_exact_current_admission_judges_the_version_the_authority_reports() -> None:
    authority, _, facts = ran(data_only("exact_current(@policy)"), {"policy": 0}, standins.data_only())
    updates = 0
    admitted = 0
    for fact in facts:
        if fact.kind == "updated" and fact.owner == "policy":
            updates += 1
        elif fact.kind == "admission" and fact.values[2]:
            observed = fact.values[1]
            assert isinstance(observed, tuple) and len(observed) == 1
            assert ordinal(observed[0]) == updates, "an admitted item observed the version current at its admission"
            admitted += 1
    assert admitted and authority.current("policy").ordinal == updates


def test_bounded_lag_admission_admits_older_items_within_the_bound() -> None:
    _, _, facts = ran(data_only("bounded_lag(@policy, max 3)"), {"policy": 0}, standins.data_only())
    admissions = [fact for fact in facts if fact.kind == "admission"]
    assert admissions and all(fact.values[2] for fact in admissions), "the stream's bounds keep every item within the lag"


def test_requested_work_reads_the_version_its_request_reserved_from_the_authority() -> None:
    """A protected live read at the request: updates wait for the reservation, and the handler reads that version."""
    made = {
        "step": StandIn(("owner_update lease",), (), increment, updates=(0,)),
        "measure": StandIn(("protected_current lease",), (), returning()),
    }
    authority, _, facts = ran(REQUESTED.replace("MODE", "protected_current"), {"m": 0}, made)
    reserved = [ordinal(f.values[0]) for f in facts if f.kind == "reserved"]
    read = [ordinal(f.values[2]) for f in facts if f.kind == "leased" and "@evaluation" in f.owner]
    assert reserved == read == [1, 2, 3], "each request held the version after its iteration's update, and its handler read it"
    assert authority.current("m").ordinal == 3
