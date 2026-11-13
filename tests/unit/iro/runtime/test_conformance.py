"""The runtime against the reference semantics on the conformance fixtures (D16).

Where a composition's outcome does not depend on the schedule, the runtime
reports exactly the reference's owner-visible facts. Where it may (a live
read racing an update, or how far a producer gets before its stop), every
run must still settle completely with the same lifecycle rules; that a
runtime trace is one the reference could produce is checked by the
interleaving explorer.
"""

from collections import Counter

import pytest

from tests.unit.iro.conformance import standins
from tests.unit.iro.conformance.test_conformance import CASES, fixture
from tests.unit.iro.runtime.harness import run_both

SCHEDULE_INDEPENDENT = ("lifetime", "lifetime_scoped")


def both(name: str):
    resources, made = CASES[name]
    return run_both(fixture(name), resources, made(), validators=standins.validators())


@pytest.mark.parametrize("name", SCHEDULE_INDEPENDENT)
def test_the_runtime_reports_exactly_the_reference_facts(name: str) -> None:
    result = both(name)
    assert result.coordinator.errors == [] and result.reference.errors == []
    reference, runtime = result.reference_facts(), result.runtime_facts()
    assert runtime - reference == Counter() and reference - runtime == Counter()


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_run_settles_completely_and_its_bookkeeping_empties(name: str) -> None:
    result = both(name)
    coordinator = result.coordinator
    assert coordinator.errors == []
    outcomes = [f for f in result.facts if f.kind == "outcome"]
    assert outcomes and all(f.values[0] == "succeeded" for f in outcomes)
    states = {}
    for fact in (f for f in result.facts if f.kind == "state"):
        states.setdefault(fact.invocation, []).append(fact.values[0])
    lifecycle = ["pending", "admitted", "running", "closing", "settling", "terminal"]
    assert all(s == lifecycle for s in states.values()), "every invocation goes through its whole lifecycle once"
    dispositions = Counter((f.owner, f.values[0]) for f in result.facts if f.kind == "settled")
    produced = {(f.owner, f.values[0]) for f in result.facts if f.kind in ("published", "nothing_produced")}
    assert all(n == 1 for n in dispositions.values()), "every item settles once"
    assert set(dispositions) | {k for k in produced if k not in dispositions} >= produced
    assert not coordinator.leases and not coordinator.products and not coordinator.outcomes
    assert all(not s.outstanding and not s.ready for s in coordinator.streams.values())
    assert all(not p.readers and p.writer is None and not p.reservations for p in coordinator.protection.values())
    assert all(not h.pending and h.active == 0 for h in coordinator.handlers.values())
    assert not coordinator._started and not coordinator._live_in and not coordinator.admissions
    assert all(t.holder is None and not t.waiting for t in coordinator.turns.values())


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_failing_implementation_fails_the_run_and_still_empties_its_bookkeeping(name: str) -> None:
    from tests.unit.iro.runtime.harness import implementation, run_runtime

    resources, made = CASES[name]
    made_standins = made()
    failing = sorted(made_standins)[0]
    installed = {n: implementation(n, s) for n, s in made_standins.items()}

    def boom(*args):
        raise ValueError("selected code failed")

    installed[failing] = type(installed[failing])(installed[failing].declaration, boom)
    outcome, facts, coordinator = run_runtime(fixture(name), resources, installed)
    called = any(f.kind == "call_entered" and f.owner for f in facts)
    assert outcome.status in ("failed", "succeeded") and called
    assert not coordinator.leases and not coordinator.products and not coordinator.outcomes
    assert all(not s.outstanding and not s.ready for s in coordinator.streams.values())
    assert all(not p.readers and p.writer is None and not p.reservations for p in coordinator.protection.values())
    assert all(not h.pending and h.active == 0 for h in coordinator.handlers.values())
    assert not coordinator.admissions and not coordinator._started and not coordinator._live_in
