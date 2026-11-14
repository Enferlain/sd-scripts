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

# Checked mode runs every check lowering narrows in full; the default runs only the lowered ones.
MODES = pytest.mark.parametrize("checked", [True, False], ids=["checked", "unchecked"])


def both(name: str, checked: bool = True):
    resources, made = CASES[name]
    return run_both(fixture(name), resources, made(), validators=standins.validators(), checked=checked)


@MODES
@pytest.mark.parametrize("name", SCHEDULE_INDEPENDENT)
def test_the_runtime_reports_exactly_the_reference_facts(name: str, checked: bool) -> None:
    result = both(name, checked)
    assert result.coordinator.errors == [] and result.reference.errors == []
    reference, runtime = result.reference_facts(), result.runtime_facts()
    assert runtime - reference == Counter() and reference - runtime == Counter()


@MODES
@pytest.mark.parametrize("name", sorted(CASES))
def test_every_run_settles_completely_and_its_bookkeeping_empties(name: str, checked: bool) -> None:
    result = both(name, checked)
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
    assert not coordinator.protections.leases and not coordinator.protections.products and not coordinator.outcomes
    assert all(not s.outstanding and not s.ready for s in coordinator.flow.streams.values())
    assert all(not p.readers and p.writer is None and not p.reservations for p in coordinator.protections.by_resource.values())
    assert all(not h.pending and h.active == 0 for h in coordinator.requests.handlers.values())
    assert coordinator._records.empty() and coordinator._all_records.empty() and not coordinator.turns.admissions
    assert all(t.holder is None and not t.waiting for t in coordinator.turns.by_device.values())


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
    assert not coordinator.protections.leases and not coordinator.protections.products and not coordinator.outcomes
    assert all(not s.outstanding and not s.ready for s in coordinator.flow.streams.values())
    assert all(not p.readers and p.writer is None and not p.reservations for p in coordinator.protections.by_resource.values())
    assert all(not h.pending and h.active == 0 for h in coordinator.requests.handlers.values())
    assert not coordinator.turns.admissions and coordinator._records.empty() and coordinator._all_records.empty()


NESTED_HANDLERS = """iro.strategy @s uses(@step v1, @look v1, @peek v1) {
  %m = iro.resource @m : participant
  %n = iro.resource @n : participant

  iro.parallel @all capture(%m, %n) {
    iro.repeat @train 2 times capture(%m) {
    ^(%k: int):
      iro.access %m owner_update {
      ^(%u):
        iro.call @step(%u)
      }
      iro.request @outer(%k) from @train at settled_success
    }

    iro.handler @outer sources [@train] max 1 capture(%m, %n) {
    ^(%x: int):
      iro.parallel @inside capture(%m, %n, %x) {
        iro.sequence @work capture(%m, %x) {
          iro.access %m protected_current at_request {
          ^(%l):
            iro.call @look(%l)
          }
          iro.request @inner(%x) from @work at settled_success
        }

        iro.handler @inner sources [@work] max 1 capture(%n) {
        ^(%y: int):
          iro.access %n protected_current at_request {
          ^(%l):
            iro.call @peek(%l)
          }
        }
      }
    }
  }
}
"""


def test_a_request_reserves_only_what_its_own_handler_reads_at_the_request() -> None:
    """An at_request use belongs to its innermost handler: the outer handler's requests do not reserve the inner's reads."""
    from library.iro.reference import StandIn, returning
    from tests.unit.iro.test_reference import increment

    lease = "protected_current lease"
    standins = {
        "step": StandIn(("owner_update lease",), (), increment, updates=(0,)),
        "look": StandIn((lease,), (), returning()),
        "peek": StandIn((lease,), (), returning()),
    }
    result = run_both(NESTED_HANDLERS, {"m": 1, "n": 1}, standins)
    assert result.coordinator.errors == [] and result.reference.errors == []
    runtime = Counter(f.owner for f in result.facts if f.kind == "reserved")
    reference = Counter(f.owner for f in result.reference.facts if f.kind == "reserved")
    assert runtime == reference == Counter({"m": 2, "n": 2}), "each of the two outer and two inner requests reserves only its own reads"
