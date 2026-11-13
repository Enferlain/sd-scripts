"""Lowering: slots, bound steps, the end checks the types allow, scope records only where something waits, and checked mode."""

import asyncio
import dataclasses

import pytest

import library.iro.prepare.lowered as lowered_module
import library.iro.runtime.coordinator as coordinator_module
from library.iro.accepted import cold_load
from library.iro.devices import Host
from library.iro.frontend import implementation
from library.iro.prepare import Target, lower, prepare
from library.iro.prepare.lowered import CALLS, INVOKE, Body, Calls, Invoke
from library.iro.runtime import Coordinator
from tests.unit.iro.accepted_runs import accepted_run
from tests.unit.iro.runtime.harness import Memory
from tests.unit.iro.runtime.test_witnesses import LONG, _installed

CHAIN = """iro.strategy @s uses(@a v1, @b v1) {
  iro.sequence @once {
    %x = iro.call @a() -> int
    %y = iro.call @b(%x, 2) -> int
  }
}
"""


def _chain_installed():
    return {
        "a": implementation(version=1, outputs=("int",), name="a")(lambda: 5),
        "b": implementation(version=1, inputs=("int", "int"), outputs=("int",), name="b")(lambda x, k: x * k),
    }


def _prepared(text: str, installed: dict, target: Target = Target()):
    run = accepted_run(text, installed=installed)
    return prepare(cold_load(run, installed=installed), installed, target)


def _invokes(body: Body) -> dict[str, Invoke]:
    """Every activity under a body, by its name."""
    found = {}
    for step in body.steps:
        if isinstance(step, Invoke):
            found[step.path.rsplit("/", 1)[-1]] = step
            for inner in step.bodies:
                found |= _invokes(inner)
    return found


def test_a_body_numbers_its_values_and_fills_in_its_constants() -> None:
    installed = _chain_installed()
    once = _invokes(lower(_prepared(CHAIN, installed), installed).root)["@once"]
    (body,) = once.bodies
    (step,) = body.steps
    assert isinstance(step, Calls) and step.kind == CALLS
    first, second = step.calls
    # %x is slot 0, the inline 2 a constant in slot 1, %y slot 2.
    assert body.entries == 0 and body.template == (None, 2, None)
    assert first.results == (0,) and second.slots == (-1, 1) and second.locals == ((0, 0, 0),)
    assert second.results == (2,)


def test_a_constant_between_fused_calls_reaches_the_later_call() -> None:
    installed = _chain_installed()
    facts = []
    coordinator = Coordinator(_prepared(CHAIN, installed), {}, installed, {"host": Host()}, observe=facts.append, checked=True)
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "succeeded", outcome.error
    assert [f.values[1] for f in facts if f.kind == "call_completed"] == [(5,), (10,)]


def _long():
    installed = _installed()
    prepared = _prepared(LONG.replace("TIMES", "3"), installed, Target(workers=2))
    return prepared, installed


def test_end_checks_read_only_where_the_types_allow_a_claim_or_a_retained_result() -> None:
    prepared, installed = _long()
    found = _invokes(lower(prepared, installed).root)
    loop, source, access = found["@loop"], found["@source"], next(s for p, s in found.items() if s.protocol == "access")
    (loop_body,) = loop.bodies
    assert loop_body.claims, "the take's claim is checked when an iteration ends"
    assert loop_body.retained, "the access's retained result is checked when an iteration ends"
    assert not source.bodies[0].claims and not source.bodies[0].retained, "a production holds neither"
    assert source.owns_claims, "a producer owns the claims it reserves until it publishes them"
    assert access.bodies[0].retained and not access.bodies[0].claims
    assert not found["@report"].owns_claims and not found["@report"].bodies[0].claims


def test_only_the_statements_something_waits_on_keep_scope_records() -> None:
    prepared, installed = _long()
    waited = lower(prepared, installed).waited
    # The handler and the stop both wait for @loop to settle in @all, the scope they share with it.
    assert {name.rsplit("/", 1)[-1]: {s.rsplit("/", 1)[-1] for s in scopes} for name, scopes in waited.items()} == {"@loop": {"@all"}}


def test_scope_records_are_kept_only_in_the_scopes_a_wait_asks_about() -> None:
    prepared, installed = _long()
    coordinator = Coordinator(prepared, {"m": Memory({"m": 0})}, installed, {"host": Host()}, checked=True)
    largest = {"kept": 0, "full": 0}

    def observe(fact) -> None:
        largest["kept"] = max(largest["kept"], len(coordinator._records.started))
        largest["full"] = max(largest["full"], len(coordinator._all_records.started))

    coordinator.observe = observe
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "succeeded", outcome.error
    assert 0 < largest["kept"] < largest["full"]


def test_checked_mode_fails_a_run_whose_lowering_skipped_a_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared, installed = _long()
    monkeypatch.setattr(lowered_module, "ClaimType", type("Never", (), {}))
    coordinator = Coordinator(prepared, {"m": Memory({"m": 0})}, installed, {"host": Host()}, checked=True)
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "failed"
    assert any("lowering excluded what the full check found" in error and "claim" in error for error in coordinator.errors)


def test_checked_mode_fails_a_run_whose_lowering_dropped_scope_records(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared, installed = _long()
    real = coordinator_module.lower
    monkeypatch.setattr(coordinator_module, "lower", lambda p, i: dataclasses.replace(real(p, i), waited={}))
    coordinator = Coordinator(prepared, {"m": Memory({"m": 0})}, installed, {"host": Host()}, checked=True)
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "failed"
    assert any("scope records" in error for error in coordinator.errors)


POLICY = """iro.strategy @s uses(@a v1) {
  iro.repeat @loop until @schedule {
  ^(%i: int):
    %x = iro.call @a(%i) -> int
  }
}
"""


def test_a_step_lowering_cannot_realize_fails_when_it_runs() -> None:
    installed = {"a": implementation(version=1, inputs=("int",), outputs=("int",), name="a")(lambda i: i)}
    prepared = _prepared(POLICY, installed)
    loop = _invokes(lower(prepared, installed).root)["@loop"]
    assert loop.kind == INVOKE and loop.problem is not None and "by policy" in loop.problem
    coordinator = Coordinator(prepared, {}, installed, {"host": Host()}, checked=True)
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "failed" and "by policy" in (outcome.error or "")


def _strip(body: Body) -> None:
    """Removes every step's operation, so a run that still works never read the strategy while executing."""
    for step in body.steps:
        object.__setattr__(step, "op", None)
        for inner in getattr(step, "bodies", ()):
            _strip(inner)


def test_execution_reads_nothing_of_the_strategy() -> None:
    prepared, installed = _long()

    def run(strip: bool):
        facts = []
        coordinator = Coordinator(prepared, {"m": Memory({"m": 0})}, installed, {"host": Host()}, observe=facts.append, checked=True)
        if strip:
            _strip(coordinator.lowered.root)
        outcome = asyncio.run(coordinator.run())
        assert outcome.status == "succeeded", outcome.error
        # The loop's and the handler's calls; how far the producer runs ahead before its stop depends on timing.
        mine = [f for f in facts if f.kind == "call_completed" and ("@loop" in f.owner or "@report" in f.owner)]
        return sorted((f.owner, f.invocation, repr(f.values)) for f in mine)

    assert run(strip=True) == run(strip=False)


TWO_RESULTS = """iro.strategy @s uses(@both v1) {
  %m = iro.resource @m : participant
  iro.sequence @once capture(%m) {
    iro.access %m protected_current {
    ^(%lease):
      iro.call @both(%lease) -> retained int, int
    }
  }
}
"""


def test_a_retaining_call_that_returns_more_than_its_retained_result_fails_when_it_runs() -> None:
    installed = {
        "both": implementation(version=1, inputs=("protected_current lease",), outputs=("retained int", "int"), name="both")(
            lambda lease: (1, 2)
        )
    }
    prepared = _prepared(TWO_RESULTS, installed)
    coordinator = Coordinator(prepared, {"m": Memory({"m": 0})}, installed, {"host": Host()}, checked=True)
    outcome = asyncio.run(coordinator.run())
    assert outcome.status == "failed" and "returns exactly one result, the retained one" in (outcome.error or "")
    assert not coordinator.leases and not coordinator.products
