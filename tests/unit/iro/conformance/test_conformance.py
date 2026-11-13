"""The conformance suite, rebuilt from the research fixtures, stand-ins and qualification cases.

Three compositions go through the same path: the written form reads back
exactly, normalization and verification accept it, its accepted run
cold-loads, and the reference semantics runs it to settlement. Each outcome
is checked against expectations computed here independently, not taken from
the reference semantics.

- lifetime: a producer; a training repeat of windows, each a repeat of
  members carrying a gradient total and then an owner update; an evaluation
  requested after the second window, which takes from the same stream.
- lifetime_scoped: the same with an extra consumer scope, a valid
  rearrangement that must pass the same rules with the same meaning.
- data_only: the design's ``@score_records``, with no handler and a policy
  the consumer updates.
- filtering: two streams with a filtering producer between them, and
  requested work that keeps consuming after the first consumer settles.
"""

import re
from fractions import Fraction
from pathlib import Path

import pytest
from xdsl.dialects.builtin import SymbolRefAttr

from library.iro.accepted import cold_load
from library.iro.diagnostics import Rejection
from library.iro.normalize import normalize
from library.iro.reference import Reference, Version
from library.iro.text import parse, print_strategy
from library.iro.operation_contracts import path
from library.iro.verify import verify
from library.iro.words.calls import CallOp
from tests.unit.iro.accepted_runs import accepted_run
from tests.unit.iro.conformance import standins

HERE = Path(__file__).parent
ROOT = HERE.parents[3]
DESIGN = (ROOT / "openspec/changes/rework-model-strategy-trainer/design.md").read_text()


def fixture(name: str) -> str:
    if name == "data_only":
        block = next(b for b in re.findall(r"```text\n(.*?)```", DESIGN, re.S) if "@score_records" in b)
        return "\n".join(line for line in block.splitlines() if not line.strip().startswith("//")) + "\n"
    return (HERE / f"{name}.iro").read_text()


CASES = {
    "lifetime": ({"encoder": 2.0, "model": 0.8}, standins.lifetime),
    "lifetime_scoped": ({"encoder": 2.0, "model": 0.8}, standins.lifetime),
    "data_only": ({"policy": 0}, standins.data_only),
    "filtering": ({"config": 0, "products": ()}, standins.filtering),
}


def run(name: str) -> Reference:
    resources, made = CASES[name]
    reference = Reference(normalize(parse(fixture(name), source=f"{name}.iro")), resources, made(), validators=standins.validators())
    reference.run()
    return reference


def drained(reference: Reference) -> None:
    assert reference.settled, "every task finished"
    assert reference.errors == []
    assert all(not s.outstanding and not s.ready for s in reference.streams.values()), "every item settled"
    assert all(not lease.active for lease in reference.leases.values()), "every lease released"
    assert not reference.products, "every retained result released"
    assert all(not r.readers and r.writer is None and not r.reservations for r in reference.resources.values())
    assert all(not h.pending and h.active == 0 for h in reference.handlers.values())
    assert all(o.state == "terminal" and o.status == "succeeded" for o in reference.outcomes.values())


def calls(reference: Reference, implementation: str) -> list[str]:
    """The paths of the calls naming an implementation."""
    return [path(op) for op in reference.strategy.walk() if isinstance(op, CallOp) and op.implementation == SymbolRefAttr(implementation)]


def results(reference: Reference, implementation: str) -> list[tuple]:
    named = set(calls(reference, implementation))
    found = []
    for fact in reference.facts_of("call_completed"):
        if fact.owner in named:
            values = fact.values[1]
            assert isinstance(values, tuple)
            found.append(values)
    return found


# The shared path: written form, acceptance and cold loading.


@pytest.mark.parametrize("name", list(CASES))
def test_each_case_reads_back_exactly_and_is_accepted(name: str) -> None:
    text = fixture(name)
    assert print_strategy(parse(text)) == text
    run_ = accepted_run(text, source=f"{name}.iro")
    loaded = cold_load(run_)
    assert loaded.verification.obligations == run_.obligations
    assert print_strategy(parse(run_.normalized)) == run_.normalized


def test_the_rearrangement_passes_the_same_rules_with_the_same_obligations() -> None:
    def kinds(name: str) -> list[tuple[str, str]]:
        return sorted((str(o.checkpoint), o.evidence) for o in verify(normalize(parse(fixture(name)))).obligations)

    assert kinds("lifetime") == [(c, e.replace("@consumer_scope", "@consumers")) for c, e in kinds("lifetime_scoped")]


# The reference semantics, against independent expectations.


def expected_weights() -> list[Fraction]:
    weight, found = Fraction(4, 5), []
    for members in ((0, 1), (2, 3)):
        total = sum((weight * (2 * (k + 1)) - (k + 1)) * (2 * (k + 1)) for k in members) / 2
        weight -= total / 100
        found.append(weight)
    return found


@pytest.mark.parametrize("name", ["lifetime", "lifetime_scoped"])
def test_the_lifetime_composition_trains_then_evaluates_once(name: str) -> None:
    reference = run(name)
    drained(reference)
    updates = [f.values[1] for f in reference.facts_of("updated") if f.owner == "model"]
    assert updates == pytest.approx([float(w) for w in expected_weights()])
    (measured,) = results(reference, "measure")
    row = measured[0]
    assert isinstance(row, dict)
    assert row["key"] == 4, "the evaluation takes the item after the four training members"
    assert row["prediction"] == pytest.approx(6.35)
    assert row["loss"] == pytest.approx(0.91125)
    evaluation = [f for f in reference.facts_of("leased") if "@evaluation" in f.owner]
    assert [f.values[2] for f in evaluation] == [Version("model", 1, "initial", 2)], "it reads the model after both updates"
    consumed = {key for key, d in reference.settlements("records").items() if d == ("consumed",)}
    assert consumed == set(range(5))
    assert all(d == ("cancelled",) for key, d in reference.settlements("records").items() if key not in consumed)


def test_the_stop_follows_the_consumers_and_the_producer_settles_before_the_strategy() -> None:
    reference = run("lifetime")

    def ended(owner: str) -> int:
        """When the outermost invocation of a statement ended; its productions and iterations share its owner."""
        outermost = min((o.invocation for o in reference.outcomes.values() if o.owner == owner), key=len)
        return next(i for i, f in enumerate(reference.facts) if f.kind == "outcome" and f.invocation == outermost)

    stop = next(i for i, f in enumerate(reference.facts) if f.kind == "stop_requested")
    assert ended("@run/@activities/@consumers") < stop < ended("@run/@activities/@producer") < ended("@run")


def test_the_data_only_control_reuses_the_protocols_with_a_moving_policy() -> None:
    reference = run("data_only")
    drained(reference)
    assert reference.resources["policy"].ordinal == 4
    selected = {row[0]["key"]: row[0] for row in results(reference, "select")}
    scores = [row[0] for row in results(reference, "score")]
    assert scores == [(2 * selected[key]["x"] + 1) ** 2 for key in range(4)], "the consumer scores items in admission order"
    updates = [f.values[1] for f in reference.facts_of("updated") if f.owner == "policy"]
    assert updates == [s % 5 for s in scores]
    assert {key for key, d in reference.settlements("records").items() if d == ("consumed",)} == {0, 1, 2, 3}


def test_filtering_forwards_even_records_to_the_requested_consumer() -> None:
    reference = run("filtering")
    drained(reference)
    assert reference.resources["products"].value == ((0, 2.0), (2, 6.0), (4, 10.0))
    raw = reference.settlements("raw")
    assert all(raw[key] == ("filtered_out",) for key in raw if key % 2 and raw[key] != ("cancelled",))
    assert [raw[key] for key in (0, 2, 4)] == [("forwarded",)] * 3
    (request,) = reference.facts_of("request_settled")
    assert request.values[1] == "completed"


def test_a_late_input_keeps_the_requested_consumer_open() -> None:
    resources, made = CASES["lifetime"]
    reference = Reference(normalize(parse(fixture("lifetime"))), resources, made())
    (encode,) = calls(reference, "encode")
    reference.hold(encode, 4)
    reference.run()
    assert not reference.settled
    training = [o for o in reference.outcomes.values() if o.owner.endswith("@training")]
    assert all(o.state == "terminal" for o in training)
    consumers = next(o for o in reference.outcomes.values() if o.owner == "@run/@activities/@consumers")
    assert consumers.state != "terminal", "the evaluation still waits for item 4"
    reference.complete(encode, 4)
    reference.run()
    drained(reference)


# What verification refuses in these compositions.


def test_a_training_only_stop_does_not_cover_the_requested_evaluation() -> None:
    text = fixture("lifetime").replace("iro.stop @producer after @consumers", "iro.stop @producer after @training")
    with pytest.raises(Rejection) as raised:
        verify(normalize(parse(text)))
    (found,) = raised.value.diagnostics
    assert found.code == "iro-life-004"
    assert "@evaluation's pending request" in found.message


def test_a_request_from_an_undeclared_source_is_refused() -> None:
    text = fixture("lifetime").replace("sources [@training]", "sources [@consumers]")
    with pytest.raises(Rejection) as raised:
        verify(normalize(parse(text)))
    assert "iro-norm-005" in raised.value.codes


def test_a_handler_template_without_a_request_makes_no_invocation() -> None:
    text = fixture("lifetime").replace(
        "        iro.choice %due {\n          iro.request @evaluation(%window) from @training at settled_success\n        }\n", ""
    )
    text = text.replace(
        "          capture(%records, %model) {\n        %claim",
        "          capture(%records, %model) {\n      ^(%window: int):\n        %claim",
    )
    assert "^(%window: int):\n        %claim" in text
    reference = Reference(normalize(parse(text, source="no_request.iro")), *_resources_and("lifetime"))
    reference.run()
    assert reference.settled and reference.errors == []
    served = [o for o in reference.outcomes.values() if o.owner.endswith("@evaluation")]
    assert len(served) == 1, "only the handler itself is invoked, never a request it was not given"


def _resources_and(name: str):
    resources, made = CASES[name]
    return resources, made()


# Scenarios of the governing requirements that belong to the language half.


def test_several_legal_interleavings_settle_with_the_same_outcome() -> None:
    import random

    baseline = run("lifetime")
    weights = [f.values[1] for f in baseline.facts_of("updated")]
    for seed in range(8):
        chooser = random.Random(seed)
        resources, made = CASES["lifetime"]
        reference = Reference(normalize(parse(fixture("lifetime"))), resources, made(), validators=standins.validators())
        while enabled := reference.enabled():
            reference.step(chooser.choice(enabled))
        drained(reference)
        assert [f.values[1] for f in reference.facts_of("updated")] == weights


def test_run_cancellation_disposes_a_pending_request_before_its_handler_runs() -> None:
    resources, made = CASES["lifetime"]
    reference = Reference(normalize(parse(fixture("lifetime"))), resources, made())
    while not reference.facts_of("request_pending"):
        assert reference.step()
    reference.cancel()
    reference.run()
    assert reference.settled
    assert [f.values[1] for f in reference.facts_of("request_disposed")] == ["cancelled"]
    assert not results(reference, "measure"), "the cancelled request's handler never measures"
    assert all(not s.outstanding for s in reference.streams.values())


def admitted(policy: str) -> Reference:
    text = fixture("data_only").replace("admit any_with_provenance(@policy)", f"admit {policy}")
    reference = Reference(normalize(parse(text)), {"policy": 0}, standins.data_only(), validators=standins.validators())
    reference.run()
    return reference


def test_exact_current_admission_rejects_items_produced_before_an_update() -> None:
    exact = admitted("exact_current(@policy)")
    drained(exact)
    rejected = [key for key, d in exact.settlements("records").items() if d == ("rejected",)]
    assert rejected, "production runs ahead of the consumer's policy updates"
    for fact in exact.facts_of("admission"):
        if fact.values[2]:
            observed = fact.values[1]
            assert isinstance(observed, tuple)
            (seen,) = observed
            assert isinstance(seen, Version)
            assert seen.ordinal == len([u for u in exact.facts_of("updated") if exact.facts.index(u) < exact.facts.index(fact)])


def test_bounded_lag_admits_older_items_within_the_bound() -> None:
    exact = admitted("exact_current(@policy)")
    lagging = admitted("bounded_lag(@policy, max 3)")
    drained(lagging)
    assert not [d for d in lagging.settlements("records").values() if d == ("rejected",)]
    assert len(exact.settlements("records")) > len(lagging.settlements("records"))


def test_the_data_only_control_reports_its_live_read_race() -> None:
    verification = verify(normalize(parse(fixture("data_only"))))
    races = [c for c in verification.contributors if c.reproducibility == "may_depend_on_schedule"]
    assert any("@policy" in c.reason for c in races), "the producer's protected read can overlap the consumer's update"
