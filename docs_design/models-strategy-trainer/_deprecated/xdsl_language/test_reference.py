"""Independent fixture expectations and control/physical lifetime witnesses."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys

from dataclasses import replace
from pathlib import Path

import pytest

from pliron_candidate.composition_conformance import expected_weights
from xdsl.dialects.builtin import ArrayAttr, StringAttr, SymbolRefAttr
from xdsl.ir import Block
from xdsl.pattern_rewriter import PatternRewriter

from .contracts import origin
from .fixtures import body, data_only, end, lifetime, node, repeat, structured
from .normalize import normalize, source_index
from .dialects import ChoiceOp, ConstantOp, OnceOp, bool_type, index_type
from .observations import compare, observe
from .reference import CallResult, Fact, Outcome, Reference, ReferenceError, ReferenceLimit, Request, StandIn, State, Update, checked
from .standins import bindings, schemas


def model_reference(*, extra_scope: bool = False, prefix: str = "", history: bool = True, windows: int = 2) -> Reference:
    return Reference(
        normalize(lifetime(extra_scope=extra_scope, prefix=prefix, windows=windows)),
        {prefix + "encoder": 2.0, prefix + "model": 0.8},
        bindings(),
        schemas=schemas(),
        history=history,
    )


def top(ref: Reference, owner: str) -> Outcome:
    return min((value for value in ref.outcomes.values() if value.owner == owner), key=lambda x: len(x.invocation))


def facts(ref: Reference, owner: str, kind: str) -> list[Fact]:
    return [fact for fact in ref.facts if fact.owner == owner and fact.kind == kind]


def returned(fact: Fact) -> tuple:
    return checked(fact.values[1], tuple)


def row(fact: Fact) -> dict:
    return checked(returned(fact)[0], dict)


def drained(ref: Reference) -> None:
    assert ref.settled
    assert all(not stream.outstanding and not stream.ready for stream in ref.streams.values())
    assert all(not cap.active for cap in ref.capabilities.values())
    assert all(not resource.readers and resource.writer is None for resource in ref.resources.values())
    assert all(not handler.pending and handler.active == 0 for handler in ref.handlers.values())


@pytest.mark.parametrize("extra_scope", [False, True])
@pytest.mark.parametrize("prefix", ["", "renamed."])
def test_lifetime_math_identity_and_distinct_settlement(extra_scope: bool, prefix: str) -> None:
    ref = model_reference(extra_scope=extra_scope, prefix=prefix)
    ref.run()
    drained(ref)
    assert not ref.errors
    weights = expected_weights()
    assert ref.resources[prefix + "model"].value == pytest.approx(float(weights[1]))
    updates = facts(ref, prefix + "model", "updated")
    assert [fact.values[1] for fact in updates] == pytest.approx([float(x) for x in weights])
    assert ref.resources[prefix + "model"].ordinal == 2
    measured = facts(ref, prefix + "measure", "physical_completed")
    assert len(measured) == 1
    result = row(measured[0])
    assert result["key"] == 4 and result["epoch"] == 2
    assert result["prediction"] == pytest.approx(6.35)
    assert result["loss"] == pytest.approx(0.91125)
    request = checked(measured[0].values[2], Request)
    assert request.payload == 1
    assert (prefix + "training", 1) in request.cause
    assert measured[0].values[3] == ((prefix + "records", 4),)
    training = top(ref, prefix + "training")
    training_end = next(i for i, fact in enumerate(ref.facts) if fact.kind == "outcome" and fact.invocation == training.invocation)
    measurement_entry = next(i for i, fact in enumerate(ref.facts) if fact.kind == "physical_entered" and fact.owner == prefix + "measure")
    assert training_end < measurement_entry
    stream = ref.streams[prefix + "records"]
    assert {key for key, value in stream.terminal.items() if value == "consumed"} == set(range(5))
    assert 0 < stream.high_water <= 3
    group = top(ref, prefix + ("consumer.scope" if extra_scope else "consumers"))
    producer = top(ref, prefix + "producer")
    root = top(ref, prefix + "run")
    group_end = next(i for i, fact in enumerate(ref.facts) if fact.kind == "outcome" and fact.invocation == group.invocation)
    producer_end = next(i for i, fact in enumerate(ref.facts) if fact.kind == "outcome" and fact.invocation == producer.invocation)
    root_end = next(i for i, fact in enumerate(ref.facts) if fact.kind == "outcome" and fact.invocation == root.invocation)
    stop = next(i for i, fact in enumerate(ref.facts) if fact.kind == "stop_requested")
    assert group_end < stop < producer_end < root_end


def test_data_only_reuses_protocols_with_dynamic_policy_and_real_associations() -> None:
    ref = Reference(normalize(data_only()), {"policy": 0}, bindings(), schemas=schemas())
    ref.run()
    drained(ref)
    assert not ref.errors
    assert set(ref.resources) == {"policy"}
    assert not ref.handlers
    assert ref.resources["policy"].ordinal == 4
    scored = facts(ref, "score", "physical_completed")
    assert len(scored) == 4
    produced = {row(fact)["key"]: row(fact) for fact in facts(ref, "select", "physical_completed")}
    for event in scored:
        stream, key = checked(event.values[3], tuple)[0]
        assert stream == "records"
        expected = (2 * produced[key]["x"] + 1) ** 2
        assert event.values[1] == (expected,)
    feedback = facts(ref, "policy", "updated")
    assert [event.values[1] for event in feedback] == [int(returned(event)[0]) % 5 for event in scored]
    assert {key for key, status in ref.streams["records"].terminal.items() if status == "consumed"} == {0, 1, 2, 3}


def test_late_input_keeps_the_requested_consumer_lifetime_open() -> None:
    ref = model_reference()
    ref.held.add(("encode", 4))
    ref.run()
    assert not ref.settled
    assert top(ref, "training").state == State.TERMINAL
    assert top(ref, "consumers").state != State.TERMINAL
    assert ref.handlers["evaluation"].next_ordinal == 1
    assert ref.handlers["evaluation"].active + len(ref.handlers["evaluation"].pending) == 1
    assert "producer" not in ref.stopped
    assert not facts(ref, "measure", "physical_entered")
    ref.held.clear()
    ref.run()
    drained(ref)
    assert row(facts(ref, "measure", "physical_completed")[0])["key"] == 4


def test_pending_request_is_owned_without_an_executing_handler() -> None:
    ref = model_reference()
    for _ in range(2000):
        enabled = [index for index in ref.enabled() if (command := ref.tasks[index].command) is not None and command.kind != "next_request"]
        if not enabled:
            break
        candidates = [index for index in enabled if index >= ref.cursor]
        ref.step(candidates[0] if candidates else enabled[0])
    assert top(ref, "training").state == State.TERMINAL
    assert len(ref.handlers["evaluation"].pending) == 1
    assert ref.handlers["evaluation"].active == 0
    assert top(ref, "consumers").state != State.TERMINAL
    ref.run()
    drained(ref)


def test_handler_template_does_not_invent_an_admitted_request() -> None:
    ref = model_reference(windows=1)
    ref.run()
    drained(ref)
    assert ref.handlers["evaluation"].next_ordinal == 0
    assert not facts(ref, "measure", "physical_entered")
    assert ref.resources["model"].ordinal == 1


@pytest.mark.parametrize("admitted", [False, True])
def test_orderly_stop_respects_the_iteration_admission_boundary(admitted: bool) -> None:
    ref = model_reference()
    for _ in range(2000):
        windows = [
            value
            for value in ref.outcomes.values()
            if value.owner == "training" and value.invocation_source == "repeat_policy" and value.invocation[-1][1] == 1
        ]
        if windows and windows[0].state == (State.RUNNING if admitted else State.PENDING):
            break
        assert ref.step()
    else:
        pytest.fail("second window did not reach its selected boundary")
    ref.request_stop("training")
    ref.run()
    drained(ref)
    assert ref.resources["model"].ordinal == (2 if admitted else 1)
    assert ref.handlers["evaluation"].next_ordinal == (1 if admitted else 0)
    if admitted:
        assert ref.handlers["evaluation"].dispositions == {0: "completed"}
    else:
        assert windows[0].status == "stopped_unadmitted"


def test_cancellation_retains_protection_until_external_physical_completion() -> None:
    ref = model_reference()
    ref.held.add(("measure", 0))
    ref.run()
    assert ref.resources["model"].readers
    ref.cancelled = True
    ref.run()
    assert not ref.settled
    assert ref.resources["model"].readers
    assert ref.handlers["evaluation"].active == 1
    ref.held.clear()
    ref.run()
    drained(ref)
    assert top(ref, "run").status == "cancelled"
    assert ref.handlers["evaluation"].dispositions == {0: "cancelled"}
    assert facts(ref, "measure", "physical_completed")


def test_evaluation_success_survives_a_later_producer_failure() -> None:
    ref = model_reference()
    ref.held.add(("encode", 6))
    ref.run()
    assert top(ref, "consumers").status == "succeeded"
    assert facts(ref, "measure", "physical_completed")
    assert not ref.settled and "producer" in ref.stopped
    ref.failures[("encode", 6)] = "late physical failure"
    ref.held.clear()
    ref.run()
    drained(ref)
    assert top(ref, "run").status == "failed"
    assert ref.handlers["evaluation"].dispositions == {0: "completed"}
    assert ref.errors == ["encode: late physical failure"]
    assert ref.streams["records"].terminal[6] == "failed"


def test_failure_before_the_trigger_closes_the_unrequested_template() -> None:
    ref = model_reference()
    ref.failures[("gradient", 1)] = "member failed"
    ref.run()
    drained(ref)
    assert not ref.handlers["evaluation"].next_ordinal
    assert ref.resources["model"].ordinal == 0
    assert top(ref, "run").status == "failed"


def test_cancellation_and_late_physical_failure_remain_separate_facts() -> None:
    ref = model_reference()
    ref.held.add(("measure", 0))
    ref.run()
    ref.cancelled = True
    ref.failures[("measure", 0)] = "late measurement failure"
    ref.held.clear()
    ref.run()
    drained(ref)
    assert top(ref, "run").status == "cancelled"
    assert top(ref, "run").cancel_requested
    assert ref.errors == ["measure: late measurement failure"]
    assert ref.handlers["evaluation"].dispositions == {0: "failed"}
    assert facts(ref, "measure", "physical_failed")


def test_invalid_output_does_not_erase_an_already_known_update() -> None:
    registry = bindings()
    registry["advance"] = StandIn(
        ("capability", "scalar"), ("scalar",), lambda args: CallResult(("invalid",), (Update(0, 0.7),)), updates=(0,)
    )
    ref = Reference(normalize(lifetime()), {"encoder": 2.0, "model": 0.8}, registry, schemas=schemas())
    ref.run()
    drained(ref)
    assert ref.resources["model"].value == 0.7 and ref.resources["model"].ordinal == 1
    assert facts(ref, "model", "updated")
    assert "output contract mismatch" in ref.errors[0]
    assert not ref.handlers["evaluation"].next_ordinal


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("domain", ["lifetime", "extra_scope", "data_only"])
def test_seeded_external_scheduling_smoke(seed: int, domain: str) -> None:
    ref = (
        Reference(normalize(data_only()), {"policy": 0}, bindings(), schemas=schemas())
        if domain == "data_only"
        else model_reference(extra_scope=domain == "extra_scope")
    )
    rng = random.Random(seed)
    for _ in range(3000):
        enabled = ref.enabled()
        if not enabled:
            break
        ref.step(rng.choice(enabled))
    drained(ref)
    assert not ref.errors
    if domain != "data_only":
        assert ref.resources["model"].value == pytest.approx(0.635)


def test_reference_rejects_an_external_choice_that_is_not_enabled() -> None:
    ref = model_reference()
    with pytest.raises(ReferenceError, match="non-enabled"):
        ref.step(1000)
    ref.run()
    drained(ref)


def test_unknown_stop_symbol_reports_a_reference_error() -> None:
    ref = model_reference()
    with pytest.raises(ReferenceError, match="unknown stop symbol"):
        ref.request_stop("missing")
    ref.run()
    drained(ref)


def test_observable_comparison_ignores_states_and_global_call_counters() -> None:
    ref = model_reference()
    ref.run()
    actual = [
        replace(fact, values=(999, *fact.values[1:])) if fact.kind.startswith("physical_") else fact
        for fact in ref.facts
        if fact.kind != "state"
    ]
    compare(observe(ref.facts), observe(actual))
    with pytest.raises(ReferenceError, match="causal"):
        compare(observe(ref.facts), observe(list(reversed(actual))))
    with pytest.raises(ReferenceError, match="owner-visible"):
        compare(observe(ref.facts), observe([fact for fact in actual if fact.kind != "request_pending"]))


def test_reference_rejects_bad_standin_output_and_undeclared_authority() -> None:
    registry = bindings()
    registry["encode"] = StandIn(("capability", "index"), ("record",), lambda args: CallResult(({},), (Update(0, 99),)))
    ref = Reference(normalize(lifetime()), {"encoder": 2.0, "model": 0.8}, registry, schemas=schemas())
    ref.run()
    drained(ref)
    assert ref.resources["encoder"].value == 2.0
    assert "undeclared stand-in update" in ref.errors[0]


def test_reference_does_not_consult_diagnostic_history() -> None:
    ref = model_reference(history=False)
    ref.run()
    drained(ref)
    assert not ref.facts and not ref.errors
    assert ref.resources["model"].value == pytest.approx(0.635)
    assert ref.handlers["evaluation"].dispositions == {0: "completed"}


def test_repeated_consumer_scopes_close_their_own_request_sources() -> None:
    author = lifetime()
    group = source_index(author)["consumers"]
    cycle = Block(arg_types=[*(value.type for value in group.operands), index_type()])
    cloned = group.clone(value_mapper=dict(zip(group.operands, cycle.args[:-1], strict=True)))
    cycle.add_op(cloned)
    end(cycle, "cycles.body")
    replacement = repeat("cycles", group.operands, cycle, 2)
    PatternRewriter(group).replace(group, replacement)
    ref = Reference(normalize(author), {"encoder": 2.0, "model": 0.8}, bindings(), schemas=schemas())
    ref.run()
    drained(ref)
    assert not ref.errors
    assert ref.resources["model"].ordinal == 4
    assert ref.handlers["evaluation"].dispositions == {0: "completed", 1: "completed"}
    measurements = facts(ref, "measure", "physical_completed")
    assert [row(fact)["key"] for fact in measurements] == [4, 9]
    assert measurements[0].invocation != measurements[1].invocation
    assert measurements[0].values[2] != measurements[1].values[2]


def test_duplicate_owned_fact_keys_cannot_be_hidden_by_dictionary_overwrite() -> None:
    ref = model_reference()
    ref.run()
    selected = next(fact for fact in ref.facts if fact.kind == "selected")
    with pytest.raises(ReferenceError, match="duplicate owner-visible"):
        observe([*ref.facts, selected])


def test_disposing_unclaimed_products_in_another_order_preserves_meaning() -> None:
    ref = model_reference()
    ref.run()
    indices = [index for index, fact in enumerate(ref.facts) if fact.kind == "disposition" and fact.values[1] == "discarded"]
    assert len(indices) >= 2
    actual = list(ref.facts)
    for target, source in zip(indices, reversed(indices), strict=True):
        actual[target] = ref.facts[source]
    compare(observe(ref.facts), observe(actual))


def test_diagnostic_origins_do_not_select_runtime_behavior_or_resource_identity() -> None:
    author = lifetime()
    for op in source_index(author).values():
        op.properties["origin"] = StringAttr("diagnostic." + origin(op))
    ref = Reference(normalize(author), {"encoder": 2.0, "model": 0.8}, bindings(), schemas=schemas())
    ref.run()
    drained(ref)
    assert not ref.errors and ref.resources["model"].value == pytest.approx(0.635)
    assert ref.handlers["diagnostic.evaluation"].dispositions == {0: "completed"}


def test_pending_work_cannot_be_hidden_by_a_handler_that_was_not_invoked() -> None:
    author = lifetime()
    handler = source_index(author)["evaluation"]
    yes, no = body(handler.operands), body(handler.operands)
    yes.add_op(handler.clone(value_mapper=dict(zip(handler.operands, yes.args, strict=True))))
    end(yes, "optional.handler.true")
    end(no, "optional.handler.false")
    condition = node(ConstantOp, "optional.handler.condition", outputs=[bool_type()], literal=StringAttr("false"))
    choice = structured(ChoiceOp, "optional.handler", [condition.results[0], *handler.operands], [yes, no])
    PatternRewriter(handler).replace(handler, [condition, choice])
    ref = Reference(normalize(author), {"encoder": 2.0, "model": 0.8}, bindings(), schemas=schemas())
    ref.run()
    assert not ref.settled
    assert ref.handlers["evaluation"].pending
    assert top(ref, "consumers").state == State.SETTLING
    assert "producer" not in ref.stopped
    ref.cancelled = True
    ref.run()
    drained(ref)
    assert ref.handlers["evaluation"].dispositions == {0: "cancelled_unfulfilled"}


def test_stop_comparison_uses_the_completed_invocation_instead_of_a_source_name() -> None:
    first = (("run", 0), ("source", 0))
    second = (("run", 0), ("source", 1))
    events = [
        Fact("source", first, "outcome", ("succeeded",)),
        Fact("producer", (("run", 0), ("control", 0)), "stop_requested", ("source", (first,))),
        Fact("source", second, "outcome", ("succeeded",)),
        Fact("producer", (("run", 0), ("control", 1)), "stop_requested", ("source", (second,))),
    ]
    compare(observe(events), observe(events))
    with pytest.raises(ReferenceError, match="causal"):
        compare(observe(events), observe([events[1], events[0], *events[2:]]))


def test_nested_persistent_declarations_are_an_explicit_reference_target_limit() -> None:
    author = lifetime()
    encoder = source_index(author)["encoder"]
    declaration = Block()
    local = encoder.clone()
    declaration.add_op(local)
    end(declaration, "local.declaration", local.results[0])
    wrapper = structured(OnceOp, "local.declaration", [], [declaration], outputs=[encoder.results[0].type])
    PatternRewriter(encoder).replace(encoder, wrapper)
    normalized = normalize(author)
    with pytest.raises(ReferenceLimit, match="root-owned persistent"):
        Reference(normalized, {"encoder": 2.0, "model": 0.8}, bindings(), schemas=schemas())


@pytest.mark.parametrize("kind", ["stop", "handler"])
def test_self_referential_source_settlement_has_a_typed_diagnostic(kind: str) -> None:
    author = lifetime()
    index = source_index(author)
    if kind == "stop":
        index["stop"].properties["source"] = SymbolRefAttr("activities")
    else:
        index["evaluation"].properties["sources"] = ArrayAttr([SymbolRefAttr("consumers")])
        index["request"].properties["source"] = SymbolRefAttr("consumers")
    normalized = normalize(author)
    with pytest.raises(ReferenceError, match="self-referential source settlement"):
        Reference(normalized, {"encoder": 2.0, "model": 0.8}, bindings(), schemas=schemas())


@pytest.mark.parametrize("optimized", [False, True])
def test_detached_reference_in_fresh_normal_and_optimized_children(tmp_path: Path, optimized: bool) -> None:
    document = tmp_path / "normalized.mlir"
    document.write_text(normalize(lifetime()).text)
    script = """
import importlib, json, pathlib, sys
before = sys._is_gil_enabled()
base = 'docs_design.models-strategy-trainer.xdsl_language'
d = importlib.import_module(base + '.dialects')
n = importlib.import_module(base + '.normalize')
r = importlib.import_module(base + '.reference')
s = importlib.import_module(base + '.standins')
ref = r.Reference(n.normalize(d.parse(pathlib.Path(sys.argv[1]).read_text())), {'encoder':2.0,'model':0.8}, s.bindings(), schemas=s.schemas())
ref.run()
print(json.dumps({'settled':ref.settled,'errors':ref.errors,'weight':ref.resources['model'].value,'epoch':ref.resources['model'].ordinal,'requests':ref.handlers['evaluation'].dispositions,'active':sum(c.active for c in ref.capabilities.values()),'torch':'torch' in sys.modules,'author':base+'.fixtures' in sys.modules,'gil_before':before,'gil_after':sys._is_gil_enabled()}))
"""
    env = dict(os.environ)
    env.pop("PYTHON_GIL", None)
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), "-c", script, str(document)], env=env, text=True, capture_output=True, check=True
    )
    facts = json.loads(result.stdout)
    assert facts == {
        "settled": True,
        "errors": [],
        "weight": 0.635,
        "epoch": 2,
        "requests": {"0": "completed"},
        "active": 0,
        "torch": False,
        "author": False,
        "gil_before": False,
        "gil_after": False,
    }
