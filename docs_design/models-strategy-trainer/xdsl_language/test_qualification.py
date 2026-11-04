"""Integrated qualification under independently selected legal schedules."""

from __future__ import annotations

import random

import pytest


pytest.importorskip("torch", reason="owner qualification requires the existing CPU target")
from .contracts import AccessMode, origin
from .normalize import normalize
from .observations import compare
from .prepare import Artifact, Policy, load, prepare
from .qualification_cases import lag_case, make_case
from .qualification_oracle import check
from .qualification_products import reference_bindings
from .qualification_runtime import execute
from .qualification_target import selected
from .reference import Reference, State, checked
from .qualification import collect
from .standins import schemas
from .test_target import assert_recorded_causality, consumer_observation


def reference(case, *, seed=0):
    ref = Reference(normalize(case.module), case.initial, reference_bindings(), schemas=schemas())
    rng = random.Random(seed)
    for _ in range(30000):
        enabled = ref.enabled()
        if not enabled:
            break
        ref.step(rng.choice(enabled))
    assert ref.settled and not ref.errors
    return ref


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["lifetime", "extra-scope", "data-only"])
@pytest.mark.parametrize("seed", range(8))
async def test_seeded_suspensions_against_independently_scheduled_reference(name, seed):
    case = make_case(name)
    plan = prepare(case.module, selected().manifest, Policy())
    runtime, report = await execute(plan, case.initial, prefix=case.prefix, seed=seed)
    check(report, name, prefix=case.prefix)
    ref = reference(case, seed=seed)
    compare(consumer_observation(ref.facts, case.prefix), consumer_observation(runtime.facts, case.prefix))
    assert_recorded_causality(runtime.facts)
    assert report["witness"]["points"]["call_complete"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["lifetime", "extra-scope", "data-only"])
async def test_erasing_histories_at_every_checkpoint_preserves_owned_results(name):
    case = make_case(name)
    plan = prepare(case.module, selected().manifest, Policy())
    runtime, report = await execute(plan, case.initial, prefix=case.prefix, erased=True, seed=13)
    check(report, name, prefix=case.prefix, erased=True)
    assert not runtime.facts and not runtime.outcomes
    ref = reference(case, seed=13)
    assert ref.resources[case.prefix + "products"].value == runtime.value(case.prefix + "products")


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["lifetime", "extra-scope"])
@pytest.mark.parametrize(
    "mode",
    ["late", "prefetched", "orderly-before", "orderly-after", "pending-cancel", "held-cancel", "held-failure", "late-producer-failure"],
)
async def test_integrated_pressure_choices_from_cold_artifacts_without_histories(name, mode):
    case = make_case(name, windows=3 if mode == "orderly-after" else 2)
    authored = prepare(case.module, selected().manifest, Policy())
    plan = load(Artifact.from_json(authored.artifact.to_json()), selected().manifest, Policy())
    _, report = await execute(plan, case.initial, prefix=case.prefix, mode=mode, erased=True, seed=7)
    check(report, name, prefix=case.prefix, mode=mode, erased=True)
    ref = pressure_reference(case, mode)
    for resource, value in report["owner"]["resources"].items():
        assert (
            ref.resources[resource].value == pytest.approx(value["value"])
            if isinstance(value["value"], int | float)
            else ref.resources[resource].value == tuple(value["value"])
        )
        assert ref.resources[resource].ordinal == value["epoch"]
    actual_requests = report["owner"]["requests"][0]["dispositions"]
    assert {str(key): status for key, status in ref.handlers[case.prefix + "evaluation"].dispositions.items()} == actual_requests


def pressure_reference(case, mode):
    """Apply fixed external choices; never read runtime traces or results."""
    ref = Reference(normalize(case.module), case.initial, reference_bindings(), schemas=schemas())
    p = case.prefix
    if mode in ("late", "prefetched", "orderly-after", "held-cancel", "held-failure", "late-producer-failure"):
        held = {
            "late": ("encode", 4),
            "prefetched": ("gradient", 2),
            "orderly-after": ("gradient", 2),
            "held-cancel": ("measure", 0),
            "held-failure": ("measure", 0),
            "late-producer-failure": ("encode", 6),
        }[mode]
        ref.held.add((p + held[0], held[1]))
        ref.run()
        assert not ref.settled
        if mode == "late":
            assert not any(value.owner == p + "consumers" and value.state == State.TERMINAL for value in ref.outcomes.values())
        if mode == "prefetched":
            assert any(ticket.key == 4 for ticket in ref.streams[p + "records"].ready)
        if mode == "orderly-after":
            ref.request_stop(p + "training")
        if mode in ("held-cancel", "held-failure"):
            ref.cancelled = True
            ref.run()
            assert ref.resources[p + "model"].readers
        if mode in ("held-failure", "late-producer-failure"):
            ref.failures[p + held[0], held[1]] = "late physical failure" if mode == "held-failure" else "late producer failure"
        ref.held.clear()
    elif mode in ("orderly-before", "pending-cancel"):
        target = p + ("training" if mode == "orderly-before" else "evaluation")
        for _ in range(10000):
            pending = [
                value
                for value in ref.outcomes.values()
                if value.owner == target
                and value.state == State.PENDING
                and value.invocation_source == ("repeat_policy" if mode == "orderly-before" else "accepted_request")
            ]
            if pending and (mode == "pending-cancel" or pending[0].invocation[-1] == (target, 1)):
                break
            assert ref.step()
        else:
            raise AssertionError("reference did not reach independently selected pending boundary")
        if mode == "orderly-before":
            ref.request_stop(target)
        else:
            ref.cancelled = True
    ref.run()
    assert ref.settled and not any(cap.active for cap in ref.capabilities.values())
    assert all(not stream.ready and not stream.outstanding for stream in ref.streams.values())
    return ref


@pytest.mark.asyncio
async def test_returning_exclusions_allow_heartbeat_and_root_cancellation():
    case = lag_case(AccessMode.SNAPSHOT, reject_forever=True)
    plan = prepare(case.module, selected().manifest, Policy())
    runtime, report = await execute(plan, case.initial, mode="rejections", seed=17)
    check(report, "rejections", mode="rejections")
    ref = Reference(normalize(case.module), case.initial, reference_bindings(), schemas=schemas())
    for _ in range(10000):
        if ref.resources["policy"].ordinal == 2:
            break
        choices = [
            index for index in ref.enabled() if (command := ref.tasks[index].command) is not None and origin(command.op) != "consume.take"
        ]
        assert choices
        ref.step(choices[0])
    assert ref.resources["policy"].ordinal == 2
    for _ in range(30000):
        if sum(status == "rejected_stale" for status in ref.streams["records"].terminal.values()) >= 128:
            break
        assert ref.step()
    ref.cancelled = True
    ref.run()
    assert ref.settled
    for execution in (runtime, ref):
        rejected = [fact.values for fact in execution.facts if fact.kind == "admission_checked"][:128]
        assert len(rejected) == 128
        assert all(
            checked(item[1], tuple)[0].ordinal == 0 and checked(item[3], tuple)[0].ordinal == 2 and item[4] == "rejected_stale"
            for item in rejected
        )
    assert [fact.values for fact in runtime.facts if fact.kind == "admission_checked"][:128] == [
        fact.values for fact in ref.facts if fact.kind == "admission_checked"
    ][:128]


@pytest.mark.asyncio
async def test_wrong_selected_implementation_passes_contract_but_fails_independent_oracle():
    case = make_case("data-only")
    plan = prepare(case.module, selected().manifest, Policy())
    _, report = await execute(plan, case.initial, wrong_score=True, erased=True)
    assert report["owner"]["completed"] and not report["owner"]["failures"]
    with pytest.raises(AssertionError) as failed:
        check(report, case.name, erased=True)
    assert failed.value.args == ((0.0, 1),)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["lifetime", "extra-scope"])
@pytest.mark.parametrize("mode", ["orderly-before", "orderly-after"])
async def test_actual_selected_attempt_admission_facts_match_reference(name, mode):
    case = make_case(name, windows=3 if mode == "orderly-after" else 2)
    plan = prepare(case.module, selected().manifest, Policy())
    runtime, report = await execute(plan, case.initial, prefix=case.prefix, mode=mode)
    check(report, name, prefix=case.prefix, mode=mode)
    ref = pressure_reference(case, mode)
    compare(consumer_observation(ref.facts, case.prefix), consumer_observation(runtime.facts, case.prefix))
    assert_recorded_causality(runtime.facts)
    attempts = [
        outcome
        for outcome in runtime.outcomes.values()
        if outcome.owner == case.prefix + "training" and outcome.invocation_source == "repeat_policy" and outcome.invocation[-1][1] == 1
    ]
    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.status == ("stopped_unadmitted" if mode == "orderly-before" else "succeeded")
    states = [fact.values[0] for fact in runtime.facts if fact.kind == "state" and fact.invocation == attempt.invocation]
    assert states == [
        State.CREATED,
        State.PENDING,
        *([State.ADMITTED, State.RUNNING] if mode == "orderly-after" else []),
        State.CLOSING,
        State.SETTLING,
        State.TERMINAL,
    ]
    outcomes = [fact.values for fact in runtime.facts if fact.kind == "outcome" and fact.invocation == attempt.invocation]
    assert len(outcomes) == 1 and len(outcomes[0]) == 7
    if mode == "orderly-before":
        assert outcomes[0] == ("stopped_unadmitted", (), None, True, "repeat_policy", False, True)


def test_fresh_normal_and_optimized_cold_children_have_independent_parent_checks(tmp_path):
    record = collect(tmp_path / "cold", full=False)
    assert len(record["verdicts"]) == 10
    assert sum(item["oracle"].startswith("rejected") for item in record["verdicts"]) == 4
