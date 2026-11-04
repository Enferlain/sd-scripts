"""Existing owner services: filtered demand, fusion and physical settlement."""

from __future__ import annotations

import asyncio

from dataclasses import replace

import pytest


pytest.importorskip("torch", reason="owner engine imports Torch")
from . import filter_case
from .filter_checks import EXPECTED, check_report
from .filter_contracts import INITIAL, MANIFEST
from .filter_runtime import SCENARIOS, execute
from .observations import compare, observe
from .prepare import Artifact, Policy, load, prepare
from .reference import ReferenceError
from .filter_target import installed
from .target import Binding, Controls, ExecutionError, Result, Runtime
from .test_filtering import check_reference, consumer_facts, reference


@pytest.mark.asyncio
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("opaque", [False, True])
async def test_owner_cold_cardinality_matches_independent_reference(fused, opaque):
    plan = prepare(filter_case.program(opaque=opaque), MANIFEST, Policy(workers=3, fuse_local=fused))
    plan = load(Artifact.from_json(plan.artifact.to_json()), MANIFEST, plan.policy)
    actual, facts, _ = await execute(plan)
    check_report(actual, "normal")
    ref = reference(fused=fused, opaque=opaque)
    ref.run()
    check_reference(ref)
    compare(observe(consumer_facts(ref.facts)), observe(consumer_facts(facts)))
    observed = observe(facts)
    assert all(observed.positions[a] < observed.positions[b] for a, b in observed.required_before)
    assert actual["resources"]["products"]["value"] == [list(item) for item in EXPECTED]
    assert actual["units"] == {} and actual["drained"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("scenario", SCENARIOS)
async def test_filter_pressure_boundaries_survive_diagnostic_erasure(fused, scenario):
    plan = prepare(filter_case.program(), MANIFEST, Policy(workers=3, fuse_local=fused))
    actual, facts, choices = await execute(plan, scenario, erase_history=True)
    check_report(actual, scenario)
    assert not facts and choices and actual["drained"]
    assert all(value for name, value in actual["conditions"].items() if name != "run_raised")
    expected_failure = scenario in {
        "late_source_failure",
        "cancel_filter",
        "fail_filter",
        "cancel_then_fail_filter",
        "cancel_local",
        "invalid_stage",
        "all_drops",
    }
    assert actual["conditions"]["run_raised"] == expected_failure
    assert all(not request["pending"] and not request["active"] for request in actual["requests"])
    if scenario == "late_source_failure":
        assert any("late physical source failure" in item for item in actual["failures"])
        assert any(item["source"] == "consumers" and item["status"] == "succeeded" for item in actual["source_receipts"])
    if scenario in ("fail_filter", "cancel_then_fail_filter"):
        assert any("physical filter failure" in item for item in actual["failures"])
    if scenario == "invalid_stage":
        assert not any(name in {"first.double", "next.double"} for name, _ in actual["physical_entered"])
    if scenario == "cancel_local":
        assert not any(name == "next.double" for name, _ in actual["physical_entered"])


@pytest.mark.asyncio
async def test_fusion_changes_local_dispatch_but_retains_stage_calls_and_observations():
    plain, grouped = (prepare(filter_case.program(), MANIFEST, Policy(workers=3, fuse_local=fused)) for fused in (False, True))
    left, right = await execute(plain), await execute(grouped)
    compare(observe(consumer_facts(left[1])), observe(consumer_facts(right[1])))
    stage_names = {prefix + suffix for prefix in ("first.", "next.") for suffix in ("extract", "offset", "double")}
    assert sum(value for key, value in left[0]["local_dispatches"].items() if key in stage_names) == 9
    assert sum(value for key, value in right[0]["local_dispatches"].items() if key.startswith("$fused/")) == 3
    assert not any(key in stage_names for key in right[0]["local_dispatches"])
    for owner in stage_names:
        assert sum(f.owner == owner and f.kind == "physical_completed" for f in left[1]) == sum(
            f.owner == owner and f.kind == "physical_completed" for f in right[1]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("fused", [False, True])
async def test_seeded_owner_interleavings(seed, fused):
    plan = prepare(filter_case.program(), MANIFEST, Policy(workers=3, fuse_local=fused))
    actual, _, choices = await execute(plan, seed=seed)
    check_report(actual, "normal")
    assert actual["drained"] and choices


@pytest.mark.asyncio
@pytest.mark.parametrize("fused", [False, True])
async def test_wrong_implementation_is_detected_after_actual_execution(fused):
    plan = prepare(filter_case.program(), MANIFEST, Policy(workers=3, fuse_local=fused))
    actual, _, _ = await execute(plan, wrong_double=True, erase_history=True)
    assert actual["completed"] and not actual["failures"]
    with pytest.raises(ReferenceError, match="independent expectation"):
        check_report(actual, "normal")


@pytest.mark.asyncio
async def test_filter_retains_original_payload_when_opaque_predicate_changes_its_detached_copy():
    selected = installed()

    def predicate(args):
        key = args[0]["key"]
        args[0]["x"] = 999
        return Result((key % 2 == 0,))

    selected = replace(
        selected,
        bindings=tuple(
            Binding(replace(item.spec, unknown_effects=True), predicate) if item.spec.name == "filter_even" else item
            for item in selected.bindings
        ),
    )
    plan = prepare(filter_case.program(), selected.manifest, Policy(workers=3, fuse_local=True))
    runtime = Runtime(plan, selected, INITIAL)
    await asyncio.wait_for(runtime.run(), 10)
    assert runtime.value("products") == EXPECTED


@pytest.mark.asyncio
async def test_cancellation_after_filter_claim_transfer_before_predicate_entry_drains():
    controls = Controls(before={("filter", 0): asyncio.Event()})
    selected = installed()
    plan = prepare(filter_case.program(), selected.manifest, Policy(workers=3))
    runtime = Runtime(plan, selected, INITIAL, controls=controls)
    task = asyncio.create_task(runtime.run())
    try:
        async with asyncio.timeout(10):
            while ("filter", 0) not in controls.reached:
                await asyncio.sleep(0)
        assert runtime.flows["raw"].active[0].owner[-1] == ("filter", 0)
        assert not controls.entered.get(("predicate", 0))
        ticket = runtime.flows["raw"].active[0]
        with pytest.raises(ExecutionError, match="current reservation"):
            runtime._dispose(replace(ticket), "forged", ())
        assert runtime.flows["raw"].active[0] is ticket
        runtime.cancel(task)
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
        assert all(not flow.active for flow in runtime.flows.values())
        assert runtime.workers.active == 0
        assert runtime.flows["raw"].terminal[0] == "cancelled"
        original = runtime.flows["raw"].settled[0]
        with pytest.raises(ExecutionError, match="duplicate item disposition"):
            runtime._dispose(ticket, "forged", ())
        assert runtime.flows["raw"].settled[0] == original
    finally:
        controls.before["filter", 0].set()
        if not task.done():
            runtime.cancel(task)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fused", [False, True])
async def test_interrupted_forwarding_receipts_are_independently_checked(fused):
    plan = prepare(filter_case.program(), MANIFEST, Policy(workers=3, fuse_local=fused))
    actual, _, _ = await execute(plan, "cancel_local", erase_history=True)
    check_report(actual, "cancel_local")
    actual["settlements"]["kept"]["2"]["provenance"] = [["raw", 3]]
    with pytest.raises(ReferenceError, match="provenance"):
        check_report(actual, "cancel_local")
