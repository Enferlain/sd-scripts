"""Independent values/outcomes constrain an actual xDSL local-group rewrite."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from dataclasses import asdict, replace

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.ir import Block, Region

from . import prepare as preparation
from .conformance import check_b
from .dialects import CallOp, ChainOp, FusedOp, MapOp, PlanOp, dump, parse
from .fixture import build, execute, registry
from .fusion import rewrite
from .prepare import NotReady, Registry, accept, origin, prepare
from .runtime import ScalarSession


def test_rewrite_uses_a_clone_and_retains_every_stage() -> None:
    accepted = accept(build())
    unfused = prepare(accepted, registry())
    fused = prepare(accepted, registry(), fuse=True)
    assert unfused.accepted == fused.accepted == accepted
    assert unfused.local.source_groups == (("data.double",), ("data.offset",))
    assert fused.local.source_groups == (("data.double", "data.offset"),)
    assert "exec.fused" not in unfused.text and "exec.fused" in fused.text
    assert len(fused.local.generated) == 1 and "invoke_stage" not in fused.local.generated[0]
    module = parse(unfused.text)
    before = dump(module)
    sources = {origin(op): op for op in parse(accepted.text).walk() if isinstance(op, MapOp)}
    transformed = rewrite(module, sources)
    assert dump(module) == before and dump(transformed) == fused.text
    assert dump(rewrite(module, {})) == before  # A pure label alone is insufficient.
    assert dump(rewrite(transformed, sources)) == dump(transformed)  # Stable fixed point.


@pytest.mark.parametrize("effect", ["pure", "may_effect"])
def test_fusion_requires_supported_resource_effects(effect: str) -> None:
    module = build()
    first = next(op for op in module.walk() if isinstance(op, MapOp))
    first.properties["effect"] = StringAttr(effect)
    bindings = dict(registry().bindings)
    bindings["double_v1"] = replace(bindings["double_v1"], effect=effect)
    ready = prepare(accept(module), Registry("effect", bindings), fuse=True)
    assert ("exec.fused" in ready.text) == (effect == "pure")
    manifest = json.loads(ready.manifest)
    assert manifest["local_lowering"] == ("fused-v1" if effect == "pure" else "unfused-v1")
    assert manifest["fused_groups"] == (1 if effect == "pure" else 0)


@pytest.mark.parametrize("case", ["origins", "profile", "order", "binding", "incomplete", "consumer", "unrequested"])
def test_corrupt_fused_group_rejects_before_execution(case: str, monkeypatch) -> None:
    original = preparation.parse

    def corrupted(text):
        module = original(text)
        if not any(isinstance(op, PlanOp) for op in module.walk()):
            return module
        groups = [op for op in module.walk() if isinstance(op, FusedOp)]
        if case == "unrequested":
            producer = next(op for op in module.walk() if isinstance(op, ChainOp) and op.role.data == "producer")
            calls = tuple(producer.body.block.ops)
            group = FusedOp(
                properties={
                    "profile": StringAttr("ordered-validated-local-v1"),
                    "origins": ArrayAttr(call.origin for call in calls if isinstance(call, CallOp)),
                },
                regions=[Region(Block(call.clone() for call in calls))],
            )
            for call in calls:
                producer.body.block.erase_op(call)
            producer.body.block.add_op(group)
            return module
        group = groups[0]
        if case == "origins":
            group.properties["origins"] = ArrayAttr([StringAttr("unrelated")])
        elif case == "profile":
            group.properties["profile"] = StringAttr("drop-validation")
        elif case == "order":
            first = group.body.block.first_op
            assert first is not None
            group.body.block.detach_op(first)
            group.body.block.add_op(first)
            group.properties["origins"] = ArrayAttr(call.origin for call in group.body.block.ops if isinstance(call, CallOp))
        elif case == "binding":
            left, right = tuple(group.body.block.ops)
            assert isinstance(left, CallOp) and isinstance(right, CallOp)
            left.properties["implementation"], right.properties["implementation"] = right.implementation, left.implementation
        elif case == "incomplete":
            first = group.body.block.first_op
            assert first is not None
            group.body.block.erase_op(first)
        else:
            consumer = next(op for op in module.walk() if isinstance(op, ChainOp) and op.role.data == "consumer")
            calls = tuple(consumer.body.block.ops)
            for call in calls:
                consumer.body.block.erase_op(call)
            consumer.body.block.add_op(
                FusedOp(
                    properties={"profile": group.profile, "origins": ArrayAttr(call.origin for call in calls if isinstance(call, CallOp))},
                    regions=[Region(Block(call.clone() for call in calls))],
                )
            )
        return module

    monkeypatch.setattr(preparation, "parse", corrupted)
    with pytest.raises(NotReady, match=r"prepared|fused|semantic"):
        prepare(accept(build()), registry(), fuse=case != "unrequested")


@pytest.mark.asyncio
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("horizon", [1, 2])
async def test_dispatch_changes_preserve_independent_feedback_contract(fuse: bool, horizon: int) -> None:
    prepared = prepare(accept(build(horizon=horizon)), registry(), fuse=fuse)
    report = asdict((await execute(prepared, diagnostic_limit=0)).report)
    check_b(report, horizon=horizon)
    assert report["local_dispatches"] == len(report["tickets"]) * (1 if fuse else 2)
    assert all(len(ticket["calls"]) == 4 for ticket in report["tickets"])


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["double_v1", "offset_v1"])
@pytest.mark.parametrize("failure", ["invalid", "raise"])
async def test_each_stage_validation_and_later_entry_survive_fusion(symbol: str, failure: str) -> None:
    reports = []
    for fuse in (False, True):
        bindings = dict(registry().bindings)

        async def bad(value):
            if failure == "raise":
                raise RuntimeError("entered pure stage failed")
            return None

        bindings[symbol] = replace(bindings[symbol], invoke=bad)
        report = asdict((await execute(prepare(accept(build(horizon=1)), Registry("fault", bindings), fuse=fuse))).report)
        check_b(report, horizon=1)
        failed = next(ticket for ticket in report["tickets"] if ticket["terminal"] == "failed")
        expected = ["data.double"] if symbol == "double_v1" else ["data.double", "data.offset"]
        assert [call["origin"] for call in failed["calls"]] == expected
        assert failed["calls"][-1]["outcome"] == ("invalid_output" if failure == "invalid" else "entered_failed")
        assert failed["publication_ack"] is None and failed["feedback_ack"] is None
        reports.append(failed)
    assert reports[0] == reports[1]  # Horizon one fixes this ticket's required observations.


@pytest.mark.asyncio
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("interrupt", ["stop", "cancel"])
@pytest.mark.parametrize("fail", [False, True])
async def test_fused_physical_use_drains_before_stage_or_session_release(fuse: bool, interrupt: str, fail: bool) -> None:
    entered, release, finished = (threading.Event() for _ in range(3))
    bindings = dict(registry().bindings)
    original = bindings["double_v1"].invoke

    async def selected(value):
        def physical():
            entered.set()
            try:
                if not release.wait(5):
                    raise TimeoutError("physical work not released")
                if fail:
                    raise RuntimeError("physical local failure after stop")
            finally:
                finished.set()

        await asyncio.to_thread(physical)
        return await original(value)

    bindings["double_v1"] = replace(bindings["double_v1"], invoke=selected)
    session = ScalarSession(prepare(accept(build()), Registry("physical", bindings), fuse=fuse), diagnostic_limit=0)
    running = asyncio.create_task(session.run())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        session.stop()
        if interrupt == "cancel":
            running.cancel()
            await asyncio.sleep(0)
            running.cancel()
        await asyncio.sleep(0)
        assert not running.done() and session.active_calls == 1 and not finished.is_set() and session.report is None
    finally:
        release.set()
        session.stop()
    if interrupt == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 5)
        report = session.report
    else:
        report = await asyncio.wait_for(running, 5)
    assert report is not None and finished.is_set() and session.active_calls == session.outstanding == 0
    check_b(asdict(report))
    assert report.status == ("cancelled" if interrupt == "cancel" else "failed" if fail else "stopped")
    if fail:
        assert report.errors and report.tickets[0].calls[-1].origin == "data.double"


@pytest.mark.asyncio
async def test_shared_wrong_binding_is_not_certified_by_fusion_agreement() -> None:
    bindings = dict(registry().bindings)
    bindings["double_v1"] = replace(bindings["double_v1"], invoke=bindings["offset_v1"].invoke)
    projections = []
    for fuse in (False, True):
        report = asdict((await execute(prepare(accept(build(horizon=1)), Registry("shared-fault", bindings), fuse=fuse))).report)
        projections.append([call["output"] for ticket in report["tickets"] for call in ticket["calls"]])
        with pytest.raises(AssertionError):
            check_b(report, horizon=1)
        bad = copy.deepcopy(report)
        bad["tickets"][0]["calls"][0]["origin"] = "unrelated"
        with pytest.raises(AssertionError):
            check_b(bad, horizon=1)
    assert projections[0] == projections[1]
