"""Matched B controls and the valid opaque-offset fixture, before alignment."""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import subprocess
import sys
import threading
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.ir import Block, Region

_prefix = "docs_design.models-strategy-trainer"
fixture = importlib.import_module(_prefix + ".xdsl_candidate.fixture")
preparation = importlib.import_module(_prefix + ".xdsl_candidate.prepare")
dialects = importlib.import_module(_prefix + ".xdsl_candidate.dialects")
fusion = importlib.import_module(_prefix + ".xdsl_candidate.fusion")
pair = importlib.import_module(_prefix + ".xdsl_candidate.pair_extension")
oracle = importlib.import_module(_prefix + ".xdsl_candidate.conformance")
extension_oracle = importlib.import_module(_prefix + ".xdsl_candidate.extension_conformance")
native_tests = importlib.import_module(_prefix + ".pliron_candidate.test_b")
native_pair = importlib.import_module(_prefix + ".pliron_candidate.pair_extension")
native_author = importlib.import_module(_prefix + ".pliron_candidate.author")
native = native_tests.native


def prepare_b(native, framework, selected, fuse=False):
    if framework == "xdsl":
        return preparation.prepare(preparation.accept(fixture.build(horizon=2)), selected, fuse=fuse)
    return native.prepare(native.accept(native_author.build(horizon=2)), selected, fuse=fuse)[0]


def xdsl_opaque(effects):
    module = pair.build_extension()
    offset = next(op for op in module.walk() if isinstance(op, dialects.MapOp) and op.source_id.data == "data.offset")
    offset.properties["effect"] = StringAttr("may_effect")

    async def invoke(value):
        effects.append(value)
        await asyncio.sleep(0)
        return value + 1

    selected = pair.extension_registry()
    bindings = dict(selected.bindings)
    bindings["offset_v1"] = replace(bindings["offset_v1"], effect="may_effect", invoke=invoke)
    return module, replace(selected, bindings=bindings)


@pytest.mark.parametrize("framework", ["xdsl", "pliron"])
@pytest.mark.parametrize("fuse", [False, True])
def test_same_b_parameters_live_and_saved_control(native, framework, fuse):
    if framework == "xdsl":
        selected = fixture.registry()
        accepted = preparation.accept(fixture.build(horizon=2))
        first = preparation.prepare(accepted, selected, capacity=1, fuse=fuse)
        cold = preparation.prepare(accepted, selected, fuse=fuse, saved_text=first.text, saved_manifest=json.loads(first.manifest))
    else:
        selected = fixture.registry()
        first, artifact = native.prepare(native.accept(native_author.build(horizon=2)), selected, fuse=fuse)
        cold, _ = native.load(artifact["accepted"], artifact["prepared"], artifact["manifest"], selected)
    assert first.horizon == cold.horizon == 2 and first.capacity == cold.capacity == 1
    assert cold.local.source_groups == ((("data.double", "data.offset"),) if fuse else (("data.double",), ("data.offset",)))
    report = asdict(asyncio.run(fixture.execute(cold, stop_after=4, diagnostic_limit=0)).report)
    oracle.check_b(report, horizon=2)
    assert report["status"] == "stopped"


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_b_saved_plan_fresh_child_blocks_authors_and_pliron(tmp_path, fuse, optimized):
    accepted = preparation.accept(fixture.build())
    ready = preparation.prepare(accepted, fixture.registry(), fuse=fuse)
    saved = tmp_path / "b.json"
    saved.write_text(json.dumps({"accepted": accepted.text, "prepared": ready.text, "manifest": json.loads(ready.manifest)}))
    script = r"""
import asyncio,importlib,json,sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,sys.argv[1]);before=sys._is_gil_enabled()
p='docs_design.models-strategy-trainer'
def forbidden(*args,**kwargs): raise RuntimeError('author or Pliron machinery entered xDSL B cold path')
fixture=importlib.import_module(p+'.xdsl_candidate.fixture');fixture.build=forbidden
author=importlib.import_module(p+'.pliron_candidate.author');author.build=forbidden
bridge=importlib.import_module(p+'.pliron_candidate.bridge')
for method in ('request','accept','prepare','load'):setattr(bridge.Native,method,forbidden)
ir=importlib.import_module(p+'.xdsl_candidate.dialects')
api=importlib.import_module(p+'.xdsl_candidate.prepare')
data=json.loads(Path(sys.argv[2]).read_text())
ready=api.prepare(api.accept(ir.parse(data['accepted'])),fixture.registry(),fuse=sys.argv[3]=='1',
                  saved_text=data['prepared'],saved_manifest=data['manifest'])
report=asdict(asyncio.run(fixture.execute(ready,stop_after=4,diagnostic_limit=0)).report)
print(json.dumps({'report':report,'gil_before':before,'gil_after':sys._is_gil_enabled(),'optimized':not __debug__}))
"""
    result = subprocess.run(
        [
            sys.executable,
            *(["-O"] if optimized else []),
            "-c",
            script,
            str(Path(__file__).resolve().parents[3]),
            str(saved),
            "1" if fuse else "0",
        ],
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    observation = json.loads(result.stdout)
    assert observation["optimized"] is optimized
    assert observation["gil_before"] is False and observation["gil_after"] is False
    oracle.check_b(observation["report"])


@pytest.mark.parametrize("framework", ["xdsl", "pliron"])
def test_valid_opaque_execution_precedes_forged_barrier_rejection(native, framework):
    effects = []
    accepted, artifact = None, None
    if framework == "xdsl":
        module, selected = xdsl_opaque(effects)
        accepted = preparation.accept(module)
        ready = preparation.prepare(accepted, selected)
    else:
        selected = native_pair.registry(opaque=True, effects=effects)
        ready, artifact = native.prepare(native.accept(native_pair.build(opaque=True)), selected)
    report = asdict(asyncio.run(fixture.execute(ready, stop_after=4)).report)
    extension_oracle.check_extension(report)
    assert effects and len(ready.local.source_groups) == 4
    # The same valid program now receives only a forged prepared grouping.
    # Pack's semantic/resource/binding declarations are left intact.
    if framework == "xdsl":
        assert accepted is not None
        decoded = dialects.parse(ready.text, extensions=selected.dialects)
        chain = next(op for op in decoded.walk() if isinstance(op, dialects.ChainOp) and op.role.data == "producer")
        calls = tuple(chain.body.block.ops)[:2]
        for call in calls:
            chain.body.block.detach_op(call)
        forged = dialects.FusedOp(
            properties={"profile": StringAttr(fusion.PROFILE), "origins": ArrayAttr(call.origin for call in calls)},
            regions=[Region(Block(calls))],
        )
        chain.body.block.insert_op_before(forged, chain.body.block.first_op)
        with pytest.raises(preparation.NotReady, match="effect/schema/resource boundary"):
            preparation.prepare(accepted, selected, fuse=True, saved_text=dialects.dump(decoded))
    else:
        assert artifact is not None
        text = artifact["prepared"]
        calls = list(re.finditer(r"(?m)^\s+exec.call[^\n]+", text))[:2]
        assert len(calls) == 2
        content = "\n".join(match.group().rstrip(";") + (";" if i == 0 else "") for i, match in enumerate(calls))
        group = (
            'exec.fused () [] [protocol: builtin.string "preserve_each_stage_guard_and_completion"] : <() -> ()> {\n^forged():\n'
            + content
            + "\n};"
        )
        forged = text[: calls[0].start()] + "\n" + group + text[calls[-1].end() :]
        with pytest.raises(native_tests.Rejected, match=r"data.offset.*ineligible fused stage"):
            native.load(artifact["accepted"], forged, artifact["manifest"], selected)


@pytest.mark.parametrize("field", ["prepared", "manifest"])
def test_xdsl_cold_claims_reject_before_resource_lowering(field):
    selected = pair.extension_registry()
    accepted = preparation.accept(pair.build_extension())
    first = preparation.prepare(accepted, selected)
    lowered = []
    original = selected.bindings["pack_pair_v1"]
    bindings = dict(selected.bindings)
    bindings["pack_pair_v1"] = replace(original, lower=lambda values: lowered.append(values) or original.lower(values))
    selected = replace(selected, bindings=bindings)
    saved_text, saved_manifest = first.text, json.loads(first.manifest)
    if field == "prepared":
        saved_text = saved_text.replace("double_v1", "offset_v1")
    else:
        saved_manifest["bindings"][0]["version"] = "incompatible"
    with pytest.raises(preparation.NotReady):
        preparation.prepare(accepted, selected, saved_text=saved_text, saved_manifest=saved_manifest)
    assert not lowered


@pytest.mark.parametrize("framework", ["xdsl", "pliron"])
def test_shared_wrong_binding_cannot_pass_independent_b_oracle(native, framework):
    async def wrong(value):
        return 2 * value + 7

    selected = fixture.registry()
    bindings = dict(selected.bindings)
    bindings["double_v1"] = replace(bindings["double_v1"], invoke=wrong)
    selected = replace(selected, bindings=bindings)
    if framework == "xdsl":
        ready = preparation.prepare(preparation.accept(fixture.build()), selected)
    else:
        ready, _ = native.prepare(native.accept(native_author.build()), selected)
    report = asdict(asyncio.run(fixture.execute(ready)).report)
    assert report["status"] == "stopped"
    with pytest.raises(AssertionError):
        oracle.check_b(report)


@pytest.mark.asyncio
@pytest.mark.parametrize("framework", ["xdsl", "pliron"])
@pytest.mark.parametrize("fuse", [False, True])
async def test_same_selected_failure_boundary(native, framework, fuse):
    async def fail(value):
        raise RuntimeError("matched offset failure")

    selected = fixture.registry()
    bindings = dict(selected.bindings)
    bindings["offset_v1"] = replace(bindings["offset_v1"], invoke=fail)
    ready = prepare_b(native, framework, replace(selected, bindings=bindings), fuse)
    report = asdict((await fixture.execute(ready, diagnostic_limit=0)).report)
    oracle.check_b(report)
    assert report["status"] == "failed"
    assert [(call["origin"], call["outcome"]) for call in report["tickets"][0]["calls"]] == [
        ("data.double", "returned"),
        ("data.offset", "entered_failed"),
    ]
    assert report["tickets"][0]["publication_ack"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("framework", ["xdsl", "pliron"])
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_same_physical_failure_after_interruption(native, framework, fuse, cancel):
    entered, release, finished = (threading.Event() for _ in range(3))

    def physical():
        entered.set()
        try:
            if not release.wait(30):
                raise RuntimeError("matched physical gate expired")
            raise RuntimeError("matched late physical failure")
        finally:
            finished.set()

    async def invoke(value):
        await asyncio.to_thread(physical)
        return value * 2

    selected = fixture.registry()
    bindings = dict(selected.bindings)
    bindings["double_v1"] = replace(bindings["double_v1"], invoke=invoke)
    ready = prepare_b(native, framework, replace(selected, bindings=bindings), fuse)
    session = native_tests.runtime.ScalarSession(ready, diagnostic_limit=0)
    task = asyncio.create_task(session.run())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        session.stop()
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not finished.is_set() and session.active_calls == 1
    finally:
        session.stop()
        release.set()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        report = session.report
    else:
        report = await asyncio.wait_for(task, 10)
    assert report is not None and finished.is_set() and session.active_calls == session.outstanding == 0
    oracle.check_b(asdict(report))
    assert any("matched late physical failure" in item for item in report.errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("framework", ["xdsl", "pliron"])
@pytest.mark.parametrize("fuse", [False, True])
async def test_same_feedback_progress_while_publication_waits(native, framework, fuse, monkeypatch):
    gate = asyncio.Event()

    async def receipt(value):
        await gate.wait()
        return value

    selected = fixture.registry()
    bindings = dict(selected.bindings)
    bindings["receipt_v1"] = replace(bindings["receipt_v1"], invoke=receipt)
    ready = prepare_b(native, framework, replace(selected, bindings=bindings), fuse)
    session = native_tests.runtime.ScalarSession(ready)
    apply = session._apply_feedback
    progressed = []

    def feedback(ticket, score):
        if not progressed:
            assert session.publication_waiting == {1} and session.buffered == 1
            apply(ticket, score)
            assert session.publication_waiting == {1} and session.feedback_count == 1
            progressed.append(True)
            session.stop()
        else:
            apply(ticket, score)

    monkeypatch.setattr(session, "_apply_feedback", feedback)
    task = asyncio.create_task(session.run())
    try:
        await asyncio.wait_for(session.wait_for(lambda: session.publication_waiting == {1}), 10)
        assert session.outstanding == 2
    finally:
        gate.set()
    report = await asyncio.wait_for(task, 10)
    assert progressed == [True]
    oracle.check_b(asdict(report))
