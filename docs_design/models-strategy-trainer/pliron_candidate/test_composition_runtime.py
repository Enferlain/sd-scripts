"""Owner integration, independent arithmetic and physical lifetime pressure."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from .composition import Registry, registry
from .composition_author import build
from .composition_conformance import check, close
from .composition_demo import pressure, until
from . import test_composition as fixtures
from .bridge import Rejected

pytest.importorskip("torch", reason="composition owner case requires optional Torch")
runtime = importlib.import_module(__name__.rsplit(".", 1)[0] + ".composition_runtime")
composition_artifact = fixtures.composition_artifact
composition_native = fixtures.composition_native


@pytest.mark.parametrize("mode", ["late", "prefetched"])
def test_lifetimes_bounds_and_both_input_availabilities(composition_native, composition_artifact, mode):
    before = sys._is_gil_enabled()
    report = asyncio.run(pressure(composition_native, composition_artifact, mode=mode))
    check(report)
    assert sys._is_gil_enabled() is before
    assert report["source"]["high_water"] == 3
    assert any(event["kind"] == "aggregate_bound_witness" for event in report["events"])
    assert any(event["kind"] == "shutdown_witness" for event in report["events"])
    if mode == "late":
        pending = next(event for event in report["events"] if event["kind"] == "pending_request_witness")
        published = next(event for event in report["events"] if event["kind"] == "published" and event["key"] == 4)
        assert pending["producer_live"] and report["scope_clocks"]["training"] < pending["clock"] < published["clock"]
    else:
        published = next(event for event in report["events"] if event["kind"] == "published" and event["key"] == 4)
        requested = next(event for event in report["events"] if event["kind"] == "evaluation_requested")
        assert published["clock"] < requested["clock"]


@pytest.mark.parametrize("stop", ["before", "after"])
def test_orderly_stop_respects_window_admission(composition_native, composition_artifact, stop):
    controls = runtime.Controls(**{f"stop_{stop}": 1})
    run = composition_native.materialize(composition_artifact, controls=controls)
    report = asyncio.run(asyncio.wait_for(run.run(), 10))
    check(report, windows=1 if stop == "before" else 2)


def test_orderly_shutdown_can_cancel_a_call_before_physical_entry(composition_native, composition_artifact, monkeypatch):
    async def exercise():
        queued = asyncio.Event()
        gate = threading.Event()
        controls = runtime.Controls(evaluation_gate=gate)
        execute = runtime.physical.Workers.execute

        async def before_entry(pool, call, stop):
            if controls.stop_signals.get(6) is stop:
                queued.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as error:
                    raise runtime.physical.NotEntered(error) from error
            return await execute(pool, call, stop)

        monkeypatch.setattr(runtime.physical.Workers, "execute", before_entry)
        run = composition_native.materialize(composition_artifact, controls=controls)
        task = asyncio.create_task(run.run())
        try:
            await asyncio.wait_for(queued.wait(), 10)
            gate.set()
            report = await asyncio.wait_for(task, 10)
            check(report)
            use = next(item for item in report["source"]["uses"] if item["key"] == 6)
            assert use["entered"] is False and use["physical_done"]
            assert report["source"]["terminal"][6] == "discarded"
        finally:
            gate.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_unentered_evaluation_input_is_not_reported_consumed(composition_native, composition_artifact, monkeypatch):
    async def exercise():
        queued = asyncio.Event()
        controls = runtime.Controls()
        execute = runtime.physical.Workers.execute

        async def before_entry(pool, call, stop):
            if not any(stop is owned for owned in controls.stop_signals.values()):
                queued.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as error:
                    raise runtime.physical.NotEntered(error) from error
            return await execute(pool, call, stop)

        monkeypatch.setattr(runtime.physical.Workers, "execute", before_entry)
        run = composition_native.materialize(composition_artifact, controls=controls)
        task = asyncio.create_task(run.run())
        try:
            await asyncio.wait_for(queued.wait(), 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
            report = run.last_report
            assert report["status"] == "cancelled" and report["released"]
            assert report["evaluation"]["physical_entered"] is False
            assert report["evaluation"]["physical_done"] and not report["evaluation_completed"]
            assert report["source"]["terminal"][4] == "closed_unconsumed"
            assert not any(item["type"] == "PhysicalFailed" for item in report["errors"])
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_partial_window_claim_survives_root_cancellation(composition_native, composition_artifact):
    async def exercise():
        gate = asyncio.Event()
        controls = runtime.Controls(second_member_gate=gate)
        run = composition_native.materialize(composition_artifact, controls=controls)
        task = asyncio.create_task(run.run())
        try:
            await asyncio.wait_for(controls.first_member.wait(), 10)
            # Finish the first contribution but prevent acquisition of member3.
            await until(lambda: len(run.learning.contributions) == 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            report = run.last_report
            assert report["status"] == "cancelled" and report["released"]
            assert report["selected_windows"] == [0, 1]
            assert report["acquired"] == {0: [0, 1], 1: [2]}
            assert report["pending"] and report["steps"] == 1
            close(report["weight"], 0.77)
            assert report["evaluation"] is None
            assert not report["source"]["active"] and report["source"]["cleanup_complete"]
        finally:
            gate.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_orderly_child_stop_finishes_a_partly_acquired_window(composition_native, composition_artifact):
    async def exercise():
        gate = asyncio.Event()
        controls = runtime.Controls(second_member_gate=gate)
        run = composition_native.materialize(composition_artifact, controls=controls)
        task = asyncio.create_task(run.run())
        try:
            await asyncio.wait_for(controls.first_member.wait(), 10)
            controls.close_training = True
            assert run.learning.acquired[1] == [2]
            gate.set()
            check(await asyncio.wait_for(task, 10))
        finally:
            gate.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_root_cancel_can_leave_a_requested_evaluation_unfulfilled(composition_native, composition_artifact):
    async def exercise():
        controls = runtime.Controls(late_record=True)
        run = composition_native.materialize(composition_artifact, controls=controls)
        task = asyncio.create_task(run.run())
        try:
            await asyncio.wait_for(run.signals.event("training").wait(), 10)
            assert controls.requested.is_set() and not run.signals.event("consumers").is_set()
            assert run.evaluation.receipt is None
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
            report = run.last_report
            assert report["status"] == "cancelled" and report["released"] and report["steps"] == 2
            assert report["request"] == {
                "cause": {"source": "training", "window": 1, "position": 2},
                "disposition": "cancelled_unfulfilled",
            }
            assert report["evaluation"] is None and not report["evaluation_completed"]
            assert {(h["channel"], h["key"]) for h in report["failure_inputs"]} == {("producer.slot.0", 1), ("producer.slot.1", 1)}
        finally:
            controls.allow_record.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["training", "evaluation"])
def test_group_failure_shuts_down_producer(composition_native, composition_artifact, phase):
    controls = runtime.Controls(fail_contribution=3 if phase == "training" else None, fail_evaluation=phase == "evaluation")
    run = composition_native.materialize(composition_artifact, controls=controls)
    report = asyncio.run(asyncio.wait_for(run.run(), 10))
    assert report["status"] == "failed" and report["released"] and report["closed"]
    assert report["source"]["cleanup_complete"] and not report["source"]["active"]
    assert report["scope_clocks"]["producer"] > report["scope_clocks"]["consumers"]
    assert not report["evaluation_completed"]
    if phase == "training":
        assert report["steps"] == 1 and report["pending"]
        assert report["evaluation"] is None
    else:
        assert report["steps"] == 2 and not report["pending"]
        assert report["evaluation"]["physical_done"] and not report["evaluation"]["returned"]
        assert report["training_mode"] and report["gradients_enabled"]


@pytest.mark.parametrize("cancel", [False, True])
def test_completed_evaluation_and_later_physical_failure_both_survive(composition_native, composition_artifact, cancel):
    report = asyncio.run(pressure(composition_native, composition_artifact, fail=True, cancel=cancel))
    if cancel:
        assert report["status"] == "cancelled" and report["released"] and not report["pending"]
        assert report["evaluation_completed"] and report["evaluation"]["physical_done"]
        assert any("physical production failure 6" in item["message"] for item in report["errors"])
        assert any(item["type"] == "CancelledError" for item in report["errors"])
        assert report["source"]["terminal"][6] == "failed"
        close(report["weight"], 0.635)
    else:
        check(report, failed_after_evaluation=True)


def test_repeated_cancellation_drains_entered_evaluation(composition_native, composition_artifact):
    async def exercise():
        gate = threading.Event()
        controls = runtime.Controls(evaluation_gate=gate)
        run = composition_native.materialize(composition_artifact, controls=controls)
        task = asyncio.create_task(run.run())
        try:
            await until(controls.evaluation_entered.is_set)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            assert not task.done() and run.ready.state.readers
            assert not run.signals.event("consumers").is_set()
            assert run.evaluation.receipt["physical_done"] is False
            gate.set()
            with pytest.raises((asyncio.CancelledError, BaseExceptionGroup)):
                await asyncio.wait_for(task, 10)
            report = run.last_report
            assert report["status"] == "cancelled" and report["released"]
            assert report["evaluation"]["physical_done"] and not report["evaluation_completed"]
            assert report["training_mode"] and report["gradients_enabled"]
            assert report["scope_clocks"]["producer"] > report["scope_clocks"]["consumers"]
        finally:
            gate.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_cold_reconstruction_ignores_inspection_projection(composition_native, composition_artifact):
    saved = json.loads(json.dumps(composition_artifact))
    saved["profile"] = {"windows": 999, "producer": "unverified"}
    saved["nodes"] = []
    run = composition_native.materialize(saved)
    check(asyncio.run(run.run()))


@pytest.mark.parametrize("boundary", ["accepted", "prepared", "manifest"])
def test_invalid_claim_does_not_invoke_model_factory(composition_native, composition_artifact, boundary):
    calls = []
    selected = registry()
    bindings = dict(selected.bindings)
    original = bindings["scale_v1"]
    bindings["scale_v1"] = replace(original, invoke=lambda *args: calls.append(args) or original.invoke(*args))
    saved = copy.deepcopy(composition_artifact)
    if boundary == "accepted":
        saved["accepted"] += " unchecked"
    elif boundary == "prepared":
        saved["prepared"] = saved["prepared"].replace("keeps_group_alive", "ignore_pending")
    else:
        saved["manifest"]["bindings"]["scale_v1"]["version"] = "2"
    with pytest.raises(Rejected):
        composition_native.materialize(saved, Registry(bindings))
    assert calls == []


def test_independent_oracle_catches_shared_wrong_objective(composition_native, composition_artifact):
    selected = registry()
    bindings = dict(selected.bindings)
    original = bindings["objective_v1"]
    bindings["objective_v1"] = replace(original, invoke=lambda *args: 2 * original.invoke(*args))
    run = composition_native.materialize(composition_artifact, Registry(bindings))
    report = asyncio.run(run.run())
    assert report["status"] == "stopped"
    with pytest.raises(AssertionError):
        check(report)


@pytest.mark.parametrize("field", ["accepted", "prepared", "manifest"])
@pytest.mark.parametrize("missing", [False, True])
def test_malformed_artifact_diagnostic_precedes_factories(composition_native, composition_artifact, field, missing):
    calls = []
    bindings = dict(registry().bindings)
    original = bindings["scale_v1"]
    bindings["scale_v1"] = replace(original, invoke=lambda *args: calls.append(args) or original.invoke(*args))
    artifact = copy.deepcopy(composition_artifact)
    if missing:
        artifact.pop(field)
    else:
        artifact[field] = [] if field == "manifest" else 7
    with pytest.raises(Rejected, match=field):
        composition_native.materialize(artifact, Registry(bindings))
    assert calls == []


@pytest.mark.parametrize("field", ["epoch", "request_members", "input_identity", "bound", "group_end", "physical_end", "consumption"])
def test_independent_observation_corruptions(composition_native, composition_artifact, field):
    report = asyncio.run(pressure(composition_native, composition_artifact))
    if field == "epoch":
        report["evaluation"]["epoch"] = 1
    elif field == "request_members":
        report["evaluation"]["cause"]["window"] = 0
    elif field == "input_identity":
        report["evaluation"]["key"] = 3
    elif field == "bound":
        report["source"]["high_water"] = 4
    elif field == "group_end":
        report["scope_clocks"]["consumers"] = report["scope_clocks"]["training"]
    elif field == "physical_end":
        report["evaluation"]["physical_done"] = False
    else:
        report["source"]["terminal"][5] = "consumed"
    with pytest.raises(AssertionError):
        check(report)


def test_histories_are_not_execution_authority(composition_native, composition_artifact):
    report = asyncio.run(pressure(composition_native, composition_artifact, erase=True))
    check(report, histories=False)
    assert report["events"] == report["windows"] == report["contributions"] == []
    assert report["source"]["products"] == report["source"]["uses"] == []
    assert report["diagnostics"] == 0
    assert report["archive_erasure_points"] == ["partial_window", "pending_request", "physical_use"]


def test_renamed_case_executes_without_name_dispatch(composition_native):
    artifact = composition_native.prepare(composition_native.accept(build(prefix="other.")))
    check(asyncio.run(pressure(composition_native, artifact)), prefix="other.")


def test_verified_projection_is_snapshotted(composition_native, composition_artifact):
    run = composition_native.materialize(composition_artifact)
    with pytest.raises(TypeError):
        run.spec["windows"] = 99
    with pytest.raises(TypeError):
        run.spec["models"][0]["initial"] = 99
    check(asyncio.run(run.run()))


@pytest.mark.parametrize("optimized", [False, True])
def test_fresh_cold_process_without_authoring_or_xdsl_ir(composition_native, composition_artifact, tmp_path, optimized):
    saved = tmp_path / "artifact.json"
    saved.write_text(json.dumps(composition_artifact))
    script = r"""
import asyncio, importlib, json, sys
sys.path.insert(0,sys.argv[1])
before=sys._is_gil_enabled()
p='docs_design.models-strategy-trainer.pliron_candidate'
def forbidden(*args,**kwargs): raise RuntimeError('authoring or xDSL machinery invoked during cold execution')
author=importlib.import_module(p+'.composition_author'); author.build=forbidden
api=importlib.import_module(p+'.composition'); api.CompositionNative.accept=forbidden; api.CompositionNative.prepare=forbidden
from xdsl.parser import Parser
from xdsl.ir import Operation
Parser.parse_module=forbidden; Operation.create=classmethod(forbidden)
preparation=importlib.import_module('docs_design.models-strategy-trainer.xdsl_candidate.prepare'); preparation.prepare=forbidden
driver=importlib.import_module(p+'.composition_demo')
from pathlib import Path
native=api.CompositionNative(Path(sys.argv[2]))
report=asyncio.run(driver.pressure(native,json.loads(Path(sys.argv[3]).read_text())))
print(json.dumps({'report':report,'gil_before':before,'gil_after':sys._is_gil_enabled(),'optimized':not __debug__}))
"""
    command = [
        sys.executable,
        *(["-O"] if optimized else []),
        "-c",
        script,
        str(Path(__file__).resolve().parents[3]),
        str(composition_native.binary),
        str(saved),
    ]
    completed = subprocess.run(command, cwd="/tmp", text=True, capture_output=True, timeout=60, check=False)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["optimized"] is optimized
    assert result["gil_before"] is False and result["gil_after"] is False
    # This oracle executes in the normal parent even when the child used -O.
    check(result["report"])
