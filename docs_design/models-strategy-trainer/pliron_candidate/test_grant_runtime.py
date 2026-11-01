"""Pliron C through existing owners and independent numerical expectations."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import subprocess
import sys
import threading
from collections import deque
from dataclasses import replace
from pathlib import Path

import pytest

from .grant import Rejected, registry
from .grant_author import build
from . import test_grant as fixtures

pytest.importorskip("torch", reason="C owner target requires optional Torch")
_prefix = "docs_design.models-strategy-trainer"
target = importlib.import_module(_prefix + ".grant_target")
oracle = importlib.import_module(_prefix + ".xdsl_candidate.grant_conformance")
physical = importlib.import_module(_prefix + ".execution_candidate.workers")
owners = importlib.import_module(_prefix + ".execution_candidate.state")
torch = target.torch
grant_native = fixtures.grant_native
grant_artifact = fixtures.grant_artifact


def ready(native, *, initial=0.8, clip=None, lowering="calls", history=256, selected=None):
    selected = registry(lowering=lowering) if selected is None else replace(selected, lowering=lowering)
    artifact = native.prepare(native.accept(build(initial=initial, clip_norm=clip)), selected.describe())
    return native.materialize(artifact, selected, history_limit=history)


def selected_registry(implementation):
    selected = registry()
    bindings = dict(selected.bindings)
    bindings["two_pass_v1"] = replace(bindings["two_pass_v1"], invoke=implementation, version="test-1")
    return replace(selected, bindings=bindings)


@pytest.mark.parametrize("lowering", ["calls", "inline"])
@pytest.mark.parametrize("initial", [0.8, 0.6])
@pytest.mark.parametrize("clip", [None, 1.0])
def test_same_c_math_and_actual_handback(grant_native, lowering, initial, clip):
    before = sys._is_gil_enabled()
    report = asyncio.run(ready(grant_native, initial=initial, clip=clip, lowering=lowering).publish().run())
    oracle.check_c(report, initial=initial, clip_norm=clip)
    assert sys._is_gil_enabled() is before


@pytest.mark.parametrize("history", [0, 1, 256])
def test_diagnostic_archive_does_not_decide_handback(grant_native, history):
    report = asyncio.run(ready(grant_native, history=history).publish().run())
    oracle.check_c(report)
    assert report["diagnostics"] <= history


@pytest.mark.parametrize(
    "case", ["restored_raises", "incomplete", "missing_gradient", "edited_after_restore", "phase_order", "final_advance"]
)
@pytest.mark.parametrize("lowering", ["calls", "inline"])
def test_failed_handback_never_authorizes_advancement(grant_native, case, lowering):
    original = registry().bindings["two_pass_v1"].invoke

    def invalid(access, models):
        if case == "phase_order":
            access.restore()
        elif case == "final_advance":
            access.advance()
        elif case == "incomplete":
            access.backward((models["model"](torch.tensor(2.0)) - 1).square())
            access.perturb(0.1)
        else:
            original(access, models)
            if case == "restored_raises":
                raise RuntimeError("failed after restoring")
            if case == "missing_gradient":
                models["model"].weight.grad = None
            else:
                with torch.no_grad():
                    models["model"].weight.add_(0.1)

    report = asyncio.run(ready(grant_native, selected=selected_registry(invalid), lowering=lowering, history=0).publish().run())
    oracle.check_unverified(report)
    if case == "restored_raises":
        assert report["selected"]["restored"] and not report["selected"]["selected_returned"]


@pytest.mark.parametrize("cancel", [False, True])
def test_restored_suspended_grant_has_no_verified_token(grant_native, cancel):
    async def exercise():
        entered, release, inspected = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original = registry().bindings["two_pass_v1"].invoke

        async def suspended(*args):
            original(*args)
            entered.set()
            await release.wait()

        run = ready(grant_native, selected=selected_registry(suspended)).publish()
        task = asyncio.create_task(run.run())
        reader = None
        try:
            await asyncio.wait_for(entered.wait(), 10)
            state = run.ready.state
            unit = state.image.units["weights"]
            assert unit.unfinished and unit.window is None and unit.steps == 0 and state.writers
            pending = unit._work
            assert pending is not None
            invocation = next(iter(pending.records))
            forged = owners.Contribution(
                invocation, pending.units, pending.prepared_keys, pending.region, tuple(pending.records), pending.destination_stamps
            )
            with pytest.raises(target.graph.NotReady, match="unverified"):
                state.advance(forged, ("weights",), invocation, region=pending.region)

            async def inspect():
                async with state.acquire((target.graph.Use("model"),)):
                    inspected.set()

            reader = asyncio.create_task(inspect())
            await asyncio.sleep(0)
            assert not inspected.is_set()
            if cancel:
                task.cancel()
        finally:
            release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
            oracle.check_unverified(run.report(), cancelled=True)
            with pytest.raises(target.graph.NotReady):
                await asyncio.wait_for(reader, 10)
        else:
            oracle.check_c(await asyncio.wait_for(task, 10))
            await asyncio.wait_for(reader, 10)
            assert inspected.is_set()

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel,fail", [(False, False), (False, True), (True, False), (True, True)])
def test_physical_completion_survives_interruption_and_failure(grant_native, cancel, fail):
    async def exercise():
        entered, release, finished, stop = (threading.Event() for _ in range(4))
        original = registry().bindings["two_pass_v1"].invoke
        pool = physical.Workers(1)

        async def selected(access, models):
            def work():
                try:
                    original(access, models)
                    entered.set()
                    if not release.wait(30):
                        raise RuntimeError("physical grant test gate expired")
                    if fail:
                        raise RuntimeError("physical grant failed after restoration")
                finally:
                    finished.set()

            with pool.lifetime():
                completion = await pool.execute(work, stop)
                if completion.cancellation is not None:
                    raise completion.cancellation

        run = ready(grant_native, selected=selected_registry(selected)).publish()
        task = asyncio.create_task(run.run())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            assert not task.done() and not finished.is_set() and run.ready.state.writers
            assert run.ready.state.image.units["weights"].window is None
            if cancel:
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done() and not finished.is_set() and stop.is_set() and pool.active == 1
        finally:
            release.set()
        if cancel:
            with pytest.raises((asyncio.CancelledError, target.backend.WorkFailed)):
                await asyncio.wait_for(task, 10)
            oracle.check_unverified(run.report(), cancelled=True)
        elif fail:
            oracle.check_unverified(await asyncio.wait_for(task, 10))
        else:
            oracle.check_c(await asyncio.wait_for(task, 10))
        assert finished.is_set() and pool.active == 0 and pool.executor is None
        if fail:
            assert any("physical grant failed" in error["message"] for error in run.report()["errors"])

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["optimizer", "reset"])
@pytest.mark.parametrize("lowering", ["calls", "inline"])
def test_retained_partial_outcome_cannot_be_retried(grant_native, phase, lowering, monkeypatch):
    run = ready(grant_native, lowering=lowering, history=0).publish()
    unit = run.ready.state.image.units["weights"]
    unit.outcomes = deque(maxlen=0)
    method = "step" if phase == "optimizer" else "zero_grad"
    original = getattr(type(unit.optimizer), method)

    def failed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("retained failure after effects")

    monkeypatch.setattr(type(unit.optimizer), method, failed)
    report = asyncio.run(run.run())
    assert report["status"] == "failed" and report["pending"] and report["released"] and report["handback_verified"]
    assert report["final_weight"] == pytest.approx(oracle.reference()["final_weight"])
    item = report["unit_outcomes"][0]
    assert (item["optimizer"], item["reset"]) == (("uncertain", "not_attempted") if phase == "optimizer" else ("returned", "uncertain"))
    assert report["steps"] == (0 if phase == "optimizer" else 1) and report["epoch"] == (1 if phase == "optimizer" else 2)
    with pytest.raises(target.graph.NotReady, match="unverified"):
        run.ready.state.advance(unit.window, ("weights",), item["invocation"])


def test_unverified_work_survives_both_history_evictions(grant_native):
    original = registry().bindings["two_pass_v1"].invoke

    def failed(*args):
        original(*args)
        raise RuntimeError("restored but not handed back")

    run = ready(grant_native, history=0, selected=selected_registry(failed)).publish()
    run.ready.state.image.units["weights"].outcomes = deque(maxlen=0)
    report = asyncio.run(run.run())
    oracle.check_unverified(report)
    assert report["diagnostics"] == 0


@pytest.mark.parametrize("field", ["accepted", "prepared", "manifest", "unknown", "unoffered"])
def test_invalid_claims_precede_factories(grant_native, grant_artifact, field):
    calls = []
    selected = registry()
    bindings = dict(selected.bindings)
    original = bindings["scale_v1"]
    bindings["scale_v1"] = replace(original, invoke=lambda *args: calls.append(args) or original.invoke(*args))
    selected = replace(selected, bindings=bindings, offered=() if field == "unoffered" else selected.offered)
    saved = copy.deepcopy(grant_artifact)
    if field == "accepted":
        saved["accepted"] += " garbage"
    elif field == "prepared":
        saved["prepared"] = saved["prepared"].replace("owner_verified_handback_not_restoration", "restoration")
    elif field == "manifest":
        saved["manifest"]["bindings"]["scale_v1"]["version"] = "changed"
    elif field == "unknown":
        saved["owner_verified_token"] = True
    with pytest.raises(Rejected):
        grant_native.materialize(saved, selected)
    assert not calls


def test_shared_wrong_optimizer_needs_independent_oracle(grant_native, monkeypatch):
    optimizer = importlib.import_module(_prefix + ".execution_candidate.optimizer")
    monkeypatch.setattr(optimizer.EagerSGD, "step", lambda self: None)
    reports = [asyncio.run(ready(grant_native, lowering=lowering).publish().run()) for lowering in ("calls", "inline")]
    assert reports[0]["final_weight"] == reports[1]["final_weight"]
    for report in reports:
        with pytest.raises(AssertionError, match="independent retained SGD"):
            oracle.check_c(report)


@pytest.mark.parametrize("case", ["gradient", "weight", "phase", "origin", "handback", "optimizer", "reset", "epoch"])
def test_independent_report_corruptions(grant_native, case):
    report = asyncio.run(ready(grant_native).publish().run())
    if case == "gradient":
        report["selected"]["selected_gradient"] += 1
    elif case == "weight":
        report["final_weight"] = 0.8
    elif case in ("phase", "origin"):
        report["selected"]["phases"][2][case] = "wrong"
    elif case == "handback":
        report["handback_verified"] = False
    elif case in ("optimizer", "reset"):
        report["unit_outcomes"][0][case] = "uncertain"
    else:
        report["epoch"] = 0
    with pytest.raises(AssertionError):
        oracle.check_c(report)


def test_inspection_is_not_authority_and_owner_lifetime_is_single(grant_native, grant_artifact):
    saved = copy.deepcopy(grant_artifact)
    saved["profile"] = {"initial": 999, "verified": True}
    run = grant_native.materialize(saved).publish()
    oracle.check_c(asyncio.run(run.run()))
    with pytest.raises(RuntimeError, match="one owner lifetime"):
        asyncio.run(run.run())


@pytest.mark.parametrize("prefix", ["", "other."])
def test_participant_unit_and_source_names_are_not_case_dispatch(grant_native, prefix):
    artifact = grant_native.prepare(grant_native.accept(build(prefix=prefix, participant="renamed.model", unit="renamed.unit")))
    report = asyncio.run(grant_native.materialize(artifact).publish().run())
    if not prefix:
        oracle.check_c(report)
    else:
        assert report["status"] == "stopped" and report["released"] and report["handback_verified"]
        assert report["steps"] == 1 and report["epoch"] == 2 and not report["pending"]
        expected = oracle.reference()
        oracle.close(report["final_weight"], expected["final_weight"], "renamed retained SGD")
        assert [item["origin"] for item in report["selected"]["phases"]] == [
            prefix + f"phase.{i}.{phase}"
            for i, phase in enumerate(("backward", "temporary_edit", "intermediate_reset", "backward", "restore"))
        ]
        assert report["unit_outcomes"][0]["work"] == (prefix + "grant-session", prefix + "selected-two-pass")


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("lowering", ["calls", "inline"])
def test_fresh_native_cold_load_blocks_authoring_and_xdsl(grant_native, tmp_path, optimized, lowering):
    selected = registry(lowering=lowering)
    artifact = grant_native.prepare(grant_native.accept(build(clip_norm=1.0)), selected.describe())
    saved = tmp_path / "artifact.json"
    saved.write_text(json.dumps(artifact))
    script = r"""
import asyncio,importlib,importlib.abc,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);before=sys._is_gil_enabled()
class NoXDsl(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='xdsl' or fullname.startswith('xdsl.') or fullname.endswith(('.grant_contract','.grant_adapter')):
   raise RuntimeError('xDSL IR machinery imported by Pliron C')
sys.meta_path.insert(0,NoXDsl())
p='docs_design.models-strategy-trainer.pliron_candidate'
def forbidden(*a,**k):raise RuntimeError('authoring/native preparation wrapper invoked in cold path')
author=importlib.import_module(p+'.grant_author');author.build=forbidden
api=importlib.import_module(p+'.grant');api.GrantNative.accept=forbidden;api.GrantNative.prepare=forbidden
native=api.GrantNative(Path(sys.argv[2]));data=json.loads(Path(sys.argv[3]).read_text())
report=asyncio.run(native.materialize(data,api.registry(lowering=sys.argv[4])).publish().run())
print(json.dumps({'report':report,'gil_before':before,'gil_after':sys._is_gil_enabled(),'xdsl_imported':'xdsl' in sys.modules,'optimized':not __debug__}))
"""
    result = subprocess.run(
        [
            sys.executable,
            *(["-O"] if optimized else []),
            "-c",
            script,
            str(Path(__file__).resolve().parents[3]),
            str(grant_native.binary),
            str(saved),
            lowering,
        ],
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["optimized"] is optimized and not observed["xdsl_imported"]
    assert observed["gil_before"] is False and observed["gil_after"] is False
    oracle.check_c(observed["report"], clip_norm=1.0)
