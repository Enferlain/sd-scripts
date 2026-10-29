"""C reuses grant/optimizer owners; optional tests require Torch."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import os
import subprocess
import sys
import threading
from collections import deque
from dataclasses import replace
from pathlib import Path

import pytest
from xdsl.dialects.builtin import StringAttr, SymbolRefAttr

from .dialects import CallOp, PlanOp, dump, integer
from .grant_conformance import check_c, check_unverified, reference
from .grant_contract import accept_grant, build_grant
from .grant_dialect import AdvanceGrantedOp, parse_grant
from .model_contract import one
from .model_dialect import RecipeOp
from .prepare import NotReady

pytest.importorskip("torch")
assert __package__ is not None
adapter = importlib.import_module(__package__ + ".grant_adapter")
model_adapter = importlib.import_module(__package__ + ".model_adapter")
torch, owners, physical = adapter.torch, model_adapter.owners, model_adapter.physical
DomainRegistry, grant_registry, prepare_grant = adapter.DomainRegistry, adapter.grant_registry, adapter.prepare_grant


def selected_registry(selected):
    bindings = dict(grant_registry().bindings)
    bindings["two_pass_v1"] = replace(bindings["two_pass_v1"], invoke=selected, version="test-1")
    return DomainRegistry("test-grant", bindings)


@pytest.mark.asyncio
@pytest.mark.parametrize("lowering", ["calls", "inline"])
@pytest.mark.parametrize("initial", [0.8, 0.6])
@pytest.mark.parametrize("clip", [None, 1.0])
async def test_granted_handback_and_retained_update_match_independent_math(lowering, initial, clip) -> None:
    prepared = prepare_grant(accept_grant(build_grant(initial=initial, clip_norm=clip)), lowering=lowering)
    report = await asyncio.wait_for(prepared.publish().run(), 5)
    check_c(report, initial=initial, clip_norm=clip)
    plan = one(parse_grant(prepared.text), PlanOp)
    assert plan.horizon is None and plan.capacity is None and plan.feedback is None
    assert "grant.contribution" in prepared.text and "grant-owner" in prepared.text


@pytest.mark.asyncio
@pytest.mark.parametrize("history", [0, 1, 256])
async def test_diagnostics_do_not_own_grant_readiness(history) -> None:
    report = await prepare_grant(accept_grant(build_grant()), history_limit=history).publish().run()
    check_c(report)
    assert report["diagnostics"] <= history


def test_default_binding_reuses_the_maintained_numerical_callable() -> None:
    maintained = adapter.examples.two_pass().root.work[0].implementation
    assert grant_registry().bindings["two_pass_v1"].invoke.__code__ is maintained.__code__


@pytest.mark.parametrize("case", ["offer", "target", "missing", "role", "version", "parameter"])
def test_unavailable_concrete_grant_rejects_before_model_load(case: str) -> None:
    module = build_grant()
    bindings = dict(grant_registry().bindings)
    loads = []
    bindings["scale_v1"] = replace(bindings["scale_v1"], invoke=lambda *args: loads.append(args))
    if case == "missing":
        del bindings["two_pass_v1"]
    elif case == "role":
        bindings["two_pass_v1"] = replace(bindings["two_pass_v1"], effect="read")
    elif case == "version":
        bindings["two_pass_v1"] = replace(bindings["two_pass_v1"], version="")
    elif case == "parameter":
        one(module, AdvanceGrantedOp).properties["parameter"] = StringAttr("unsupported")
    accepted = accept_grant(module)
    with pytest.raises(NotReady):
        prepare_grant(
            accepted,
            DomainRegistry("unavailable", bindings),
            offered=case != "offer",
            target="unsupported" if case == "target" else adapter.TARGET_C,
        )
    assert not loads and accept_grant(parse_grant(accepted.text)) == accepted


@pytest.mark.parametrize("case", ["completion", "symbol", "ancestry", "target", "protocol", "capacity"])
def test_prepared_recipe_is_checked_before_generated_owner_program(case: str, monkeypatch) -> None:
    bindings = dict(grant_registry().bindings)
    bindings["alternate"] = bindings["two_pass_v1"]

    def corrupted(module):
        call = next(op for op in module.walk() if isinstance(op, CallOp) and op.role.data == "grant")
        if case == "completion":
            call.properties["completion"] = StringAttr("return_is_handback")
        elif case == "symbol":
            call.properties["implementation"] = SymbolRefAttr("alternate")
        elif case == "ancestry":
            call.properties["ancestors"] = type(call.ancestors)([StringAttr("other")])
        elif case == "target":
            one(module, PlanOp).properties["target"] = StringAttr("other")
        elif case == "protocol":
            one(module, RecipeOp).properties["protocol"] = StringAttr("skip_owner_verification")
        else:
            one(module, PlanOp).properties["capacity"] = integer(1)
        return dump(module)

    monkeypatch.setattr(adapter, "dump", corrupted)
    with pytest.raises(NotReady, match=r"prepared|flow coordination"):
        prepare_grant(accept_grant(build_grant()), DomainRegistry("extra", bindings))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["restored_raises", "incomplete", "missing_gradient", "edited_after_restore", "phase_order", "final_advance"]
)
@pytest.mark.parametrize("lowering", ["calls", "inline"])
async def test_invalid_or_failed_handback_never_authorizes_retained_advance(case: str, lowering: str) -> None:
    original = grant_registry().bindings["two_pass_v1"].invoke

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

    report = (
        await prepare_grant(accept_grant(build_grant()), selected_registry(invalid), lowering=lowering, history_limit=0).publish().run()
    )
    check_unverified(report)
    selected = report["selected"]
    if case == "restored_raises":
        assert selected["restored"] and not selected["selected_returned"]
        assert report["final_weight"] == pytest.approx(0.8)
        assert report["final_gradient"] == pytest.approx(reference()["final_gradient"])
    elif case == "incomplete":
        assert selected["selected_returned"] and not selected["restored"]
        assert report["final_weight"] == pytest.approx(reference()["temporary"])
    elif case == "edited_after_restore":
        assert selected["selected_returned"] and selected["restored"]
        assert report["final_weight"] == pytest.approx(0.9)
    elif case == "missing_gradient":
        assert selected["selected_returned"] and selected["restored"] and report["final_gradient"] is None


@pytest.mark.asyncio
async def test_active_failed_grant_survives_both_completed_and_diagnostic_history_eviction() -> None:
    original = grant_registry().bindings["two_pass_v1"].invoke

    def failed(*args):
        original(*args)
        raise RuntimeError("restored but not handed back")

    run = prepare_grant(accept_grant(build_grant()), selected_registry(failed), history_limit=0).publish()
    run.ready.state.image.units["weights"].outcomes = deque(maxlen=0)
    report = await run.run()
    check_unverified(report)
    assert report["diagnostics"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_restored_suspended_grant_is_not_ready_and_holds_current_use(cancel: bool) -> None:
    entered, release, inspected = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = grant_registry().bindings["two_pass_v1"].invoke

    async def suspended(*args):
        original(*args)
        entered.set()
        await release.wait()

    run = prepare_grant(accept_grant(build_grant()), selected_registry(suspended)).publish()
    running = asyncio.create_task(run.run())
    waiter = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        state = run.ready.state
        unit = state.image.units["weights"]
        parameter = state.image.bindings["model"].model.weight
        assert float(parameter.detach()) == pytest.approx(0.8) and parameter.grad is not None
        assert unit.unfinished and unit.window is None and unit.steps == 0
        pending = unit._work
        assert pending is not None
        invocation = next(iter(pending.records))
        forged = owners.Contribution(
            invocation, pending.units, pending.prepared_keys, pending.region, tuple(pending.records), pending.destination_stamps
        )
        with pytest.raises(adapter.graph.NotReady, match="unverified"):
            state.advance(forged, ("weights",), invocation, region=pending.region)
        assert unit.steps == 0

        async def reader():
            async with state.acquire((adapter.graph.Use("model"),)):
                inspected.set()

        waiter = asyncio.create_task(reader())
        await asyncio.sleep(0)
        assert not inspected.is_set() and state.writers
        if cancel:
            running.cancel()
    finally:
        release.set()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 5)
        check_unverified(run.report(), cancelled=True)
        with pytest.raises(adapter.graph.NotReady):
            await asyncio.wait_for(waiter, 5)
        assert not inspected.is_set()
    else:
        check_c(await asyncio.wait_for(running, 5))
        await asyncio.wait_for(waiter, 5)
        assert inspected.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel,fail", [(False, False), (False, True), (True, False), (True, True)])
async def test_entered_physical_grant_drains_under_repeated_interruption(cancel: bool, fail: bool) -> None:
    entered, release, finished, stop = (threading.Event() for _ in range(4))
    original = grant_registry().bindings["two_pass_v1"].invoke
    pool = physical.Workers(1)

    async def selected(access, models):
        def work():
            try:
                original(access, models)
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("physical grant not released")
                if fail:
                    raise RuntimeError("selected physical grant failed after restoration")
            finally:
                finished.set()

        with pool.lifetime():
            completion = await pool.execute(work, stop)
            if completion.cancellation is not None:
                raise completion.cancellation

    run = prepare_grant(accept_grant(build_grant()), selected_registry(selected)).publish()
    running = asyncio.create_task(run.run())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert not finished.is_set() and not running.done()
        unit = run.ready.state.image.units["weights"]
        assert unit.unfinished and unit.window is None and unit.steps == 0
        assert run.ready.state.writers and pool.active == 1
        if cancel:
            running.cancel()
            await asyncio.sleep(0)
            running.cancel()
            await asyncio.sleep(0)
            assert not running.done() and stop.is_set() and not finished.is_set()
            assert run.ready.state.writers and pool.active == 1
    finally:
        release.set()
    if cancel:
        with pytest.raises((asyncio.CancelledError, adapter.backend.WorkFailed)):
            await asyncio.wait_for(running, 5)
        check_unverified(run.report(), cancelled=True)
    elif fail:
        check_unverified(await asyncio.wait_for(running, 5))
    else:
        check_c(await asyncio.wait_for(running, 5))
    assert finished.is_set() and pool.active == 0 and pool.executor is None
    if fail:
        assert any("physical grant failed" in item["message"] for item in run.report()["errors"])


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["optimizer", "reset"])
@pytest.mark.parametrize("lowering", ["calls", "inline"])
async def test_retained_owner_failure_preserves_partial_outcome_and_cannot_retry(phase, lowering, monkeypatch) -> None:
    run = prepare_grant(accept_grant(build_grant()), lowering=lowering, history_limit=0).publish()
    runtime = run.ready.state.image.units["weights"]
    runtime.outcomes = deque(maxlen=0)
    original = type(runtime.optimizer).step if phase == "optimizer" else type(runtime.optimizer).zero_grad

    def failed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError(f"retained {phase} failed after effects")

    monkeypatch.setattr(type(runtime.optimizer), "step" if phase == "optimizer" else "zero_grad", failed)
    report = await run.run()
    assert report["status"] == "failed" and report["released"] and report["pending"] and report["handback_verified"]
    assert report["final_weight"] == pytest.approx(reference()["final_weight"])
    outcome = report["unit_outcomes"][0]
    assert outcome["contribution"] == "granted_ready" and outcome["advance_invocation"] is not None
    assert (outcome["optimizer"], outcome["reset"]) == (
        ("uncertain", "not_attempted") if phase == "optimizer" else ("returned", "uncertain")
    )
    assert report["steps"] == (0 if phase == "optimizer" else 1)
    assert report["epoch"] == (1 if phase == "optimizer" else 2)
    with pytest.raises(adapter.graph.NotReady, match="unverified"):
        run.ready.state.advance(runtime.window, ("weights",), outcome["invocation"])


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["gradient", "weight", "phase", "origin", "handback", "step", "optimizer", "reset", "pending", "epoch"])
async def test_independent_oracle_rejects_corrupted_reports(case: str) -> None:
    bad = copy.deepcopy(await prepare_grant(accept_grant(build_grant())).publish().run())
    if case == "gradient":
        bad["selected"]["selected_gradient"] += 1
    elif case == "weight":
        bad["final_weight"] = 0.8
    elif case in ("phase", "origin"):
        bad["selected"]["phases"][2][case] = "wrong"
    elif case == "handback":
        bad["handback_verified"] = False
    elif case == "step":
        bad["steps"] = 2
    elif case in ("optimizer", "reset"):
        bad["unit_outcomes"][0][case] = "uncertain"
    elif case == "pending":
        bad["pending"] = True
    else:
        bad["epoch"] = 0
    with pytest.raises(AssertionError):
        check_c(bad)


@pytest.mark.asyncio
async def test_common_wrong_optimizer_escapes_lowering_equivalence_but_not_oracle(monkeypatch) -> None:
    optimizer = importlib.import_module("docs_design.models-strategy-trainer.execution_candidate.optimizer")
    monkeypatch.setattr(optimizer.EagerSGD, "step", lambda self: None)
    reports = [await prepare_grant(accept_grant(build_grant()), lowering=lowering).publish().run() for lowering in ("calls", "inline")]
    assert reports[0]["final_weight"] == reports[1]["final_weight"] and reports[0]["unit_outcomes"] == reports[1]["unit_outcomes"]
    for report in reports:
        with pytest.raises(AssertionError, match="independent retained SGD"):
            check_c(report)


@pytest.mark.asyncio
async def test_run_has_one_owner_lifetime() -> None:
    run = prepare_grant(accept_grant(build_grant())).publish()
    check_c(await run.run())
    with pytest.raises(RuntimeError, match="one owner lifetime"):
        await run.run()


def test_saved_clipped_document_reconstructs_with_author_disabled(tmp_path: Path) -> None:
    prepared = prepare_grant(accept_grant(build_grant(clip_norm=1.0)))
    semantic, manifest = tmp_path / "accepted.mlir", tmp_path / "bindings.json"
    semantic.write_text(prepared.accepted.text)
    manifest.write_text(prepared.manifest)
    script = """
import importlib
contract=importlib.import_module('docs_design.models-strategy-trainer.xdsl_candidate.grant_contract')
contract.build_grant=lambda **kwargs: (_ for _ in ()).throw(AssertionError('author used'))
demo=importlib.import_module('docs_design.models-strategy-trainer.xdsl_candidate.grant_demo')
demo.main()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, "--load", str(semantic), "--manifest", str(manifest), "--dump-dir", str(tmp_path / "loaded")],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    check_c(result["report"], clip_norm=1.0)
    assert result["accepted_digest"] == prepared.accepted.digest
    assert (tmp_path / "loaded" / "prepared.mlir").read_text() == prepared.text
