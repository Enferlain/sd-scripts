"""A reuses real owner services; only its optional tests need Torch."""

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

from xdsl.dialects.builtin import StringAttr

from .dialects import CallOp, integer
from .model_conformance import check_a, check_admission_failure
from .model_contract import accept_model, build_model, one
from .model_dialect import AddressOp, AdvanceOp
from .prepare import NotReady

# This is an optional-backend test module. Keep every ordinary import above;
# then load the integration only when its numerical dependency exists.
pytest.importorskip("torch")
assert __package__ is not None
adapter = importlib.import_module(__package__ + ".model_adapter")
Controls, DomainRegistry = adapter.Controls, adapter.DomainRegistry
model_registry, prepare_model = adapter.model_registry, adapter.prepare_model


@pytest.mark.asyncio
@pytest.mark.parametrize("lowering", ["calls", "inline"])
@pytest.mark.parametrize("workers", [False, True])
@pytest.mark.parametrize("history", [0, 1, 256])
async def test_owner_backend_matches_independent_arithmetic_and_associations(lowering, workers, history) -> None:
    prepared = prepare_model(accept_model(build_model()), lowering=lowering, workers=workers, controls=Controls(history_limit=history))
    run = prepared.publish()
    report = await asyncio.wait_for(run.run(), 5)
    check_a(report)
    assert report["diagnostics"] <= history
    assert "model_data.address" in prepared.text and "model_data.recipe" in prepared.text
    assert "await" not in prepared.accepted.text and "training mode" not in prepared.accepted.text


@pytest.mark.asyncio
async def test_changed_production_order_preserves_the_declared_learning_order() -> None:
    reports = []
    for reverse in (False, True):
        run = prepare_model(accept_model(build_model()), controls=Controls(reverse_production=reverse)).publish()
        report = await run.run()
        check_a(report)
        reports.append(report)
    actual_orders = [[item["sample"] for item in report["source"]["products"] if item["branch"] == "left"] for report in reports]
    assert actual_orders == [[1, 2, 4, 5], [5, 4, 2, 1]]
    semantic = [
        [{name: value for name, value in item.items() if name != "invocation"} for item in report["learning"]["contributions"]]
        for report in reports
    ]
    assert semantic[0] == semantic[1]  # Incidental global invocation numbering may differ.


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["missing", "bad_sample", "wrong_member", "stale_source"])
@pytest.mark.parametrize("sample", [1, 2, 4, 5])
async def test_atomic_admission_rejects_without_consuming_either_member(case: str, sample: int) -> None:
    controls = (
        Controls(omit_sample=sample, history_limit=0)
        if case == "missing"
        else Controls(corrupt_sample=sample, history_limit=0)
        if case == "bad_sample"
        else Controls(swap_member=sample, history_limit=0)
        if case == "wrong_member"
        else Controls(stale_sample=sample, history_limit=0)
    )
    run = prepare_model(accept_model(build_model()), controls=controls).publish()
    report = await asyncio.wait_for(run.run(), 5)
    check_admission_failure(report, sample, missing=case == "missing")
    assert report["diagnostics"] == 0


@pytest.mark.asyncio
async def test_pending_owner_associations_survive_both_diagnostic_and_completed_history_eviction() -> None:
    run = prepare_model(accept_model(build_model()), controls=Controls(corrupt_sample=2, history_limit=0)).publish()
    for runtime in run.ready.state.image.units.values():
        runtime.outcomes = deque(maxlen=0)
    report = await asyncio.wait_for(run.run(), 5)
    check_admission_failure(report, 2, missing=False)
    assert len(report["unit_outcomes"]) == 1 and len(report["unit_outcomes"][0]["inputs"]) == 2


@pytest.mark.parametrize("case", ["count", "horizon", "binding", "parameter"])
def test_target_failures_preserve_semantic_acceptance(case: str) -> None:
    module = build_model(count=1 if case == "count" else 2, horizon=2 if case == "horizon" else 4)
    registry = model_registry()
    if case == "binding":
        registry = DomainRegistry("missing", {})
    if case == "parameter":
        one(module, AdvanceOp).properties["parameter"] = StringAttr("unknown")
    accepted = accept_model(module)
    with pytest.raises(NotReady):
        prepare_model(accepted, registry)


@pytest.mark.parametrize("case", ["membership", "binding_version"])
def test_prepared_address_and_binding_corruption_rejected(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    original = adapter.parse_model

    def corrupted(text):
        module = original(text)
        if "exec.plan" in text:
            if case == "membership":
                one_address = next(op for op in module.walk() if isinstance(op, AddressOp))
                one_address.properties["sample"] = integer(100)
            else:
                next(op for op in module.walk() if isinstance(op, CallOp)).properties["binding_version"] = StringAttr("other")
        return module

    monkeypatch.setattr(adapter, "parse_model", corrupted)
    with pytest.raises(NotReady, match=r"(membership|binding)"):
        prepare_model(accept_model(build_model()))


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["encode_v1", "loss_v1", "measure_v1"])
async def test_invalid_selected_output_remains_entered_failure(symbol: str) -> None:
    bindings = dict(model_registry().bindings)
    entered = []

    def invalid(*args):
        entered.append(True)
        return None

    bindings[symbol] = replace(bindings[symbol], invoke=invalid)
    run = prepare_model(accept_model(build_model()), DomainRegistry("invalid", bindings)).publish()
    report = await asyncio.wait_for(run.run(), 5)
    assert entered and report["status"] == "failed" and report["released"] and report["errors"]
    assert any("invalid returned" in item["message"] for item in report["errors"])
    if symbol == "measure_v1":
        assert not report["evaluation"]["measurements"][0]["completed"]
    if symbol == "encode_v1":
        assert any(item["outcome"] == "invalid_output" for item in report["source"]["stages"])


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["encode_v1", "measure_v1"])
@pytest.mark.parametrize("fail", [False, True])
async def test_cancellation_waits_for_physical_current_use(target: str, fail: bool) -> None:
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    registry = model_registry()
    bindings = dict(registry.bindings)
    original = bindings[target].invoke

    def blocked(*args):
        entered.set()
        try:
            if not release.wait(5):
                raise TimeoutError("physical probe not released")
            if fail:
                raise RuntimeError("selected physical failure after interruption")
            return original(*args)
        finally:
            finished.set()

    bindings[target] = replace(bindings[target], invoke=blocked)
    run = prepare_model(accept_model(build_model()), DomainRegistry("blocked", bindings)).publish()
    running = asyncio.create_task(run.run())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        running.cancel()
        await asyncio.sleep(0)
        running.cancel()
        await asyncio.sleep(0)
        assert not running.done() and not finished.is_set()
        assert run.ready.state.readers or run.ready.state.writers
    finally:
        release.set()
    with pytest.raises((asyncio.CancelledError, BaseExceptionGroup)):
        await asyncio.wait_for(running, 5)
    assert finished.is_set() and run.report()["released"]
    assert run.report()["status"] == "cancelled"
    if target == "measure_v1":
        assert not run.report()["evaluation"]["measurements"][0]["completed"]
    if fail:
        assert any("selected physical failure" in item["message"] for item in run.report()["errors"])


@pytest.mark.asyncio
async def test_evaluation_trigger_is_distinct_from_physical_completion() -> None:
    requested, entered, release, finished = (threading.Event() for _ in range(4))
    bindings = dict(model_registry().bindings)
    original = bindings["measure_v1"].invoke

    def delayed(model):
        entered.set()
        try:
            if not release.wait(5):
                raise TimeoutError("measurement not released")
            return original(model)
        finally:
            finished.set()

    bindings["measure_v1"] = replace(bindings["measure_v1"], invoke=delayed)
    run = prepare_model(
        accept_model(build_model()), DomainRegistry("delayed", bindings), controls=Controls(evaluation_requested=requested)
    ).publish()
    running = asyncio.create_task(run.run())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert requested.is_set() and not finished.is_set() and not running.done()
        evidence = run.ready.state.owners["a.evaluation"].measurements[0]
        assert evidence["trigger_window"] == 0 and evidence["epoch"] in (1, 2) and not evidence["completed"]
        assert run.ready.state.readers or run.ready.state.writers
    finally:
        release.set()
    report = await asyncio.wait_for(running, 5)
    check_a(report)
    assert finished.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "product",
        "policy_epoch",
        "sample",
        "window",
        "loss",
        "advance",
        "reset",
        "eval_epoch",
        "eval_trigger",
        "eval_inputs",
        "source",
        "claim",
    ],
)
async def test_independent_oracle_rejects_corrupted_reports(case: str) -> None:
    report = await prepare_model(accept_model(build_model())).publish().run()
    bad = copy.deepcopy(report)
    if case == "product":
        bad["source"]["products"][0]["value"] += 1
    elif case == "policy_epoch":
        bad["learning"]["contributions"][1]["epoch"] += 1
    elif case == "sample":
        bad["learning"]["contributions"][0]["sample"] = 2
    elif case == "window":
        bad["learning"]["contributions"][2]["window"] = 0
    elif case == "loss":
        bad["learning"]["contributions"][0]["loss"] += 1
    elif case == "advance":
        bad["learning"]["windows"][0]["weight"] += 0.1
    elif case == "reset":
        bad["unit_outcomes"][0]["reset"] = "not_attempted"
    elif case == "eval_epoch":
        bad["evaluation"]["measurements"][0]["epoch"] = 0
    elif case == "eval_trigger":
        bad["evaluation"]["measurements"][0]["trigger_window"] = 1
    elif case == "eval_inputs":
        bad["evaluation"]["measurements"][0]["inputs"] = []
    elif case == "source":
        next(item for item in bad["source"]["products"] if item["branch"] == "right")["sources"][0][2] = 99
    else:
        bad["claimed"]["left.0"] = []
    with pytest.raises(AssertionError):
        check_a(bad)


@pytest.mark.asyncio
async def test_shared_wrong_loss_is_detected_beyond_backend_equivalence() -> None:
    bindings = dict(model_registry().bindings)
    original = bindings["loss_v1"].invoke
    bindings["loss_v1"] = replace(bindings["loss_v1"], invoke=lambda *args: original(*args) * 2)
    registry = DomainRegistry("shared-fault", bindings)
    reports = [
        await prepare_model(accept_model(build_model()), registry, lowering=lowering).publish().run() for lowering in ("calls", "inline")
    ]
    assert reports[0]["learning"]["windows"] == reports[1]["learning"]["windows"]
    for report in reports:
        with pytest.raises(AssertionError, match="loss"):
            check_a(report)


def test_model_document_reconstructs_without_author_in_fresh_process(tmp_path: Path) -> None:
    prepared = prepare_model(accept_model(build_model()))
    semantic, manifest = tmp_path / "accepted.mlir", tmp_path / "bindings.json"
    semantic.write_text(prepared.accepted.text)
    manifest.write_text(prepared.manifest)
    script = """
import importlib
contract=importlib.import_module('docs_design.models-strategy-trainer.xdsl_candidate.model_contract')
contract.build_model=lambda: (_ for _ in ()).throw(AssertionError('author used'))
demo=importlib.import_module('docs_design.models-strategy-trainer.xdsl_candidate.model_demo')
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
    check_a(result["report"])
    assert result["accepted_digest"] == prepared.accepted.digest
    assert (tmp_path / "loaded" / "prepared.mlir").read_text() == prepared.text
