"""A's independent Pliron path uses the same actual owner and physical services."""

from __future__ import annotations

import asyncio
import importlib
import json
import subprocess
import sys
import threading
from collections import deque
from dataclasses import replace
from pathlib import Path

import pytest

from .correlation import Rejected, contracts, describe, registry
from .correlation_author import build
from . import test_correlation as document_tests

correlation_native = document_tests.correlation_native

pytest.importorskip("torch", reason="A owner case requires optional Torch")
_prefix = "docs_design.models-strategy-trainer"
target = importlib.import_module(_prefix + ".model_target")
oracle = importlib.import_module(_prefix + ".xdsl_candidate.model_conformance")


def ready(native, *, selected=None, controls=None, lowering="calls", workers=True, fuse=False):
    selected = registry() if selected is None else selected
    options = {"lowering": lowering, "workers": workers, "fuse": fuse}
    artifact = native.prepare(native.accept(build()), describe(selected, **options))
    return native.materialize(artifact, selected, controls=controls, **options)


@pytest.mark.parametrize("lowering", ["calls", "inline"])
@pytest.mark.parametrize("workers", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("history", [0, 256])
def test_independent_arithmetic_and_owner_associations(correlation_native, lowering, workers, fuse, history):
    prepared = ready(correlation_native, lowering=lowering, workers=workers, fuse=fuse, controls=target.Controls(history_limit=history))
    report = asyncio.run(prepared.publish().run())
    oracle.check_a(report)
    assert report["diagnostics"] <= history
    assert prepared.local.source_groups == ((("data.double", "data.offset"),) if fuse else (("data.double",), ("data.offset",)))


@pytest.mark.parametrize("fuse", [False, True])
def test_production_order_is_not_learning_order(correlation_native, fuse):
    for reverse in (False, True):
        report = asyncio.run(ready(correlation_native, fuse=fuse, controls=target.Controls(reverse_production=reverse)).publish().run())
        oracle.check_a(report)
        order = [item["sample"] for item in report["source"]["products"] if item["branch"] == "left"]
        assert order == ([5, 4, 2, 1] if reverse else [1, 2, 4, 5])


@pytest.mark.parametrize("case", ["missing", "bad_sample", "wrong_member", "stale_source"])
@pytest.mark.parametrize("sample", [1, 2, 4, 5])
def test_atomic_join_rejects_neither_member_claimed(correlation_native, case, sample):
    field = {"missing": "omit_sample", "bad_sample": "corrupt_sample", "wrong_member": "swap_member", "stale_source": "stale_sample"}[case]
    controls = target.Controls(history_limit=0, **{field: sample})
    report = asyncio.run(ready(correlation_native, controls=controls).publish().run())
    oracle.check_admission_failure(report, sample, missing=case == "missing")


@pytest.mark.parametrize("sample", [2, 5])
def test_pending_associations_survive_history_eviction(correlation_native, sample):
    run = ready(correlation_native, controls=target.Controls(corrupt_sample=sample, history_limit=0)).publish()
    for unit in run.ready.state.image.units.values():
        unit.outcomes = deque(maxlen=0)
    report = asyncio.run(run.run())
    oracle.check_admission_failure(report, sample, missing=False)
    assert len(report["unit_outcomes"]) == 1 and len(report["unit_outcomes"][0]["inputs"]) == 2


@pytest.mark.parametrize("symbol", ["encode_v1", "loss_v1", "measure_v1"])
def test_invalid_selected_output_is_entered_failure(correlation_native, symbol):
    selected = registry()
    bindings = dict(selected.bindings)
    bindings[symbol] = replace(bindings[symbol], invoke=lambda *args: None)
    report = asyncio.run(ready(correlation_native, selected=replace(selected, bindings=bindings)).publish().run())
    assert report["status"] == "failed" and report["released"]
    assert any("invalid returned" in item["message"] for item in report["errors"])
    if symbol == "measure_v1":
        assert not report["evaluation"]["measurements"][0]["completed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["encode_v1", "measure_v1"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
async def test_physical_current_use_retains_protection_through_cancellation(correlation_native, symbol, fail, fuse):
    entered, release, finished = (threading.Event() for _ in range(3))
    selected = registry()
    bindings = dict(selected.bindings)
    original = bindings[symbol].invoke

    def blocked(*args):
        entered.set()
        try:
            if not release.wait(10):
                raise TimeoutError("physical A probe not released")
            if fail:
                raise RuntimeError("selected physical failure after interruption")
            return original(*args)
        finally:
            finished.set()

    bindings[symbol] = replace(bindings[symbol], invoke=blocked)
    run = ready(correlation_native, selected=replace(selected, bindings=bindings), fuse=fuse).publish()
    running = asyncio.create_task(run.run())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        running.cancel()
        await asyncio.sleep(0)
        running.cancel()
        await asyncio.sleep(0)
        assert not running.done() and not finished.is_set()
        assert run.ready.state.readers or run.ready.state.writers
        if symbol == "measure_v1":
            assert not run.ready.state.image.bindings["learner"].model.training
    finally:
        release.set()
    with pytest.raises((asyncio.CancelledError, BaseExceptionGroup)):
        await asyncio.wait_for(running, 10)
    report = run.report()
    assert finished.is_set() and report["released"] and report["status"] == "cancelled"
    if fail:
        assert any("selected physical failure" in item["message"] for item in report["errors"])
    if symbol == "measure_v1":
        assert not report["evaluation"]["measurements"][0]["completed"]
        assert run.ready.state.image.bindings["learner"].model.training


@pytest.mark.asyncio
async def test_request_cause_does_not_claim_physical_completion(correlation_native):
    requested, entered, release = (threading.Event() for _ in range(3))
    selected = registry()
    bindings = dict(selected.bindings)
    original = bindings["measure_v1"].invoke

    def held(model):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("measurement gate expired")
        return original(model)

    bindings["measure_v1"] = replace(bindings["measure_v1"], invoke=held)
    run = ready(
        correlation_native, selected=replace(selected, bindings=bindings), controls=target.Controls(evaluation_requested=requested)
    ).publish()
    running = asyncio.create_task(run.run())
    try:
        assert await asyncio.to_thread(entered.wait, 5) and requested.is_set()
        measured = run.ready.state.owners["a.evaluation"].measurements[0]
        assert measured["trigger_window"] == 0 and measured["epoch"] in (1, 2) and not measured["completed"]
        assert not running.done() and (run.ready.state.readers or run.ready.state.writers)
        assert not run.ready.state.image.bindings["learner"].model.training
    finally:
        release.set()
    oracle.check_a(await asyncio.wait_for(running, 10))
    assert run.ready.state.image.bindings["learner"].model.training


@pytest.mark.parametrize("field", ["accepted", "prepared", "manifest", "unknown"])
def test_bad_claims_do_not_invoke_model_factories(correlation_native, field):
    artifact = correlation_native.prepare(correlation_native.accept(build()))
    selected = registry()
    bindings, called = dict(selected.bindings), []
    original = bindings["scale_v1"].invoke
    bindings["scale_v1"] = replace(bindings["scale_v1"], invoke=lambda *args: called.append(args) or original(*args))
    if field == "accepted":
        artifact["accepted"] += " garbage"
    elif field == "prepared":
        artifact["prepared"] = artifact["prepared"].replace("validate_all_members_and_current_source_before_any_claim", "claim_first")
    elif field == "manifest":
        artifact["manifest"]["bindings"]["scale_v1"]["version"] = "forged"
    else:
        artifact["owner_verified"] = True
    with pytest.raises(Rejected):
        correlation_native.materialize(artifact, replace(selected, bindings=bindings))
    assert not called


@pytest.mark.parametrize("case", ["manifest", "prepared"])
def test_xdsl_saved_claims_reject_before_factories(case):
    adapter = importlib.import_module(_prefix + ".xdsl_candidate.model_adapter")
    contract = importlib.import_module(_prefix + ".xdsl_candidate.model_contract")
    accepted = contract.accept_model(contract.build_model())
    prepared = adapter.prepare_model(accepted)
    selected = registry()
    bindings, called = dict(selected.bindings), []
    original = bindings["scale_v1"].invoke
    bindings["scale_v1"] = replace(bindings["scale_v1"], invoke=lambda *args: called.append(args) or original(*args))
    saved, manifest = prepared.text, json.loads(prepared.manifest)
    if case == "manifest":
        manifest["bindings"][0]["version"] = "forged"
    else:
        saved = saved.replace(adapter.OWNER_PROTOCOL, "claim_before_checking")
    with pytest.raises(adapter.NotReady):
        adapter.prepare_model(accepted, replace(selected, bindings=bindings), saved_text=saved, saved_manifest=manifest)
    assert not called


def test_shared_wrong_loss_needs_independent_oracle(correlation_native):
    selected = registry()
    bindings = dict(selected.bindings)
    original = bindings["loss_v1"].invoke
    bindings["loss_v1"] = replace(bindings["loss_v1"], invoke=lambda *args: original(*args) * 2)
    selected = replace(selected, bindings=bindings)
    reports = [asyncio.run(ready(correlation_native, selected=selected, lowering=mode).publish().run()) for mode in ("calls", "inline")]
    assert reports[0]["learning"]["windows"] == reports[1]["learning"]["windows"]
    for report in reports:
        with pytest.raises(AssertionError, match="loss"):
            oracle.check_a(report)


def test_saved_inspection_fields_do_not_authorize_and_run_is_single(correlation_native):
    artifact = correlation_native.prepare(correlation_native.accept(build()))
    artifact["profile"] = {"models": [], "ready": True}
    run = correlation_native.materialize(artifact).publish()
    oracle.check_a(asyncio.run(run.run()))
    with pytest.raises(RuntimeError, match="one owner lifetime"):
        asyncio.run(run.run())


@pytest.mark.parametrize("case", ["product", "sample", "loss", "epoch", "source"])
def test_independent_expectations_reject_corruption(correlation_native, case):
    report = asyncio.run(ready(correlation_native).publish().run())
    if case == "product":
        report["source"]["products"][0]["value"] += 1
    elif case == "sample":
        report["learning"]["contributions"][0]["sample"] = 99
    elif case == "loss":
        report["learning"]["contributions"][0]["loss"] += 1
    elif case == "epoch":
        report["evaluation"]["measurements"][0]["epoch"] = 0
    else:
        next(item for item in report["source"]["products"] if item["branch"] == "right")["sources"][0][2] = 99
    with pytest.raises(AssertionError):
        oracle.check_a(report)


@pytest.mark.parametrize("framework", ["pliron", "xdsl"])
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("lowering", ["calls", "inline"])
def test_fresh_saved_plan_with_authoring_and_other_ir_disabled(correlation_native, tmp_path, framework, optimized, fuse, lowering):
    if framework == "pliron":
        artifact = correlation_native.prepare(correlation_native.accept(build()), contracts(fuse=fuse, lowering=lowering))
    else:
        adapter = importlib.import_module(_prefix + ".xdsl_candidate.model_adapter")
        contract = importlib.import_module(_prefix + ".xdsl_candidate.model_contract")
        prepared = adapter.prepare_model(contract.accept_model(contract.build_model()), fuse=fuse, lowering=lowering)
        artifact = {"accepted": prepared.accepted.text, "prepared": prepared.text, "manifest": json.loads(prepared.manifest)}
    saved = tmp_path / "artifact.json"
    saved.write_text(json.dumps(artifact))
    script = r"""
import asyncio,importlib,importlib.abc,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);before=sys._is_gil_enabled()
p='docs_design.models-strategy-trainer'
def forbidden(*a,**k):raise RuntimeError('author or other frontend entered saved A path')
class Boundary(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if sys.argv[2]=='xdsl' and '.pliron_candidate' in fullname:raise RuntimeError('Pliron imported by xDSL A')
  if sys.argv[2]=='pliron' and fullname.endswith(('.model_adapter','.model_contract','.model_dialect')):raise RuntimeError('xDSL A IR imported by Pliron A')
sys.meta_path.insert(0,Boundary())
fixture=importlib.import_module(p+'.xdsl_candidate.fixture');fixture.build=forbidden
data=json.loads(Path(sys.argv[3]).read_text());fuse=sys.argv[4]=='1';lowering=sys.argv[5]
if sys.argv[2]=='pliron':
 api=importlib.import_module(p+'.pliron_candidate.correlation');api.CorrelationNative.accept=forbidden;api.CorrelationNative.prepare=forbidden
 importlib.import_module(p+'.pliron_candidate.correlation_author').build=forbidden
 prep=importlib.import_module(p+'.xdsl_candidate.prepare');prep.accept=forbidden;prep.prepare=forbidden;prep.parse=forbidden
 ready=api.CorrelationNative(Path(sys.argv[6])).materialize(data,fuse=fuse,lowering=lowering)
else:
 contract=importlib.import_module(p+'.xdsl_candidate.model_contract');contract.build_model=forbidden
 api=importlib.import_module(p+'.xdsl_candidate.model_adapter')
 accepted=contract.accept_model(importlib.import_module(p+'.xdsl_candidate.model_dialect').parse_model(data['accepted']))
 ready=api.prepare_model(accepted,fuse=fuse,lowering=lowering,saved_text=data['prepared'],saved_manifest=data['manifest'])
report=asyncio.run(ready.publish().run())
print(json.dumps({'report':report,'optimized':not __debug__,'gil_before':before,'gil_after':sys._is_gil_enabled()}))
"""
    result = subprocess.run(
        [
            sys.executable,
            *(["-O"] if optimized else []),
            "-c",
            script,
            str(Path(__file__).resolve().parents[3]),
            framework,
            str(saved),
            str(int(fuse)),
            lowering,
            str(correlation_native.binary),
        ],
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["optimized"] is optimized and not observed["gil_before"] and not observed["gil_after"]
    oracle.check_a(observed["report"])
