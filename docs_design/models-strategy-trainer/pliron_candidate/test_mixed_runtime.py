"""Native mixed preparation exercises actual shared ownership and causal checks."""

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

from .mixed import Rejected, describe, registry
from .mixed_author import build
from . import test_mixed as document_tests

mixed_native = document_tests.mixed_native

pytest.importorskip("torch", reason="mixed owner case requires optional Torch")
prefix = "docs_design.models-strategy-trainer"
target = importlib.import_module(prefix + ".mixed_target")
oracle = importlib.import_module(prefix + ".xdsl_candidate.mixed_conformance")


def ready(native, *, selected=None, controls=None, lowering="calls", fuse=False):
    selected = registry() if selected is None else selected
    options = {"lowering": lowering, "fuse": fuse}
    saved = native.prepare(native.accept(build()), describe(selected, **options))
    return native.materialize(saved, selected, controls=controls, **options)


async def held(prepared, *, interrupt=False):
    run = prepared.publish()
    task = asyncio.create_task(run.run())
    controls = prepared.controls
    witness = None
    try:
        async with asyncio.timeout(10):
            while not (
                prepared.bridge.checkpoint.is_set() and controls.entered.is_set() and run.ready.state.owners["mixed.source"].revision == 3
            ):
                assert not task.done(), run.last_report
                await asyncio.sleep(0.001)
        source = run.ready.state.owners["mixed.source"]
        assert any(p["key"] == 5 and p["returned"] is None for p in source.products)
        stage = next(p for p in source.stages if p["origin"] == "data.encode" and p["sample"] == 6)
        run.stop()
        while not run.ready.state.feeds["mixed.left"].closed:
            await asyncio.sleep(0)
        if interrupt:
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0.005)
                assert not task.done() and run.ready.state.readers and run.ready.state.active_producers
        witness = {
            "running": not task.done(),
            "protected": bool(run.ready.state.readers),
            "physical_end": stage["physical_end"],
            "release": stage["release"],
            "feedback_revision": source.revision,
        }
        assert witness["running"] and witness["protected"] and witness["physical_end"] is witness["release"] is None
        controls.release.set()
        if interrupt:
            with pytest.raises((asyncio.CancelledError, BaseExceptionGroup)):
                await asyncio.wait_for(task, 10)
            return run.last_report, witness
        return await asyncio.wait_for(task, 10), witness
    finally:
        controls.release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("lowering", ["calls", "inline"])
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("history", [0, 256])
def test_independent_production_epoch_arithmetic_and_feedback(mixed_native, lowering, fuse, history):
    prepared = ready(mixed_native, lowering=lowering, fuse=fuse, controls=target.MixedControls(block_key=6, history_limit=history))
    report = asyncio.run(target.exercise(prepared))
    oracle.check_mixed(report)
    assert report["diagnostics"] <= history
    assert prepared.local.source_groups == ((("data.double", "data.offset"),) if fuse else (("data.double",), ("data.offset",)))


@pytest.mark.parametrize("lowering", ["calls", "inline"])
@pytest.mark.parametrize("fuse", [False, True])
def test_prefetched_production_state_is_not_replaced_with_latest(mixed_native, lowering, fuse):
    selected = registry()
    bindings = dict(selected.bindings)
    original_map, original_encode = bindings["double_v1"].invoke, bindings["encode_v1"].invoke
    produced, lock = threading.Event(), threading.Lock()
    count = 0

    async def held_first(value):
        if value == 0 and not await asyncio.to_thread(produced.wait, 5):
            raise TimeoutError("initial production did not finish")
        return await original_map(value)

    def observe_production(*args):
        nonlocal count
        value = original_encode(*args)
        with lock:
            count += 1
            if count == 3:
                produced.set()
        return value

    bindings["double_v1"] = replace(bindings["double_v1"], invoke=held_first)
    bindings["encode_v1"] = replace(bindings["encode_v1"], invoke=observe_production)
    prepared = ready(
        mixed_native,
        selected=replace(selected, bindings=bindings),
        lowering=lowering,
        fuse=fuse,
        controls=target.MixedControls(block_key=6),
    )
    report = asyncio.run(target.exercise(prepared))
    oracle.check_mixed(report)
    initial = [row for row in report["source"]["stages"] if row["origin"] == "data.encode" and row["sample"] < 3]
    assert len(initial) == 3 and all(row["sources"][0][2] == 0 for row in initial)
    assert [row["epoch"] for row in report["learning"]["contributions"]] == [0, 1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("lowering", ["calls", "inline"])
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("interrupt,fail", [(False, False), (False, True), (True, False), (True, True)])
async def test_blocked_publication_and_physical_read_draining(mixed_native, lowering, fuse, interrupt, fail):
    prepared = ready(
        mixed_native, lowering=lowering, fuse=fuse, controls=target.MixedControls(block_key=6, history_limit=0, fail_physical=fail)
    )
    report, witness = await held(prepared, interrupt=interrupt)
    outcome = "cancelled" if interrupt else "failed" if fail else "stopped"
    oracle.check_mixed(report, outcome=outcome)
    assert witness["feedback_revision"] == 3 and report["diagnostics"] == 0
    assert any("mixed physical failure" in item["message"] for item in report["errors"]) == fail


@pytest.mark.parametrize("key", [0, 1, 2])
@pytest.mark.parametrize("case", ["identity", "stamp", "member"])
def test_atomic_admission_claims_neither_bad_member(mixed_native, key, case):
    controls = target.MixedControls(corrupt_key=key, corrupt_stamp=case == "stamp", swap_member=case == "member", history_limit=0)
    report = asyncio.run(ready(mixed_native, controls=controls).publish().run())
    assert report["status"] == "failed" and report["released"] and not report["source"]["active"]
    assert all(not any(ordinal == key for ordinal, _ in uses) for uses in report["claimed"].values())
    assert report["steps"] <= key


@pytest.mark.parametrize("symbol", ["encode_v1", "mixed_loss_v1", "dynamic_filter_v1", "learned_bias_v1"])
def test_invalid_selected_results_preserve_failure_and_settle(mixed_native, symbol):
    selected = registry()
    bindings = dict(selected.bindings)
    bindings[symbol] = replace(bindings[symbol], invoke=lambda *args: None)
    report = asyncio.run(ready(mixed_native, selected=replace(selected, bindings=bindings)).publish().run())
    assert report["status"] == "failed" and report["released"] and report["errors"] and not report["source"]["active"]
    if symbol == "learned_bias_v1":
        assert report["learning"]["windows"] and not report["source"]["updates"]
    else:
        assert report["steps"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("lowering", ["calls", "inline"])
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("interrupt", [False, True])
async def test_observation_erasure_leaves_owner_feedback_and_math(mixed_native, monkeypatch, lowering, fuse, interrupt):
    class Discarded(list):
        def append(self, item):
            pass

    original_source = target.MixedSource
    original_learning = target.LearningFacts

    def source_factory():
        source = original_source()
        for name, value in vars(source).items():
            if isinstance(value, list):
                setattr(source, name, Discarded())
        return source

    monkeypatch.setattr(target, "MixedSource", source_factory)
    monkeypatch.setattr(target, "LearningFacts", lambda: original_learning(contributions=Discarded(), windows=Discarded()))
    controls = target.MixedControls(block_key=6, history_limit=0, fail_physical=interrupt)
    run = ready(mixed_native, controls=controls, lowering=lowering, fuse=fuse).publish()
    task = asyncio.create_task(run.run())
    try:
        async with asyncio.timeout(10):
            while not (
                run.prepared.bridge.checkpoint.is_set()
                and controls.entered.is_set()
                and run.ready.state.owners["mixed.source"].revision == 3
            ):
                assert not task.done(), run.last_report
                await asyncio.sleep(0.001)
        run.stop()
        while not run.ready.state.feeds["mixed.left"].closed:
            await asyncio.sleep(0)
        assert not task.done() and run.ready.state.readers
        if interrupt:
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0.005)
                assert not task.done() and run.ready.state.readers
        controls.release.set()
        if interrupt:
            with pytest.raises((asyncio.CancelledError, BaseExceptionGroup)):
                await asyncio.wait_for(task, 10)
            report = run.last_report
            assert any("mixed physical failure" in item["message"] for item in report["errors"])
        else:
            report = await asyncio.wait_for(task, 10)
        assert report["released"] and report["steps"] == 3 and not report["pending"] and report["diagnostics"] == 0
        assert report["learning"] == {"contributions": [], "windows": []} and report["source"]["revision"] == 3
        assert all(not value for value in report["source"].values() if isinstance(value, list))
        unit = next(iter(run.ready.state.image.units.values()))
        weights = [0.8]
        outcomes = sorted(unit.outcomes, key=lambda value: value.invocation)
        assert len(outcomes) == 3
        for ordinal, outcome in enumerate(outcomes):
            assert outcome.optimizer == outcome.reset == "returned"
            assert [(use.channel, use.key) for use in outcome.input_handoffs] == [("mixed.left", ordinal), ("mixed.right", ordinal)]
            epoch = next(use for use in outcome.input_handoffs if use.channel == "mixed.right").provenance[0].numerical_epoch
            assert 0 <= epoch <= ordinal
            feature = 2 * ordinal + 1 + weights[epoch] * ordinal / 4
            weights.append(weights[-1] - 0.001 * (weights[-1] * feature - ordinal / 2) * feature)
        assert float(unit.parameters[0].detach()) == pytest.approx(weights[-1], rel=2e-6, abs=2e-6)
    finally:
        controls.release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("admit_first", [False, True])
async def test_long_exclusions_allow_heartbeat_and_applied_feedback(mixed_native, admit_first):
    calls = 0
    beats = []
    feedback_at = []
    limit = 20000

    def filtered(key, revision):
        nonlocal calls
        calls += 1
        if calls == 1:
            asyncio.get_running_loop().call_soon(lambda: beats.append(calls))
        if revision and not feedback_at:
            feedback_at.append(calls)
        if calls == limit:
            raise RuntimeError("excluded-stretch probe complete")
        return admit_first and key == 0

    selected = registry()
    bindings = dict(selected.bindings)
    bindings["dynamic_filter_v1"] = replace(bindings["dynamic_filter_v1"], invoke=filtered)
    run = ready(mixed_native, selected=replace(selected, bindings=bindings)).publish()
    report = await run.run()
    assert report["status"] == "failed" and report["released"] and calls == limit
    assert beats and beats[0] < limit
    if admit_first:
        assert feedback_at and feedback_at[0] < limit and report["steps"] == report["source"]["revision"] == 1
    else:
        assert report["steps"] == 0


@pytest.mark.asyncio
async def test_long_exclusions_are_cancellable(mixed_native):
    calls = 0
    cancelled_at = []

    def cancel():
        cancelled_at.append(calls)
        task.cancel()

    def filtered(key, revision):
        nonlocal calls
        calls += 1
        if calls == 1:
            asyncio.get_running_loop().call_soon(cancel)
        if calls == 20000:
            raise RuntimeError("cancellation starved")
        return False

    selected = registry()
    bindings = dict(selected.bindings)
    bindings["dynamic_filter_v1"] = replace(bindings["dynamic_filter_v1"], invoke=filtered)
    run = ready(mixed_native, selected=replace(selected, bindings=bindings)).publish()
    task = asyncio.create_task(run.run())
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled_at and cancelled_at[0] < 20000 and calls < 20000
    assert run.last_report["status"] == "cancelled" and run.last_report["released"] and not run.last_report["source"]["active"]


@pytest.mark.parametrize("claim", ["protocol", "manifest", "unused_binding", "substitution"])
def test_bad_saved_claims_never_load_models(mixed_native, claim):
    selected = registry()
    loaded = []
    bindings = dict(selected.bindings)
    bindings["scale_v1"] = replace(bindings["scale_v1"], invoke=lambda *args: loaded.append(args))
    if claim == "substitution":
        bindings["another_encoder"] = bindings["encode_v1"]
    selected = replace(selected, bindings=bindings)
    saved = mixed_native.prepare(mixed_native.accept(build()), describe(selected))
    if claim == "protocol":
        saved["prepared"] = saved["prepared"].replace("producer_owned_policy_validate_ack_retire_once", "archive_authority")
    elif claim == "manifest":
        saved["manifest"]["registry"] = "different"
    elif claim == "unused_binding":
        saved["manifest"]["bindings"]["measure_v1"]["version"] = "other"
    else:
        lines = saved["prepared"].splitlines()
        index = next(i for i, line in enumerate(lines) if "ac_exec.declaration" in line and 'kind: builtin.string "encode"' in line)
        lines[index] = lines[index].replace(
            'implementation: builtin.string "encode_v1"', 'implementation: builtin.string "another_encoder"'
        )
        saved["prepared"] = "\n".join(lines)
    with pytest.raises(Rejected):
        mixed_native.materialize(saved, selected)
    assert not loaded


@pytest.mark.parametrize("case", ["manifest", "prepared"])
def test_xdsl_saved_correspondence_precedes_factories(case):
    a = importlib.import_module(prefix + ".xdsl_candidate.mixed_adapter")
    contract = importlib.import_module(prefix + ".xdsl_candidate.mixed_contract")
    accepted = contract.accept_mixed(contract.build_mixed())
    original = a.prepare_mixed(accepted)
    loaded = []
    selected = a.mixed_registry()
    bindings = dict(selected.bindings)
    bindings["scale_v1"] = replace(bindings["scale_v1"], invoke=lambda *args: loaded.append(args))
    options = {"saved_text": original.text, "saved_manifest": json.loads(original.manifest)}
    if case == "manifest":
        options["saved_manifest"]["registry"] = "other"
    else:
        module = a.parse_mixed(original.text)
        module.attributes["unchecked"] = importlib.import_module("xdsl.dialects.builtin").StringAttr("forged")
        options["saved_text"] = a.dump(module)
    with pytest.raises(a.NotReady):
        a.prepare_mixed(accepted, replace(selected, bindings=bindings), **options)
    assert not loaded


@pytest.mark.parametrize("lowering", ["calls", "inline"])
def test_actual_shared_wrong_mapping_is_rejected(mixed_native, lowering):
    selected = registry()
    bindings = dict(selected.bindings)

    async def wrong(value):
        return 2 * value + 3

    bindings["double_v1"] = replace(bindings["double_v1"], invoke=wrong)
    for fuse in (False, True):
        prepared = ready(
            mixed_native,
            selected=replace(selected, bindings=bindings),
            lowering=lowering,
            fuse=fuse,
            controls=target.MixedControls(block_key=6),
        )
        report = asyncio.run(target.exercise(prepared))
        assert report["status"] == "stopped"
        with pytest.raises(AssertionError):
            oracle.check_mixed(report)


def test_independent_report_corruptions(mixed_native):
    report = asyncio.run(target.exercise(ready(mixed_native, controls=target.MixedControls(block_key=6))))
    oracle.check_mixed(report)
    for case in (
        "selection",
        "exclusion",
        "weight",
        "feedback",
        "source_epoch",
        "input",
        "release",
        "unit",
        "discard",
        "horizon",
        "publication",
    ):
        changed = copy.deepcopy(report)
        if case == "selection":
            changed["source"]["selections"][-1]["value"] += 1
        elif case == "exclusion":
            changed["source"]["excluded"].clear()
        elif case == "weight":
            changed["learning"]["windows"][1]["weight"] += 1
        elif case == "feedback":
            changed["source"]["updates"][1]["epoch"] = 99
        elif case == "source_epoch":
            next(item for item in changed["source"]["stages"] if item["origin"] == "data.encode")["sources"][0][2] = 4
        elif case == "input":
            changed["learning"]["contributions"][0]["inputs"][1]["window"] = 99
        elif case == "release":
            next(item for item in changed["source"]["stages"] if item["origin"] == "data.encode")["release"] = 0
        elif case == "unit":
            changed["unit_outcomes"][0]["optimizer"] = "not_attempted"
        elif case == "discard":
            changed["remaining"]["mixed.right"] = [3]
        elif case == "horizon":
            changed["source"]["max_active"] = 4
        else:
            changed["source"]["products"][0]["selected"] += 1
        with pytest.raises(AssertionError):
            oracle.check_mixed(changed)


def test_checkpoint_control_and_single_owner_lifetime(mixed_native):
    run = ready(mixed_native).publish()
    with pytest.raises(ValueError, match="declared learning checkpoint"):
        run.stop()
    assert not run.prepared.bridge.stop.is_set()
    run.started = True
    with pytest.raises(RuntimeError, match="one owner lifetime"):
        asyncio.run(run.run())


@pytest.mark.parametrize("frontend", ["native", "xdsl"])
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("lowering", ["calls", "inline"])
@pytest.mark.parametrize("fuse", [False, True])
def test_fresh_saved_plan_without_other_frontend(mixed_native, tmp_path: Path, frontend, optimized, lowering, fuse):
    if frontend == "native":
        saved = mixed_native.prepare(mixed_native.accept(build()), describe(registry(), lowering=lowering, fuse=fuse))
    else:
        a = importlib.import_module(prefix + ".xdsl_candidate.mixed_adapter")
        c = importlib.import_module(prefix + ".xdsl_candidate.mixed_contract")
        p = a.prepare_mixed(c.accept_mixed(c.build_mixed()), lowering=lowering, fuse=fuse)
        saved = {"accepted": p.accepted.text, "prepared": p.text, "manifest": json.loads(p.manifest)}
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(saved))
    code = r"""
import asyncio,importlib,importlib.abc,json,sys
prefix='docs_design.models-strategy-trainer'
frontend,binary,path,lowering,fuse=sys.argv[1:];fuse=fuse=='1'
blocked=('.mixed_adapter','.mixed_contract','.mixed_dialect','.model_adapter','.model_contract','.model_dialect')
class BlockOther(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if (frontend=='native' and fullname.endswith(blocked)) or (frontend=='xdsl' and '.pliron_candidate' in fullname):
   raise AssertionError('other frontend imported: '+fullname)
sys.meta_path.insert(0,BlockOther());before=sys._is_gil_enabled()
saved=json.load(open(path))
target=importlib.import_module(prefix+'.mixed_target')
def forbidden(*args,**kw):raise AssertionError('author or disabled IR entry point called')
if frontend=='native':
 m=importlib.import_module(prefix+'.pliron_candidate.mixed')
 importlib.import_module(prefix+'.pliron_candidate.mixed_author').build=forbidden
 m.MixedNative.accept=forbidden;m.MixedNative.prepare=forbidden
 for name in ('.dialects','.prepare'):
  module=importlib.import_module(prefix+'.xdsl_candidate'+name)
  for entry in ('parse','accept','prepare'):
   if hasattr(module,entry):setattr(module,entry,forbidden)
 from pathlib import Path
 prepared=m.MixedNative(Path(binary)).materialize(saved,controls=target.MixedControls(block_key=6),lowering=lowering,fuse=fuse)
else:
 m=importlib.import_module(prefix+'.xdsl_candidate.mixed_adapter');c=importlib.import_module(prefix+'.xdsl_candidate.mixed_contract')
 c.build_mixed=forbidden
 accepted=c.accept_mixed(m.parse_mixed(saved['accepted']))
 prepared=m.prepare_mixed(accepted,controls=target.MixedControls(block_key=6),lowering=lowering,fuse=fuse,saved_text=saved['prepared'],saved_manifest=saved['manifest'])
report=asyncio.run(target.exercise(prepared))
print(json.dumps({'report':report,'gil_before':before,'gil_after':sys._is_gil_enabled()}))
"""
    result = subprocess.run(
        [
            sys.executable,
            *(["-O"] if optimized else []),
            "-c",
            code,
            frontend,
            str(mixed_native.binary),
            str(path),
            lowering,
            str(int(fuse)),
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    oracle.check_mixed(data["report"])
    assert data["gil_before"] is data["gil_after"] is False
