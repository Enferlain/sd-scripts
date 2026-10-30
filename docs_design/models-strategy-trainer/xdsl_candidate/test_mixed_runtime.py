"""Mixed pressure and independent fault controls through the existing Engine."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from xdsl.dialects.builtin import ArrayAttr, StringAttr

from .dialects import CallOp, FusedOp, PlanOp, integer
from .mixed_conformance import check_mixed
from .mixed_contract import accept_mixed, build_mixed
from .model_dialect import RecipeOp
from .prepare import NotReady

pytest.importorskip("torch")
assert __package__ is not None
a = importlib.import_module(__package__ + ".mixed_adapter")
demo = importlib.import_module(__package__ + ".mixed_demo")


@pytest.mark.asyncio
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("lowering", ["calls", "inline"])
@pytest.mark.parametrize("history", [0, 256])
async def test_mixed_one_engine_feedback_and_physical_stop(fuse, lowering, history) -> None:
    prepared = a.prepare_mixed(
        accept_mixed(build_mixed()), fuse=fuse, lowering=lowering, controls=a.MixedControls(history_limit=history, block_key=6)
    )
    report = await demo.exercise(prepared)
    check_mixed(report)
    if history == 0:
        assert report["diagnostics"] == 0
    assert prepared.local.source_groups == ((("data.double", "data.offset"),) if fuse else (("data.double",), ("data.offset",)))


@pytest.mark.asyncio
@pytest.mark.parametrize("physical_failure", [False, True])
async def test_stop_and_repeated_cancellation_drain_the_real_model_read(physical_failure) -> None:
    controls = a.MixedControls(block_key=6, fail_physical=physical_failure, history_limit=0)
    prepared = a.prepare_mixed(accept_mixed(build_mixed()), controls=controls, fuse=True)
    run = prepared.publish()
    task = asyncio.create_task(run.run())
    try:
        async with asyncio.timeout(10):
            while not (
                prepared.bridge.checkpoint.is_set()
                and controls.entered.is_set()
                and len(run.ready.state.owners["mixed.source"].updates) == 3
            ):
                assert not task.done(), run.last_report
                await asyncio.sleep(0.001)
        assert any(item["key"] == 5 and item["returned"] is None for item in run.ready.state.owners["mixed.source"].products)
        run.stop()
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0.005)
            assert not task.done() and run.ready.state.readers and run.ready.state.active_producers
        controls.release.set()
        with pytest.raises((asyncio.CancelledError, BaseExceptionGroup)):
            await asyncio.wait_for(task, 10)
        report = run.last_report
        check_mixed(report, outcome="cancelled")
        assert report["diagnostics"] == 0
        if physical_failure:
            assert any("mixed physical failure" in item["message"] for item in report["errors"])
    finally:
        controls.release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_retains_physical_failure_and_completed_updates() -> None:
    controls = a.MixedControls(block_key=6, fail_physical=True)
    report = await demo.exercise(a.prepare_mixed(accept_mixed(build_mixed()), controls=controls))
    check_mixed(report, outcome="failed")
    assert any("mixed physical failure" in item["message"] for item in report["errors"])


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["identity", "stamp", "member"])
async def test_correlated_member_rejection_claims_neither_port(case) -> None:
    controls = a.MixedControls(corrupt_key=1, corrupt_stamp=case == "stamp", swap_member=case == "member")
    report = await a.prepare_mixed(accept_mixed(build_mixed()), controls=controls).publish().run()
    assert report["status"] == "failed" and report["released"] and report["steps"] <= 1
    assert all(not any(key == 1 for key, _ in claimed) for claimed in report["claimed"].values())
    assert not report["source"]["active"]


@pytest.mark.parametrize(
    "case", ["target", "horizon", "capacity", "digest", "protocol", "ancestry", "symbol", "order", "missing", "meaning"]
)
def test_mixed_prepared_correspondence_before_model_load(case, monkeypatch) -> None:
    original = a.parse_mixed
    loaded = []
    bindings = dict(a.mixed_registry().bindings)
    bindings["scale_v1"] = replace(bindings["scale_v1"], invoke=lambda *args: loaded.append(args))

    def corrupt(text):
        module = original(text)
        roots = [op for op in module.walk() if isinstance(op, PlanOp)]
        if not roots:
            return module
        root = roots[0]
        recipe = next(op for op in root.body.block.ops if isinstance(op, RecipeOp))
        calls = [op for op in root.walk() if isinstance(op, CallOp)]
        maps = [op for op in calls if op.role.data == "map"]
        if case in ("target", "digest"):
            root.properties["accepted_digest" if case == "digest" else case] = StringAttr("changed")
        elif case in ("capacity", "horizon"):
            root.properties[case] = integer(99)
        elif case == "protocol":
            recipe.properties["protocol"] = StringAttr("changed")
        elif case == "ancestry":
            maps[0].properties["ancestors"] = ArrayAttr([StringAttr("wrong")])
        elif case == "symbol":
            maps[0].properties["implementation"], maps[1].properties["implementation"] = maps[1].implementation, maps[0].implementation
        elif case == "order":
            group = next(op for op in root.body.block.ops if isinstance(op, FusedOp))
            first = group.body.block.first_op
            group.body.block.detach_op(first)
            group.body.block.add_op(first)
            group.properties["origins"] = ArrayAttr(call.origin for call in group.body.block.ops)
        elif case == "missing":
            root.body.block.erase_op(calls[0])
        else:
            from .mixed_dialect import StopCheckpointOp

            next(op for op in recipe.walk() if isinstance(op, StopCheckpointOp)).properties["after_windows"] = integer(4)
        return module

    monkeypatch.setattr(a, "parse_mixed", corrupt)
    with pytest.raises(NotReady):
        a.prepare_mixed(accept_mixed(build_mixed()), a.DomainRegistry("load-probe", bindings), fuse=True)
    assert not loaded


def test_accepted_but_unsupported_mixed_profiles_are_target_failures() -> None:
    for options in ({"horizon": 4}, {"after_windows": 4}):
        accepted = accept_mixed(build_mixed(**options))
        with pytest.raises(NotReady, match="mixed CPU target"):
            a.prepare_mixed(accepted)


@pytest.mark.asyncio
async def test_shared_wrong_implementations_do_not_make_the_oracle_agree() -> None:
    bindings = dict(a.mixed_registry().bindings)

    async def wrong(value):
        return 2 * value + 3

    bindings["double_v1"] = replace(bindings["double_v1"], invoke=wrong)
    reports = []
    for fuse in (False, True):
        report = await demo.exercise(
            a.prepare_mixed(
                accept_mixed(build_mixed()), a.DomainRegistry("shared-wrong", bindings), fuse=fuse, controls=a.MixedControls(block_key=6)
            )
        )
        assert report["status"] == "stopped"
        with pytest.raises(AssertionError):
            check_mixed(report)
        reports.append(report)
    assert [item["left"] for item in reports[0]["learning"]["contributions"]] == [
        item["left"] for item in reports[1]["learning"]["contributions"]
    ]


@pytest.mark.asyncio
async def test_independent_checker_detects_corrupted_mixed_reports() -> None:
    report = await demo.exercise(a.prepare_mixed(accept_mixed(build_mixed()), controls=a.MixedControls(block_key=6)))
    check_mixed(report)
    for case in ("selection", "exclusion", "weight", "feedback", "source_epoch", "input", "release", "unit", "discard"):
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
        else:
            changed["remaining"]["mixed.right"] = [3]
        with pytest.raises(AssertionError):
            check_mixed(changed)


@pytest.mark.parametrize("fuse", [False, True])
def test_fresh_process_reconstructs_mixed_without_author(tmp_path: Path, fuse: bool) -> None:
    prepared = a.prepare_mixed(accept_mixed(build_mixed()), fuse=fuse)
    (tmp_path / "accepted.mlir").write_text(prepared.accepted.text)
    (tmp_path / "bindings.json").write_text(prepared.manifest)
    code = "import importlib; m=importlib.import_module('docs_design.models-strategy-trainer.xdsl_candidate.mixed_demo'); m.build_mixed=lambda: (_ for _ in ()).throw(AssertionError('author called')); m.main()"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "--load",
            str(tmp_path / "accepted.mlir"),
            "--manifest",
            str(tmp_path / "bindings.json"),
            *(["--fuse"] if fuse else []),
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=os.environ | {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    check_mixed(result["report"])
    assert result["accepted_digest"] == prepared.accepted.digest


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["encode_v1", "mixed_loss_v1", "dynamic_filter_v1", "learned_bias_v1"])
async def test_invalid_selected_domain_results_fail_and_drain(symbol) -> None:
    bindings = dict(a.mixed_registry().bindings)
    bindings[symbol] = replace(bindings[symbol], invoke=lambda *args: None)
    report = await a.prepare_mixed(accept_mixed(build_mixed()), a.DomainRegistry("invalid", bindings), fuse=True).publish().run()
    assert report["status"] == "failed" and report["released"] and report["errors"]
    assert not report["source"]["active"]
    if symbol == "learned_bias_v1":
        assert 1 <= report["steps"] <= 3  # Already admitted work may finish before owner failure reaches it.
        assert [item["key"] for item in report["source"]["selections"]] == [0, 1, 2]
    else:
        assert report["steps"] == 0
    if symbol == "encode_v1":
        assert any(item["outcome"] == "invalid_output" for item in report["source"]["stages"] if item["origin"] == "data.encode")
    if symbol == "learned_bias_v1":
        assert report["learning"]["windows"] and not report["source"]["updates"]


def test_checkpoint_control_cannot_implicitly_stop_before_declared_point() -> None:
    run = a.prepare_mixed(accept_mixed(build_mixed())).publish()
    with pytest.raises(NotReady, match="declared learning checkpoint"):
        run.stop()
    assert not run.prepared.bridge.stop.is_set()


@pytest.mark.asyncio
async def test_pressure_driver_rejects_missing_gate_promptly() -> None:
    prepared = a.prepare_mixed(accept_mixed(build_mixed()))
    with pytest.raises(ValueError, match="block_key=6"):
        await demo.exercise(prepared)
    assert not prepared.attempt.state.running


def test_unfused_order_corruption_rejects_before_model_loading(monkeypatch) -> None:
    original = a.parse_mixed

    def corrupt(text):
        module = original(text)
        root = next((op for op in module.walk() if isinstance(op, PlanOp)), None)
        if root is not None:
            first, second = [op for op in root.body.block.ops if isinstance(op, CallOp) and op.role.data == "map"]
            root.body.block.detach_op(first)
            root.body.block.insert_op_after(first, second)
        return module

    monkeypatch.setattr(a, "parse_mixed", corrupt)
    with pytest.raises(NotReady, match="stage order"):
        a.prepare_mixed(accept_mixed(build_mixed()))


def test_accepted_alternative_parameter_is_an_unsupported_target() -> None:
    from .model_dialect import AdvanceOp

    module = build_mixed()
    next(op for op in module.walk() if isinstance(op, AdvanceOp)).properties["parameter"] = StringAttr("bias")
    with pytest.raises(NotReady, match="mixed CPU target"):
        a.prepare_mixed(accept_mixed(module))


def test_mixed_owner_target_rejects_accepted_non_scalar_local_schema() -> None:
    from .dialects import FlowType, LocalOp, MapOp

    module = build_mixed()
    custom = FlowType(StringAttr("another-payload"))
    next(op for op in module.walk() if isinstance(op, MapOp)).output._type = custom
    next(op for op in module.walk() if isinstance(op, LocalOp)).body.block.args[0]._type = custom
    accepted = accept_mixed(module)
    with pytest.raises(NotReady, match="local schema/resources"):
        a.prepare_mixed(accepted)


@pytest.mark.asyncio
@pytest.mark.parametrize("lowering", ["calls", "inline"])
async def test_live_feedback_and_model_ownership_do_not_depend_on_observation_archives(monkeypatch, lowering) -> None:
    class DiscardedObservations(list):
        def append(self, item):
            pass

    original_source, original_learning = a.MixedSource, a.LearningFacts

    def source_factory():
        source = original_source()
        for name, value in vars(source).items():
            if isinstance(value, list):
                setattr(source, name, DiscardedObservations())
        return source

    monkeypatch.setattr(a, "MixedSource", source_factory)
    monkeypatch.setattr(
        a, "LearningFacts", lambda: original_learning(contributions=DiscardedObservations(), windows=DiscardedObservations())
    )
    controls = a.MixedControls(block_key=6, history_limit=0)
    prepared = a.prepare_mixed(accept_mixed(build_mixed()), controls=controls, lowering=lowering)
    run = prepared.publish()
    task = asyncio.create_task(run.run())
    try:
        async with asyncio.timeout(10):
            while not (
                prepared.bridge.checkpoint.is_set() and controls.entered.is_set() and run.ready.state.owners["mixed.source"].revision == 3
            ):
                assert not task.done(), run.last_report
                await asyncio.sleep(0.001)
        run.stop()
        while not run.ready.state.feeds["mixed.left"].closed:
            await asyncio.sleep(0)
        assert not task.done() and run.ready.state.readers
        controls.release.set()
        report = await asyncio.wait_for(task, 10)
        assert report["status"] == "stopped" and report["released"] and report["steps"] == 3
        assert report["learning"] == {"contributions": [], "windows": []} and report["diagnostics"] == 0
        assert report["source"]["revision"] == 3 and not report["source"]["active"]
        assert all(not value for value in report["source"].values() if isinstance(value, list))
        # Use retained owner input facts, not an observational recorder, to
        # reconstruct the actual production epochs and independent updates.
        unit = next(iter(run.ready.state.image.units.values()))
        weights = [0.8]
        outcomes = sorted(unit.outcomes, key=lambda outcome: outcome.invocation)
        assert len(outcomes) == 3
        for ordinal, outcome in enumerate(outcomes):
            assert outcome.optimizer == outcome.reset == "returned"
            assert [(item.channel, item.key) for item in outcome.input_handoffs] == [("mixed.left", ordinal), ("mixed.right", ordinal)]
            right = next(item for item in outcome.input_handoffs if item.channel == "mixed.right")
            epoch = right.provenance[0].numerical_epoch
            assert 0 <= epoch <= ordinal
            feature = 2 * ordinal + 1 + weights[epoch] * ordinal / 4
            error = weights[-1] * feature - ordinal / 2
            weights.append(weights[-1] - 0.001 * error * feature)
        assert float(unit.parameters[0].detach()) == pytest.approx(weights[-1], rel=2e-6, abs=2e-6)
    finally:
        controls.release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
