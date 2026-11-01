"""Retain C numerical and failure witnesses from native-verified programs."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import threading
from collections import deque
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

PREFIX = "docs_design.models-strategy-trainer"


async def capture(binary: Path, output: Path):
    api = importlib.import_module(PREFIX + ".pliron_candidate.grant")
    author = importlib.import_module(PREFIX + ".pliron_candidate.grant_author")
    target = importlib.import_module(PREFIX + ".grant_target")
    physical = importlib.import_module(PREFIX + ".execution_candidate.workers")
    oracle = importlib.import_module(PREFIX + ".xdsl_candidate.grant_conformance")
    reports = {}

    def prepare(selected=None, *, initial=0.8, clip=None):
        selected = api.registry() if selected is None else selected
        artifact = api.GrantNative(binary).prepare(
            api.GrantNative(binary).accept(author.build(initial=initial, clip_norm=clip)), selected.describe()
        )
        run = api.GrantNative(binary).materialize(artifact, selected, history_limit=0).publish()
        return artifact, run

    def changed(function):
        selected = api.registry()
        bindings = dict(selected.bindings)
        bindings["two_pass_v1"] = replace(bindings["two_pass_v1"], invoke=function, version="probe-1")
        return replace(selected, bindings=bindings)

    for lowering in ("calls", "inline"):
        for initial in (0.8, 0.6):
            for clip in (None, 1.0):
                label = f"{lowering}-i{initial}-" + ("plain" if clip is None else "clip")
                artifact, run = prepare(api.registry(lowering=lowering), initial=initial, clip=clip)
                report = await run.run()
                oracle.check_c(report, initial=initial, clip_norm=clip)
                reports[label] = {"kind": "normal", "initial": initial, "clip": clip, "report": report}
                directory = output / "cases" / label
                directory.mkdir(parents=True, exist_ok=True)
                for name in ("accepted", "prepared"):
                    (directory / (name + ".pliron")).write_text(artifact[name].rstrip("\n") + "\n")
                (directory / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")

    original = api.registry().bindings["two_pass_v1"].invoke
    for case in ("restored_raises", "incomplete", "missing_gradient", "edited_after_restore", "phase_order", "final_advance"):

        def invalid(access, models, case=case):
            if case == "phase_order":
                access.restore()
            elif case == "final_advance":
                access.advance()
            elif case == "incomplete":
                access.backward((models["model"](target.torch.tensor(2.0)) - 1).square())
                access.perturb(0.1)
            else:
                original(access, models)
                if case == "restored_raises":
                    raise RuntimeError("failed after restoring")
                if case == "missing_gradient":
                    models["model"].weight.grad = None
                else:
                    with target.torch.no_grad():
                        models["model"].weight.add_(0.1)

        _, run = prepare(changed(invalid))
        report = await run.run()
        oracle.check_unverified(report)
        reports[case] = {"kind": "unverified", "report": report}

    def restored_failure(*args):
        original(*args)
        raise RuntimeError("restored but not handed back")

    _, run = prepare(changed(restored_failure))
    run.ready.state.image.units["weights"].outcomes = deque(maxlen=0)
    report = await run.run()
    oracle.check_unverified(report)
    assert report["diagnostics"] == 0
    reports["unverified_no_history"] = {"kind": "unverified", "report": report}

    for cancel, fail in ((False, False), (False, True), (True, False), (True, True)):
        entered, release, finished, stop = (threading.Event() for _ in range(4))
        pool = physical.Workers(1)

        async def selected(access, models, entered=entered, release=release, finished=finished, stop=stop, fail=fail, pool=pool):
            def work():
                try:
                    original(access, models)
                    entered.set()
                    if not release.wait(30):
                        raise RuntimeError("physical capture gate expired")
                    if fail:
                        raise RuntimeError("physical grant failed after restoration")
                finally:
                    finished.set()

            with pool.lifetime():
                completion = await pool.execute(work, stop)
                if completion.cancellation is not None:
                    raise completion.cancellation

        _, run = prepare(changed(selected))
        task = asyncio.create_task(run.run())
        witness = {}
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            unit = run.ready.state.image.units["weights"]
            witness = {
                "writer_held": bool(run.ready.state.writers),
                "pending_without_token": unit.unfinished and unit.window is None,
                "steps_before_return": unit.steps,
                "active_workers": pool.active,
                "run_live": not task.done(),
            }
            assert all(witness[k] for k in ("writer_held", "pending_without_token", "run_live"))
            if cancel:
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                witness["stop_seen"] = stop.is_set()
                assert not task.done() and stop.is_set() and run.ready.state.writers
        finally:
            release.set()
        try:
            report = await asyncio.wait_for(task, 10)
        except asyncio.CancelledError, target.backend.WorkFailed:
            if not cancel:
                raise
            report = run.last_report
        if cancel or fail:
            oracle.check_unverified(report, cancelled=cancel)
        else:
            oracle.check_c(report)
        assert finished.is_set() and pool.active == 0 and pool.executor is None
        if fail:
            assert any("physical grant failed" in item["message"] for item in report["errors"])
        reports[f"physical-cancel{int(cancel)}-fail{int(fail)}"] = {
            "kind": "physical",
            "cancel": cancel,
            "fail": fail,
            "witness": witness,
            "report": report,
        }

    for phase in ("optimizer", "reset"):
        _, run = prepare()
        unit = run.ready.state.image.units["weights"]
        unit.outcomes = deque(maxlen=0)
        method = "step" if phase == "optimizer" else "zero_grad"
        operation = getattr(type(unit.optimizer), method)

        def failed(self, *args, _operation=operation, **kwargs):
            _operation(self, *args, **kwargs)
            raise RuntimeError("retained failure after effects")

        with patch.object(type(unit.optimizer), method, failed):
            report = await run.run()
        assert report["status"] == "failed" and report["pending"] and report["handback_verified"] and report["released"]
        oracle.close(report["final_weight"], oracle.reference()["final_weight"], "returned C weight")
        try:
            run.ready.state.advance(unit.window, ("weights",), report["unit_outcomes"][0]["invocation"])
        except target.graph.NotReady:
            retry_rejected = True
        else:
            raise RuntimeError("retained failure incorrectly retryable")
        reports["partial_" + phase] = {"kind": "partial", "phase": phase, "retry_rejected": retry_rejected, "report": report}

    optimizer = importlib.import_module(PREFIX + ".execution_candidate.optimizer")
    with patch.object(optimizer.EagerSGD, "step", lambda self: None):
        for lowering in ("calls", "inline"):
            _, run = prepare(api.registry(lowering=lowering))
            report = await run.run()
            try:
                oracle.check_c(report)
            except AssertionError:
                pass
            else:
                raise RuntimeError("independent oracle accepted shared no-op optimizer")
            reports["wrong_optimizer_" + lowering] = {"kind": "wrong_computation", "report": report}
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("capture checker assertions must be enabled")
    before = sys._is_gil_enabled()
    output = Path(__file__).resolve().parent
    reports = asyncio.run(capture(args.binary.resolve(), output))
    (output / "reports.json").write_text(json.dumps(reports, indent=2) + "\n")
    record = {
        "python": sys.version,
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "xdsl_imported": "xdsl" in sys.modules,
        "reports": len(reports),
        "cases": {name: item["report"]["status"] for name, item in reports.items()},
    }
    (output / "capture_run.json").write_text(json.dumps(record, indent=2) + "\n")
    if before or record["gil_after"] or record["xdsl_imported"]:
        raise RuntimeError("C capture requires ordinary GIL-disabled execution without xDSL import")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
