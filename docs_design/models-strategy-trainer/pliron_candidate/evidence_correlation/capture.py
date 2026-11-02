"""Retain actual A numerical, admission, request and physical-use witnesses."""

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

PREFIX = "docs_design.models-strategy-trainer"


async def capture(binary: Path, output: Path) -> dict:
    api = importlib.import_module(PREFIX + ".pliron_candidate.correlation")
    author = importlib.import_module(PREFIX + ".pliron_candidate.correlation_author")
    target = importlib.import_module(PREFIX + ".model_target")
    oracle = importlib.import_module(PREFIX + ".xdsl_candidate.model_conformance")
    native = api.CorrelationNative(binary)
    reports = {}

    def prepare(selected=None, controls=None, **options):
        selected = api.registry() if selected is None else selected
        artifact = native.prepare(native.accept(author.build()), api.describe(selected, **options))
        return artifact, native.materialize(artifact, selected, controls=controls, **options).publish()

    for mode in ("calls", "inline"):
        for fuse in (False, True):
            for workers in (False, True):
                label = f"{mode}-f{int(fuse)}-w{int(workers)}"
                artifact, run = prepare(controls=target.Controls(history_limit=0), lowering=mode, fuse=fuse, workers=workers)
                report = await run.run()
                oracle.check_a(report)
                reports[label] = {"kind": "normal", "report": report}
                directory = output / "cases" / label
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
                if mode == "calls" and workers:
                    for field in ("accepted", "prepared"):
                        (directory / (field + ".pliron")).write_text(artifact[field].rstrip("\n") + "\n")
    for case in ("missing", "bad_sample", "wrong_member", "stale_source"):
        field = {"missing": "omit_sample", "bad_sample": "corrupt_sample", "wrong_member": "swap_member", "stale_source": "stale_sample"}[
            case
        ]
        for sample in (1, 2, 4, 5):
            _, run = prepare(controls=target.Controls(history_limit=0, **{field: sample}))
            report = await run.run()
            oracle.check_admission_failure(report, sample, missing=case == "missing")
            reports[f"{case}-{sample}"] = {"kind": "admission", "sample": sample, "missing": case == "missing", "report": report}
    for sample in (2, 5):
        _, run = prepare(controls=target.Controls(history_limit=0, corrupt_sample=sample))
        for unit in run.ready.state.image.units.values():
            unit.outcomes = deque(maxlen=0)
        report = await run.run()
        oracle.check_admission_failure(report, sample, missing=False)
        assert len(report["unit_outcomes"]) == 1
        reports[f"evicted-{sample}"] = {"kind": "admission", "sample": sample, "missing": False, "evicted": True, "report": report}
    for symbol in ("encode_v1", "measure_v1"):
        entered, release, finished = (threading.Event() for _ in range(3))
        selected = api.registry()
        bindings = dict(selected.bindings)

        def failed(*args, entered=entered, release=release, finished=finished):
            entered.set()
            try:
                if not release.wait(20):
                    raise TimeoutError("A physical capture gate expired")
                raise RuntimeError("A physical failure after interruption")
            finally:
                finished.set()

        bindings[symbol] = replace(bindings[symbol], invoke=failed)
        _, run = prepare(replace(selected, bindings=bindings), target.Controls(history_limit=0), fuse=True)
        task = asyncio.create_task(run.run())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            witness = {
                "protected": bool(run.ready.state.readers or run.ready.state.writers),
                "run_live": not task.done(),
                "physical_done": finished.is_set(),
            }
            assert witness == {"protected": True, "run_live": True, "physical_done": False}
            if symbol == "measure_v1":
                witness["evaluation_mode_held"] = not run.ready.state.image.bindings["learner"].model.training
                assert witness["evaluation_mode_held"]
        finally:
            release.set()
        try:
            await asyncio.wait_for(task, 10)
        except asyncio.CancelledError, BaseExceptionGroup:
            pass
        else:
            raise RuntimeError("interrupted A physical work returned normally")
        report = run.report()
        assert finished.is_set() and report["released"] and report["status"] == "cancelled"
        assert any("A physical failure" in item["message"] for item in report["errors"])
        if symbol == "measure_v1":
            witness["mode_restored"] = run.ready.state.image.bindings["learner"].model.training
            assert witness["mode_restored"]
        reports[symbol + "-physical"] = {"kind": "physical", "symbol": symbol, "witness": witness, "report": report}
    requested, entered, release = (threading.Event() for _ in range(3))
    selected = api.registry()
    bindings = dict(selected.bindings)
    original = bindings["measure_v1"].invoke

    def held(model):
        entered.set()
        if not release.wait(20):
            raise TimeoutError("A requested evaluation gate expired")
        return original(model)

    bindings["measure_v1"] = replace(bindings["measure_v1"], invoke=held)
    _, run = prepare(replace(selected, bindings=bindings), target.Controls(history_limit=0, evaluation_requested=requested))
    task = asyncio.create_task(run.run())
    try:
        assert await asyncio.to_thread(entered.wait, 10) and requested.is_set()
        observation = run.ready.state.owners["a.evaluation"].measurements[0]
        witness = {
            "requested": requested.is_set(),
            "completed": observation["completed"],
            "epoch": observation["epoch"],
            "protected": bool(run.ready.state.readers or run.ready.state.writers),
            "evaluation_mode_held": not run.ready.state.image.bindings["learner"].model.training,
        }
        assert not task.done() and not witness["completed"] and witness["protected"] and witness["evaluation_mode_held"]
    finally:
        release.set()
    report = await asyncio.wait_for(task, 10)
    oracle.check_a(report)
    witness["mode_restored"] = run.ready.state.image.bindings["learner"].model.training
    assert witness["mode_restored"]
    reports["held-evaluation"] = {"kind": "normal", "witness": witness, "report": report}
    for symbol in ("encode_v1", "loss_v1", "measure_v1"):
        selected = api.registry()
        bindings = dict(selected.bindings)
        bindings[symbol] = replace(bindings[symbol], invoke=lambda *args: None)
        _, run = prepare(replace(selected, bindings=bindings), target.Controls(history_limit=0))
        report = await run.run()
        assert report["status"] == "failed" and report["released"] and report["errors"]
        reports[symbol + "-invalid"] = {"kind": "invalid_output", "symbol": symbol, "report": report}
    selected = api.registry()
    bindings = dict(selected.bindings)
    original = bindings["loss_v1"].invoke
    bindings["loss_v1"] = replace(bindings["loss_v1"], invoke=lambda *args: original(*args) * 2)
    for mode in ("calls", "inline"):
        _, run = prepare(replace(selected, bindings=bindings), target.Controls(history_limit=0), lowering=mode)
        report = await run.run()
        try:
            oracle.check_a(report)
        except AssertionError:
            pass
        else:
            raise RuntimeError("oracle accepted shared wrong loss")
        reports["wrong-loss-" + mode] = {"kind": "wrong_computation", "report": report}
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        parser.error("capture assertions must be enabled")
    before = sys._is_gil_enabled()
    output = Path(__file__).resolve().parent
    reports = asyncio.run(capture(args.binary.resolve(), output))
    (output / "reports.json").write_text(json.dumps(reports, indent=2) + "\n")
    result = {
        "python": sys.version,
        "reports": len(reports),
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "model_adapter_imported": any(name.endswith(".model_adapter") for name in sys.modules),
        "evaluation_epochs": sorted(
            {item["report"]["evaluation"]["measurements"][0]["epoch"] for item in reports.values() if item["kind"] == "normal"}
        ),
    }
    if before or result["gil_after"] or result["model_adapter_imported"]:
        raise RuntimeError("capture requires ordinary GIL-disabled execution without xDSL A frontend")
    (output / "capture_run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
