"""Regenerate the bounded saved-case reports; run with the recorded owner env."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY))
PREFIX = "docs_design.models-strategy-trainer.pliron_candidate"


async def capture(binary: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    api = importlib.import_module(PREFIX + ".composition")
    author = importlib.import_module(PREFIX + ".composition_author")
    runtime = importlib.import_module(PREFIX + ".composition_runtime")
    driver = importlib.import_module(PREFIX + ".composition_demo")
    oracle = importlib.import_module(PREFIX + ".composition_conformance")
    native = api.CompositionNative(binary)
    artifact = native.prepare(native.accept(author.build()))
    reports = {}
    for name, options in {
        "late": {},
        "prefetched": {"mode": "prefetched"},
        "late_failure": {"fail": True},
        "cancel_failure": {"fail": True, "cancel": True},
        "erased": {"erase": True},
    }.items():
        report = await driver.pressure(native, artifact, **options)
        if name != "cancel_failure":
            oracle.check(report, failed_after_evaluation=options.get("fail", False), histories=not options.get("erase", False))
        else:
            assert report["status"] == "cancelled" and report["evaluation_completed"] and report["released"]
            assert report["source"]["terminal"][6] == "failed"
            assert any("physical production failure 6" in error["message"] for error in report["errors"])
        reports[name] = report
    for name, controls in {
        "stop_before": runtime.Controls(stop_before=1),
        "stop_after": runtime.Controls(stop_after=1),
        "training_failure": runtime.Controls(fail_contribution=3),
        "evaluation_failure": runtime.Controls(fail_evaluation=True),
    }.items():
        run = native.materialize(artifact, controls=controls)
        report = await asyncio.wait_for(run.run(), 15)
        if name.startswith("stop_"):
            oracle.check(report, windows=1 if name == "stop_before" else 2)
        else:
            assert report["status"] == "failed" and report["released"] and not report["evaluation_completed"]
            assert report["source"]["cleanup_complete"] and not report["source"]["active"]
        reports[name] = report
    for name in ("partial_cancel", "pending_cancel"):
        gate = asyncio.Event()
        controls = runtime.Controls(second_member_gate=gate) if name == "partial_cancel" else runtime.Controls(late_record=True)
        run = native.materialize(artifact, controls=controls)
        task = asyncio.create_task(run.run())
        try:
            if name == "partial_cancel":
                await asyncio.wait_for(controls.first_member.wait(), 10)
                await driver.until(lambda run=run: run.ready.state.image.units["unit"].unfinished)
            else:
                await asyncio.wait_for(run.signals.event("training").wait(), 10)
                assert controls.requested.is_set() and not run.signals.event("consumers").is_set()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, 15)
            report = run.last_report
            assert report is not None and report["status"] == "cancelled" and report["released"]
            if name == "partial_cancel":
                assert report["pending"] and report["steps"] == 1 and report["acquired"][1] == [2]
                oracle.close(report["weight"], 0.77)
            else:
                assert report["request"]["disposition"] == "cancelled_unfulfilled" and report["evaluation"] is None
                oracle.close(report["weight"], 0.635)
            reports[name] = report
        finally:
            gate.set()
            controls.allow_record.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    return artifact, reports


def save(output: Path, artifact: dict[str, Any], reports: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(exist_ok=True)
    for name in ("accepted", "prepared"):
        (output / (name + ".pliron")).write_text(artifact[name])
    for name, value in {"artifact": artifact, "bindings": artifact["manifest"]}.items():
        (output / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    for name, report in reports.items():
        (output / "reports" / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {key: report[key] for key in ("status", "steps", "released")} for name, report in reports.items()}, indent=2))


def main() -> None:
    if not __debug__:
        raise RuntimeError("capture assertions require a normal interpreter")
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    before = sys._is_gil_enabled()
    artifact, reports = asyncio.run(capture(args.binary))
    save(args.output, artifact, reports)
    after = sys._is_gil_enabled()
    if before or after:
        raise RuntimeError("recorded capture requires ordinarily GIL-disabled execution")
    (args.output / "capture_run.json").write_text(
        json.dumps({"gil_before": before, "gil_after": after, "python": sys.version}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
