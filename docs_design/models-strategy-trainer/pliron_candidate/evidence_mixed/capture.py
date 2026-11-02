"""Retain actual mixed executions; reuse the tested pressure driver, not its oracles."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

candidate = importlib.import_module("docs_design.models-strategy-trainer.pliron_candidate.mixed")
helpers = importlib.import_module("docs_design.models-strategy-trainer.pliron_candidate.test_mixed_runtime")
MixedNative, describe, registry = candidate.MixedNative, candidate.describe, candidate.registry
build = importlib.import_module("docs_design.models-strategy-trainer.pliron_candidate.mixed_author").build
held, ready, target = helpers.held, helpers.ready, helpers.target


def erased_source(original):
    class Discarded(list):
        def append(self, item):
            pass

    def create():
        source = original()
        for name, value in vars(source).items():
            if isinstance(value, list):
                setattr(source, name, Discarded())
        return source

    return Discarded, create


async def archive(native, lowering, interrupt):
    original_source, original_learning = target.MixedSource, target.LearningFacts
    discarded, source_factory = erased_source(original_source)
    target.MixedSource = source_factory
    target.LearningFacts = lambda: original_learning(contributions=discarded(), windows=discarded())
    controls = target.MixedControls(block_key=6, history_limit=0, fail_physical=interrupt)
    task = None
    try:
        run = ready(native, lowering=lowering, fuse=True, controls=controls).publish()
        task = asyncio.create_task(run.run())
        async with asyncio.timeout(10):
            while not (
                run.prepared.bridge.checkpoint.is_set()
                and controls.entered.is_set()
                and run.ready.state.owners["mixed.source"].revision == 3
            ):
                if task.done():
                    raise RuntimeError("archive probe ended before held read/checkpoint")
                await asyncio.sleep(0.001)
        run.stop()
        while not run.ready.state.feeds["mixed.left"].closed:
            await asyncio.sleep(0)
        witness = {"running": not task.done(), "protected": bool(run.ready.state.readers)}
        if interrupt:
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0.005)
                if task.done() or not run.ready.state.readers:
                    raise RuntimeError("archive probe released physical work early")
        controls.release.set()
        if interrupt:
            with contextlib.suppress(asyncio.CancelledError, BaseExceptionGroup):
                await asyncio.wait_for(task, 10)
            report = run.last_report
        else:
            report = await asyncio.wait_for(task, 10)
        unit = next(iter(run.ready.state.image.units.values()))
        owner = {
            "weight": float(unit.parameters[0].detach()),
            "outcomes": [
                {
                    "invocation": outcome.invocation,
                    "optimizer": outcome.optimizer,
                    "reset": outcome.reset,
                    "inputs": [
                        {
                            "channel": use.channel,
                            "ordinal": use.key,
                            "sources": [(stamp.participant_ref, stamp.binding, stamp.numerical_epoch) for stamp in use.provenance],
                        }
                        for use in outcome.input_handoffs
                    ],
                }
                for outcome in sorted(unit.outcomes, key=lambda item: item.invocation)
            ],
        }
        labels = {ref: name for name, ref in run.ready.state.accepted.participant_refs.items()}
        for outcome in owner["outcomes"]:
            for use in outcome["inputs"]:
                use["sources"] = [(labels.get(ref, ref), binding, epoch) for ref, binding, epoch in use["sources"]]
        return {"kind": "archive", "report": report, "owner": owner, "witness": witness, "interrupted": interrupt}
    finally:
        controls.release.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        target.MixedSource, target.LearningFacts = original_source, original_learning


async def exclusion(native, mode):
    calls = 0
    heartbeats, feedback_at, cancelled_at = [], [], []
    limit = 20000
    task = None

    def cancel():
        cancelled_at.append(calls)
        if task is None:
            raise RuntimeError("exclusion cancellation has no owning task")
        task.cancel()

    def filtered(key, revision):
        nonlocal calls
        calls += 1
        if calls == 1:
            asyncio.get_running_loop().call_soon(lambda: heartbeats.append(calls))
            if mode == "cancel":
                asyncio.get_running_loop().call_soon(cancel)
        if revision and not feedback_at:
            feedback_at.append(calls)
        if calls == limit:
            raise RuntimeError("finite exclusion probe complete")
        return mode == "feedback" and key == 0

    selected = registry()
    bindings = dict(selected.bindings)
    bindings["dynamic_filter_v1"] = replace(bindings["dynamic_filter_v1"], invoke=filtered)
    run = ready(native, selected=replace(selected, bindings=bindings)).publish()
    task = asyncio.create_task(run.run())
    with contextlib.suppress(asyncio.CancelledError, BaseExceptionGroup):
        await task
    report = run.last_report
    return {
        "kind": "exclusion",
        "mode": mode,
        "calls": calls,
        "limit": limit,
        "heartbeats": heartbeats,
        "first_feedback_at": feedback_at,
        "cancelled_at": cancelled_at,
        "result": {key: report[key] for key in ("status", "released", "steps", "pending", "errors")},
        "revision": report["source"]["revision"],
        "active": report["source"]["active"],
    }


async def capture(native, destination):
    reports = {}
    for lowering in ("calls", "inline"):
        for fuse in (False, True):
            options = {"lowering": lowering, "fuse": fuse}
            saved = native.prepare(native.accept(build()), describe(registry(), **options))
            folder = destination / "cases" / f"{lowering}-f{int(fuse)}"
            folder.mkdir(parents=True, exist_ok=True)
            for name, text in (
                ("artifact.json", json.dumps(saved, indent=2)),
                ("accepted.pliron", saved["accepted"]),
                ("prepared.pliron", saved["prepared"]),
            ):
                (folder / name).write_text(text.rstrip() + "\n")
            prepared = native.materialize(saved, controls=target.MixedControls(block_key=6, history_limit=0), **options)
            reports[folder.name] = {"kind": "normal", "report": await target.exercise(prepared)}
    for interrupt in (False, True):
        prepared = ready(
            native, lowering="inline", fuse=True, controls=target.MixedControls(block_key=6, history_limit=0, fail_physical=True)
        )
        report, witness = await held(prepared, interrupt=interrupt)
        reports[f"physical-{int(interrupt)}"] = {"kind": "physical", "report": report, "witness": witness, "interrupted": interrupt}
    for case in ("identity", "stamp", "member"):
        controls = target.MixedControls(corrupt_key=1, corrupt_stamp=case == "stamp", swap_member=case == "member", history_limit=0)
        reports["admission-" + case] = {"kind": "admission", "report": await ready(native, controls=controls).publish().run()}
    for symbol in ("encode_v1", "mixed_loss_v1", "dynamic_filter_v1", "learned_bias_v1"):
        selected = registry()
        bindings = dict(selected.bindings)
        bindings[symbol] = replace(bindings[symbol], invoke=lambda *args: None)
        reports[symbol + "-invalid"] = {
            "kind": "invalid",
            "symbol": symbol,
            "report": await ready(native, selected=replace(selected, bindings=bindings)).publish().run(),
        }
    for lowering in ("calls", "inline"):
        for interrupt in (False, True):
            reports[f"archive-{lowering}-{int(interrupt)}"] = await archive(native, lowering, interrupt)
    for mode in ("empty", "feedback", "cancel"):
        reports["exclusion-" + mode] = await exclusion(native, mode)
    for lowering in ("calls", "inline"):
        selected = registry()
        bindings = dict(selected.bindings)

        async def wrong(value):
            return 2 * value + 3

        bindings["double_v1"] = replace(bindings["double_v1"], invoke=wrong)
        prepared = ready(
            native, selected=replace(selected, bindings=bindings), lowering=lowering, fuse=True, controls=target.MixedControls(block_key=6)
        )
        reports["wrong-" + lowering] = {"kind": "wrong", "report": await target.exercise(prepared)}
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("capture requires assertions enabled; optimized children are separate tests")
    args.out.mkdir(parents=True, exist_ok=True)
    before = sys._is_gil_enabled()
    reports = asyncio.run(capture(MixedNative(args.binary), args.out))
    (args.out / "reports.json").write_text(json.dumps(reports, indent=2) + "\n")
    receipt = {
        "python": sys.version,
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "reports": len(reports),
        "mixed_adapter_imported": any(name.endswith(".mixed_adapter") for name in sys.modules),
        "driver": "Existing target.exercise and test pressure helpers; all retained reports are actual executions.",
    }
    (args.out / "capture_run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
