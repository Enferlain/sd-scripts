"""Reproduce late-input/prefetched-input witnesses around the existing engine."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .composition import CompositionNative
from .composition_author import build
from .composition_conformance import check


async def until(predicate: Callable[[], bool], *, seconds: float = 10) -> None:
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.001)


async def pressure(
    native: CompositionNative,
    artifact: dict[str, Any],
    *,
    mode: str = "late",
    fail: bool = False,
    cancel: bool = False,
    erase: bool = False,
):
    runtime = importlib.import_module(__name__.rsplit(".", 1)[0] + ".composition_runtime")
    release_producer, release_evaluation = threading.Event(), threading.Event()
    controls = runtime.Controls(
        late_record=mode == "late",
        prefetch=mode == "prefetched",
        hold_keys={6: release_producer},
        evaluation_gate=release_evaluation,
        fail_key=6 if fail else None,
        erase_histories=erase,
    )
    run = native.materialize(artifact, controls=controls)
    if erase:
        controls.second_member_gate = asyncio.Event()
    task = asyncio.create_task(run.run())
    try:
        if erase:
            await asyncio.wait_for(controls.first_member.wait(), 10)
            await until(lambda: run.ready.state.image.units["unit"].unfinished)
            run.erase_histories()
            controls.erasure_points.append("partial_window")
            controls.second_member_gate.set()
        if mode == "late":
            await asyncio.wait_for(run.signals.event(run.spec["training"]).wait(), 10)
            assert controls.requested.is_set() and run.evaluation.receipt is None
            assert not run.signals.event(run.spec["group"]).is_set() and not task.done()
            model = run.ready.state.image.bindings[run.spec["learner"]].model
            assert model.training and not run.ready.state.readers
            run.journal.mark(
                "pending_request_witness",
                producer_live=bool(run.ready.state.active_producers),
                group_live=not run.signals.event(run.spec["group"]).is_set(),
            )
            if erase:
                run.erase_histories()
                controls.erasure_points.append("pending_request")
            controls.allow_record.set()
        await until(controls.evaluation_entered.is_set)
        await until(lambda: controls.key_entered.get(6, threading.Event()).is_set())
        await until(lambda: run.source.active.get(4) == "claimed")
        assert run.source.active == {4: "claimed", 5: "ready", 6: "physical"}
        if erase:
            run.erase_histories()
            controls.erasure_points.append("physical_use")
        run.journal.mark("aggregate_bound_witness", active=sorted(run.source.active))
        release_evaluation.set()
        await asyncio.wait_for(run.signals.event(run.spec["group"]).wait(), 10)
        await until(lambda: controls.stop_signals[6].is_set())
        assert run.evaluation.completed and not task.done()
        assert run.ready.state.readers and run.ready.state.active_producers
        run.journal.mark(
            "shutdown_witness",
            producer_physical_live=6 in run.source.physical,
            protected=bool(run.ready.state.readers),
            producer_stopped=controls.stop_signals[6].is_set(),
            run_live=not task.done(),
        )
        if cancel:
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
        release_producer.set()
        try:
            report = await asyncio.wait_for(task, 10)
        except asyncio.CancelledError, BaseExceptionGroup:
            if not cancel or run.last_report is None:
                raise
            report = run.last_report
        return report
    finally:
        if controls.second_member_gate is not None:
            controls.second_member_gate.set()
        controls.allow_record.set()
        release_evaluation.set()
        release_producer.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--mode", choices=("late", "prefetched"), default="late")
    parser.add_argument("--load", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--cancel", action="store_true")
    parser.add_argument("--erase-histories", action="store_true")
    args = parser.parse_args()
    native = CompositionNative(args.binary)
    artifact = json.loads(args.load.read_text()) if args.load else native.prepare(native.accept(build()))
    report = asyncio.run(pressure(native, artifact, mode=args.mode, fail=args.fail, cancel=args.cancel, erase=args.erase_histories))
    if not args.cancel:
        check(report, failed_after_evaluation=args.fail, histories=not args.erase_histories)
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        for name in ("accepted", "prepared"):
            (args.dump_dir / (name + ".pliron")).write_text(artifact[name] + "\n")
        (args.dump_dir / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
        (args.dump_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "weight", "steps", "evaluation", "released")}, indent=2))


if __name__ == "__main__":
    main()
