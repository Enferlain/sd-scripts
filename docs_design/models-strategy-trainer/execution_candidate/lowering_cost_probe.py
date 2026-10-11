"""Paired CPU lowering measurements, not a production performance gate.

The cheap case exposes framework cost without model compute hiding it. Both
paths include the same leases, live context, guards, bounded tracing and effects.
The joint case includes actual tiny Torch/SGD work; it is not dispatch-only.
No capture, GPU, threaded execution or numerical compilation is measured.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import inspect
import json
import platform
import statistics
import subprocess
import sys
import sysconfig
import time
import tracemalloc
from typing import cast

import torch

from .examples import joint
from .graph import Block, Call, Contract, Output, OwnedState, Run
from .prepare import Context, Lowering, Ready, prepare


def cheap() -> Run:
    def first(context: Context) -> int:
        owner = cast(dict[str, int], context.states["counter"])
        owner["calls"] += 1
        owner["last"] = owner["calls"] % 2
        return owner["last"]

    def next_value(context: Context, value: int) -> int:
        owner = cast(dict[str, int], context.states["counter"])
        owner["calls"] += 1
        owner["last"] = value + 1
        return owner["last"]

    return Run(
        Block(
            "cheap",
            (
                Call("first", first, outputs=(Output("v0", int),), states=("counter",)),
                *(
                    Call(f"next-{index}", next_value, (f"v{index - 1}",), (Output(f"v{index}", int),), states=("counter",))
                    for index in range(1, 4)
                ),
            ),
        ),
        states=(OwnedState("counter", lambda: {"calls": 0, "last": 0}),),
    )


async def invoke(ready: Ready, iterations: int) -> None:
    # The measured scope is prepared root/block execution. Engine entry/exit,
    # acceptance, loading and code generation are outside this measurement.
    root = ready.root
    for _ in range(iterations):
        await root()


async def timed(ready: Ready, iterations: int) -> float:
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        started = time.perf_counter_ns()
        await invoke(ready, iterations)
        return (time.perf_counter_ns() - started) / iterations
    finally:
        if was_enabled:
            gc.enable()


async def memory(ready: Ready, iterations: int = 200) -> dict[str, int]:
    # Diagnostic only, outside all timings. Cumulative allocation rate is NOT
    # inferred from these peak/retained byte counts.
    tracemalloc.start()
    try:
        gc.collect()
        starting, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        await invoke(ready, iterations)
        _, peak = tracemalloc.get_traced_memory()
        gc.collect()
        retained, _ = tracemalloc.get_traced_memory()
        return {"iterations": iterations, "peak_above_start_bytes": peak - starting, "retained_after_gc_bytes": retained - starting}
    finally:
        tracemalloc.stop()


async def worker(iterations: int, samples: int, warmup: int, run_index: int):
    accepted = {"cheap": Contract().accept(cheap()), "joint": Contract().accept(joint())}
    prepared = {
        case: {kind: prepare(description, lowering=kind).publish() for kind in ("calls", "inline")}
        for case, description in accepted.items()
    }
    # All Torch optimizer construction happens before either case is timed;
    # native-extension GIL fallback, if any, is shared by the compared paths.
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "gil_enabled": sys._is_gil_enabled(),
        "gc_during_timing": False,
        "lowering_source_sha256": hashlib.sha256(inspect.getsource(Lowering).encode()).hexdigest(),
        "cheap_source_sha256": hashlib.sha256(inspect.getsource(cheap).encode()).hexdigest(),
    }
    cases = {}
    for case, variants in prepared.items():
        count = iterations if case == "cheap" else min(iterations, 500)
        for ready in variants.values():
            await invoke(ready, warmup)
        timings: dict[str, list[float]] = {"calls": [], "inline": []}
        orders = []
        for sample in range(samples):
            order = ("calls", "inline") if (sample + run_index) % 2 == 0 else ("inline", "calls")
            orders.append(order)
            for kind in order:
                timings[kind].append(await timed(variants[kind], count))
        excess = [baseline - inline for baseline, inline in zip(timings["calls"], timings["inline"], strict=True)]
        medians = {kind: statistics.median(values) for kind, values in timings.items()}
        assert variants["calls"].state.invocation == variants["inline"].state.invocation == warmup + samples * count
        if case == "cheap":
            assert variants["calls"].state.owners == variants["inline"].state.owners
            assert variants["calls"].state.owners["counter"] == {"calls": 4 * (warmup + samples * count), "last": 4}
        else:
            for kind in variants:
                image = variants[kind].state.image
                assert image is not None and all(unit.steps == warmup + samples * count for unit in image.units.values())
        cases[case] = {
            "iterations_per_sample": count,
            "warmup_each": warmup,
            "order": orders,
            "ns_per_invocation": timings,
            "median_ns": medians,
            "paired_excess_ns": excess,
            "median_excess_ns": statistics.median(excess),
            "paired_excess_mad_ns": statistics.median(abs(value - statistics.median(excess)) for value in excess),
            "memory": {kind: await memory(ready) for kind, ready in variants.items()},
        }
    return {"run_index": run_index, "environment": environment, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-index", type=int, default=0)
    args = parser.parse_args()
    if min(args.iterations, args.samples, args.warmup, args.runs) < 1:
        parser.error("counts must be positive")
    if args.worker:
        result = asyncio.run(worker(args.iterations, args.samples, args.warmup, args.run_index))
    else:
        runs = []
        for run_index in range(args.runs):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "docs_design.models-strategy-trainer.execution_candidate.lowering_cost_probe",
                    "--worker",
                    "--iterations",
                    str(args.iterations),
                    "--samples",
                    str(args.samples),
                    "--warmup",
                    str(args.warmup),
                    "--run-index",
                    str(run_index),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            runs.append(json.loads(completed.stdout))
        result = {
            "scope": "matched prepared CPU block dispatch with guards, live contexts, protected access and bounded traces; joint includes compute",
            "retention": "trace=256 entries; unit outcomes=32; owner counter fixed-size; allocation diagnostics not timed",
            "claims": "comparative local evidence only, not G3.7 acceptance, GPU performance or free-threaded safety",
            "runs": runs,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
