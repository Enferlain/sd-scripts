"""Matched GIL-on/off Python input work through the actual CPU candidate.

This is a bounded experiment, not a data pipeline or performance acceptance
gate. Workers normalize/tokenize/pack synthetic records; the run admits their
actual results and executes the same tiny training computation in every case.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import sysconfig
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch

from .examples import scale
from .graph import (
    Advance,
    Block,
    Call,
    Channel,
    Choice,
    Contract,
    Differentiate,
    Join,
    Output,
    OwnedState,
    Producer,
    Repeat,
    Run,
    Together,
    Unit,
    Use,
)
from .prepare import Context, Engine, ProviderContext, prepare
from .state import Product
from .workers import NotEntered, PhysicalFailed, Workers


class ProcessingStopped(InterruptedError):
    """This selected input implementation observed its worker stop request."""


@dataclass(frozen=True)
class PackedText:
    key: int
    sample_ids: tuple[int, ...]
    packs: tuple[tuple[int, ...], ...]
    segments: tuple[tuple[tuple[int, int, int], ...], ...]
    checksum: int

    @property
    def token_count(self) -> int:
        return sum(map(len, self.packs))


@dataclass(frozen=True)
class WorkTiming:
    start_ns: int
    end_ns: int
    cpu_ns: int
    thread: int


def process_text(key: int, records: tuple[tuple[int, str], ...], budget: int, stop: threading.Event) -> tuple[PackedText, WorkTiming]:
    if budget < 1:
        raise ValueError("Packing budget must be positive")
    start, cpu = time.perf_counter_ns(), time.thread_time_ns()
    packs, spans, tokens, boundaries = [], [], [], []
    checksum = 0
    for sample_id, text in records:
        if stop.is_set():
            raise ProcessingStopped("Input processing was stopped")
        words, word = [], []
        for character in text:
            if character.isalnum():
                word.append(character.lower())
            elif word:
                words.append("".join(word))
                word.clear()
        if word:
            words.append("".join(word))
        ids = []
        for value in words:
            token = 2166136261
            for character in value:
                token = ((token ^ ord(character)) * 16777619) & 0xFFFFFFFF
            ids.append(token)
            checksum = (checksum * 33 + token) & 0xFFFFFFFF
        offset = 0
        while offset < len(ids):
            count = min(budget - len(tokens), len(ids) - offset)
            tokens.extend(ids[offset : offset + count])
            boundaries.append((sample_id, offset, count))
            offset += count
            if len(tokens) == budget:
                packs.append(tuple(tokens))
                spans.append(tuple(boundaries))
                tokens.clear()
                boundaries.clear()
    if tokens:
        packs.append(tuple(tokens))
        spans.append(tuple(boundaries))
    result = PackedText(key, tuple(sample for sample, _ in records), tuple(packs), tuple(spans), checksum)
    timing = WorkTiming(start, time.perf_counter_ns(), time.thread_time_ns() - cpu, threading.get_ident())
    return result, timing


def corpus(jobs: int, records: int) -> tuple[tuple[tuple[int, str], ...], ...]:
    vocabulary = ("Blue", "objects", "beside", "a", "window", "TEXTURE", "soft", "light", "motion", "sample", "with", "details")
    return tuple(
        tuple(
            (
                job * records + row,
                ", ".join(vocabulary[(row + job + index * 7) % len(vocabulary)] for index in range(64 + row % 17)) + f". {job}-{row}!",
            )
            for row in range(records)
        )
        for job in range(jobs)
    )


@dataclass
class ProviderStats:
    timings: list[WorkTiming] = field(default_factory=list)
    max_pending: int = 0


@dataclass
class TrainingStats:
    inputs: list[tuple[int, int, int]] = field(default_factory=list)


def arrangement(data: tuple[tuple[tuple[int, str], ...], ...], workers: int, budget: int, window: int = 4) -> Run:
    async def source(context: ProviderContext):
        stats = context.states["provider"]
        assert isinstance(stats, ProviderStats)
        pool = Workers(workers)  # Selected provider owns its private execution lifetime.
        with pool.lifetime():
            async with asyncio.TaskGroup() as tasks:
                pending = {}

                async def execute(key, stop):
                    try:
                        finished = await pool.execute(lambda: process_text(key, data[key], budget, stop), stop)
                    except NotEntered as error:
                        raise error.cause from error
                    except PhysicalFailed as error:
                        if error.cancellation is not None and isinstance(error.cause, ProcessingStopped):
                            raise error.cancellation from error
                        raise  # Keep real processing failure AND interruption evidence.
                    if finished.cancellation is not None:
                        raise finished.cancellation
                    return finished.value

                def submit(key):
                    stop = threading.Event()
                    pending[key] = tasks.create_task(execute(key, stop))
                    stats.max_pending = max(stats.max_pending, len(pending))

                for key in range(min(window, len(data))):
                    submit(key)
                for key in range(len(data)):
                    value, timing = await pending.pop(key)
                    stats.timings.append(timing)
                    # Computation may finish out of order. This provider's
                    # selected publication order preserves bounded admission.
                    await context.publish("text", Product(key, value))
                    following = key + window
                    if following < len(data):
                        submit(following)

    def admissible(product, projection):
        value = product.value
        return (
            isinstance(value, PackedText)
            and value.key == product.key
            and value.sample_ids == tuple(sample for sample, _ in data[value.key])
            and all(len(pack) <= budget for pack in value.packs)
        )

    def objective(context: Context, value: PackedText):
        stats = context.states["training"]
        assert isinstance(stats, TrainingStats)
        stats.inputs.append((value.key, value.token_count, value.checksum))
        feature = torch.tensor(value.token_count / 1000.0 + (value.checksum % 97) / 97.0)
        return (context.models["model"](feature) - 0.1).square()

    block = Block(
        "learn",
        (
            Join("admit", ("text",), (Output("packed", PackedText),), admissible),
            Call("objective", objective, ("packed",), (Output("loss", torch.Tensor),), (Use("model"),), ("training",)),
            Differentiate("backward", "loss", ("weights",), "gradient"),
            Advance("advance", "gradient", ("weights",)),
        ),
    )
    return Run(
        Together(
            "run",
            (
                Producer("text-production", ("text",), source, states=("provider",), stop_when="learning", remaining="discard"),
                Repeat(
                    "learning",
                    (block,),
                    lambda counts, state: Choice("learn", counts.get("learn", 0)) if counts.get("learn", 0) < len(data) else None,
                    "policy",
                ),
            ),
        ),
        (scale("model", 0.25),),
        (Unit("weights", (("model", "weight"),), learning_rate=0.00001),),
        (OwnedState("provider", ProviderStats), OwnedState("training", TrainingStats), OwnedState("policy", lambda: None)),
        (Channel("text", 2),),
    )


async def trial(workers: int, jobs: int, records: int, budget: int, lowering: Literal["calls", "inline"] = "calls") -> dict:
    data = corpus(jobs, records)  # Fixture creation and preparation are outside timing.
    ready = prepare(Contract().accept(arrangement(data, workers, budget)), lowering=lowering).publish()
    stop, delays = asyncio.Event(), []

    async def observe():
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            due = loop.time() + 0.005
            try:
                await asyncio.wait_for(stop.wait(), 0.005)
            except TimeoutError:
                delays.append(max(0.0, loop.time() - due))

    monitor = asyncio.create_task(observe())
    start = time.perf_counter_ns()
    try:
        await Engine(ready).run()
    finally:
        elapsed = time.perf_counter_ns() - start
        stop.set()
        await monitor
    state = ready.state
    provider, training = state.owners["provider"], state.owners["training"]
    assert isinstance(provider, ProviderStats) and isinstance(training, TrainingStats)
    assert state.image is not None and state.completed
    assert len(training.inputs) == jobs and [row[0] for row in training.inputs] == list(range(jobs))
    assert state.image.units["weights"].steps == jobs
    assert provider.max_pending <= 4 and state.feeds["text"].high_water <= 2
    assert not state.progress.leases and not state.progress.waits and not state.progress.demands
    state.capture()
    cpu = sum(timing.cpu_ns for timing in provider.timings)
    span = max(timing.end_ns for timing in provider.timings) - min(timing.start_ns for timing in provider.timings)
    ordered_delays = sorted(delays)
    return {
        "wall_s": elapsed / 1e9,
        "records_per_s": jobs * records * 1e9 / elapsed,
        "worker_cpu_s": cpu / 1e9,
        "worker_cpu_per_wall": cpu / elapsed,
        "worker_cpu_per_span": cpu / span,
        "worker_threads": len({timing.thread for timing in provider.timings}),
        "wake_delay_p95_ms": ordered_delays[min(len(delays) - 1, int(0.95 * len(delays)))] * 1000 if delays else None,
        "wake_delay_max_ms": max(delays) * 1000 if delays else None,
        "max_private_pending": provider.max_pending,
        "max_ready": state.feeds["text"].high_water,
        "outcome": {
            "inputs": training.inputs,
            "weight": float(state.image.bindings["model"].model.get_parameter("weight").detach()),
            "steps": jobs,
        },
        "work_ns": [asdict(timing) for timing in provider.timings],
    }


def sources() -> dict[str, str]:
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("parallel_work_probe.py", "prepare.py", "state.py", "progress.py", "optimizer.py", "workers.py")
    }


def measure(args) -> dict:
    assert sysconfig.get_config_var("Py_GIL_DISABLED") == 1
    expected = args.gil == "on"
    assert sys._is_gil_enabled() == expected
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    asyncio.run(trial(args.workers, 2, 16, args.budget, args.lowering))
    result = asyncio.run(trial(args.workers, args.jobs, args.records, args.budget, args.lowering))
    assert sys._is_gil_enabled() == expected
    assert "triton" not in sys.modules and "torch._dynamo" not in sys.modules
    result.update({"workers": args.workers, "gil": args.gil, "source_hashes": sources()})
    result["environment"] = {
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "load": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "torch_threads": torch.get_num_threads(),
        "gil_enabled_after": sys._is_gil_enabled(),
        "python_gil_override": os.environ.get("PYTHON_GIL"),
        "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip() if Path("/sys/fs/cgroup/cpu.max").exists() else None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--gil", choices=("on", "off"), default="off")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--records", type=int, default=512)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--lowering", choices=("calls", "inline"), default="calls")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.jobs, args.records, args.budget, args.runs) < 1:
        parser.error("Work sizes and run count must be positive")
    if args.measure:
        result = measure(args)
    else:
        assert __spec__ is not None
        cases = (("on", 1), ("off", 1), ("on", 2), ("off", 2))
        samples = []
        environment = os.environ.copy()
        environment.pop("PYTHON_GIL", None)
        for repeat in range(args.runs):
            for gil, workers in cases if repeat % 2 == 0 else reversed(cases):
                command = [
                    sys.executable,
                    *(("-X", "gil=1") if gil == "on" else ()),
                    "-m",
                    __spec__.name,
                    "--measure",
                    "--gil",
                    gil,
                    "--workers",
                    str(workers),
                    "--jobs",
                    str(args.jobs),
                    "--records",
                    str(args.records),
                    "--budget",
                    str(args.budget),
                    "--lowering",
                    args.lowering,
                ]
                child = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=180)
                if child.returncode:
                    raise RuntimeError(child.stdout + child.stderr)
                sample = json.loads(child.stdout)
                if samples and sample["outcome"] != samples[0]["outcome"]:
                    raise AssertionError("Matched runs changed admitted inputs or optimization outcomes")
                samples.append(sample)
                print(
                    f"run={repeat + 1} gil={gil} workers={workers}: {sample['wall_s']:.3f}s, CPU/span={sample['worker_cpu_per_span']:.2f}",
                    flush=True,
                )
        result = {"arguments": vars(args) | {"output": str(args.output) if args.output else None}, "samples": samples}
        for gil, workers in cases:
            values = [sample["wall_s"] for sample in samples if sample["gil"] == gil and sample["workers"] == workers]
            print(f"median gil={gil} workers={workers}: {statistics.median(values):.3f}s", flush=True)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n")
    elif args.measure:
        print(encoded)


if __name__ == "__main__":
    main()
