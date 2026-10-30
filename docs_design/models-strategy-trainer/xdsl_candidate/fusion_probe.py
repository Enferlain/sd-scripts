"""Bounded matched local-dispatch costs, separate from complete workflow runs.

This uses the real scalar async bindings and a small validating observer. It
does not measure training, the full owner runtime or steady-state throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Any

from .conformance import check_b
from .fixture import build, execute, registry
from .local_runtime import LocalMetrics, LocalProgram
from .prepare import accept, prepare


@dataclass
class Counter:
    entered: int = 0
    physical: int = 0
    validated: int = 0
    released: int = 0


@dataclass
class ProbeFrame:
    counter: Counter

    def __enter__(self) -> ProbeFrame:
        self.counter.entered += 1
        return self

    def physical_end(self) -> None:
        self.counter.physical += 1

    def validate(self, value: Any) -> int:
        if type(value) is not int:
            raise AssertionError("invalid scalar probe output")
        self.counter.validated += 1
        return value

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None) -> None:
        self.counter.released += 1


async def batch(program: LocalProgram, count: int) -> dict[str, Any]:
    counter, metrics = Counter(), LocalMetrics()
    started = perf_counter()
    for value in range(count):
        actual = await program.run(value, lambda source, incoming: ProbeFrame(counter), metrics)
        assert actual == 2 * value + 1  # Independent arithmetic, never lowering agreement alone.
    elapsed = perf_counter() - started
    assert counter.entered == counter.physical == counter.validated == counter.released == 2 * count
    return {"seconds": elapsed, "dispatches": metrics.dispatches, "stages": asdict(counter)}


async def measure(directory: Path, count: int, pairs: int) -> dict[str, Any]:
    accepted = accept(build(horizon=1))
    prepared = {mode: prepare(accepted, registry(), fuse=mode == "fused") for mode in ("unfused", "fused")}
    cold: dict[str, list[dict[str, float]]] = {mode: [] for mode in prepared}
    hot: dict[str, list[dict[str, Any]]] = {mode: [] for mode in prepared}
    for _ in range(2):
        for artifact in prepared.values():
            await batch(artifact.local, count)
    for pair in range(pairs):
        for mode in ("unfused", "fused") if pair % 2 == 0 else ("fused", "unfused"):
            started = perf_counter()
            snapshot = accept(build(horizon=1))
            accepted_at = perf_counter()
            prepare(snapshot, registry(), fuse=mode == "fused")
            prepared_at = perf_counter()
            cold[mode].append({"accept_seconds": accepted_at - started, "prepare_seconds": prepared_at - accepted_at})
            hot[mode].append(await batch(prepared[mode].local, count))
    reports = {}
    for mode, artifact in prepared.items():
        report = asdict((await execute(artifact, diagnostic_limit=0)).report)
        check_b(report, horizon=1)
        reports[mode] = report
        (directory / f"{mode}.mlir").write_text(artifact.text)
        (directory / f"{mode}-bindings.json").write_text(artifact.manifest)
    (directory / "accepted.mlir").write_text(accepted.text)
    (directory / "reports.json").write_text(json.dumps(reports, indent=2) + "\n")
    medians = {mode: statistics.median(item["seconds"] for item in samples) for mode, samples in hot.items()}
    return {
        "scope": "isolated scalar local dispatch with real yielding bindings and a validating counter; full B conformance reports separately retained",
        "python": platform.python_version(),
        "xdsl": version("xdsl"),
        "gil_enabled": sys._is_gil_enabled(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "recorded_at": datetime.now(UTC).isoformat(),
        "count_per_batch": count,
        "pairs": pairs,
        "warmup_batches_per_mode": 2,
        "order": "alternating within pairs",
        "cold": cold,
        "hot": hot,
        "median_local_seconds": medians,
        "unfused_over_fused_ratio": medians["unfused"] / medians["fused"],
        "source_groups": {mode: artifact.local.source_groups for mode, artifact in prepared.items()},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob("*.py"))},
        "limits": [
            "One local CPU process and seven pairs by default; no statistical or production-speedup claim.",
            "Hot probe excludes complete session admission/feedback, model owners and GPU work.",
            "Both paths use the same observer contract; A now shares that guarded-stage ABI but is not timed here.",
            "Physical binding calls and per-stage validation remain; only local dispatch grouping changes.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--pairs", type=int, default=7)
    args = parser.parse_args()
    if args.count < 1 or args.pairs < 1:
        parser.error("positive count/pairs required")
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(measure(args.dump_dir, args.count, args.pairs))
    (args.dump_dir / "costs.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("scope", "gil_enabled", "median_local_seconds", "unfused_over_fused_ratio")}))


if __name__ == "__main__":
    main()
