"""Collect finished baseline runs into compact, reviewable records.

    uv run python collect_runs.py <runs-root> <destination>

``<runs-root>`` holds one folder per run as written by the pinned trainer
(benchmark report, Hydra folder, TensorBoard logs, saved product). For each run
this writes ``<destination>/<run>/`` with the composed config, the Markdown
report and ``summary.json``: timing, memory, per-step loss and learning rate,
and a check of the saved product.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import sys
from collections import Counter
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalars(log_root: Path) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    for folder in sorted(log_root.glob("*/training")):
        accumulator = EventAccumulator(str(folder), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags()["scalars"]:
            series[tag] = [event.value for event in accumulator.Scalars(tag)]
    return series


def product(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
        handle.seek(0)
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    metadata = header.pop("__metadata__", {})
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "tensors": len(header),
        "dtypes": dict(Counter(entry["dtype"] for entry in header.values())),
        "elements": sum(_count(entry["shape"]) for entry in header.values()),
        "metadata_keys": sorted(metadata),
    }


def _count(shape: list[int]) -> int:
    total = 1
    for size in shape:
        total *= size
    return total


def collect(run: Path, destination: Path) -> dict[str, object] | None:
    reports = sorted(run.glob("benchmark_report_*.json"))
    if not reports:
        return None
    report = json.loads(reports[-1].read_text())
    monitor = report.get("resource_monitor") or {}
    start, end = monitor.get("session_start") or {}, monitor.get("session_end") or {}
    epoch = next((row for row in monitor.get("phases", []) if row.get("phase") == "training.epoch.0"), {})
    series = scalars(run / "logs")
    target = destination / run.name
    target.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "overrides.yaml"):
        source = run / "hydra" / ".hydra" / name
        if source.exists():
            shutil.copyfile(source, target / name)
    shutil.copyfile(reports[-1].with_suffix(".md"), target / "benchmark_report.md")
    summary = {
        "run": run.name,
        "status": report.get("status"),
        "config_name": report.get("hydra_config_name"),
        "git_sha": start.get("git_sha"),
        "git_dirty": start.get("git_dirty"),
        "mode": report.get("mode_name"),
        "strategy": report.get("strategy_name"),
        "optimizer": report.get("optimizer_name"),
        "global_step": report.get("global_step"),
        "train_seconds_per_step": report.get("train_seconds_per_step"),
        "total_duration_s": report.get("total_duration_s"),
        "milestones": (report.get("runtime_trace") or {}).get("milestones"),
        "phase_totals": (report.get("runtime_trace") or {}).get("phase_totals"),
        "system": report.get("system"),
        "memory_mb": {
            "allocated_end": end.get("gpu_allocated_mb"),
            "reserved_end": end.get("gpu_reserved_mb"),
            "training_epoch_peak_allocated": epoch.get("gpu_peak_allocated_mb"),
            "training_epoch_reserved_end": epoch.get("gpu_reserved_end_mb"),
            "cpu_rss_end": end.get("cpu_rss_mb"),
        },
        "series": series,
        "products": [product(path) for path in sorted(run.glob("*.safetensors"))],
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    root, destination = Path(sys.argv[1]), Path(sys.argv[2])
    for run in sorted(path for path in root.iterdir() if path.is_dir()):
        summary = collect(run, destination)
        if summary is not None:
            losses = summary["series"].get("loss/average") or [float("nan")]  # type: ignore[union-attr]
            print(run.name, summary["status"], summary["global_step"], summary["train_seconds_per_step"], losses[-1])
