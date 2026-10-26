"""Measure two access realizations of the same prepared CPU blocks.

Both keep the shared lease, freshness, effects and failure rules. The prepared
variant omits region borrowing and empty input demands in ordinary blocks.
These are local diagnostics, not production or whole-training performance gates.
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
from datetime import UTC, datetime
from pathlib import Path

import torch

from .examples import joint
from .graph import Contract
from .lowering_cost_probe import cheap, invoke, memory, timed
from .prepare import prepare


def environment():
    folder = Path(__file__).resolve().parent
    files = ("access_cost_probe.py", "lowering_cost_probe.py", "prepare.py", "state.py", "access.py", "progress.py", "graph.py")
    return {
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_git": torch.version.git_version,
        "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "gil_enabled": sys._is_gil_enabled(),
        "dynamo_imported": "torch._dynamo" in sys.modules,
        "triton_imported": "triton" in sys.modules,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "source_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in files},
        "gc_during_timing": False,
        "host_load_and_cpu_affinity_controlled": False,
    }


def qualify_environment() -> None:
    if "PYTHON_GIL" in os.environ or sys._is_gil_enabled() or "torch._dynamo" in sys.modules or "triton" in sys.modules:
        raise RuntimeError("This probe requires ordinary GIL-disabled imports without Dynamo/Triton or a GIL override")


async def worker(iterations: int, samples: int, warmup: int, run_index: int):
    if not __debug__:
        raise RuntimeError("Optimized Python would remove this diagnostic's numerical assertions")
    qualify_environment()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = datetime.now(UTC).isoformat()
    descriptions = {"cheap": Contract().accept(cheap()), "joint": Contract().accept(joint())}
    variants = {
        name: {kind: prepare(accepted, lowering="inline", access=kind).publish() for kind in ("general", "prepared")}
        for name, accepted in descriptions.items()
    }
    qualify_environment()
    measured_environment = environment()
    cases = {}
    for name, preparations in variants.items():
        count = iterations if name == "cheap" else min(iterations, 300)
        for ready in preparations.values():
            await invoke(ready, warmup)
        assert all(ready.access == kind for kind, ready in preparations.items())
        values: dict[str, list[float]] = {"general": [], "prepared": []}
        orderings = []
        for sample in range(samples):
            order = ("general", "prepared") if (run_index + sample) % 2 == 0 else ("prepared", "general")
            orderings.append(order)
            for kind in order:
                values[kind].append(await timed(preparations[kind], count))
        total = warmup + samples * count
        general, specialized = preparations["general"], preparations["prepared"]
        assert general.state.invocation == specialized.state.invocation == total
        assert general.state.owners == specialized.state.owners
        if name == "cheap":
            assert general.state.owners["counter"] == {"calls": 4 * total, "last": 4}
        else:
            left, right = general.state.image, specialized.state.image
            assert left is not None and right is not None
            assert all(runtime.steps == total for image in (left, right) for runtime in image.units.values())
            for participant in left.bindings:
                for path, tensor in left.bindings[participant].model.state_dict().items():
                    assert torch.equal(tensor, right.bindings[participant].model.state_dict()[path])
        assert all(not ready.state.progress.leases and not ready.state.progress.demands for ready in preparations.values())
        savings = [a - b for a, b in zip(values["general"], values["prepared"], strict=True)]
        ratios = [b / a for a, b in zip(values["general"], values["prepared"], strict=True)]
        median_saving = statistics.median(savings)
        cases[name] = {
            "iterations_per_sample": count,
            "access_realizations": {kind: ready.access for kind, ready in preparations.items()},
            "warmup_each": warmup,
            "order": orderings,
            "ns_per_invocation": values,
            "median_ns": {kind: statistics.median(row) for kind, row in values.items()},
            "paired_saving_ns": savings,
            "median_saving_ns": median_saving,
            "paired_saving_mad_ns": statistics.median(abs(value - median_saving) for value in savings),
            "median_paired_prepared_over_general": statistics.median(ratios),
            "memory": {kind: await memory(ready) for kind, ready in preparations.items()},
            "conformance": "equal numerical/owner results and invocation/unit progress; leases and demands drained",
        }
    qualify_environment()
    return {
        "run_index": run_index,
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "environment": measured_environment,
        "gil_enabled_after": sys._is_gil_enabled(),
        "imports_after": {"dynamo_imported": "torch._dynamo" in sys.modules, "triton_imported": "triton" in sys.modules},
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    if min(args.runs, args.iterations, args.samples, args.warmup) < 1:
        parser.error("counts must be positive")
    if args.worker:
        result = asyncio.run(worker(args.iterations, args.samples, args.warmup, args.run_index))
    else:
        child_environment = dict(os.environ)
        child_environment.pop("PYTHON_GIL", None)
        runs = []
        for index in range(args.runs):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "docs_design.models-strategy-trainer.execution_candidate.access_cost_probe",
                    "--worker",
                    "--run-index",
                    str(index),
                    "--iterations",
                    str(args.iterations),
                    "--samples",
                    str(args.samples),
                    "--warmup",
                    str(args.warmup),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=child_environment,
            )
            runs.append(json.loads(completed.stdout))
        result = {
            "scope": "matched inline CPU root/block execution including shared protection, scopes, effects and bounded diagnostic history",
            "difference": "prepared ordinary blocks omit retained-region borrowing and empty input demands; protection is unchanged",
            "claims": "no live-trained producer or snapshot comparison; not G3.7, production, GPU or whole-training performance evidence",
            "runs_launched_serially": True,
            "runs": runs,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
