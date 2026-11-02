"""Run the frozen CPU protocol, retaining every sample and process failure."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parent
PREFIX = "docs_design.models-strategy-trainer.evidence_comparison"


@dataclass(frozen=True)
class ProcessObservation:
    returncode: int
    stdout: str
    stderr: str
    exception: str | None = None
    timed_out: bool = False


def run_process(command, *, env, timeout) -> ProcessObservation:
    def decoded(value):
        return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout, check=False)
        return ProcessObservation(result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired as error:
        message = f"TimeoutExpired: {error}"
        return ProcessObservation(124, decoded(error.stdout), decoded(error.stderr) + "\n" + message, message, True)
    except OSError as error:
        message = f"{type(error).__name__}: {error}"
        return ProcessObservation(127, "", message, message)


def provenance(binaries):
    manifest = ROOT.parent / "pliron_candidate/native/Cargo.toml"
    return {
        "native_binary_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries.glob("rmst-pliron-*") if p.is_file() and not p.suffix
        },
        "native_binary_directory": str(binaries),
        "native_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "native_lock_sha256": hashlib.sha256(manifest.with_name("Cargo.lock").read_bytes()).hexdigest(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "cpu_model": next(
            (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
            "unknown",
        ),
        "rust_env": {name: os.environ.get(name) for name in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_PROFILE_RELEASE_OPT_LEVEL")},
    }


def quantile(values, p):
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summary(rows):
    groups = {}
    for row in rows:
        key = "|".join(str(row[k]) for k in ("case", "framework", "fuse", "phase"))
        groups.setdefault(key, []).append(row)
    result = {}
    for key, items in groups.items():
        ok = [row["elapsed_ns"] for row in items if row["status"] == "passed"]
        result[key] = {
            "attempted": len(items),
            "passed": len(ok),
            "failed": len(items) - len(ok),
            "median_ns": statistics.median(ok) if ok else None,
            "iqr_ns": quantile(ok, 0.75) - quantile(ok, 0.25) if ok else None,
        }
    return result


def metadata():
    torch = sys.modules.get("torch")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "gil": sys._is_gil_enabled(),
        "torch_imported": torch is not None,
        "torch_version": str(torch.__version__) if torch else None,
        "torch_threads": torch.get_num_threads() if torch else None,
        "torch_interop_threads": torch.get_num_interop_threads() if torch else None,
        "cuda_initialized": torch.cuda.is_initialized() if torch else False,
        "triton_native_imported": any(name.startswith("triton._C") for name in sys.modules),
        "peak_parent_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_env": {
            key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED")
        },
        "gc_enabled": gc.isenabled(),
    }


def threads():
    torch = sys.modules.get("torch")
    if torch:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)


def worker(args, protocol):
    start = perf_counter_ns()
    api = importlib.import_module(PREFIX + ".cases")
    frameworks = [args.framework] if args.framework else ["xdsl", "pliron"]
    cases = {framework: api.Case(args.case, framework, args.binaries, measured=True) for framework in frameworks}
    threads()
    imports_ns = perf_counter_ns() - start
    if args.fresh:
        candidate = cases[args.framework]
        api.NATIVE_EVENTS.clear()
        start = perf_counter_ns()
        report = candidate.invocation()
        elapsed = perf_counter_ns() - start
        candidate.check(report)
        return {
            "import_and_setup_ns": imports_ns,
            "operation_ns": elapsed,
            "native": list(api.NATIVE_EVENTS),
            "metadata": metadata(),
            "status": "passed",
        }
    samples = []
    warmups = []
    artifacts = {}
    reports = {}
    for fuse in protocol["rewrite"][args.case]:
        saved = {f: c.prepare(fuse) for f, c in cases.items()}
        for framework, artifact in saved.items():
            artifacts[framework + "-f" + str(int(fuse))] = artifact
        for phase in protocol["phases"]:
            gc.collect()
            for index in range(protocol["warmups"] + protocol["measured_pairs"]):
                order = frameworks if index % 2 == 0 else frameworks[::-1]
                for framework in order:
                    candidate = cases[framework]
                    artifact = saved[framework]
                    api.NATIVE_EVENTS.clear()
                    functions = {
                        "author": candidate.build,
                        "accept": lambda candidate=candidate: candidate.accept(candidate.build()),
                        "prepare": lambda candidate=candidate, fuse=fuse: candidate.prepare(fuse),
                        "artifact_json_roundtrip": lambda artifact=artifact: json.loads(json.dumps(artifact)),
                        "cold": lambda candidate=candidate, artifact=artifact, fuse=fuse: candidate.cold(artifact, fuse),
                        "materialize": lambda candidate=candidate, artifact=artifact, fuse=fuse: candidate.materialize(artifact, fuse),
                        "end_to_end": lambda candidate=candidate, fuse=fuse: candidate.invocation(fuse),
                    }
                    # Author construction is outside the accept span; acceptance
                    # keeps the actual frontend's print/parse/verify choreography.
                    authored = candidate.build() if phase == "accept" else None
                    if phase == "accept":
                        functions[phase] = lambda candidate=candidate, authored=authored: candidate.accept(authored)
                    row = {
                        "case": args.case,
                        "framework": framework,
                        "fuse": fuse,
                        "phase": phase,
                        "index": index - protocol["warmups"],
                        "order": order.index(framework),
                        "status": "passed",
                    }
                    start = perf_counter_ns()
                    try:
                        value = functions[phase]()
                        row["elapsed_ns"] = perf_counter_ns() - start
                        if phase == "end_to_end":
                            candidate.check(value)
                            reports[framework + "-f" + str(int(fuse))] = value
                        if sys._is_gil_enabled():
                            raise RuntimeError("GIL enabled in declared isolated process")
                    except BaseException as error:
                        row["elapsed_ns"] = perf_counter_ns() - start
                        row.update(status="failed", error=str(error), trace=traceback.format_exc())
                    row["native"] = list(api.NATIVE_EVENTS)
                    (warmups if index < protocol["warmups"] else samples).append(row)
    return {
        "samples": samples,
        "warmups": warmups,
        "artifacts": artifacts,
        "reports": reports,
        "import_and_setup_ns": imports_ns,
        "metadata": metadata(),
    }


def driver(args, protocol):
    env = os.environ | {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    output = args.out or ROOT
    output.mkdir(parents=True, exist_ok=True)
    path = output / "measurements.json"
    if path.exists():
        raise RuntimeError("measurements already exists; retain it, use a separately declared new attempt rather than overwrite")
    result = {
        "schema": 1,
        "protocol_sha256": hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest(),
        "command": sys.argv,
        "harness_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")},
        **provenance(args.binaries),
        "warm": [],
        "fresh": [],
        "failures": [],
    }
    for case in protocol["cases"]:
        python = args.torch_free_python if case in ("B", "extension") else args.owner_python
        command = [str(python), "-m", PREFIX + ".measure", "--worker", "--case", case, "--binaries", str(args.binaries)]
        start = perf_counter_ns()
        p = run_process(command, env=env, timeout=600)
        entry = {
            "case": case,
            "command": command,
            "exit_code": p.returncode,
            "wall_ns": perf_counter_ns() - start,
            "stderr": p.stderr,
            "exception": p.exception,
            "timed_out": p.timed_out,
        }
        try:
            entry["result"] = json.loads(p.stdout)
        except ValueError:
            entry["stdout"] = p.stdout
        result["warm"].append(entry)
        if p.returncode or "result" not in entry:
            result["failures"].append(entry)
        print("warm", case, "exit", p.returncode, flush=True)
        for index in range(protocol["fresh_starts_per_candidate_case"]):
            for framework in ["xdsl", "pliron"] if index % 2 == 0 else ["pliron", "xdsl"]:
                command2 = [*command, "--fresh", "--framework", framework]
                start = perf_counter_ns()
                p = run_process(command2, env=env, timeout=60)
                entry = {
                    "case": case,
                    "framework": framework,
                    "index": index,
                    "fuse": False,
                    "phase": "fresh_process",
                    "command": command2,
                    "elapsed_ns": perf_counter_ns() - start,
                    "status": "passed" if p.returncode == 0 else "failed",
                    "exit_code": p.returncode,
                    "stderr": p.stderr,
                    "exception": p.exception,
                    "timed_out": p.timed_out,
                }
                try:
                    entry["result"] = json.loads(p.stdout)
                except ValueError:
                    entry["stdout"] = p.stdout
                    entry["status"] = "failed"
                result["fresh"].append(entry)
                if entry["status"] == "failed":
                    result["failures"].append(entry)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print("fresh", case, "completed", protocol["fresh_starts_per_candidate_case"] * 2, flush=True)
    rows = [row for entry in result["warm"] for row in entry.get("result", {}).get("samples", [])] + result["fresh"]
    result["summary"] = summary(rows)
    result["sample_failures"] = [row for row in rows if row["status"] == "failed"]
    result["warmup_failures"] = [
        row for entry in result["warm"] for row in entry.get("result", {}).get("warmups", []) if row["status"] == "failed"
    ]
    path.write_text(json.dumps(result, indent=2) + "\n")
    for entry in result["warm"]:
        for name, artifact in entry.get("result", {}).get("artifacts", {}).items():
            folder = output / "cases" / entry["case"] / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
            for key in ("accepted", "prepared"):
                (folder / (key + (".mlir" if name.startswith("xdsl") else ".pliron"))).write_text(artifact[key].rstrip() + "\n")
            (folder / "report.json").write_text(json.dumps(entry["result"]["reports"].get(name), indent=2) + "\n")
    if result["failures"] or result["sample_failures"] or result["warmup_failures"]:
        raise RuntimeError("retained comparison failures; no resampling performed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--torch-free-python", default="/tmp/sd-scripts-xdsl-b-env/bin/python")
    parser.add_argument("--owner-python", default="/tmp/sd-scripts-pliron-composition-owner-env/bin/python")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--framework")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if args.worker and (args.case not in protocol["cases"] or (args.fresh and not args.framework)):
        parser.error("--worker requires a declared --case; --fresh also requires --framework")
    if not args.worker and (args.case or args.framework or args.fresh):
        parser.error("--case/--framework/--fresh require --worker")
    if args.worker:
        result = worker(args, protocol)
        print(json.dumps(result))
        if any(row["status"] == "failed" for row in (*result.get("samples", []), *result.get("warmups", []))):
            raise SystemExit(1)
    else:
        driver(args, protocol)


if __name__ == "__main__":
    main()
