"""Collect the separately frozen natural endpoint without replacing v1 samples."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from time import perf_counter_ns

from .measure import PREFIX, ROOT, metadata, provenance, run_process, summary, threads


def worker(args, protocol, base):
    api = importlib.import_module(PREFIX + ".cases")
    start = perf_counter_ns()
    frameworks = [args.framework] if args.fresh else ["xdsl", "pliron"]
    cases = {f: api.Case(args.case, f, args.binaries, measured=True) for f in frameworks}
    threads()
    imports = perf_counter_ns() - start
    rows = []
    warm = []
    for fuse in [False] if args.fresh else base["rewrite"][args.case]:
        for index in range(1 if args.fresh else protocol["warmups"] + protocol["measured_pairs"]):
            order = frameworks if index % 2 == 0 else frameworks[::-1]
            for framework in order:
                candidate = cases[framework]
                api.NATIVE_EVENTS.clear()
                row = {
                    "case": args.case,
                    "framework": framework,
                    "fuse": fuse,
                    "phase": "natural",
                    "index": index if args.fresh else index - protocol["warmups"],
                    "status": "passed",
                }
                start = perf_counter_ns()
                try:
                    report = candidate.natural_invocation(fuse)
                    row["elapsed_ns"] = perf_counter_ns() - start
                    candidate.check(report)
                    if sys._is_gil_enabled():
                        raise RuntimeError("GIL enabled")
                except BaseException as error:
                    row.setdefault("elapsed_ns", perf_counter_ns() - start)
                    row.update(status="failed", error=str(error), trace=traceback.format_exc())
                row["native"] = list(api.NATIVE_EVENTS)
                (warm if not args.fresh and index < protocol["warmups"] else rows).append(row)
    return {"rows": rows, "warmups": warm, "import_and_setup_ns": imports, "metadata": metadata()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--torch-free-python", default="/tmp/sd-scripts-xdsl-b-env/bin/python")
    parser.add_argument("--owner-python", default="/tmp/sd-scripts-pliron-composition-owner-env/bin/python")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--framework")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "endpoint_protocol.json").read_text())
    base = json.loads((ROOT / "protocol.json").read_text())
    if args.worker and (args.case not in protocol["cases"] or (args.fresh and not args.framework)):
        parser.error("--worker requires a declared --case; --fresh also requires --framework")
    if not args.worker and (args.case or args.framework or args.fresh):
        parser.error("--case/--framework/--fresh require --worker")
    if args.worker:
        result = worker(args, protocol, base)
        print(json.dumps(result))
        if any(row["status"] == "failed" for row in (*result["rows"], *result["warmups"])):
            raise SystemExit(1)
        return
    output = args.out or ROOT
    output.mkdir(parents=True, exist_ok=True)
    path = output / "natural_measurements.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite natural endpoint measurements")
    env = os.environ | {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    data = {
        "protocol_sha256": hashlib.sha256((ROOT / "endpoint_protocol.json").read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ("cases.py", "measure.py", "natural.py")
        },
        **provenance(args.binaries),
        "warm": [],
        "fresh": [],
        "failures": [],
    }
    for case in protocol["cases"]:
        python = args.torch_free_python if case in ("B", "extension") else args.owner_python
        command = [python, "-m", PREFIX + ".natural", "--worker", "--case", case, "--binaries", str(args.binaries)]
        start = perf_counter_ns()
        p = run_process(command, env=env, timeout=180)
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
        data["warm"].append(entry)
        if p.returncode or "result" not in entry:
            data["failures"].append(entry)
        print("natural warm", case, p.returncode, flush=True)
        for index in range(protocol["fresh_starts_per_candidate_case"]):
            for framework in ["xdsl", "pliron"] if index % 2 == 0 else ["pliron", "xdsl"]:
                cmd = [*command, "--fresh", "--framework", framework]
                start = perf_counter_ns()
                p = run_process(cmd, env=env, timeout=60)
                entry = {
                    "case": case,
                    "framework": framework,
                    "fuse": False,
                    "phase": "fresh_natural",
                    "index": index,
                    "elapsed_ns": perf_counter_ns() - start,
                    "status": "passed" if p.returncode == 0 else "failed",
                    "exit_code": p.returncode,
                    "stderr": p.stderr,
                    "exception": p.exception,
                    "timed_out": p.timed_out,
                    "command": cmd,
                }
                try:
                    entry["result"] = json.loads(p.stdout)
                except ValueError:
                    entry.update(status="failed", stdout=p.stdout)
                data["fresh"].append(entry)
                if entry["status"] == "failed":
                    data["failures"].append(entry)
        path.write_text(json.dumps(data, indent=2) + "\n")
        print("natural fresh", case, "completed", flush=True)
    rows = [row for entry in data["warm"] for row in entry.get("result", {}).get("rows", [])] + data["fresh"]
    data["summary"] = summary(rows)
    data["sample_failures"] = [row for row in rows if row["status"] == "failed"]
    path.write_text(json.dumps(data, indent=2) + "\n")
    if data["failures"] or data["sample_failures"]:
        raise RuntimeError("natural endpoint failures retained; no resampling")


if __name__ == "__main__":
    main()
