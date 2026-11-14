"""Retain cold process observations and independent product/lifetime verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from . import product_case, product_reference
from .prepare import Policy, document, prepare


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]


def check(report: dict, *, wrong: bool = False, corrupt: bool = False) -> None:
    if corrupt:
        if not report.get("rejected") or report["factories"]:
            raise RuntimeError("cold corruption crossed readiness/factory boundary")
        return
    scenario, kind, derived = report["scenario"], report["kind"], report["derived"]
    normal = scenario == "normal"
    expected = product_reference.expected(kind, derived) if normal else 0.0
    if report["result"] != expected:
        raise ValueError("independent numerical expectation failed")
    if wrong:
        raise RuntimeError("wrong implementation unexpectedly satisfied oracle")
    if report["raised"] == normal or report["live_products"] or not report["drained"] or not report["history_empty"]:
        raise RuntimeError("product lifetime/drain boundary failed")
    completion_closure = scenario in ("cancel_at_completion", "failure_at_completion")
    names = [] if scenario == "cancel_make" or completion_closure else ["derive", "make"] if derived else ["make"]
    if [item["source"] for item in report["receipts"]] != names:
        raise RuntimeError("retained source/release order mismatch")
    disposition = "released" if normal else "failed" if scenario == "failure" else "cancelled"
    version = {"owner": "storage", "incarnation": 1, "lineage": "initial", "ordinal": 0}
    for receipt in report["receipts"]:
        if receipt["ordinal"] != 0 or receipt["disposition"] != disposition or receipt["versions"] != [version]:
            raise RuntimeError("retained dependency/disposition provenance failed")
    if completion_closure:
        if not report["conditions"].get("completion_closed_before_admission") or report["consumer_calls"]:
            raise RuntimeError("closing completion checkpoint authorized dependent admission")
    elif not normal:
        conditions = report["conditions"]
        if any(
            not conditions[name]
            for name in ("physical_entered", "storage_protected", "run_waits_for_physical", "writer_blocked", "no_early_receipts")
        ):
            raise RuntimeError("physical retention witness failed")
        if conditions["products_before_settlement"] != len(names):
            raise RuntimeError("physical consumer did not retain every dependency")
        failed = scenario in ("cancel_failure", "failure")
        if report["storage_unusable"] != failed or report["writer_acquired"] == failed:
            raise RuntimeError("owner usability was confused with physical settlement")


def collect(folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise RuntimeError("qualification requires an empty output directory")
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in PACKAGE.glob("*.py")}
    (folder / "source-hashes.json").write_text(document(sources))
    artifacts = folder / "artifacts"
    artifacts.mkdir()
    jobs = []
    for kind in ("buffer", "autograd"):
        for derived in (False, True):
            name = kind + ("-derived" if derived else "-direct")
            plan = prepare(
                product_case.program(kind, derived=derived), product_case.manifest(kind, derived=derived), Policy(fuse_local=True)
            )
            artifact = artifacts / (name + ".json")
            artifact.write_text(plan.artifact.to_json())
            ref = product_reference.reference(kind, derived=derived, history=False)
            ref.run()
            if not ref.settled or ref.retained or ref.resources["result"].value != product_reference.expected(kind, derived):
                raise RuntimeError("independent logical product reference failed")
            (artifacts / (name + "-reference.json")).write_text(
                document({"value": ref.resources["result"].value, "receipts": ref.product_receipts})
            )
            for optimized in (False, True):
                for scenario in (
                    "normal",
                    "cancel",
                    "cancel_failure",
                    "failure",
                    "cancel_make",
                    "cancel_at_completion",
                    "failure_at_completion",
                ):
                    jobs.append((artifact, {"kind": kind, "derived": derived, "scenario": scenario}, optimized, name + "-" + scenario))
                if not derived:
                    jobs.append(
                        (artifact, {"kind": kind, "derived": derived, "scenario": "normal", "wrong": True}, optimized, name + "-wrong")
                    )
            forged = json.loads(plan.artifact.prepared)
            next(fragment for fragment in forged["fragments"] if fragment["source"] == "make")["realization"]["retains"] = []
            corrupt = artifacts / (name + "-forged.json")
            corrupt.write_text(replace(plan.artifact, prepared=document(forged)).to_json())
            for optimized in (False, True):
                jobs.append(
                    (corrupt, {"kind": kind, "derived": derived, "scenario": "normal", "corrupt": True}, optimized, name + "-corrupt")
                )

    def run(job):
        artifact, options, optimized, name = job
        child = folder / (name + ("-optimized" if optimized else "-normal"))
        child.mkdir()
        choices = child / "choices.json"
        choices.write_text(document(options))
        command = [
            sys.executable,
            *(["-O"] if optimized else []),
            "-m",
            "xdsl_language.product_child",
            str(artifact.resolve()),
            str(choices.resolve()),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(PACKAGE.parent), *sys.path))
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=90, check=False)
        (child / "stdout.json").write_text(result.stdout)
        (child / "stderr").write_text(result.stderr)
        (child / "process.json").write_text(document({"command": command, "exit": result.returncode}))
        if result.returncode:
            raise RuntimeError(f"cold product child failed at {child}: {result.stderr}")
        report = json.loads(result.stdout)
        env = report["environment"]
        if (
            env["gil_before"]
            or env["gil_after"]
            or env["forced_gil"] is not None
            or env["optimized"] != optimized
            or env["authoring_loaded"]
            or env["blocked"] != ["product_case", "product_reference", "fixtures"]
        ):
            raise RuntimeError("cold ordinary-import/authoring boundary failed")
        wrong = options.get("wrong", False)
        if wrong:
            if report["raised"] or report["live_products"] or not report["drained"]:
                raise RuntimeError("wrong implementation failed before a valid completed numerical result")
            try:
                check(report, wrong=True)
            except ValueError:
                verdict = "wrong numerical implementation rejected"
            else:
                raise RuntimeError("independent negative control did not reject")
        else:
            check(report, corrupt=options.get("corrupt", False))
            verdict = "corruption rejected before factories" if options.get("corrupt") else "passed"
        if not options.get("corrupt") and not wrong and report["receipts"]:
            damaged = json.loads(json.dumps(report))
            damaged["receipts"][0]["versions"][0]["ordinal"] = 99
            try:
                check(damaged)
            except RuntimeError:
                pass
            else:
                raise RuntimeError("corrupted retained provenance escaped oracle")
        (child / "verdict.json").write_text(document({"verdict": verdict}))
        return {"case": name, "optimized": optimized, "verdict": verdict}

    with ThreadPoolExecutor(max_workers=2) as pool:
        verdicts = list(pool.map(run, jobs))
    current = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in PACKAGE.glob("*.py")}
    if current != sources:
        raise RuntimeError("sources changed during qualification")
    summary = {
        "processes": len(verdicts),
        "verdicts": verdicts,
        "scope": "scoped CPU storage/physical uses; no escaping ownership, GPU or reclamation claim",
    }
    (folder / "qualification.json").write_text(document(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    result = collect(options.output.resolve())
    print(json.dumps({"processes": result["processes"], "output": str(options.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
