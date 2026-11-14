"""Retain real child observations, independent checks and immutable inputs."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys

from dataclasses import replace
from pathlib import Path

from . import filter_case, filter_oracle
from .filter_checks import check_products, check_report, check_settlement
from .filter_contracts import INITIAL, MANIFEST
from .filter_runtime import SCENARIOS, execute
from .local_rewrite import fuse
from .normalize import normalize
from .observations import compare, observe
from .prepare import Artifact, Policy, document, prepare
from .qualification import PACKAGE, ROOT, file_hashes, source_hashes
from .reference import Reference, ReferenceError


def child(artifact: Path, options: dict, folder: Path, optimized: bool) -> dict:
    folder.mkdir(parents=True)
    choices = folder / "choices.json"
    choices.write_text(document(options))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(PACKAGE.parent), *sys.path))
    command = [sys.executable, *(["-O"] if optimized else []), "-m", "xdsl_language.filter_child", str(artifact), str(choices)]
    run = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60, check=False)
    (folder / "stdout.json").write_text(run.stdout)
    (folder / "stderr").write_text(run.stderr)
    (folder / "process.json").write_text(document({"command": command, "exit": run.returncode}))
    if run.returncode:
        raise RuntimeError(f"cold filtering failed at {folder}: {run.stderr}")
    report = json.loads(run.stdout)
    env = report["environment"]
    if (
        env["optimized"] != optimized
        or env["authoring_loaded"]
        or env["authoring_blocked"] != ["filter_case", "fixtures", "qualification_cases"]
        or env["gil_before"]
        or env["gil_after"]
        or env["forced_gil"] is not None
    ):
        raise RuntimeError("cold child optimization, authoring isolation or ordinary-import GIL boundary failed")
    return report


def consumers(facts):
    return observe([item for item in facts if any(name == "consumers" for name, _ in item.invocation)])


async def schedules(folder: Path) -> list[dict]:
    records = []
    for fused in (False, True):
        plan = prepare(filter_case.program(), MANIFEST, Policy(workers=3, fuse_local=fused))
        for seed in range(8):
            ref = Reference(plan.verification.normalized, INITIAL, filter_oracle.bindings(), schemas=filter_oracle.SCHEMAS)
            scheduler = random.Random(seed)
            for _ in range(12000):
                enabled = ref.enabled()
                if not enabled:
                    break
                ref.step(scheduler.choice(enabled))
            if not ref.settled or ref.errors:
                raise ReferenceError(f"reference interleaving did not settle: {ref.errors}")
            check_products(ref.resources["products"].value)
            check_settlement(ref.streams["raw"].settled, ref.streams["kept"].settled)
            owner, facts, choices = await execute(plan, seed=seed)
            check_report(owner, "normal")
            compare(consumers(ref.facts), consumers(facts))
            actual = observe(facts)
            if any(actual.positions[a] >= actual.positions[b] for a, b in actual.required_before):
                raise ReferenceError("actual owner trace violated physical/disposition or completion causality")
            path = folder / f"{'fused' if fused else 'unfused'}-{seed}.json.gz"
            path.write_bytes(
                gzip.compress(
                    document(
                        {"seed": seed, "owner": owner, "choices": choices, "reference_facts": ref.facts, "owner_facts": facts}
                    ).encode(),
                    mtime=0,
                )
            )
            records.append(
                {
                    "file": str(path.relative_to(folder.parent)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "reference_facts": len(ref.facts),
                    "owner_facts": len(facts),
                    "oracle": "matched consumer observations and causal edges",
                }
            )
    return records


def collect(output: Path) -> dict:
    if not __debug__:
        raise RuntimeError("qualification parent checks must run with assertions enabled")
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new evidence folder; failed qualifications are retained")
    output.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    old = {name: file_hashes(PACKAGE / name, ROOT) for name in ("evidence_prepared", "evidence_qualification")}
    verdicts, paths = [], {}
    for fused in (False, True):
        for opaque in (False, True):
            label = ("fused" if fused else "unfused") + ("-opaque" if opaque else "")
            folder = output / "artifacts" / label
            folder.mkdir(parents=True)
            plan = prepare(filter_case.program(opaque=opaque), MANIFEST, Policy(workers=3, fuse_local=fused))
            paths[fused, opaque] = folder / "artifact.json"
            (folder / "artifact.json").write_text(plan.artifact.to_json())
            for name, content in (
                ("accepted.mlir", plan.artifact.accepted),
                ("normalized.mlir", plan.artifact.normalized),
                ("prepared.json", plan.artifact.prepared),
            ):
                (folder / name).write_text(content)
    base = Artifact.from_json(paths[False, False].read_text())
    forged_claim = json.loads(base.prepared)
    forged_claim["target"]["workers"] = 99
    claim_path = output / "corrupt-claim.json"
    claim_path.write_text(replace(base, prepared=document(forged_claim)).to_json())
    opaque = Artifact.from_json(paths[True, True].read_text())
    dishonest = {spec.name: replace(spec, unknown_effects=False).contract for spec in MANIFEST.bindings}
    group_path = output / "corrupt-group.json"
    group_path.write_text(replace(opaque, normalized=fuse(normalize(filter_case.program(opaque=True)), dishonest).text).to_json())
    artifact_hashes = file_hashes(output / "artifacts", output)
    negatives = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (claim_path, group_path)}
    for optimized in (False, True):
        suffix = "optimized" if optimized else "normal"
        for fused in (False, True):
            for scenario in SCENARIOS:
                name = f"{'fused' if fused else 'unfused'}-{scenario}-{suffix}"
                report = child(paths[fused, False], {"fused": fused, "scenario": scenario, "seed": 7}, output / "runs" / name, optimized)
                check_report(report["owner"], scenario)
                verdicts.append({"run": name, "oracle": "passed"})
            name = f"{'fused' if fused else 'unfused'}-opaque-{suffix}"
            report = child(paths[fused, True], {"fused": fused, "scenario": "normal"}, output / "runs" / name, optimized)
            check_report(report["owner"], "normal")
            verdicts.append({"run": name, "oracle": "passed valid opaque execution"})
            name = f"wrong-{'fused' if fused else 'unfused'}-{suffix}"
            report = child(
                paths[fused, False], {"fused": fused, "scenario": "normal", "wrong_double": True}, output / "runs" / name, optimized
            )
            try:
                check_report(report["owner"], "normal")
            except ReferenceError as mismatch:
                if "independent expectation" not in str(mismatch) or report["owner"]["failures"] or not report["owner"]["completed"]:
                    raise
                verdicts.append({"run": name, "oracle": "rejected actual wrong arithmetic", "mismatch": str(mismatch)})
            else:
                raise ReferenceError("independent oracle accepted wrong arithmetic")
        for label, path, fused in (("corrupt-claim", claim_path, False), ("corrupt-group", group_path, True)):
            name = label + "-" + suffix
            report = child(path, {"fused": fused, "corrupt": True}, output / "runs" / name, optimized)
            if (
                not report["rejected"]
                or report["factory_calls"]
                or (label == "corrupt-group" and "unknown effects" not in report["reason"])
            ):
                raise ReferenceError("forged preparation/group crossed the protected boundary")
            verdicts.append({"run": name, "oracle": "rejected before factories", "reason": report["reason"]})
    (output / "schedules").mkdir()
    explored = asyncio.run(schedules(output / "schedules"))
    if (
        source_hashes() != sources
        or file_hashes(output / "artifacts", output) != artifact_hashes
        or any(hashlib.sha256((output / name).read_bytes()).hexdigest() != digest for name, digest in negatives.items())
    ):
        raise RuntimeError("sources, reports or detached artifacts changed during qualification")
    if any(file_hashes(PACKAGE / name, ROOT) != hashes for name, hashes in old.items()):
        raise RuntimeError("historical qualification/preparation evidence changed")
    record = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_report_hashes": sources,
        "artifact_hashes": artifact_hashes,
        "negative_artifact_hashes": negatives,
        "historical_evidence_hashes": old,
        "verdicts": verdicts,
        "schedules": explored,
        "limits": "Bounded CPU host reference/owner evidence. Physical calls and validations remain per stage; local dispatch counts establish a different path, not speed or device/production qualification.",
    }
    (output / "qualification.json").write_text(document(record))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    record = collect(options.output)
    print(document({"runs": len(record["verdicts"]), "schedules": len(record["schedules"]), "evidence": str(options.output)}))


if __name__ == "__main__":
    main()
