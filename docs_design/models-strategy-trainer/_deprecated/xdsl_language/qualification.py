"""Retain detached cold artifacts, real child outputs and parent verdicts."""

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

from .prepare import Artifact, Policy, document, prepare
from .contracts import AccessMode
from .normalize import normalize
from .observations import compare, observe
from .qualification_cases import Case, lag_case, make_case
from .qualification_oracle import check
from .qualification_target import selected
from .qualification_products import reference_bindings
from .qualification_runtime import execute
from .reference import Reference
from .standins import schemas


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
MODES = (
    "normal",
    "late",
    "prefetched",
    "orderly-before",
    "orderly-after",
    "pending-cancel",
    "held-cancel",
    "held-failure",
    "late-producer-failure",
)


def consumer_observation(facts, prefix: str = ""):
    return observe(
        [
            fact
            for fact in facts
            if not any(owner == prefix + "producer" for owner, _ in fact.invocation)
            and fact.owner not in {prefix + "run", prefix + "activities"}
        ]
    )


def source_hashes() -> dict[str, str]:
    reports = [
        ROOT / "openspec/changes/rework-model-strategy-trainer/research" / name
        for name in ("ir_architecture_report.md", "xdsl_language_0_1_proposal.md")
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (*sorted(PACKAGE.glob("*.py")), *reports)
    }


async def collect_schedules(output: Path) -> list[dict]:
    folder = output / "schedules"
    folder.mkdir()
    records = []
    for name in ("lifetime", "extra-scope", "data-only"):
        case = make_case(name)
        plan = prepare(case.module, selected().manifest, Policy())
        for seed in range(8):
            runtime, report = await execute(plan, case.initial, prefix=case.prefix, seed=seed)
            check(report, name, prefix=case.prefix)
            reference = Reference(normalize(case.module), case.initial, reference_bindings(), schemas=schemas())
            rng = random.Random(seed)
            for _ in range(30000):
                enabled = reference.enabled()
                if not enabled:
                    break
                reference.step(rng.choice(enabled))
            assert reference.settled and not reference.errors
            compare(consumer_observation(reference.facts, case.prefix), consumer_observation(runtime.facts, case.prefix))
            observed = observe(runtime.facts)
            assert all(observed.positions[before] < observed.positions[after] for before, after in observed.required_before)
            raw = document(
                {"name": name, "seed": seed, "runtime": report, "target_facts": runtime.facts, "reference_facts": reference.facts}
            ).encode()
            path = folder / f"{name}-{seed}.json.gz"
            path.write_bytes(gzip.compress(raw, mtime=0))
            records.append(
                {
                    "file": str(path.relative_to(output)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "target_facts": len(runtime.facts),
                    "reference_facts": len(reference.facts),
                    "verdict": "consumer facts and causal edges match",
                }
            )
    return records


def write_case(case: Case, folder: Path) -> Artifact:
    artifact = prepare(case.module, selected().manifest, Policy()).artifact
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "accepted.mlir").write_text(artifact.accepted)
    (folder / "normalized.mlir").write_text(artifact.normalized)
    (folder / "prepared.json").write_text(artifact.prepared)
    (folder / "artifact.json").write_text(artifact.to_json())
    return artifact


def file_hashes(folder: Path, relative_to: Path) -> dict[str, str]:
    return {
        str(path.relative_to(relative_to)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def child(artifact: Path, choices: dict, folder: Path, *, optimized: bool = False) -> dict:
    if not __debug__:
        raise RuntimeError("parent qualification checks require assertions enabled")
    folder.mkdir(parents=True, exist_ok=True)
    options = folder / "choices.json"
    options.write_text(document(choices))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(PACKAGE.parent), *sys.path))
    command = [sys.executable, *(["-O"] if optimized else []), "-m", "xdsl_language.qualification_child", str(artifact), str(options)]
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=45, check=False)
    (folder / "stdout.json").write_text(completed.stdout)
    (folder / "stderr").write_text(completed.stderr)
    receipt = {"command": command, "exit": completed.returncode, "parent_assertions": __debug__}
    (folder / "process.json").write_text(document(receipt))
    if completed.returncode:
        raise AssertionError(f"child failed ({folder}): {completed.stderr}")
    report = json.loads(completed.stdout)
    observed = report["environment"]
    assert observed["optimized"] == optimized
    assert observed["authoring_blocked"] and not observed["authoring_loaded"]
    assert not observed["gil_before"] and not observed["gil_after"] and observed["forced_gil"] is None
    return report


def collect(output: Path, *, full: bool = True) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("qualification output must be new or empty; failed attempts are retained")
    output.mkdir(parents=True, exist_ok=True)
    frozen_sources = source_hashes()
    frozen_artifacts: dict[str, str] = {}
    verdicts = []
    for name in ("lifetime", "extra-scope", "data-only"):
        modes = MODES if full and name != "data-only" else ("normal",)
        for mode in modes:
            case = make_case(name, windows=3 if mode == "orderly-after" else 2)
            artifact_folder = output / "artifacts" / (name + ("-three-windows" if mode == "orderly-after" else ""))
            if not artifact_folder.exists():
                write_case(case, artifact_folder)
                frozen_artifacts.update(file_hashes(artifact_folder, output))
            choices = {"name": name, "initial": case.initial, "prefix": case.prefix, "mode": mode, "seed": 7}
            for optimized in (False, True):
                label = f"{name}-{mode}-{'optimized' if optimized else 'normal'}"
                report = child(artifact_folder / "artifact.json", choices, output / "runs" / label, optimized=optimized)
                check(report, name, prefix=case.prefix, mode=mode, erased=True)
                verdicts.append({"run": label, "oracle": "passed"})
    case = make_case("data-only")
    if full:
        rejections = lag_case(AccessMode.SNAPSHOT, reject_forever=True)
        folder = output / "artifacts" / "rejections"
        write_case(rejections, folder)
        frozen_artifacts.update(file_hashes(folder, output))
        for optimized in (False, True):
            suffix = "optimized" if optimized else "normal"
            report = child(
                folder / "artifact.json",
                {"name": "rejections", "initial": rejections.initial, "mode": "rejections", "seed": 7},
                output / "runs" / ("rejections-" + suffix),
                optimized=optimized,
            )
            check(report, "rejections", mode="rejections", erased=True)
            verdicts.append({"run": "rejections-" + suffix, "oracle": "passed"})
    base = output / "artifacts" / "data-only"
    artifact = Artifact.from_json((base / "artifact.json").read_text())
    prepared = json.loads(artifact.prepared)
    # A valid JSON document with a forged realization claim, not malformed input.
    prepared["target"]["workers"] = 99
    forged = replace(artifact, prepared=document(prepared))
    forged_path = output / "forged.json"
    forged_path.write_text(forged.to_json())
    forged_hash = hashlib.sha256(forged_path.read_bytes()).hexdigest()
    for optimized in (False, True):
        suffix = "optimized" if optimized else "normal"
        choices = {"name": case.name, "initial": case.initial, "prefix": "", "mode": "normal", "seed": 7, "wrong_score": True}
        report = child(base / "artifact.json", choices, output / "runs" / ("wrong-implementation-" + suffix), optimized=optimized)
        assert report["owner"]["completed"] and not report["owner"]["failures"]
        try:
            check(report, case.name, erased=True)
        except AssertionError as failed:
            assert failed.args == ((0.0, 1),), "wrong implementation failed an unrelated oracle obligation"
            verdicts.append({"run": "wrong-implementation-" + suffix, "oracle": "rejected wrong arithmetic", "mismatch": str(failed)})
        else:
            raise AssertionError("independent oracle accepted the wrong implementation")
        report = child(forged_path, {**choices, "corrupt": True}, output / "runs" / ("corrupted-claim-" + suffix), optimized=optimized)
        assert report["rejected"] and report["factory_calls"] == []
        verdicts.append({"run": "corrupted-claim-" + suffix, "oracle": "rejected before resource construction"})
    schedules = asyncio.run(collect_schedules(output)) if full else []
    if source_hashes() != frozen_sources:
        raise RuntimeError("source/report bytes changed during qualification; retain this attempt and start a new one")
    if file_hashes(output / "artifacts", output) != frozen_artifacts or hashlib.sha256(forged_path.read_bytes()).hexdigest() != forged_hash:
        raise RuntimeError("artifact bytes changed during qualification; retain this attempt and start a new one")
    record = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "verdicts": verdicts,
        "source_report_hashes": frozen_sources,
        "schedules": schedules,
        "artifact_hashes": frozen_artifacts,
        "negative_artifact_hashes": {"forged.json": forged_hash},
        "limits": "CPU host owner target only; bounded seeded interleavings, not exhaustive. No device, performance or full language/production qualification.",
    }
    (output / "qualification.json").write_text(document(record))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    record = collect(options.output.resolve(), full=not options.smoke)
    print(document({"output": str(options.output), "verdicts": len(record["verdicts"])}), end="")


if __name__ == "__main__":
    main()
