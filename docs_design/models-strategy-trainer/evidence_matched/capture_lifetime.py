"""Capture the existing lifetime witnesses through independent xDSL machinery."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path


def main() -> None:
    if not __debug__:
        raise RuntimeError("capture oracles require assertions enabled")
    before = sys._is_gil_enabled()
    prefix = "docs_design.models-strategy-trainer"
    api = importlib.import_module(prefix + ".xdsl_candidate.lifetime")
    author = importlib.import_module(prefix + ".xdsl_candidate.lifetime_author")
    ir = importlib.import_module(prefix + ".xdsl_candidate.lifetime_ir")
    witness = importlib.import_module(prefix + ".pliron_candidate.evidence_composition.capture")
    candidate = api.LifetimeCandidate()
    artifact = candidate.prepare(candidate.accept(ir.printed(author.build())))
    reports = asyncio.run(witness.capture_cases(candidate, artifact))
    output = Path(__file__).resolve().parent / "lifetime_xdsl"
    (output / "reports").mkdir(parents=True, exist_ok=True)
    for name in ("accepted", "prepared"):
        (output / (name + ".mlir")).write_text(artifact[name] + "\n")
    (output / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    for name, report in reports.items():
        (output / "reports" / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    result = {
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "python": sys.version,
        "reports": {name: {key: report[key] for key in ("status", "steps", "released")} for name, report in reports.items()},
    }
    (output / "capture_run.json").write_text(json.dumps(result, indent=2) + "\n")
    if before or result["gil_after"]:
        raise RuntimeError("capture requires ordinarily GIL-disabled execution")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
