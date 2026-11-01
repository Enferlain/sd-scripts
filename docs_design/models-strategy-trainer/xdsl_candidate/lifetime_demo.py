"""Execute or cold-load the matched lifetime case through its xDSL handoff."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path

from .lifetime import LifetimeCandidate
from .lifetime_author import build
from .lifetime_ir import printed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--mode", choices=("late", "prefetched"), default="late")
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--cancel", action="store_true")
    parser.add_argument("--erase-histories", action="store_true")
    args = parser.parse_args()
    if not __debug__:
        parser.error("run assertion-based oracle in normal parent Python")
    api = LifetimeCandidate()
    artifact = json.loads(args.load.read_text()) if args.load else api.prepare(api.accept(printed(build())))
    shared = importlib.import_module("docs_design.models-strategy-trainer.pliron_candidate.composition_demo")
    oracle = importlib.import_module("docs_design.models-strategy-trainer.pliron_candidate.composition_conformance")
    report = asyncio.run(shared.pressure(api, artifact, mode=args.mode, fail=args.fail, cancel=args.cancel, erase=args.erase_histories))
    if not args.cancel:
        oracle.check(report, failed_after_evaluation=args.fail, histories=not args.erase_histories)
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        for name in ("accepted", "prepared"):
            (args.dump_dir / f"{name}.mlir").write_text(artifact[name] + "\n")
        for name, value in (("artifact", artifact), ("report", report)):
            (args.dump_dir / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "weight", "steps", "evaluation", "released")}, indent=2))


if __name__ == "__main__":
    main()
