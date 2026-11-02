"""Run native-verified A through the existing correlation/optimization owners."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

from .correlation import CorrelationNative, contracts
from .correlation_author import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--lowering", choices=("calls", "inline"), default="calls")
    parser.add_argument("--no-workers", action="store_true")
    args = parser.parse_args()
    if not __debug__:
        parser.error("run the oracle in normal Python; optimized children use a normal parent checker")
    before = sys._is_gil_enabled()
    options = {"lowering": args.lowering, "workers": not args.no_workers, "fuse": args.fuse}
    native = CorrelationNative(args.binary.resolve())
    artifact = json.loads(args.load.read_text()) if args.load else native.prepare(native.accept(build()), contracts(**options))
    report = asyncio.run(native.materialize(artifact, **options).publish().run())
    importlib.import_module("docs_design.models-strategy-trainer.xdsl_candidate.model_conformance").check_a(report)
    result = {"python": sys.version, "pliron": "0.20.0", "gil_before": before, "gil_after": sys._is_gil_enabled(), "report": report}
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        for name in ("accepted", "prepared"):
            (args.dump_dir / (name + ".pliron")).write_text(artifact[name].rstrip("\n") + "\n")
        for name, value in (("artifact", artifact), ("report", report), ("run", result)):
            (args.dump_dir / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
