"""Author or cold-load C; native verification precedes existing owner services."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

from .grant import GrantNative, registry
from .grant_author import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--initial", type=float, default=0.8)
    parser.add_argument("--clip-norm", type=float)
    parser.add_argument("--lowering", choices=("calls", "inline"), default="calls")
    args = parser.parse_args()
    if not __debug__:
        parser.error("run the independent oracle in normal Python; optimized execution is checked by a normal parent")
    before = sys._is_gil_enabled()
    native, selected = GrantNative(args.binary.resolve()), registry(lowering=args.lowering)
    artifact = (
        json.loads(args.load.read_text())
        if args.load
        else native.prepare(native.accept(build(initial=args.initial, clip_norm=args.clip_norm)), selected.describe())
    )
    verified = native.verify_saved(artifact, selected.describe())
    report = asyncio.run(native.materialize(artifact, selected).publish().run())
    oracle = importlib.import_module("docs_design.models-strategy-trainer.xdsl_candidate.grant_conformance")
    profile = verified["profile"]
    oracle.check_c(report, initial=profile["initial"], learning_rate=profile["learning_rate"], clip_norm=profile["clip_norm"])
    result = {
        "pliron": "0.20.0",
        "python": sys.version,
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "xdsl_imported": "xdsl" in sys.modules,
        "report": report,
    }
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        for name in ("accepted", "prepared"):
            (args.dump_dir / (name + ".pliron")).write_text(verified[name].rstrip("\n") + "\n")
        for name, value in (("artifact", verified), ("report", report), ("run", result)):
            (args.dump_dir / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
