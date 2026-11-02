"""Cold-load and execute the mixed pressure case through native Pliron checks."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

from .mixed import MixedNative, describe, registry, target_module
from .mixed_author import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--lowering", choices=("calls", "inline"), default="calls")
    options = parser.parse_args()
    before = sys._is_gil_enabled()
    native, selected = MixedNative(options.binary), registry()
    settings = {"lowering": options.lowering, "fuse": options.fuse}
    artifact = (
        json.loads(options.load.read_text()) if options.load else native.prepare(native.accept(build()), describe(selected, **settings))
    )
    target = target_module()
    prepared = native.materialize(artifact, selected, controls=target.MixedControls(block_key=6), **settings)
    report = asyncio.run(target.exercise(prepared))
    importlib.import_module("docs_design.models-strategy-trainer.xdsl_candidate.mixed_conformance").check_mixed(report)
    result = {"python": sys.version, "pliron": "0.20.0", "gil_before": before, "gil_after": sys._is_gil_enabled(), "report": report}
    if options.dump_dir:
        options.dump_dir.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("artifact.json", json.dumps(artifact, indent=2) + "\n"),
            ("accepted.pliron", artifact["accepted"].rstrip() + "\n"),
            ("prepared.pliron", artifact["prepared"].rstrip() + "\n"),
            ("run.json", json.dumps(result, indent=2) + "\n"),
        ):
            (options.dump_dir / name).write_text(value)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
