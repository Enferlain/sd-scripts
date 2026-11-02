"""Reconstruct and run the bounded mixed pressure case from structural IR."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

from .mixed_adapter import MixedControls, prepare_mixed
from .mixed_conformance import check_mixed
from .mixed_contract import accept_mixed, build_mixed
from .mixed_dialect import parse_mixed


exercise = importlib.import_module("docs_design.models-strategy-trainer.mixed_target").exercise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--lowering", choices=("calls", "inline"), default="calls")
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--fuse", action="store_true")
    options = parser.parse_args()
    if options.prepared and not options.load:
        parser.error("--prepared requires --load")
    accepted = accept_mixed(parse_mixed(options.load.read_text()) if options.load else build_mixed())
    prepared = prepare_mixed(
        accepted,
        fuse=options.fuse,
        controls=MixedControls(block_key=6),
        lowering=options.lowering,
        saved_text=options.prepared.read_text() if options.prepared else None,
        saved_manifest=json.loads(options.manifest.read_text()) if options.manifest else None,
    )
    report = asyncio.run(exercise(prepared))
    check_mixed(report)
    result = {
        "python": platform.python_version(),
        "gil_enabled": sys._is_gil_enabled(),
        "xdsl": version("xdsl"),
        "torch": version("torch"),
        "accepted_digest": accepted.digest,
        "report": report,
    }
    if options.dump_dir:
        options.dump_dir.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("accepted.mlir", accepted.text),
            ("prepared.mlir", prepared.text),
            ("bindings.json", prepared.manifest),
            ("report.json", json.dumps(result, indent=2) + "\n"),
        ):
            (options.dump_dir / name).write_text(value)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
