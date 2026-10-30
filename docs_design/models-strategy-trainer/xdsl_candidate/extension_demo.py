"""Independent local operation/schema through B's existing preparation/runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from .dialects import parse
from .extension_conformance import check_extension
from .fixture import execute
from .pair_extension import build_extension, extension_registry
from .prepare import accept, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--fuse", action="store_true")
    options = parser.parse_args()
    registry = extension_registry()
    accepted = accept(parse(options.load.read_text(), extensions=registry.dialects) if options.load else build_extension())
    prepared = prepare(accepted, registry, fuse=options.fuse)
    if options.manifest and json.loads(options.manifest.read_text()) != json.loads(prepared.manifest):
        raise ValueError("installed extension manifest differs")
    report = asdict(asyncio.run(execute(prepared)).report)
    check_extension(report)
    result = {
        "python": platform.python_version(),
        "gil_enabled": sys._is_gil_enabled(),
        "xdsl": version("xdsl"),
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
