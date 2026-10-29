"""Author or load structural IR, prepare, execute, and retain inspectable facts."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .conformance import check_b
from .dialects import parse
from .fixture import build, execute, registry
from .prepare import accept, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--stop-after", type=int, default=4)
    options = parser.parse_args()
    if options.stop_after < 1:
        parser.error("--stop-after must be positive")
    start = perf_counter()
    accepted = accept(parse(options.load.read_text()) if options.load else build())
    accepted_seconds = perf_counter() - start
    start = perf_counter()
    prepared = prepare(accepted, registry())
    prepared_seconds = perf_counter() - start
    if options.manifest and json.loads(options.manifest.read_text()) != json.loads(prepared.manifest):
        raise ValueError("binding manifest differs from available registry")
    execution = asyncio.run(execute(prepared, stop_after=options.stop_after))
    report = asdict(execution.report)
    check_b(report, horizon=prepared.horizon)
    result = {
        "python": platform.python_version(),
        "gil_enabled": sys._is_gil_enabled(),
        "xdsl": version("xdsl"),
        "accepted_digest": accepted.digest,
        "accept_seconds": accepted_seconds,
        "prepare_seconds": prepared_seconds,
        "execute_seconds": execution.seconds,
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
