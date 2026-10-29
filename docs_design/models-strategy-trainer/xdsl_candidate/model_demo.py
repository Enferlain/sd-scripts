"""Run A from Python authoring or a saved xDSL document and binding manifest."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .model_adapter import prepare_model
from .model_conformance import check_a
from .model_contract import accept_model, build_model
from .model_dialect import parse_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    options = parser.parse_args()
    started = perf_counter()
    accepted = accept_model(parse_model(options.load.read_text()) if options.load else build_model())
    accepted_seconds = perf_counter() - started
    started = perf_counter()
    prepared = prepare_model(accepted)
    prepared_seconds = perf_counter() - started
    if options.manifest and json.loads(options.manifest.read_text()) != json.loads(prepared.manifest):
        raise ValueError("binding manifest differs from installed model registry")
    started = perf_counter()
    report = asyncio.run(prepared.publish().run())
    execute_seconds = perf_counter() - started
    check_a(report)
    result = {
        "python": platform.python_version(),
        "gil_enabled": sys._is_gil_enabled(),
        "xdsl": version("xdsl"),
        "torch": version("torch"),
        "accepted_digest": accepted.digest,
        "accept_seconds": accepted_seconds,
        "prepare_seconds": prepared_seconds,
        "execute_seconds": execute_seconds,
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
