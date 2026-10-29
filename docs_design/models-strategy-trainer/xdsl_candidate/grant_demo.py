"""Run C from Python authoring or saved xDSL and a versioned binding manifest."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .grant_adapter import prepare_grant
from .grant_conformance import check_c
from .grant_contract import accept_grant, build_grant
from .grant_dialect import AdvanceGrantedOp, parse_grant
from .model_contract import one
from .model_dialect import ModelOp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--clip-norm", type=float)
    options = parser.parse_args()
    if options.load and options.clip_norm is not None:
        parser.error("--clip-norm authors a new document; loaded IR supplies its own clipping bound")
    started = perf_counter()
    module = parse_grant(options.load.read_text()) if options.load else build_grant(clip_norm=options.clip_norm)
    accepted = accept_grant(module)
    accept_seconds = perf_counter() - started
    started = perf_counter()
    prepared = prepare_grant(accepted)
    prepare_seconds = perf_counter() - started
    if options.manifest and json.loads(options.manifest.read_text()) != json.loads(prepared.manifest):
        raise ValueError("binding manifest differs from installed grant registry")
    started = perf_counter()
    report = asyncio.run(prepared.publish().run())
    execute_seconds = perf_counter() - started
    model, advance = one(module, ModelOp), one(module, AdvanceGrantedOp)
    check_c(
        report,
        initial=model.initial.value.data,
        learning_rate=advance.learning_rate.value.data,
        clip_norm=advance.clip_norm.value.data if advance.clip_norm is not None else None,
    )
    result = {
        "python": platform.python_version(),
        "gil_enabled": sys._is_gil_enabled(),
        "xdsl": version("xdsl"),
        "torch": version("torch"),
        "accepted_digest": accepted.digest,
        "accept_seconds": accept_seconds,
        "prepare_seconds": prepare_seconds,
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
