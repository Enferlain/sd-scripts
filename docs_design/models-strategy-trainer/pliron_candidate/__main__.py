"""Reproduce B from Python authoring or cold Pliron documents."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .author import build
from .bridge import Native
from . import pair_extension

_prefix = "docs_design.models-strategy-trainer.xdsl_candidate"
fixture = importlib.import_module(_prefix + ".fixture")
oracle = importlib.import_module(_prefix + ".conformance")
extension_oracle = importlib.import_module(_prefix + ".extension_conformance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--extension", action="store_true", help="use independent pair/frozen-resource extension")
    parser.add_argument("--opaque", action="store_true", help="keep a valid opaque-effect stage as a grouping barrier")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--stop-after", type=int, default=4)
    parser.add_argument("--load", type=Path, help="load a saved artifact.json and reverify its raw IR and bindings")
    parser.add_argument("--dump-dir", type=Path)
    args = parser.parse_args()
    if args.stop_after < 1 or args.horizon < 1:
        parser.error("stop-after and horizon must be positive")
    if args.load and args.fuse:
        parser.error("--fuse selects preparation; a loaded plan already records its groups")
    if args.opaque and not args.extension:
        parser.error("--opaque selects the extension's diagnostic barrier case")
    if not __debug__:
        parser.error("run this oracle driver with assertions enabled; optimized runtime is tested by a normal parent")
    before = sys._is_gil_enabled()
    native = Native(args.binary.resolve())
    registry = pair_extension.registry(opaque=args.opaque) if args.extension else fixture.registry()
    started = perf_counter()
    if args.load:
        stored = json.loads(args.load.read_text())
        prepared, artifact = native.load(stored["accepted"], stored["prepared"], stored["manifest"], registry)
    else:
        text = pair_extension.build(horizon=args.horizon, opaque=args.opaque) if args.extension else build(horizon=args.horizon)
        accepted = native.accept(text)
        prepared, artifact = native.prepare(accepted, registry, fuse=args.fuse)
    preparation_seconds = perf_counter() - started
    execution = asyncio.run(fixture.execute(prepared, stop_after=args.stop_after, diagnostic_limit=0))
    report = asdict(execution.report)
    if args.extension:
        extension_oracle.check_extension(report)
    else:
        oracle.check_b(report, horizon=prepared.horizon)
    result = {
        "pliron": artifact["pliron"],
        "python": sys.version,
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "torch_imported": "torch" in sys.modules,
        "status": report["status"],
        "tickets": len(report["tickets"]),
        "dispatch_groups": prepared.local.source_groups,
        "local_dispatches": report["local_dispatches"],
        "analysis": artifact["analysis"],
        "resource_analysis": artifact["resource_analysis"],
        "preparation_seconds_including_processes": preparation_seconds,
        "session_seconds": execution.seconds,
    }
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("accepted.pliron", artifact["accepted"]),
            ("prepared.pliron", artifact["prepared"]),
            ("bindings.json", prepared.manifest),
        ):
            (args.dump_dir / name).write_text(value)
        for name, value in (("artifact.json", artifact), ("report.json", report), ("run.json", result)):
            (args.dump_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
