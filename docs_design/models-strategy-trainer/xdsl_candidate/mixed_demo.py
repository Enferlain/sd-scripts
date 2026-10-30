"""Reconstruct and run the bounded mixed pressure case from structural IR."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

from .mixed_adapter import MixedControls, prepare_mixed
from .mixed_conformance import check_mixed
from .mixed_contract import accept_mixed, build_mixed
from .mixed_dialect import parse_mixed


async def exercise(prepared):
    if prepared.controls.block_key != 6:
        raise ValueError("mixed pressure driver requires block_key=6")
    run = prepared.publish()
    task = asyncio.create_task(run.run())
    controls = prepared.controls
    try:
        async with asyncio.timeout(10):
            while not (prepared.bridge.checkpoint.is_set() and controls.entered.is_set()):
                if task.done():
                    return await task
                await asyncio.sleep(0.001)
            source = run.ready.state.owners["mixed.source"]
            while not (len(source.updates) == 3 and any(item["key"] == 5 and item["returned"] is None for item in source.products)):
                await asyncio.sleep(0.001)
        run.stop()
        # Let cancellation reach the selected Worker before physical release.
        while not run.ready.state.feeds["mixed.left"].closed:
            await asyncio.sleep(0)
        assert not task.done() and run.ready.state.readers, "owner released a live model read"
        controls.release.set()
        return await asyncio.wait_for(task, 10)
    finally:
        controls.release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--fuse", action="store_true")
    options = parser.parse_args()
    accepted = accept_mixed(parse_mixed(options.load.read_text()) if options.load else build_mixed())
    prepared = prepare_mixed(accepted, fuse=options.fuse, controls=MixedControls(block_key=6))
    if options.manifest and json.loads(options.manifest.read_text()) != json.loads(prepared.manifest):
        raise ValueError("installed mixed binding manifest differs")
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
