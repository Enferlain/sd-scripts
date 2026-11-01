"""Capture actual paired B/opaque controls, without constructing a shared IR."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        parser.error("independent oracle assertions must be enabled")
    before = sys._is_gil_enabled()
    prefix = "docs_design.models-strategy-trainer"
    controls = importlib.import_module(prefix + ".xdsl_candidate.test_matched_controls")
    bridge = importlib.import_module(prefix + ".pliron_candidate.bridge")
    api = bridge.Native(args.binary.resolve())
    output = Path(__file__).resolve().parent
    receipts = []
    for framework in ("xdsl", "pliron"):
        for variant in ("plain", "fused", "opaque_unfused"):
            effects = []
            if framework == "xdsl":
                if variant == "opaque_unfused":
                    authored, selected = controls.xdsl_opaque(effects)
                else:
                    authored, selected = controls.fixture.build(), controls.fixture.registry()
                accepted = controls.preparation.accept(authored)
                ready = controls.preparation.prepare(accepted, selected, fuse=variant == "fused")
                artifact = {
                    "framework": framework,
                    "accepted": accepted.text,
                    "prepared": ready.text,
                    "manifest": json.loads(ready.manifest),
                    "fuse": variant == "fused",
                }
                ready = controls.preparation.prepare(
                    accepted, selected, fuse=variant == "fused", saved_text=artifact["prepared"], saved_manifest=artifact["manifest"]
                )
            else:
                if variant == "opaque_unfused":
                    authored = controls.native_pair.build(opaque=True)
                    selected = controls.native_pair.registry(opaque=True, effects=effects)
                else:
                    authored, selected = controls.native_author.build(), controls.fixture.registry()
                ready, artifact = api.prepare(api.accept(authored), selected, fuse=variant == "fused")
                ready, _ = api.load(artifact["accepted"], artifact["prepared"], artifact["manifest"], selected)
            report = asdict(asyncio.run(controls.fixture.execute(ready, stop_after=4, diagnostic_limit=0)).report)
            if variant == "opaque_unfused":
                controls.extension_oracle.check_extension(report)
                if not effects:
                    raise RuntimeError("opaque control did not execute its declared effect")
            else:
                controls.oracle.check_b(report)
            directory = output / "controls" / framework / variant
            directory.mkdir(parents=True, exist_ok=True)
            for name, value in (("artifact.json", artifact), ("report.json", report)):
                (directory / name).write_text(json.dumps(value, indent=2) + "\n")
            receipts.append(
                {
                    "framework": framework,
                    "variant": variant,
                    "horizon": ready.horizon,
                    "capacity": ready.capacity,
                    "groups": ready.local.source_groups,
                    "status": report["status"],
                    "tickets": len(report["tickets"]),
                    "opaque_effects": effects,
                    "artifact": str((directory / "artifact.json").relative_to(output)),
                    "report": str((directory / "report.json").relative_to(output)),
                }
            )
    fixture = output.parent / "comparison_fixtures.json"
    result = {
        "schema": 1,
        "bead": "sd-scripts-syv.31.25.1",
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "python": sys.version,
        "torch_imported": "torch" in sys.modules,
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "controls": receipts,
        "limits": "Live/cold controls only, not full matched coverage, transformation alignment or performance measurement. Negative controls and physical progress/drain gates are in test_matched_controls.py and the retained test result.",
    }
    (output / "control_setup.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
