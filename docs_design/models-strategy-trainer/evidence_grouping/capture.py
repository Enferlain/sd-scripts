"""Actual paired extension runs and held frozen-read failure witnesses."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import threading
from dataclasses import asdict, replace
from pathlib import Path

PREFIX = "docs_design.models-strategy-trainer"


async def capture(binary: Path, output: Path) -> dict:
    # Reuse fixture installation helpers, never a shared pre-IR program.
    cases = importlib.import_module(PREFIX + ".xdsl_candidate.test_grouping_comparison")
    runtime = importlib.import_module(PREFIX + ".xdsl_candidate.runtime")
    payloads = importlib.import_module(PREFIX + ".xdsl_candidate.payloads")
    native = cases.native_tests.Native(binary)
    reports, controls = {}, []
    for framework in ("xdsl", "pliron"):
        for opaque in (False, True):
            for fuse in (False, True):
                ready, artifact, selected, effects = cases.variant(native, framework, opaque=opaque, fuse=fuse)
                ready = cases.cold(native, framework, artifact, selected, fuse=fuse)
                report = asdict((await cases.execute(ready, diagnostic_limit=0)).report)
                cases.check_extension(report)
                label = f"{framework}-{'opaque' if opaque else 'frozen'}-{'fused' if fuse else 'unfused'}"
                directory = output / "cases" / label
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
                analysis = (
                    ready.local_analysis.describe()
                    if framework == "xdsl"
                    else {
                        "cache": "Pliron AnalysisManager/Passes",
                        "resource_analysis": artifact["resource_analysis"],
                        "observations": artifact["analysis"],
                    }
                )
                # Cold loading obtains fresh legality; it need not rerun a
                # rewrite already present in the saved document.
                reports[label] = {
                    "kind": "normal",
                    "framework": framework,
                    "opaque": opaque,
                    "groups": ready.local.source_groups,
                    "effects": effects,
                    "report": report,
                }
                controls.append(
                    {
                        "label": label,
                        "artifact": str((directory / "artifact.json").relative_to(output)),
                        "groups": ready.local.source_groups,
                        "analysis": analysis,
                    }
                )

        extension = cases.pair if framework == "xdsl" else cases.native_pair
        for fault in ("invalid_output", "domain", "undeclared", "hidden_lane"):
            selected = extension.extension_registry() if framework == "xdsl" else extension.registry()
            bindings = dict(selected.bindings)
            original = bindings["pack_pair_v1"]

            def lower(resources, fault=fault, original=original):
                selected_call = original.lower(resources)

                async def invoke(value):
                    if fault in ("domain", "undeclared"):
                        raise payloads.LocalOutcomeError(
                            "negative_input" if fault == "domain" else "unknown_outcome", "injected domain failure"
                        )
                    result = await selected_call(value)
                    return (result[0], True) if fault == "invalid_output" else (result[0], result[1] + 9)

                return invoke

            bindings["pack_pair_v1"] = replace(original, lower=lower)
            ready, _, _, _ = cases.variant(native, framework, fuse=True, selected=replace(selected, bindings=bindings))
            report = asdict((await cases.execute(ready, diagnostic_limit=0)).report)
            if fault == "hidden_lane":
                try:
                    cases.check_extension(report)
                except AssertionError:
                    pass
                else:
                    raise RuntimeError("independent oracle accepted bad hidden lane")
            else:
                assert report["status"] == "failed" and report["errors"]
                assert report["tickets"][0]["calls"][-1]["origin"] == "pair.pack"
            reports[framework + "-" + fault] = {
                "kind": "wrong_payload" if fault == "hidden_lane" else "failure",
                "framework": framework,
                "fault": fault,
                "report": report,
            }

        # The existing broader matrix also tests orderly stop, successful
        # cancellation and both dispatch modes. Retain the combined hard case
        # here: repeated cancellation while the fused pack's worker later fails.
        entered, release, finished = (threading.Event() for _ in range(3))
        selected = extension.extension_registry() if framework == "xdsl" else extension.registry()
        bindings = dict(selected.bindings)

        def held(resources, entered=entered, release=release, finished=finished):
            snapshot = resources["pair_coefficients"]

            async def invoke(value):
                def physical():
                    try:
                        assert snapshot.values == (1, 2)
                        entered.set()
                        if not release.wait(20):
                            raise TimeoutError("grouped frozen-read gate expired")
                        raise RuntimeError("physical frozen-resource failure after cancellation")
                    finally:
                        finished.set()

                return await asyncio.to_thread(physical)

            return invoke

        bindings["pack_pair_v1"] = replace(bindings["pack_pair_v1"], lower=held)
        ready, _, _, _ = cases.variant(native, framework, fuse=True, selected=replace(selected, bindings=bindings))
        session = runtime.ScalarSession(ready, diagnostic_limit=0)
        task = asyncio.create_task(session.run())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            session.stop()
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            witness = {"active_calls": session.active_calls, "run_live": not task.done(), "physical_done": finished.is_set()}
            assert witness == {"active_calls": 1, "run_live": True, "physical_done": False}
        finally:
            release.set()
            session.stop()
        try:
            await asyncio.wait_for(task, 10)
        except asyncio.CancelledError:
            pass
        else:
            raise RuntimeError("cancelled grouped read returned normally")
        assert finished.is_set() and session.active_calls == session.outstanding == 0 and session.report is not None
        report = asdict(session.report)
        assert any("physical frozen-resource failure" in error for error in report["errors"])
        reports[framework + "-physical"] = {"kind": "physical", "framework": framework, "witness": witness, "report": report}
    return {"controls": controls, "reports": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        parser.error("capture checks require assertions")
    before = sys._is_gil_enabled()
    output = Path(__file__).resolve().parent
    result = asyncio.run(capture(args.binary.resolve(), output))
    (output / "reports.json").write_text(json.dumps(result["reports"], indent=2) + "\n")
    result = {
        "schema": 1,
        "bead": "sd-scripts-syv.31.25.4",
        "controls": result["controls"],
        "reports": len(result["reports"]),
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "python": sys.version,
        "torch_imported": "torch" in sys.modules,
        "limit": "Correctness/grouping observations only; counts are not a framework performance result.",
    }
    if before or result["gil_after"] or result["torch_imported"]:
        raise RuntimeError("requires ordinary GIL-disabled Torch-free execution")
    (output / "capture_run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "controls"}, indent=2))


if __name__ == "__main__":
    main()
