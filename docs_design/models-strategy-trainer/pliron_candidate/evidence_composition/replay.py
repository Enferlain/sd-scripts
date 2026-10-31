"""Replay saved observations with framework/runtime imports forbidden."""

from __future__ import annotations

import copy
import importlib.abc
import json
import runpy
import sys
from pathlib import Path


class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = (
            "torch",
            "xdsl",
            "docs_design.models-strategy-trainer.execution_candidate",
            "docs_design.models-strategy-trainer.pliron_candidate.composition_runtime",
            "docs_design.models-strategy-trainer.pliron_candidate.composition_author",
        )
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise RuntimeError("independent replay imported " + fullname)
        return None


def main() -> None:
    if not __debug__:
        raise RuntimeError("independent oracle must run with assertions enabled")
    before = sys._is_gil_enabled()
    sys.meta_path.insert(0, BlockFrameworks())
    root = Path(__file__).resolve().parent
    oracle = runpy.run_path(str(root.parent / "composition_conformance.py"))
    reports = {p.stem: json.loads(p.read_text()) for p in (root / "reports").glob("*.json")}
    check, close = oracle["check"], oracle["close"]
    accepted = []
    for name in ("late", "prefetched", "late_failure", "erased", "stop_before", "stop_after"):
        check(
            reports[name],
            failed_after_evaluation=name == "late_failure",
            histories=name != "erased",
            windows=1 if name == "stop_before" else 2,
        )
        accepted.append(name)
    for name in ("cancel_failure", "partial_cancel", "pending_cancel", "training_failure", "evaluation_failure"):
        report = reports[name]
        assert report["released"] and report["closed"] and report["source"]["cleanup_complete"]
        assert not report["source"]["active"] and not report["source"]["physical_active"]
        if name in ("partial_cancel", "training_failure"):
            assert report["steps"] == 1 and report["pending"] and report["evaluation"] is None
            close(report["weight"], 0.77)
        else:
            assert report["steps"] == 2 and not report["pending"]
            close(report["weight"], 0.635)
        if name == "pending_cancel":
            assert report["request"]["disposition"] == "cancelled_unfulfilled" and report["evaluation"] is None
        elif name == "cancel_failure":
            assert report["evaluation_completed"] and report["source"]["terminal"]["6"] == "failed"
            assert any(e["type"] == "CancelledError" for e in report["errors"])
            assert any("physical production failure 6" in e["message"] for e in report["errors"])
        elif name == "evaluation_failure":
            assert report["evaluation"]["physical_done"] and not report["evaluation_completed"]
        accepted.append(name)
    rejected = []
    for name in ("weight", "epoch", "cause", "input", "bound", "group", "physical", "consumption", "request_disposition"):
        report = copy.deepcopy(reports["late"])
        if name == "weight":
            report["weight"] += 0.1
        elif name == "epoch":
            report["evaluation"]["epoch"] = 1
        elif name == "cause":
            report["evaluation"]["cause"]["window"] = 0
        elif name == "input":
            report["evaluation"]["key"] = 3
        elif name == "bound":
            report["source"]["high_water"] = 4
        elif name == "group":
            report["scope_clocks"]["consumers"] = report["scope_clocks"]["training"]
        elif name == "physical":
            report["evaluation"]["physical_done"] = False
        elif name == "consumption":
            report["source"]["terminal"]["5"] = "consumed"
        else:
            report["request"]["disposition"] = "cancelled_unfulfilled"
        try:
            check(report)
        except AssertionError:
            rejected.append(name)
        else:
            raise RuntimeError("oracle accepted corruption: " + name)
    assert "torch" not in sys.modules and "xdsl" not in sys.modules
    result = {
        "accepted": accepted,
        "rejected_corruptions": rejected,
        "assertions_enabled": __debug__,
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "torch_imported": False,
        "xdsl_imported": False,
    }
    (root / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
