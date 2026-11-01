"""Independent replay of matched controls and lifetime observations; stdlib only."""

from __future__ import annotations

import copy
import importlib
import importlib.abc
import json
import sys
from pathlib import Path

PREFIX = "docs_design.models-strategy-trainer"


class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        leaf = fullname.rsplit(".", 1)[-1]
        if (
            fullname.split(".", 1)[0] in ("torch", "xdsl")
            or fullname.startswith(PREFIX + ".execution_candidate")
            or leaf
            in {
                "runtime",
                "local_runtime",
                "prepare",
                "graph",
                "state",
                "workers",
                "fixture",
                "bridge",
                "lifetime",
                "lifetime_author",
                "lifetime_ir",
                "composition",
                "composition_author",
                "composition_runtime",
            }
        ):
            raise ImportError("independent replay imported " + fullname)
        return None


def main() -> None:
    if not __debug__:
        raise RuntimeError("oracle assertions must be enabled")
    before = sys._is_gil_enabled()
    sys.meta_path.insert(0, NoRuntime())
    b = importlib.import_module(PREFIX + ".xdsl_candidate.conformance")
    extension = importlib.import_module(PREFIX + ".xdsl_candidate.extension_conformance")
    life = importlib.import_module(PREFIX + ".pliron_candidate.composition_conformance")
    root = Path(__file__).resolve().parent
    accepted, rejected = [], []

    def rejects(name, function, report):
        try:
            function(report)
        except AssertionError:
            rejected.append(name)
        else:
            raise RuntimeError("oracle accepted deliberate corruption: " + name)

    for framework in ("xdsl", "pliron"):
        for variant in ("plain", "fused", "opaque_unfused"):
            report = json.loads((root / "controls" / framework / variant / "report.json").read_text())
            checker = extension.check_extension if variant == "opaque_unfused" else b.check_b
            checker(report)
            accepted.append(f"{framework}/{variant}")
            bad = copy.deepcopy(report)
            if variant == "opaque_unfused":
                bad["tickets"][0]["calls"][2]["output"][1] += 1
            elif variant == "plain":
                bad["tickets"][0]["calls"][0]["output"] += 1
            else:
                bad["tickets"][0]["policy_revision"] += 1
            rejects(f"{framework}/{variant}", checker, bad)
    reports = {p.stem: json.loads(p.read_text()) for p in (root / "lifetime_xdsl" / "reports").glob("*.json")}
    for name in ("late", "prefetched", "late_failure", "erased", "stop_before", "stop_after"):
        life.check(
            reports[name],
            failed_after_evaluation=name == "late_failure",
            histories=name != "erased",
            windows=1 if name == "stop_before" else 2,
        )
        accepted.append("lifetime/" + name)
    for name in ("cancel_failure", "partial_cancel", "pending_cancel", "training_failure", "evaluation_failure"):
        report = reports[name]
        assert report["released"] and report["closed"] and report["source"]["cleanup_complete"]
        assert not report["source"]["active"] and not report["source"]["physical_active"]
        assert report["status"] == ("failed" if name.endswith("failure") and name != "cancel_failure" else "cancelled")
        partial = name in ("partial_cancel", "training_failure")
        assert report["steps"] == (1 if partial else 2) and report["pending"] is partial
        life.close(report["weight"], life.expected_weights()[0 if partial else 1])
        if name == "pending_cancel":
            assert report["request"]["disposition"] == "cancelled_unfulfilled" and report["evaluation"] is None
        elif name == "cancel_failure":
            assert report["evaluation_completed"] and report["source"]["terminal"]["6"] == "failed"
            assert any(e["type"] == "CancelledError" for e in report["errors"])
            assert any("physical production failure 6" in e["message"] for e in report["errors"])
        elif name == "evaluation_failure":
            assert report["evaluation"]["physical_done"] and not report["evaluation_completed"]
        else:
            assert report["evaluation"] is None
        accepted.append("lifetime/" + name)
    for field in ("weight", "epoch", "cause", "input", "bound", "group", "physical", "consumption", "request_disposition"):
        report = copy.deepcopy(reports["late"])
        if field == "weight":
            report["weight"] += 0.1
        elif field == "epoch":
            report["evaluation"]["epoch"] = 1
        elif field == "cause":
            report["evaluation"]["cause"]["window"] = 0
        elif field == "input":
            report["evaluation"]["key"] = 3
        elif field == "bound":
            report["source"]["high_water"] = 4
        elif field == "group":
            report["scope_clocks"]["consumers"] = report["scope_clocks"]["training"]
        elif field == "physical":
            report["evaluation"]["physical_done"] = False
        elif field == "consumption":
            report["source"]["terminal"]["5"] = "consumed"
        else:
            report["request"]["disposition"] = "cancelled_unfulfilled"
        rejects("lifetime/" + field, life.check, report)
    result = {
        "accepted": accepted,
        "rejected_corruptions": rejected,
        "assertions_enabled": __debug__,
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "torch_imported": "torch" in sys.modules,
        "xdsl_imported": "xdsl" in sys.modules,
        "boundary": "Saved observations only; not live execution or artifact compatibility.",
    }
    (root / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
