"""Replay C observations with all framework, target and runtime imports blocked."""

from __future__ import annotations

import copy
import importlib.abc
import json
import runpy
import sys
from pathlib import Path


class BlockExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = (
            "torch",
            "xdsl",
            "docs_design.models-strategy-trainer.execution_candidate",
            "docs_design.models-strategy-trainer.grant_target",
            "docs_design.models-strategy-trainer.pliron_candidate",
            "docs_design.models-strategy-trainer.xdsl_candidate",
        )
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise RuntimeError("independent C replay imported " + fullname)
        return None


def partial(item, reference, close) -> None:
    """An entered retained call may have effects without returning successfully."""
    report, phase = item["report"], item["phase"]
    assert phase in ("optimizer", "reset") and item["retry_rejected"]
    assert report["status"] == "failed" and report["errors"]
    assert report["released"] and report["pending"] and report["handback_verified"]
    assert report["unusable"] == ["model"] and report["diagnostics"] == 0
    close(report["final_weight"], reference()["final_weight"], "retained call's physical effects")
    (outcome,) = report["unit_outcomes"]
    assert outcome["contribution"] == "granted_ready" and outcome["input_count"] == 0
    assert outcome["invocation"] > 0 and outcome["advance_invocation"] == outcome["invocation"]
    assert outcome["work"] == ["grant-session", "selected-two-pass"]
    assert (outcome["optimizer"], outcome["reset"]) == (
        ("uncertain", "not_attempted") if phase == "optimizer" else ("returned", "uncertain")
    )
    assert report["steps"] == (0 if phase == "optimizer" else 1)
    assert report["epoch"] == (1 if phase == "optimizer" else 2)
    if phase == "optimizer":
        close(report["final_gradient"], reference()["final_gradient"], "reset not attempted")
    else:
        assert report["final_gradient"] is None


def main() -> None:
    if not __debug__:
        raise RuntimeError("independent C oracle requires assertions enabled")
    before = sys._is_gil_enabled()
    sys.meta_path.insert(0, BlockExecution())
    root = Path(__file__).resolve().parent
    oracle = runpy.run_path(str(root.parents[1] / "xdsl_candidate" / "grant_conformance.py"))
    check, unverified, reference, close = (oracle[name] for name in ("check_c", "check_unverified", "reference", "close"))
    reports = json.loads((root / "reports.json").read_text())
    accepted, wrong_computation, corruptions = [], [], []

    def reject(name, check_case):
        try:
            check_case()
        except AssertionError:
            return name
        raise RuntimeError("independent C oracle accepted " + name)

    for name, item in reports.items():
        report, kind = item["report"], item["kind"]
        if kind == "normal":
            check(report, initial=item["initial"], clip_norm=item["clip"])
        elif kind == "unverified":
            unverified(report)
        elif kind == "physical":
            witness = item["witness"]
            assert witness["writer_held"] and witness["pending_without_token"] and witness["run_live"]
            assert witness["steps_before_return"] == 0 and witness["active_workers"] == 1
            if item["cancel"]:
                assert witness["stop_seen"]
            if item["cancel"] or item["fail"]:
                unverified(report, cancelled=item["cancel"])
            else:
                check(report)
            if item["fail"]:
                assert any("physical grant failed" in error["message"] for error in report["errors"])
        elif kind == "partial":
            partial(item, reference, close)
        elif kind == "wrong_computation":
            wrong_computation.append(reject(name, lambda report=report: check(report)))
            continue
        else:
            raise RuntimeError("unknown report kind: " + kind)
        accepted.append(name)

    for name in ("gradient", "weight", "phase", "origin", "handback", "step", "optimizer", "reset", "pending", "epoch"):
        report = copy.deepcopy(reports["calls-i0.8-plain"]["report"])
        if name == "gradient":
            report["selected"]["selected_gradient"] += 1
        elif name == "weight":
            report["final_weight"] = 0.8
        elif name in ("phase", "origin"):
            report["selected"]["phases"][2][name] = "wrong"
        elif name == "handback":
            report["handback_verified"] = False
        elif name == "step":
            report["steps"] = 2
        elif name in ("optimizer", "reset"):
            report["unit_outcomes"][0][name] = "uncertain"
        elif name == "pending":
            report["pending"] = True
        else:
            report["epoch"] = 0
        corruptions.append(reject(name, lambda report=report: check(report)))
    for phase in ("optimizer", "reset"):
        item = copy.deepcopy(reports["partial_" + phase])
        item["report"]["unit_outcomes"][0][phase] = "returned"
        corruptions.append(reject("partial_" + phase, lambda item=item: partial(item, reference, close)))

    result = {
        "accepted": accepted,
        "rejected_wrong_computations": wrong_computation,
        "rejected_observation_corruptions": corruptions,
        "assertions_enabled": __debug__,
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "torch_imported": "torch" in sys.modules,
        "xdsl_imported": "xdsl" in sys.modules,
        "runtime_imported": any(".execution_candidate" in name or name.endswith(".grant_target") for name in sys.modules),
    }
    assert len(accepted) == 21 and len(wrong_computation) == 2 and len(corruptions) == 12
    assert not any(result[key] for key in ("gil_before", "gil_after", "torch_imported", "xdsl_imported", "runtime_imported"))
    (root / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
