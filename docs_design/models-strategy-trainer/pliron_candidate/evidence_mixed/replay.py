"""Independent replay of actual mixed facts, without frameworks or owner runtime."""

from __future__ import annotations

import copy
import importlib
import importlib.abc
import json
import math
import sys
from pathlib import Path


class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "torch"
            or fullname.startswith(("torch.", "xdsl."))
            or fullname == "xdsl"
            or ".execution_candidate" in fullname
            or fullname.endswith(
                (".mixed_target", ".model_target", ".mixed_adapter", ".mixed_contract", ".mixed_dialect", ".mixed", ".mixed_author")
            )
        ):
            raise AssertionError("replay imported framework/runtime: " + fullname)


sys.meta_path.insert(0, BlockRuntime())
oracle = importlib.import_module("docs_design.models-strategy-trainer.xdsl_candidate.mixed_conformance")


def check(item):
    kind = item["kind"]
    if kind == "exclusion":
        result = item["result"]
        mode = item["mode"]
        limit = item["limit"]
        assert result["released"] and not result["pending"] and not item["active"] and result["errors"]
        assert item["heartbeats"] and 0 < item["heartbeats"][0] < limit
        if mode == "cancel":
            assert result["status"] == "cancelled" and result["steps"] == 0 and item["cancelled_at"]
            assert 0 < item["cancelled_at"][0] <= item["calls"] < limit
        else:
            assert result["status"] == "failed" and item["calls"] == limit
            assert result["steps"] == item["revision"] == int(mode == "feedback")
            if mode == "feedback":
                assert item["first_feedback_at"] and 0 < item["first_feedback_at"][0] < limit
        return
    report = item["report"]
    if kind == "archive":
        assert report["status"] == ("cancelled" if item["interrupted"] else "stopped")
        assert report["released"] and report["steps"] == 3 and not report["pending"] and report["diagnostics"] == 0
        assert item["witness"]["running"] and item["witness"]["protected"]
        assert report["learning"] == {"contributions": [], "windows": []} and report["source"]["revision"] == 3
        assert all(not value for value in report["source"].values() if isinstance(value, list))
        if item["interrupted"]:
            assert any("mixed physical failure" in error["message"] for error in report["errors"])
            assert any(error["type"] == "CancelledError" for error in report["errors"])
        else:
            assert not report["errors"]
        weights = [0.8]
        outcomes = item["owner"]["outcomes"]
        assert len(outcomes) == 3
        for ordinal, outcome in enumerate(outcomes):
            assert outcome["optimizer"] == outcome["reset"] == "returned"
            uses = outcome["inputs"]
            assert [(use["channel"], use["ordinal"]) for use in uses] == [("mixed.left", ordinal), ("mixed.right", ordinal)]
            assert uses[0]["sources"] == [] and len(uses[1]["sources"]) == 1
            source = uses[1]["sources"][0]
            assert source[:2] == ["learner", 1] and 0 <= source[2] <= ordinal
            # These fixtures admit initial keys 0,1,2 before any feedback. The
            # owner facts retain their ordinals and actual production epochs.
            feature = 2 * ordinal + 1 + weights[source[2]] * ordinal / 4
            weights.append(weights[-1] - 0.001 * (weights[-1] * feature - ordinal / 2) * feature)
        assert math.isfinite(item["owner"]["weight"]) and math.isclose(item["owner"]["weight"], weights[-1], rel_tol=2e-6, abs_tol=2e-6)
        return
    if kind == "admission":
        assert report["status"] == "failed" and report["released"] and report["errors"] and report["steps"] <= 1
        assert not report["source"]["active"]
        assert all(not any(ordinal == 1 for ordinal, _ in uses) for uses in report["claimed"].values())
        return
    if kind == "invalid":
        assert report["status"] == "failed" and report["released"] and report["errors"] and not report["source"]["active"]
        if item["symbol"] == "learned_bias_v1":
            assert report["learning"]["windows"] and not report["source"]["updates"]
        else:
            assert report["steps"] == 0
        return
    if kind == "physical":
        oracle.check_mixed(report, outcome="cancelled" if item["interrupted"] else "failed")
        assert any("mixed physical failure" in error["message"] for error in report["errors"])
        witness = item["witness"]
        assert witness["running"] and witness["protected"] and witness["physical_end"] is witness["release"] is None
        assert witness["feedback_revision"] == 3
        return
    assert kind in ("normal", "wrong")
    oracle.check_mixed(report)


def main():
    root = Path(__file__).parent
    before = sys._is_gil_enabled()
    records = json.loads((root / "reports.json").read_text())
    accepted = []
    wrong = []
    corruptions = []
    for name, item in records.items():
        if item["kind"] == "wrong":
            assert item["report"]["status"] == "stopped"
            try:
                check(item)
            except AssertionError:
                wrong.append(name)
            else:
                raise AssertionError("wrong computation passed: " + name)
        else:
            check(item)
            accepted.append(name)

    def reject(name, item):
        try:
            check(item)
        except AssertionError, KeyError, IndexError, StopIteration:
            corruptions.append(name)
        else:
            raise AssertionError("corruption passed: " + name)

    for case in (
        "selection",
        "exclusion",
        "weight",
        "feedback",
        "source_epoch",
        "input",
        "release",
        "unit",
        "discard",
        "horizon",
        "publication",
    ):
        item = copy.deepcopy(records["calls-f0"])
        report = item["report"]
        if case == "selection":
            report["source"]["selections"][-1]["value"] += 1
        elif case == "exclusion":
            report["source"]["excluded"].clear()
        elif case == "weight":
            report["learning"]["windows"][1]["weight"] += 1
        elif case == "feedback":
            report["source"]["updates"][1]["epoch"] = 99
        elif case == "source_epoch":
            next(row for row in report["source"]["stages"] if row["origin"] == "data.encode")["sources"][0][2] = 4
        elif case == "input":
            report["learning"]["contributions"][0]["inputs"][1]["window"] = 99
        elif case == "release":
            next(row for row in report["source"]["stages"] if row["origin"] == "data.encode")["release"] = 0
        elif case == "unit":
            report["unit_outcomes"][0]["optimizer"] = "not_attempted"
        elif case == "discard":
            report["remaining"]["mixed.right"] = [3]
        elif case == "horizon":
            report["source"]["max_active"] = 4
        else:
            report["source"]["products"][0]["selected"] += 1
        reject(case, item)
    for case in ("archive_weight", "archive_epoch", "archive_feedback", "archive_protection"):
        item = copy.deepcopy(records["archive-calls-1"])
        if case == "archive_weight":
            item["owner"]["weight"] += 1
        elif case == "archive_epoch":
            item["owner"]["outcomes"][1]["inputs"][1]["sources"][0][2] = 99
        elif case == "archive_feedback":
            item["report"]["source"]["revision"] = 2
        else:
            item["witness"]["protected"] = False
        reject(case, item)
    item = copy.deepcopy(records["physical-1"])
    item["witness"]["release"] = 0
    reject("early_physical_release", item)
    item = copy.deepcopy(records["admission-identity"])
    item["report"]["claimed"]["mixed.left"].append([1, 999])
    reject("partial_claim", item)
    for case, mode in (("starved_heartbeat", "empty"), ("starved_feedback", "feedback"), ("starved_cancel", "cancel")):
        item = copy.deepcopy(records["exclusion-" + mode])
        if mode == "empty":
            item["heartbeats"] = []
        elif mode == "feedback":
            item["first_feedback_at"] = [item["limit"]]
        else:
            item["cancelled_at"] = [item["limit"]]
        reject(case, item)
    result = {
        "accepted": accepted,
        "rejected_wrong_computations": wrong,
        "rejected_observation_corruptions": corruptions,
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "torch_imported": "torch" in sys.modules,
        "xdsl_imported": "xdsl" in sys.modules,
        "runtime_imported": any(".execution_candidate" in name for name in sys.modules),
    }
    (root / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
