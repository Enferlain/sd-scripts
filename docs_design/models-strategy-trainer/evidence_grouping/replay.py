"""Independent replay of paired grouping observations; no IR or runtime imports."""

from __future__ import annotations

import copy
import importlib
import importlib.abc
import json
import sys
from pathlib import Path

PREFIX = "docs_design.models-strategy-trainer"


class NoExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname.split(".", 1)[0] in {"torch", "xdsl"}
            or fullname.startswith(PREFIX + ".execution_candidate")
            or (
                fullname.rsplit(".", 1)[-1]
                in {
                    "fixture",
                    "bridge",
                    "prepare",
                    "runtime",
                    "local_runtime",
                    "fusion",
                    "payloads",
                    "pair_extension",
                    "dialects",
                    "local_contract",
                    "test_grouping_comparison",
                }
            )
        ):
            raise ImportError("independent replay imported " + fullname)
        return None


def normal(item, check) -> None:
    report = item["report"]
    check(report)
    assert report["status"] == "stopped" and not report["diagnostics"]
    leaves = ["data.double", "data.offset", "pair.pack", "pair.first"]
    allowed = [[name] for name in leaves]
    assert item["groups"] in (allowed, [[leaves[0]], [leaves[1]], leaves[2:]] if item["opaque"] else [leaves])
    assert report["local_dispatches"] == len(report["tickets"]) * len(item["groups"])
    assert item["effects"] == ([ticket["calls"][1]["input"] for ticket in report["tickets"]] if item["opaque"] else [])


def failure(item) -> None:
    report = item["report"]
    physical = item["kind"] == "physical"
    assert report["status"] == ("cancelled" if physical else "failed") and report["errors"]
    assert 0 < report["stop_clock"] < report["released"] and 0 < report["high_water"] <= 2 and not report["diagnostics"]
    expected = (
        "entered_failed"
        if physical
        else {"invalid_output": "invalid_output", "domain": "negative_input", "undeclared": "undeclared_outcome"}[item["fault"]]
    )
    failed = False
    for ticket in report["tickets"]:
        assert ticket["selected"] < report["stop_clock"] and ticket["retired"] < report["released"]
        assert ticket["value"] == ticket["key"] + ticket["policy_bias"]
        assert ticket["terminal"] in ("failed", "abandoned")
        assert ticket["publication_ack"] is None and ticket["feedback_ack"] is None
        calls = ticket["calls"]
        assert len(calls) <= 3
        assert [call["origin"] for call in calls] == ["data.double", "data.offset", "pair.pack"][: len(calls)]
        previous = ticket["selected"]
        for index, call in enumerate(calls):
            assert previous < call["entered"] < call["physical_end"] < call["released"] < ticket["retired"]
            value = ticket["value"]
            assert call["role"] == "map" and call["input"] == (value, 2 * value, 2 * value + 1)[index]
            if index < 2:
                assert call["outcome"] == "returned" and call["output"] == (2 * value, 2 * value + 1)[index]
            else:
                assert call["outcome"] == expected and ticket["terminal"] == "failed"
                failed = True
            previous = call["released"]
    assert failed and all(decision["kind"] == "selected" for decision in report["decisions"])
    if physical:
        assert item["witness"] == {"active_calls": 1, "run_live": True, "physical_done": False}
        assert any("physical frozen-resource failure" in error for error in report["errors"])


def main() -> None:
    if not __debug__:
        raise RuntimeError("independent oracle requires assertions enabled")
    before = sys._is_gil_enabled()
    sys.meta_path.insert(0, NoExecution())
    check = importlib.import_module(PREFIX + ".xdsl_candidate.extension_conformance").check_extension
    base_check = importlib.import_module(PREFIX + ".xdsl_candidate.conformance").check_b
    root = Path(__file__).resolve().parent
    reports = json.loads((root / "reports.json").read_text())
    accepted, wrong, corruptions = [], [], []

    def reject(name, function):
        try:
            function()
        except AssertionError:
            return name
        raise RuntimeError("oracle accepted deliberate corruption " + name)

    for name, item in reports.items():
        if item["kind"] == "normal":
            normal(item, check)
        elif item["kind"] in ("failure", "physical"):
            failure(item)
        else:
            assert item["kind"] == "wrong_payload"
            wrong.append(reject(name, lambda item=item: check(item["report"])))
            projected = copy.deepcopy(item["report"])
            for ticket in projected["tickets"]:
                ticket["calls"] = [call for call in ticket["calls"] if call["origin"] not in ("pair.pack", "pair.first")]
            base_check(projected)  # Demonstrably insufficient without the independent payload check.
            continue
        accepted.append(name)

    for framework in ("xdsl", "pliron"):
        for case in ("payload", "origin", "physical", "missing_leaf", "feedback", "dispatch"):
            item = copy.deepcopy(reports[framework + "-frozen-fused"])
            report, calls = item["report"], item["report"]["tickets"][0]["calls"]
            if case == "payload":
                calls[2]["output"][1] += 1
                calls[3]["input"][1] += 1
            elif case == "origin":
                calls[2]["origin"] = "another.source"
            elif case == "physical":
                calls[2]["physical_end"] = calls[2]["released"]
            elif case == "missing_leaf":
                del calls[3]
            elif case == "feedback":
                next(decision for decision in report["decisions"] if decision["kind"] == "feedback")["bias"] += 1
            else:
                report["local_dispatches"] += 1
            corruptions.append(reject(framework + "/" + case, lambda item=item: normal(item, check)))
        for case in ("relabel", "early_release"):
            item = copy.deepcopy(reports[framework + "-domain"])
            if case == "relabel":
                item["report"]["tickets"][0]["calls"][-1]["outcome"] = "returned"
            else:
                item["report"]["released"] = item["report"]["tickets"][0]["calls"][-1]["physical_end"]
            corruptions.append(reject(framework + "/" + case, lambda item=item: failure(item)))
        item = copy.deepcopy(reports[framework + "-physical"])
        item["witness"]["physical_done"] = True
        corruptions.append(reject(framework + "/lost_physical_obligation", lambda item=item: failure(item)))

    result = {
        "accepted": accepted,
        "rejected_wrong_computations": wrong,
        "rejected_observation_corruptions": corruptions,
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "torch_imported": "torch" in sys.modules,
        "xdsl_imported": "xdsl" in sys.modules,
        "runtime_imported": any(name.endswith((".runtime", ".local_runtime", ".prepare", ".bridge")) for name in sys.modules),
    }
    assert len(accepted) == 16 and len(wrong) == 2 and len(corruptions) == 18
    assert not any(result[name] for name in ("gil_before", "gil_after", "torch_imported", "xdsl_imported", "runtime_imported"))
    (root / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
