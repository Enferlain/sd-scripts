"""Replay actual A observations without importing either IR or its runtime."""

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
                    "model_target",
                    "model_adapter",
                    "model_contract",
                    "model_dialect",
                    "fixture",
                    "prepare",
                    "correlation",
                    "correlation_author",
                    "local_runtime",
                    "runtime",
                }
            )
        ):
            raise ImportError("independent A replay imported " + fullname)
        return None


def check(item, oracle) -> None:
    report, kind = item["report"], item["kind"]
    assert report["diagnostics"] == 0
    if kind == "normal":
        oracle.check_a(report)
        if "witness" in item:
            witness = item["witness"]
            assert witness["requested"] and not witness["completed"] and witness["protected"]
            assert witness["evaluation_mode_held"] and witness["mode_restored"]
            assert witness["epoch"] == report["evaluation"]["measurements"][0]["epoch"]
    elif kind == "admission":
        oracle.check_admission_failure(report, item["sample"], missing=item["missing"])
        if item.get("evicted"):
            assert report["pending"] and len(report["unit_outcomes"]) == 1
    elif kind == "physical":
        witness = item["witness"]
        assert report["status"] == "cancelled" and report["released"] and report["errors"]
        assert witness["protected"] and witness["run_live"] and not witness["physical_done"]
        assert any("A physical failure" in error["message"] for error in report["errors"])
        if item["symbol"] == "measure_v1":
            assert witness["evaluation_mode_held"] and witness["mode_restored"]
            (measurement,) = report["evaluation"]["measurements"]
            assert not measurement["returned"] and not measurement["completed"]
            assert (measurement["trigger_window"], measurement["trigger_position"]) == (0, 1)
            assert measurement["epoch"] in (1, 2)
            oracle.close(measurement["weight"], oracle.reference()[1][measurement["epoch"]], "held current state")
            oracle.check_inputs(measurement["inputs"], 0, 2)
        else:
            assert report["steps"] == 0 and not report["learning"]["contributions"]
            assert any(stage["origin"] == "data.encode" and stage["outcome"] == "entered_failed" for stage in report["source"]["stages"])
    elif kind == "invalid_output":
        assert report["status"] == "failed" and report["released"] and report["errors"]
        assert any("invalid returned" in error["message"] for error in report["errors"])
        if item["symbol"] == "measure_v1":
            (measurement,) = report["evaluation"]["measurements"]
            assert not measurement["returned"] and not measurement["completed"]
        elif item["symbol"] == "encode_v1":
            assert any(stage["origin"] == "data.encode" and stage["outcome"] == "invalid_output" for stage in report["source"]["stages"])
        else:
            assert item["symbol"] == "loss_v1" and not report["learning"]["contributions"] and report["steps"] == 0
    else:
        raise RuntimeError("unknown observation kind " + kind)


def main() -> None:
    if not __debug__:
        raise RuntimeError("independent A oracle requires assertions")
    before = sys._is_gil_enabled()
    sys.meta_path.insert(0, NoExecution())
    oracle = importlib.import_module(PREFIX + ".xdsl_candidate.model_conformance")
    output = Path(__file__).resolve().parent
    reports = json.loads((output / "reports.json").read_text())
    accepted, wrong, corruptions = [], [], []

    def reject(name, function):
        try:
            function()
        except AssertionError:
            return name
        raise RuntimeError("independent A oracle accepted " + name)

    for name, item in reports.items():
        if item["kind"] == "wrong_computation":
            assert item["report"]["status"] == "stopped"
            wrong.append(reject(name, lambda item=item: oracle.check_a(item["report"])))
        else:
            check(item, oracle)
            accepted.append(name)
    assert reports["wrong-loss-calls"]["report"]["learning"]["windows"] == reports["wrong-loss-inline"]["report"]["learning"]["windows"]
    for case in (
        "product",
        "epoch",
        "sample",
        "window",
        "loss",
        "advance",
        "reset",
        "eval_epoch",
        "eval_trigger",
        "eval_inputs",
        "source",
        "claim",
    ):
        item = copy.deepcopy(reports["calls-f0-w1"])
        report = item["report"]
        if case == "product":
            report["source"]["products"][0]["value"] += 1
        elif case == "epoch":
            report["learning"]["contributions"][1]["epoch"] += 1
        elif case == "sample":
            report["learning"]["contributions"][0]["sample"] = 2
        elif case == "window":
            report["learning"]["contributions"][2]["window"] = 0
        elif case == "loss":
            report["learning"]["contributions"][0]["loss"] += 1
        elif case == "advance":
            report["learning"]["windows"][0]["weight"] += 0.1
        elif case == "reset":
            report["unit_outcomes"][0]["reset"] = "not_attempted"
        elif case == "eval_epoch":
            report["evaluation"]["measurements"][0]["epoch"] = 0
        elif case == "eval_trigger":
            report["evaluation"]["measurements"][0]["trigger_window"] = 1
        elif case == "eval_inputs":
            report["evaluation"]["measurements"][0]["inputs"] = []
        elif case == "source":
            next(product for product in report["source"]["products"] if product["branch"] == "right")["sources"][0][2] = 99
        else:
            report["claimed"]["left.0"] = []
        corruptions.append(reject(case, lambda item=item: check(item, oracle)))
    item = copy.deepcopy(reports["bad_sample-2"])
    item["report"]["claimed"]["left.1"].append([0, 999])
    corruptions.append(reject("premature_member_claim", lambda: check(item, oracle)))
    item = copy.deepcopy(reports["evicted-2"])
    item["report"]["pending"] = False
    corruptions.append(reject("lost_pending_contribution", lambda: check(item, oracle)))
    item = copy.deepcopy(reports["encode_v1-physical"])
    item["witness"]["protected"] = False
    corruptions.append(reject("early_physical_release", lambda: check(item, oracle)))
    item = copy.deepcopy(reports["measure_v1-physical"])
    item["witness"]["mode_restored"] = False
    corruptions.append(reject("failed_mode_restoration", lambda: check(item, oracle)))
    item = copy.deepcopy(reports["measure_v1-invalid"])
    item["report"]["evaluation"]["measurements"][0]["completed"] = True
    corruptions.append(reject("false_measurement_completion", lambda: check(item, oracle)))
    result = {
        "accepted": accepted,
        "rejected_wrong_computations": wrong,
        "rejected_observation_corruptions": corruptions,
        "gil_before": before,
        "gil_after": sys._is_gil_enabled(),
        "torch_imported": "torch" in sys.modules,
        "xdsl_imported": "xdsl" in sys.modules,
        "runtime_imported": any(name.endswith((".model_target", ".model_adapter", ".prepare")) for name in sys.modules),
    }
    assert len(accepted) == 32 and len(wrong) == 2 and len(corruptions) == 17
    assert not any(result[name] for name in ("gil_before", "gil_after", "torch_imported", "xdsl_imported", "runtime_imported"))
    (output / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
