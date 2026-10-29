"""Independent A fixture expectations. No Torch, xDSL or runtime imports.

The arithmetic, logical sample membership and required causal relationships
come from the fixture definition, not another execution of the same bindings.
Backend invocation numbers and incidental production/request scheduling vary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

KEYS = (1, 2, 4, 5)


def close(actual: float, expected: float, name: str) -> None:
    assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=2e-6, abs_tol=2e-6), f"{name}: {actual} != {expected}"


def reference() -> tuple[list[dict[str, float]], list[float]]:
    weight = 0.8
    contributions, weights = [], [weight]
    for index in range(2):
        gradient = 0.0
        for sample in KEYS[index * 2 : index * 2 + 2]:
            left, right = 2 * sample + 1, 1.5 * sample / 4
            feature = left + right
            error = weight * feature - sample / 2
            contributions.append({"left": left, "right": right, "weight": weight, "loss": error * error / 2})
            gradient += error * feature  # derivative of the explicitly halved loss
        weight -= 0.05 * gradient
        weights.append(weight)
    return contributions, weights


def check_inputs(items: list[dict[str, Any]], window: int, slots: int, *, sources: bool = True) -> None:
    expected = {(f"{branch}.{slot}", window) for slot in range(slots) for branch in ("left", "right")}
    assert len(items) == len(expected) and {(item["channel"], item["window"]) for item in items} == expected, (
        "input/window association differs"
    )
    if sources:
        for item in items:
            expected_sources = [["encoder", 1, 0]] if item["channel"].startswith("right") else []
            assert item["sources"] == expected_sources, "actual source provenance differs"


def check_a(report: Mapping[str, Any]) -> None:
    assert report["status"] == "stopped" and not report["errors"] and report["error"] is None, "unexpected session outcome"
    assert report["released"] and not report["pending"] and report["steps"] == 2, "unfinished/released owner state differs"
    assert report["source"]["excluded"] == [3, 6], "exclusion agreement differs"
    assert report["addresses"] == [{"window": index // 2, "slot": index % 2, "sample": sample} for index, sample in enumerate(KEYS)], (
        "logical/physical address correspondence differs"
    )
    products = report["source"]["products"]
    assert len(products) == 8 and {(item["sample"], item["branch"]) for item in products} == {
        (sample, branch) for sample in KEYS for branch in ("left", "right")
    }, "fanout/filter membership differs"
    for product in products:
        index = KEYS.index(product["sample"])
        assert (product["window"], product["slot"]) == (index // 2, index % 2), "product address differs"
        right = product["branch"] == "right"
        close(product["value"], 1.5 * product["sample"] / 4 if right else 2 * product["sample"] + 1, "prepared data")
        assert product["sources"] == ([["encoder", 1, 0]] if right else []), "product provenance differs"
    stages = report["source"]["stages"]
    assert len(stages) == 12 and {(item["sample"], item["origin"]) for item in stages} == {
        (sample, origin) for sample in KEYS for origin in ("data.double", "data.offset", "data.encode")
    }, "stage source attribution differs"
    for item in stages:
        sample = item["sample"]
        expected = {
            "data.double": (sample, 2 * sample),
            "data.offset": (2 * sample, 2 * sample + 1),
            "data.encode": (sample, 1.5 * sample / 4),
        }[item["origin"]]
        assert item["outcome"] == "returned", "false stage success"
        close(item["input"], expected[0], "stage input")
        close(item["output"], expected[1], "stage output")
    expected, weights = reference()
    contributions = report["learning"]["contributions"]
    assert [item["sample"] for item in contributions] == list(KEYS), "admitted sample order/membership differs"
    outcomes = {item["invocation"]: item for item in report["unit_outcomes"]}
    assert len(outcomes) == len(contributions) == 4, "lost contribution outcomes"
    advances: list[int] = []
    for index, (item, expected_item) in enumerate(zip(contributions, expected, strict=True)):
        window, slot = index // 2, index % 2
        assert (item["window"], item["slot"], item["epoch"]) == (window, slot, window), "contribution/window provenance differs"
        for name in ("left", "right", "weight", "loss"):
            close(item[name], expected_item[name], name)
        check_inputs(item["inputs"], window, slot + 1)
        outcome = outcomes[item["invocation"]]
        assert (outcome["contribution"], outcome["optimizer"], outcome["reset"]) == ("ready", "returned", "returned"), (
            "optimization completion differs"
        )
        assert outcome["advance_invocation"] > item["invocation"], "advancement preceded contribution"
        check_inputs(outcome["inputs"], window, slot + 1, sources=False)
        advances.append(outcome["advance_invocation"])
    assert advances[0] == advances[1] < advances[2] == advances[3], "wrong advancement grouping"
    windows = report["learning"]["windows"]
    assert len(windows) == 2, "window completion lost"
    for index, window in enumerate(windows):
        assert (window["window"], window["epoch"]) == (index, index + 1), "advanced provenance differs"
        close(window["weight"], weights[index + 1], "independent SGD update")
        check_inputs(window["inputs"], index, 2)
    evaluations = report["evaluation"]["measurements"]
    assert len(evaluations) == 1, "requested evaluation membership differs"
    evaluation = evaluations[0]
    assert (evaluation["trigger_window"], evaluation["trigger_position"]) == (0, 1), "request association differs"
    assert evaluation["returned"] and evaluation["completed"] and evaluation["binding"] == 1 and evaluation["epoch"] in (1, 2), (
        "evaluation start/completion provenance differs"
    )
    close(evaluation["weight"], weights[evaluation["epoch"]], "actual evaluated state")
    close(evaluation["value"], evaluation["weight"], "measurement")
    check_inputs(evaluation["inputs"], 0, 2)  # trigger membership, independently of evaluated epoch
    assert all(not remaining for remaining in report["remaining"].values()), "unconsumed selected input"
    assert set(report["claimed"]) == {f"{branch}.{slot}" for branch in ("left", "right") for slot in (0, 1)}, "missing feed outcomes"
    for channel, claimed in report["claimed"].items():
        slot = int(channel[-1])
        assert [tuple(item) for item in claimed] == [(window, contributions[2 * window + slot]["invocation"]) for window in range(2)], (
            "claimed association differs"
        )


def check_admission_failure(report: Mapping[str, Any], sample: int, *, missing: bool) -> None:
    index = KEYS.index(sample)
    window, slot = index // 2, index % 2
    assert report["status"] == "failed" and report["errors"] and report["released"], "dishonest rejection/final release"
    assert len(report["learning"]["contributions"]) == index and report["steps"] == index // 2, "work continued past rejected admission"
    assert window in report["remaining"][f"left.{slot}"], "rejected join consumed the first member"
    assert (window not in report["remaining"][f"right.{slot}"]) == missing, "unexpected second member disposition"
    for branch in ("left", "right"):
        assert not any(key == window for key, _ in report["claimed"][f"{branch}.{slot}"]), "rejected join consumed a member"
    assert report["pending"] == bool(slot), "pending contribution owner lost"
    if slot:
        assert any(item["optimizer"] == "not_attempted" for item in report["unit_outcomes"]), "unadvanced input association lost"
        assert any(item["window"] == window for item in report["associations_on_failure"]), "active association lost on failure"
