"""Independent fixture arithmetic and observable lifetime obligations; stdlib only."""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any


def close(actual: Any, expected: Any) -> None:
    assert type(actual) in (int, float) and math.isfinite(actual), actual
    assert math.isclose(actual, float(expected), rel_tol=2e-6, abs_tol=2e-6), (actual, expected)


def expected_weights() -> tuple[Fraction, Fraction]:
    weight = Fraction(4, 5)
    result = []
    for members in ((0, 1), (2, 3)):
        gradient = sum((weight * (2 * (k + 1)) - (k + 1)) * (2 * (k + 1)) for k in members) / 2
        weight -= gradient / 100
        result.append(weight)
    return Fraction(result[0]), Fraction(result[1])


def check(
    report: Mapping[str, Any], *, prefix: str = "", windows: int = 2, failed_after_evaluation: bool = False, histories: bool = True
) -> None:
    assert windows in (1, 2)
    assert report["schema"] == 1
    assert report["status"] == ("failed" if failed_after_evaluation else "stopped"), (report["status"], report["errors"])
    assert report["released"] and report["closed"] and not report["pending"]
    assert report["source"]["cleanup_complete"] and not report["source"]["active"]
    assert not report["source"]["physical_active"]
    assert report["steps"] == windows
    weights = expected_weights()
    close(report["weight"], weights[windows - 1])
    assert report["training_mode"] and report["gradients_enabled"]
    assert report["selected_windows"] == list(range(windows))
    acquired = {int(key): value for key, value in report["acquired"].items()}
    assert acquired == {window: [window * 2, window * 2 + 1] for window in range(windows)}
    source = report["source"]
    assert 0 < source["high_water"] <= 3
    assert 0 < source["ready_high_water"] <= 2
    terminal = {int(key): value for key, value in source["terminal"].items()}
    assert set(terminal) == set(range(source["cursor"]))
    consumed = set(range(5 if windows == 2 else 2))
    assert {key for key, value in terminal.items() if value == "consumed"} == consumed
    assert all(value in ("consumed", "discarded", "closed_unconsumed", "failed") for value in terminal.values())
    if not failed_after_evaluation:
        assert all(terminal[key] == "discarded" for key in set(terminal) - consumed)
        assert not report["errors"]
    else:
        assert terminal[6] == "failed"
        assert any("physical production failure 6" in error["message"] for error in report["errors"])
    clock = report["scope_clocks"]
    assert clock[prefix + "consumers"] > max(clock[prefix + "training"], clock[prefix + "evaluation"])
    assert clock[prefix + "producer"] > clock[prefix + "consumers"]
    assert clock[prefix + "run"] > clock[prefix + "producer"]
    evaluation = report["evaluation"]
    if windows == 1:
        assert evaluation is None and not report["evaluation_completed"]
        assert report["request"] is None
    else:
        assert report["evaluation_completed"] and evaluation["returned"] and evaluation["physical_done"]
        assert evaluation["physical_entered"]
        assert report["request"] == {"cause": evaluation["cause"], "disposition": "completed"}
        assert evaluation["cause"] == {"source": prefix + "training", "window": 1, "position": 2}
        assert evaluation["key"] == 4 and evaluation["epoch"] == 2
        assert evaluation["evaluation_mode"] and evaluation["gradients"] is False
        close(evaluation["weight"], weights[1])
        prediction = weights[1] * 10
        close(evaluation["prediction"], prediction)
        close(evaluation["loss"], (prediction - 5) ** 2 / 2)
        assert {(item["channel"], item["key"]) for item in evaluation["inputs"]} == {
            (prefix + "producer.slot.0", 1),
            (prefix + "producer.slot.1", 1),
            (prefix + "producer.evaluation", 1),
        }
        assert all(item["sources"] == [[prefix + "encoder", 1, 0]] for item in evaluation["inputs"])
    if not histories:
        return
    assert len(report["windows"]) == windows
    for index, window in enumerate(report["windows"]):
        assert window["window"] == index and window["epoch"] == index + 1
        close(window["weight"], weights[index])
    assert [item["key"] for item in report["contributions"]] == list(range(2 * windows))
    for item in report["contributions"]:
        key, index = item["key"], item["window"]
        weight = Fraction(4, 5) if index == 0 else weights[0]
        assert index == key // 2 and item["epoch"] == index
        close(item["x"], 2 * (key + 1))
        close(item["target"], key + 1)
        close(item["weight"], weight)
        close(item["loss"], (weight * (2 * (key + 1)) - (key + 1)) ** 2 / 4)
    for product in source["products"]:
        close(product["value"], 2 * (product["key"] + 1))
        assert product["sources"] == [[prefix + "encoder", 1, 0]]
    assert all(use["physical_done"] and use["epoch"] == 0 for use in source["uses"])
    outstanding: set[int] = set()
    ready: set[int] = set()
    requests = 0
    for event in report["events"]:
        kind = event["kind"]
        key = event.get("key")
        if kind == "selected":
            assert key not in outstanding
            outstanding.add(key)
            assert event["outstanding"] == len(outstanding) <= 3
        elif kind == "published":
            assert key in outstanding and key not in ready
            ready.add(key)
            assert event["ready"] == len(ready) <= 2
        elif kind == "claimed":
            assert key in outstanding and key in ready
            ready.remove(key)
        elif kind in ("consumed", "disposed"):
            assert key in outstanding
            outstanding.remove(key)
            ready.discard(key)
        elif kind == "evaluation_requested":
            requests += 1
            assert event["source"] == prefix + "training" and event["window"] == 1 and event["position"] == 2
        elif kind == "pending_request_witness":
            assert event["producer_live"] and event["group_live"]
            assert clock[prefix + "training"] < event["clock"] < clock[prefix + "evaluation"]
        elif kind == "aggregate_bound_witness":
            assert event["active"] == [4, 5, 6]
        elif kind == "shutdown_witness":
            assert event["producer_physical_live"] and event["protected"] and event["producer_stopped"] and event["run_live"]
            assert clock[prefix + "consumers"] < event["clock"] < clock[prefix + "producer"]
    assert not outstanding and not ready
    assert requests == (1 if windows == 2 else 0)
