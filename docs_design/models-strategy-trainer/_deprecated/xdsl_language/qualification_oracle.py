"""Assertion-enabled, independent expectations for cold owner receipts."""

from __future__ import annotations

import math

from fractions import Fraction


def close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def check(report: dict, name: str, *, prefix: str = "", mode: str = "normal", erased: bool = False) -> None:
    if not __debug__:
        raise RuntimeError("qualification oracle requires assertions enabled")
    owner, witness = report["owner"], report["witness"]
    assert owner["drained"], owner
    if erased:
        assert witness["archives_empty"] and witness["erasures"] > 0
    flows = owner["flows"]
    for flow in flows.values():
        assert not flow["active"] and not flow["ready"]
        assert set(map(int, flow["terminal"])) == set(range(flow["selected"]))
    resources = owner["resources"]
    if name == "rejections":
        assert resources["policy"] == {"value": 2, "epoch": 2}
        assert owner["cancel_requested"] and not owner["completed"]
        assert witness["heartbeats_during_exclusions"] > 1
        flow = flows["records"]
        assert flow["high_water"] <= 3 and flow["permits"] == 3
        assert all(flow["terminal"][str(key)] == "rejected_stale" for key in range(128))
        assert not any(status == "consumed" for status in flow["terminal"].values())
        return
    products = resources[prefix + "products"]["value"]
    if name == "data-only":
        assert not owner["units"] and not owner["requests"]
        policy = 0
        assert len(products) == 4
        for key, product in enumerate(products):
            # Horizon one prevents the next policy read until this product is
            # consumed. Neither a runtime trace nor its admitted list is input.
            x = key + policy
            value = (2 * x + 1) ** 2
            assert product["key"] == key and product["input"] == {"key": key, "x": x, "target": 0}
            close(product["value"], value)
            policy = value % 5
        assert resources["policy"] == {"value": policy, "epoch": 4}
        assert owner["completed"] and not owner["failures"]
        flow = flows["records"]
        assert flow["high_water"] <= 1 and flow["permits"] == 1
        consumed = {int(key) for key, status in flow["terminal"].items() if status == "consumed"}
        assert consumed == set(range(4))
        assert all(status == ("consumed" if int(key) < 4 else "discarded") for key, status in flow["terminal"].items())
        return
    assert name in ("lifetime", "extra-scope")
    windows = 1 if mode == "orderly-before" else 2
    weight = Fraction(4, 5)
    for window in range(windows):
        gradient = sum((weight * (2 * (key + 1)) - (key + 1)) * (2 * (key + 1)) for key in range(2 * window, 2 * window + 2)) / 2
        weight -= Fraction(1, 100) * gradient
    close(resources[prefix + "model"]["value"], float(weight))
    assert resources[prefix + "model"]["epoch"] == windows
    assert len(owner["units"]) == 1
    unit = next(iter(owner["units"].values()))
    assert unit["steps"] == windows and not unit["unfinished"]
    assert len(unit["outcomes"]) == windows
    assert all(item["optimizer"] == item["reset"] == "returned" for item in unit["outcomes"])
    cancelled = mode in ("pending-cancel", "held-cancel", "held-failure")
    failed = mode in ("held-failure", "late-producer-failure")
    published = windows == 2 and not cancelled
    assert len(products) == int(published)
    if published:
        result = products[0]
        assert result["kind"] == "measurement" and result["key"] == 4 and result["epoch"] == 2
        close(result["weight"], float(weight))
        close(result["prediction"], float(weight * 10))
        close(result["loss"], float((weight * 10 - 5) ** 2 / 2))
    assert resources[prefix + "products"]["epoch"] == int(published)
    assert len(owner["requests"]) == 1
    request = owner["requests"][0]
    assert not request["pending"] and not request["active"] and request["permits"] == 1
    assert len(request["issued"]) == int(windows == 2)
    if windows == 2:
        issued = request["issued"]["0"]
        assert issued["payload"] == 1 and [prefix + "training", 1] in issued["cause"]
        assert request["dispositions"] == {"0": "failed" if mode == "held-failure" else "cancelled" if cancelled else "completed"}
    else:
        assert not request["dispositions"]
    flow = flows[prefix + "records"]
    assert flow["high_water"] <= 3 and flow["ready_high_water"] <= 2 and flow["permits"] == 3
    consumed = {int(key) for key, status in flow["terminal"].items() if status == "consumed"}
    assert consumed == set(range(2 * windows + int(published)))
    for key, status in flow["terminal"].items():
        expected = (
            "consumed"
            if int(key) in consumed
            else "failed"
            if mode == "late-producer-failure" and int(key) == 6
            else "cancelled"
            if cancelled
            else "discarded"
        )
        assert status == expected, (key, status, expected)
    if mode == "late-producer-failure":
        assert flow["terminal"]["6"] == "failed" and witness["completed_before_failure"]
        assert any("late producer failure" in item for item in owner["failures"])
        assert owner["source_receipts"][0]["status"] == "succeeded"
    if mode == "held-failure":
        assert any("late physical failure" in item for item in owner["failures"])
        assert prefix + "model" in owner["unusable"]
    elif mode == "held-cancel":
        assert prefix + "model" not in owner["unusable"]
    if mode in ("held-cancel", "held-failure"):
        assert witness["protected_while_cancelled"]
    if mode == "late":
        assert witness["pending_keeps_group_open"] and witness["measurement_not_entered"]
    if mode == "prefetched":
        assert witness["evaluation_input_prefetched"]
    assert owner["completed"] == (not cancelled and not failed)
    assert owner["cancel_requested"] == cancelled
    assert bool(owner["failures"]) == (cancelled or failed)
