"""Independent mixed-fixture arithmetic and causal oracle; no framework imports."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def close(actual: float, expected: float, what: str) -> None:
    assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=2e-6, abs_tol=2e-6), f"{what}: {actual} != {expected}"


def check_mixed(report: Mapping[str, Any], *, outcome: str = "stopped", pressure: bool = True) -> None:
    assert report["status"] == outcome and report["released"] and not report["pending"], "mixed owner lifetime/outcome differs"
    if outcome == "stopped":
        assert not report["errors"] and report["error"] is None, "hidden selected failure"
    else:
        assert report["errors"], "lost interruption/failure evidence"
    assert report["checkpoint"] and report["stop_requested"] and report["steps"] == 3, "stop did not follow the explicit checkpoint"
    source, learning = report["source"], report["learning"]
    selections, updates = source["selections"], source["updates"]
    contributions, windows = learning["contributions"], learning["windows"]
    assert [item["key"] for item in selections] == list(range(source["cursor"])), "cursor skipped or duplicated"
    assert [item["ordinal"] for item in updates] == list(range(3)), "missing/duplicate/out-of-order feedback"
    assert len(contributions) == len(windows) == 3, "learning continued beyond checkpoint"
    assert [item["ordinal"] for item in contributions] == list(range(3)), "wrong learning order"
    admitted = [item for item in selections if item["ordinal"] is not None]
    assert [item["ordinal"] for item in admitted] == list(range(len(admitted))), "ticket/address membership differs"
    selected = {item["key"]: item for item in selections}
    stages = {(item["sample"], item["origin"]): item for item in source["stages"]}
    assert len(stages) == len(source["stages"]), "duplicate stage/source attribution"
    weights = [0.8]
    for index, item in enumerate(contributions):
        key, value = admitted[index]["key"], admitted[index]["value"]
        assert (item["key"], item["selected"], item["revision"], item["epoch"]) == (key, value, admitted[index]["revision"], index), (
            "admitted source/policy differs"
        )
        encoded = stages[key, "data.encode"]
        stamp = encoded["sources"]
        assert len(stamp) == 1 and stamp[0][:2] == ["learner", 1] and 0 <= stamp[0][2] <= index, "invalid physical production provenance"
        left, right = 2 * value + 1, weights[stamp[0][2]] * value / 4
        feature = left + right
        error = weights[-1] * feature - value / 2
        for name, expected in (("left", left), ("right", right), ("weight", weights[-1]), ("loss", error * error / 2)):
            close(item[name], expected, name)
        inputs = item["inputs"]
        assert [(fact["channel"], fact["window"]) for fact in inputs] == [("mixed.left", index), ("mixed.right", index)], (
            "input association differs"
        )
        assert inputs[0]["sources"] == [] and inputs[1]["sources"] == stamp, "claim lost actual source snapshot"
        weights.append(weights[-1] - 0.001 * error * feature)
        window, update = windows[index], updates[index]
        assert (window["ordinal"], window["key"], window["epoch"]) == (index, key, index + 1), "post-step association differs"
        close(window["weight"], weights[-1], "independent SGD update")
        close(update["weight"], weights[-1], "feedback weight")
        assert (update["ordinal"], update["key"], update["before_revision"], update["revision"], update["epoch"], update["invocation"]) == (
            index,
            key,
            index,
            index + 1,
            index + 1,
            window["invocation"],
        ), "feedback source/epoch differs"
        assert update["bias"] == int(abs(weights[-1]) * 10) % 5, "wrong learned policy update"
        expected_inputs = [["mixed.left", index], ["mixed.right", index]]
        assert window["inputs"] == update["inputs"] == expected_inputs, "feedback lost advanced input membership"
        assert item["at"] < window["at"] < update["at"], "feedback preceded actual learning completion"
    for item in selections:
        earlier = [update for update in updates if update["at"] < item["at"]]
        revision = len(earlier)
        bias = earlier[-1]["bias"] if earlier else 0
        assert (item["revision"], item["bias"], item["value"]) == (revision, bias, item["key"] + bias), (
            "selection ignored actual applied policy"
        )
        excluded = revision > 0 and item["key"] % 4 == 0
        assert (item["ordinal"] is None) == excluded, "dynamic exclusion differs"
    assert source["excluded"] == [item["key"] for item in selections if item["ordinal"] is None], "excluded record reached fanout"
    assert not source["active"] and source["revision"] == 3 and source["bias"] == updates[-1]["bias"], (
        "live selection/policy did not settle"
    )
    assert 1 <= source["max_active"] <= 3, "aggregate horizon exceeded"
    # Reconstruct live admission independently from selection and feedback clocks.
    active: set[int] = set()
    for _, kind, key in sorted(
        [(item["at"], "admit", item["key"]) for item in admitted] + [(item["at"], "retire", item["key"]) for item in updates]
    ):
        if kind == "admit":
            assert key not in active
            active.add(key)
        else:
            assert key in active
            active.remove(key)
        assert len(active) <= 3, "report concealed a horizon violation"
    assert {item["key"] for item in source["settlements"]} == active, "unacknowledged ticket disposition lost"
    events = source["events"]
    stop = next(item["at"] for item in events if item["event"] == "stop")
    drained = next(item["at"] for item in events if item["event"] == "producer_drained")
    assert all(item["at"] < stop for item in selections), "selection continued after stop"
    products = source["products"]
    for item in products:
        selection = selected[item["key"]]
        assert (item["ordinal"], item["selected"], item["revision"], item["bias"]) == (
            selection["ordinal"],
            selection["value"],
            selection["revision"],
            selection["bias"],
        ), "publication changed ticket"
        assert selection["ordinal"] is not None and item["branch"] in ("left", "right"), "excluded or misrouted publication"
        stage = stages[item["key"], "data.offset" if item["branch"] == "left" else "data.encode"]
        close(item["value"], stage["output"], "physical product")
        if item["branch"] == "right":
            assert item["sources"] == stage["sources"] and stage["release"] < item["attempt"], "source borrow crossed blocked publication"
        else:
            assert item["sources"] == [], "spurious model authority on pure branch"
    for (key, name), stage in stages.items():
        value = selected[key]["value"]
        if name == "data.encode":
            if stage["entered"] is not None:
                assert stage["entered"] < stage["physical_end"] < stage["release"] < drained, "physical read released early"
                stamp = stage["sources"][0]
                assert stamp[:2] == ["learner", 1] and 0 <= stamp[2] <= 3, "source binding/epoch changed"
                # A protected read cannot straddle a subsequent returned update.
                for window in windows[stamp[2] :]:
                    assert stage["release"] < window["at"], "read overlapped an update of its binding"
                if stage["outcome"] in ("returned", "returned_cancelled"):
                    close(stage["output"], weights[stamp[2]] * value / 4, "actual production state")
        else:
            assert name in ("data.double", "data.offset") and stage["outcome"] in ("returned", "entered_failed"), "unknown local stage"
            if stage["outcome"] == "returned":
                expected_in, expected_out = (value, value * 2) if name == "data.double" else (value * 2, value * 2 + 1)
                close(stage["input"], expected_in, "local input")
                close(stage["output"], expected_out, "local output")
    outcomes = {item["invocation"]: item for item in report["unit_outcomes"]}
    assert len(outcomes) == 3, "optimization owner outcomes lost"
    for index, contribution in enumerate(contributions):
        item = outcomes[contribution["invocation"]]
        assert (item["contribution"], item["optimizer"], item["reset"]) == ("ready", "returned", "returned"), (
            "false optimization completion"
        )
        assert item["advance_invocation"] == windows[index]["invocation"] > contribution["invocation"], "wrong retained advancement"
        assert item["inputs"] == [{"channel": name, "window": index} for name in ("mixed.left", "mixed.right")], "unit lost inputs"
    for name in ("mixed.left", "mixed.right"):
        assert report["claimed"][name] == [[index, contributions[index]["invocation"]] for index in range(3)], "atomic claims differ"
        assert report["ports_closed"][name] and report["high_water"][name] <= 1, "bounded ports did not close"
        branch = name.removeprefix("mixed.")
        unclaimed = {item["ordinal"] for item in products if item["branch"] == branch and item["returned"] is not None} - {0, 1, 2}
        assert set(report["remaining"][name]).isdisjoint(report["discarded"][name]), "input both retained and discarded"
        assert set(report["remaining"][name]) | set(report["discarded"][name]) == unclaimed, "unclaimed handoff disposition lost"
        if outcome == "stopped":
            assert not report["remaining"][name], "normal stop did not perform declared discard"
    if pressure:
        assert 4 in source["excluded"] and selected[0]["revision"] == 0, "dynamic policy exclusion not exercised"
        assert any(item["key"] > 2 and item["revision"] > 0 and item["value"] != item["key"] for item in selections), (
            "feedback did not alter later selection"
        )
        assert any(item["returned"] is None and item["attempt"] < stop for item in products), "no blocked publication at stop"
        assert any(
            stage["entered"] < stop < stage["physical_end"]
            for (_, name), stage in stages.items()
            if name == "data.encode" and stage["entered"] is not None
        ), "no live physical model use at stop"
