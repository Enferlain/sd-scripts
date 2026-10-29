"""Independent test-only oracle for B's scalar arithmetic and causal facts.

No xDSL/runtime/binding imports. This accepts JSON-shaped observations so later
implementations need not copy the current engine or diagnostic event log.
Only this fixture's selected arithmetic/provenance is assumed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def check_b(report: Mapping[str, Any], *, horizon: int = 2) -> None:
    assert report["status"] in ("stopped", "cancelled", "failed"), "unknown session outcome"
    assert (report["status"] != "stopped" or not report["errors"]) and (report["status"] != "failed" or report["errors"]), (
        "dishonest failure outcome"
    )
    tickets = report["tickets"]
    assert tickets and [ticket["key"] for ticket in tickets] == list(range(len(tickets))), "missing/duplicate ticket identity"
    stop, release = report["stop_clock"], report["released"]
    assert 0 < stop < release, "selection stop must precede session release"
    by_key = {ticket["key"]: ticket for ticket in tickets}
    transitions: list[tuple[int, str, Mapping[str, Any]]] = []
    selected_keys: set[int] = set()
    feedback_keys: set[int] = set()
    for decision in report["decisions"]:
        assert decision["key"] in by_key, "decision lost ticket association"
        transitions.append((decision["clock"], decision["kind"], decision))
    for ticket in tickets:
        key, value = ticket["key"], ticket["value"]
        assert ticket["source_revision"] == 0, "source revision changed"
        assert ticket["selected"] < stop and ticket["retired"] < release, "work outlived selected lifetime"
        assert value == key + ticket["policy_bias"], "selection value/provenance differs"
        inputs = (value, 2 * value, 2 * value + 1, (2 * value + 1) ** 2)
        outputs = (2 * value, 2 * value + 1, (2 * value + 1) ** 2, (2 * value + 1) ** 2)
        origins = ("data.double", "data.offset", "scorer", "publisher")
        roles = ("map", "map", "score", "publish")
        previous = ticket["selected"]
        calls = ticket["calls"]
        assert 0 < len(calls) <= 4, "lost selected invocation evidence"
        for index, call in enumerate(calls):
            assert (call["origin"], call["role"]) == (origins[index], roles[index]), "source mapping differs"
            assert call["input"] == inputs[index], "stage input differs"
            assert previous < call["entered"] < call["physical_end"] < call["released"] < ticket["retired"], "physical lifetime differs"
            assert call["outcome"] in ("returned", "entered_failed", "invalid_output"), "unknown call outcome"
            if call["outcome"] == "returned":
                assert call["output"] == outputs[index], "independent arithmetic differs"
            else:
                assert index == len(calls) - 1, "continued after failed invocation"
            previous = call["released"]
        if ticket["terminal"] == "acknowledged":
            assert len(calls) == 4 and all(call["outcome"] == "returned" for call in calls), "false successful ticket"
            assert previous < ticket["publication_ack"] < ticket["feedback_ack"] < ticket["retired"], (
                "publication/feedback ordering differs"
            )
        else:
            assert ticket["terminal"] in ("failed", "abandoned") and ticket["feedback_ack"] is None, "false feedback success"
            if ticket["terminal"] == "failed":
                assert calls[-1]["outcome"] != "returned", "failed ticket lost failure attribution"
            transitions.append((ticket["retired"], "retired", ticket))
    active: set[int] = set()
    revision = bias = peak = 0
    for _, kind, item in sorted(transitions, key=lambda transition: transition[0]):
        key = item["key"]
        ticket = by_key[key]
        if kind == "selected":
            assert key not in selected_keys, "duplicate selection"
            assert item["clock"] == ticket["selected"] and item["clock"] < stop, "selection boundary differs"
            assert (item["revision"], item["bias"]) == (revision, bias), "actual policy at selection differs"
            assert (ticket["policy_revision"], ticket["policy_bias"]) == (revision, bias), "relabelled policy provenance"
            active.add(key)
            selected_keys.add(key)
            peak = max(peak, len(active))
            assert peak <= horizon, "aggregate outstanding bound exceeded"
        elif kind == "feedback":
            assert key in active and key not in feedback_keys, "feedback is duplicate or lacks selected work"
            assert ticket["terminal"] == "acknowledged" and item["clock"] == ticket["feedback_ack"], "unacknowledged feedback"
            revision += 1
            bias = ((2 * ticket["value"] + 1) ** 2) % 5
            assert (item["revision"], item["bias"]) == (revision, bias), "feedback decision differs"
            active.remove(key)
            feedback_keys.add(key)
        else:
            assert kind == "retired" and key in active, "unknown or duplicate retirement"
            active.remove(key)
    assert selected_keys == set(by_key) and not active, "lost active association"
    assert feedback_keys == {ticket["key"] for ticket in tickets if ticket["terminal"] == "acknowledged"}, "lost feedback evidence"
    assert peak == report["high_water"], "incorrect outstanding accounting"
