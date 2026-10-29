"""Independent scalar C oracle. No xDSL, Torch or owner/runtime imports."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def close(actual: float, expected: float, name: str) -> None:
    assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=2e-6, abs_tol=2e-6), f"{name}: {actual} != {expected}"


def reference(initial: float = 0.8, learning_rate: float = 0.05, clip_norm: float | None = None) -> dict[str, float]:
    # Each loss is (2*w-1)^2. The temporary step follows the first gradient;
    # final SGD applies the second gradient to the restored original weight.
    first_gradient = 4 * (2 * initial - 1)
    temporary = initial + 0.1 * first_gradient
    final_gradient = 4 * (2 * temporary - 1)
    applied = final_gradient
    if clip_norm is not None:
        applied *= min(1.0, clip_norm / (abs(final_gradient) + 1e-6))
    return {
        "initial": initial,
        "first_gradient": first_gradient,
        "temporary": temporary,
        "final_gradient": final_gradient,
        "final_weight": initial - learning_rate * applied,
    }


def check_c(report: Mapping[str, Any], *, initial: float = 0.8, learning_rate: float = 0.05, clip_norm: float | None = None) -> None:
    expected = reference(initial, learning_rate, clip_norm)
    assert report["status"] == "stopped" and not report["errors"], "unexpected grant session outcome"
    assert report["released"] and not report["pending"] and not report["unusable"], "unfinished/released ownership differs"
    assert report["steps"] == 1 and report["epoch"] == 2 and report["handback_verified"], "retained advancement differs"
    assert report["final_gradient"] is None, "retained final reset missing"
    selected = report["selected"]
    assert selected["entered"] and selected["selected_returned"] and selected["restored"], "false selected handback"
    phases = ("backward", "temporary_edit", "intermediate_reset", "backward", "restore")
    assert selected["phases"] == [{"phase": phase, "origin": f"phase.{index}.{phase}"} for index, phase in enumerate(phases)], (
        "phase/source correspondence differs"
    )
    close(selected["initial_weight"], expected["initial"], "initial weight")
    close(selected["selected_end_weight"], expected["initial"], "restored weight")
    close(selected["selected_gradient"], expected["final_gradient"], "independent second gradient")
    close(report["final_weight"], expected["final_weight"], "independent retained SGD")
    records = report["unit_outcomes"]
    assert len(records) == 1, "lost granted contribution outcome"
    item = records[0]
    assert (item["contribution"], item["optimizer"], item["reset"]) == ("granted_ready", "returned", "returned"), (
        "optimization completion differs"
    )
    assert item["invocation"] > 0 and item["advance_invocation"] == item["invocation"], "wrong addressed grant/advancement"
    assert item["work"] in (("grant-session", "selected-two-pass"), ["grant-session", "selected-two-pass"]) and item["input_count"] == 0, (
        "grant source/input association differs"
    )


def check_unverified(report: Mapping[str, Any], *, cancelled: bool = False) -> None:
    assert report["status"] == ("cancelled" if cancelled else "failed") and report["errors"], "false failed-handback outcome"
    assert report["released"] and report["pending"] and report["unusable"] == ["model"], "failed grant ownership lost"
    assert report["steps"] == 0 and report["epoch"] == 0 and not report["handback_verified"], "unverified grant authorized advancement"
    assert report["selected"]["entered"], "selected grant never entered"
    if cancelled:
        assert not report["selected"]["selected_returned"], "interrupted selected grant returned successfully"
    records = report["unit_outcomes"]
    assert len(records) == 1, "active unverified grant outcome lost"
    item = records[0]
    assert (item["contribution"], item["optimizer"], item["reset"], item["advance_invocation"]) == (
        "unverified_grant",
        "not_attempted",
        "not_attempted",
        None,
    ), "failed handback was promoted"
    assert item["work"] in (("grant-session", "selected-two-pass"), ["grant-session", "selected-two-pass"]) and item["input_count"] == 0
