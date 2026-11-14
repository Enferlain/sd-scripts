"""Independent expected products and conservation checks over owner receipts."""

from __future__ import annotations

from collections.abc import Mapping

from .contracts import Version
from .reference import ReferenceError, Settlement


EXPECTED = ((0, 2), (2, 6), (4, 10))


def check_products(value: object, expected: tuple = EXPECTED) -> None:
    if value != expected:
        raise ReferenceError(f"filter products differ from independent expectation: {value!r}")


def check_settlement(raw: Mapping[int, Settlement], kept: Mapping[int, Settlement], expected: tuple | None = EXPECTED) -> None:
    consumed = {key for key, item in kept.items() if item.disposition == "consumed"}
    if (expected is not None and consumed != {key for key, _ in expected}) or not consumed <= {key for key, _ in EXPECTED}:
        raise ReferenceError("retained input identity/disposition does not match required demand")
    for key in consumed:
        output, source = kept[key], raw[key]
        if output.provenance != (("raw", key),) or output.versions != (Version("config", 1, "initial", 0),):
            raise ReferenceError("retained provenance/observed versions were changed")
        if source.disposition != "forwarded" or source.forwarded_to != (("kept", key),):
            raise ReferenceError("retained value has no corresponding input forwarding receipt")
    if expected:
        for key in range(max(key for key, _ in expected)):
            if key % 2 and (key not in raw or raw[key].disposition != "filtered_out"):
                raise ReferenceError("required dropped input lost its filtering disposition")
    for key, source in raw.items():
        if source.disposition == "filtered_out" and (
            source.forwarded_to
            or key not in kept
            or kept[key].disposition != "no_output"
            or source.versions != (Version("config", 1, "initial", 0),)
        ):
            raise ReferenceError("dropped input did not settle without a downstream item")
        if source.disposition == "forwarded":
            output = kept.get(key)
            if (
                source.forwarded_to != (("kept", key),)
                or output is None
                or output.provenance != (("raw", key),)
                or output.versions != source.versions
                or source.versions != (Version("config", 1, "initial", 0),)
            ):
                raise ReferenceError("forwarding provenance/versions changed during settlement")


def check_conservation(*, selected: int, settled: Mapping[int, Settlement], active: int, ready: int, high_water: int, horizon: int) -> None:
    if active or ready or set(settled) != set(range(selected)) or high_water > horizon:
        raise ReferenceError("filter flow admission, settlement or bound conservation failed")


def check_report(report: dict, scenario: str) -> None:
    """Recheck actual serialized owner state, including in the parent process."""
    required = {
        "late_input": ("demand_survived_drop",),
        "prefetched": ("aggregate_bound_exhausted",),
        "held_shutdown": ("group_finished_before_producers", "filter_closed_before_source"),
        "late_source_failure": ("group_finished_before_producers", "filter_closed_before_source"),
        "cancel_filter": ("claim_protected_through_physical_filter",),
        "fail_filter": ("claim_protected_through_physical_filter",),
        "cancel_then_fail_filter": ("claim_protected_through_physical_filter",),
        "orderly_requested": ("entered_iteration_owns_input",),
        "cancel_local": ("entered_iteration_owns_input",),
        "all_drops": ("progress_during_exclusions",),
        "invalid_stage": ("claim_retained_during_sibling_failure",),
    }.get(scenario, ())
    if set(report["conditions"]) != {*required, "run_raised"} or any(report["conditions"][key] is not True for key in required):
        raise ReferenceError("required external filtering boundary was not observed")
    failure = scenario in {
        "late_source_failure",
        "cancel_filter",
        "fail_filter",
        "cancel_then_fail_filter",
        "cancel_local",
        "invalid_stage",
        "all_drops",
    }
    if report["conditions"]["run_raised"] != failure or not report["drained"] or report["units"]:
        raise ReferenceError("filter run outcome or physical/owner settlement differs")
    for flow in report["flows"].values():
        if (
            flow["active"]
            or flow["ready"]
            or flow["permits"] != 3
            or flow["high_water"] > 3
            or set(map(int, flow["terminal"])) != set(range(flow["selected"]))
        ):
            raise ReferenceError("serialized filter admission and drain conservation failed")
    if any(item["pending"] or item["active"] or item["permits"] != 1 for item in report["requests"]):
        raise ReferenceError("requested work has no final settlement")

    def receipts(name):
        return {
            int(key): Settlement(
                item["disposition"],
                tuple(Version(**version) for version in item["versions"]),
                tuple(tuple(link) for link in item["provenance"]),
                tuple(tuple(link) for link in item["forwarded_to"]),
            )
            for key, item in report["settlements"][name].items()
        }

    value = tuple(tuple(item) for item in report["resources"]["products"]["value"])
    expected = None
    if scenario in {"normal", "late_input", "prefetched", "held_shutdown", "late_source_failure"}:
        expected = EXPECTED
        check_products(value)
        if not any(item["source"] == "consumers" and item["status"] == "succeeded" for item in report["source_receipts"]):
            raise ReferenceError("consumer completion receipt is absent")
    if scenario == "orderly_requested":
        expected = EXPECTED[:2]
        check_products(value, EXPECTED[:2])
    if scenario == "cancel_local":
        expected = EXPECTED[:1]
        check_products(value, EXPECTED[:1])
    check_settlement(receipts("raw"), receipts("kept"), expected)
    if scenario == "late_source_failure" and not any("late physical source failure" in message for message in report["failures"]):
        raise ReferenceError("late producer failure was lost")
    if scenario in {"fail_filter", "cancel_then_fail_filter"} and not any(
        "physical filter failure" in message for message in report["failures"]
    ):
        raise ReferenceError("entered filter failure was lost")
    if scenario == "invalid_stage" and any(name in {"first.double", "next.double"} for name, _ in report["physical_entered"]):
        raise ReferenceError("invalid output authorized a later fused stage")
    if scenario == "invalid_stage" and receipts("raw")[1].disposition != "discarded":
        raise ReferenceError("sibling failure must discard an interrupted filter input")
    if scenario == "cancel_local" and any(name == "next.double" for name, _ in report["physical_entered"]):
        raise ReferenceError("cancellation authorized a later fused stage")
