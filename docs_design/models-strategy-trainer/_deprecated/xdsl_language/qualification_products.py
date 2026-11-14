"""Explicit owned products for qualification; no numerical implementation reuse."""

from __future__ import annotations

from .reference import CallResult, Capability, StandIn, Update
from .standins import bindings, numeric, record


def products(value: object) -> tuple:
    if not isinstance(value, tuple | list) or any(not isinstance(item, dict) for item in value):
        raise TypeError("published products require a sequence of typed records")
    return tuple(value)


def publish_measurement(args: tuple[object, ...]) -> CallResult:
    cap, value = args[0], record(args[1])
    if not isinstance(cap, Capability):
        raise TypeError("publication requires owner authority")
    return CallResult((value,), (Update(0, (*products(cap.value), {"kind": "measurement", **value})),))


def publish_score(args: tuple[object, ...]) -> CallResult:
    cap, item, value = args[0], record(args[1]), numeric(args[2])
    if not isinstance(cap, Capability):
        raise TypeError("publication requires owner authority")
    product = {"kind": "score", "key": item["key"], "input": item, "value": value}
    return CallResult((value,), (Update(0, (*products(cap.value), product)),))


def reference_bindings() -> dict[str, StandIn]:
    return {
        **bindings(),
        "publish_measurement": StandIn(("capability", "measurement"), ("measurement",), publish_measurement, updates=(0,)),
        "publish_score": StandIn(("capability", "record", "scalar"), ("scalar",), publish_score, updates=(0,)),
    }
