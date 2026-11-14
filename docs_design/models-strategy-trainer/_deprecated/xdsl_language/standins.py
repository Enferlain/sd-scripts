"""Declared fixture computations; not model bindings or a lifecycle oracle."""

from __future__ import annotations

from collections.abc import Callable

from .reference import CallResult, Capability, StandIn, Update


def capability(value: object) -> Capability:
    if not isinstance(value, Capability):
        raise TypeError("fixture expected an owner-issued capability")
    return value


def record(value: object) -> dict:
    if not isinstance(value, dict):
        raise TypeError("fixture expected a record")
    return value


def numeric(value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError("fixture expected a numeric scalar")
    return float(value)


def integer(value: object) -> int:
    if type(value) is not int:
        raise TypeError("fixture expected an integer")
    return value


def encode(args: tuple[object, ...]) -> CallResult:
    view, key = capability(args[0]), integer(args[1])
    return CallResult(({"key": key, "x": numeric(view.value) * (key + 1), "target": key + 1},))


def gradient(args: tuple[object, ...]) -> CallResult:
    view, item = capability(args[0]), record(args[1])
    total = numeric(args[2]) + (numeric(view.value) * item["x"] - item["target"]) * item["x"] / 2
    return CallResult((total,))


def advance(args: tuple[object, ...]) -> CallResult:
    value = numeric(capability(args[0]).value) - numeric(args[1]) / 100
    return CallResult((value,), (Update(0, value),))


def equal(args: tuple[object, ...]) -> CallResult:
    return CallResult((args[0] == args[1],))


def measure(args: tuple[object, ...]) -> CallResult:
    view, item = capability(args[0]), record(args[1])
    prediction = numeric(view.value) * item["x"]
    return CallResult(
        (
            {
                "key": item["key"],
                "epoch": view.version.ordinal,
                "weight": view.value,
                "prediction": prediction,
                "loss": (prediction - item["target"]) ** 2 / 2,
            },
        )
    )


def select(args: tuple[object, ...]) -> CallResult:
    view, key = capability(args[0]), integer(args[1])
    return CallResult(({"key": key, "x": key + int(numeric(view.value)), "target": 0},))


def score(args: tuple[object, ...]) -> CallResult:
    return CallResult(((2 * record(args[0])["x"] + 1) ** 2,))


def feedback(args: tuple[object, ...]) -> CallResult:
    score = int(numeric(args[1]))
    return CallResult((score,), (Update(0, score % 5),))


def bindings() -> dict[str, StandIn]:
    return {
        "encode": StandIn(("capability", "index"), ("record",), encode),
        "gradient": StandIn(("capability", "record", "scalar"), ("scalar",), gradient),
        "advance": StandIn(("capability", "scalar"), ("scalar",), advance, updates=(0,)),
        "equal": StandIn(("index", "index"), ("bool",), equal),
        "measure": StandIn(("capability", "record"), ("measurement",), measure),
        "select": StandIn(("capability", "index"), ("record",), select),
        "score": StandIn(("record",), ("scalar",), score),
        "feedback": StandIn(("capability", "scalar"), ("scalar",), feedback, updates=(0,)),
    }


def schemas() -> dict[str, Callable[[object], bool]]:
    def numeric_record(value: object) -> bool:
        return (
            isinstance(value, dict)
            and type(value.get("key")) is int
            and all(type(value.get(key)) in (int, float) for key in ("x", "target"))
        )

    def measurement(value: object) -> bool:
        return (
            isinstance(value, dict)
            and type(value.get("key")) is int
            and type(value.get("epoch")) is int
            and all(type(value.get(key)) in (int, float) for key in ("weight", "prediction", "loss"))
        )

    return {"record": numeric_record, "measurement": measurement}
