"""Declared fixture computations for the conformance cases, ported from the research stand-ins.

They honor their implementations' contracts and supply fixture values; they
are not model code. Leases arrive as the state they give access to.
"""

from collections.abc import Callable

from library.iro.reference import CallResult, StandIn, Update

LEASE = "protected_current lease"
SNAPSHOT = "snapshot lease"
UPDATE = "owner_update lease"


def number(value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"a fixture expected a number, got {value!r}")
    return float(value)


def integer(value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"a fixture expected an int, got {value!r}")
    return value


def record(value: object) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"a fixture expected a record, got {value!r}")
    return value


# The lifetime composition: x = encoder * (key + 1), target = key + 1.


def encode(given: tuple[object, ...]) -> CallResult:
    scale, key = number(given[0]), integer(given[1])
    return CallResult(({"key": key, "x": scale * (key + 1), "target": key + 1},))


def gradient(given: tuple[object, ...]) -> CallResult:
    weight, item, total = number(given[0]), record(given[1]), number(given[2])
    return CallResult((total + (weight * item["x"] - item["target"]) * item["x"] / 2,))


def advance(given: tuple[object, ...]) -> CallResult:
    weight = number(given[0]) - number(given[1]) / 100
    return CallResult((weight,), (Update(0, weight),))


def equal(given: tuple[object, ...]) -> CallResult:
    return CallResult((given[0] == given[1],))


def measure(given: tuple[object, ...]) -> CallResult:
    weight, item = number(given[0]), record(given[1])
    prediction = weight * item["x"]
    return CallResult(({"key": item["key"], "prediction": prediction, "loss": (prediction - item["target"]) ** 2 / 2},))


def lifetime() -> dict[str, StandIn]:
    return {
        "encode": StandIn((SNAPSHOT, "int"), ("record",), encode),
        "gradient": StandIn((LEASE, "record", "scalar"), ("scalar",), gradient),
        "advance": StandIn((UPDATE, "scalar"), ("scalar",), advance, updates=(0,)),
        "equal": StandIn(("int", "int"), ("bool",), equal),
        "measure": StandIn((LEASE, "record"), ("measurement",), measure),
    }


# The data-only control: the design's @score_records.


def select(given: tuple[object, ...]) -> CallResult:
    policy, key = integer(given[0]), integer(given[1])
    return CallResult(({"key": key, "x": key + policy},))


def score(given: tuple[object, ...]) -> CallResult:
    return CallResult(((2 * record(given[0])["x"] + 1) ** 2,))


def feedback(given: tuple[object, ...]) -> CallResult:
    return CallResult((), (Update(0, int(number(given[1])) % 5),))


def data_only() -> dict[str, StandIn]:
    return {
        "select": StandIn((LEASE, "int"), ("record",), select),
        "score": StandIn(("record",), ("scalar",), score),
        "feedback": StandIn((UPDATE, "scalar"), (), feedback, updates=(0,)),
    }


# Filtering: keep records whose x is even, then publish 2 * (x + 1).


def produce(given: tuple[object, ...]) -> CallResult:
    offset, key = integer(given[0]), integer(given[1])
    return CallResult(({"key": key, "x": key + offset},))


def publish(given: tuple[object, ...]) -> CallResult:
    published, item, value = given[0], record(given[1]), number(given[2])
    assert isinstance(published, tuple)
    return CallResult((), (Update(0, (*published, (item["key"], value))),))


def filtering() -> dict[str, StandIn]:
    return {
        "produce": StandIn((SNAPSHOT, "int"), ("record",), produce),
        "even": StandIn(("record",), ("bool",), lambda given: CallResult((record(given[0])["x"] % 2 == 0,))),
        "extract": StandIn(("record",), ("scalar",), lambda given: CallResult((float(record(given[0])["x"]),))),
        "offset": StandIn(("scalar",), ("scalar",), lambda given: CallResult((number(given[0]) + 1,))),
        "double": StandIn(("scalar",), ("scalar",), lambda given: CallResult((number(given[0]) * 2,))),
        "publish": StandIn((UPDATE, "record", "scalar"), (), publish, updates=(0,)),
    }


def validators() -> dict[str, Callable[[object], bool]]:
    return {
        "record": lambda value: isinstance(value, dict) and type(value.get("key")) is int,
        "scalar": lambda value: type(value) in (int, float),
        "measurement": lambda value: isinstance(value, dict) and {"key", "prediction", "loss"} <= set(value),
    }
