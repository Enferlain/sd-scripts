"""Selected CPU computations; lifecycle and scheduling stay in the owner target."""

from __future__ import annotations

import importlib
import math

import torch

from .prepare import BindingSpec, ResourceSpec
from .target import Binding, Capability, Effect, Installed, ResourceBinding, Result


# Reuse the previous case's numerical functions, never its whole-case lower().
math_bindings = importlib.import_module("docs_design.models-strategy-trainer.pliron_candidate.composition_runtime")


def capability(value: object) -> Capability:
    if not isinstance(value, Capability) or not value.active:
        raise TypeError("selected CPU computation requires an active capability")
    return value


def numeric(value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise TypeError("selected CPU computation requires a finite scalar")
    return float(value)


def record(value: object) -> dict:
    if not isinstance(value, dict) or type(value.get("key")) is not int:
        raise TypeError("selected CPU computation requires a record")
    return value


def model(value: object) -> torch.nn.Module:
    if not isinstance(value, torch.nn.Module):
        raise TypeError("selected computation requires the prepared model")
    return value


def as_record(value: object):
    item = record(value)
    return math_bindings.Record(item["key"], torch.tensor(numeric(item["x"]), dtype=torch.float64), numeric(item["target"]))


def encode(args: tuple[object, ...]) -> Result:
    cap, key = capability(args[0]), int(numeric(args[1]))
    value = math_bindings.encode(cap.value, key)
    return Result(({"key": key, "x": float(value), "target": key + 1},))


def gradient(args: tuple[object, ...]) -> Result:
    cap = capability(args[0])
    with torch.enable_grad():
        loss = math_bindings.objective(cap.value, as_record(args[1]), 2)
        value = torch.autograd.grad(loss, model(cap.value).get_parameter("weight"))[0]
    return Result((numeric(args[2]) + float(value),))


def advance(args: tuple[object, ...]) -> Result:
    cap, value = capability(args[0]), numeric(args[1])
    return Result((float(model(cap.value).get_parameter("weight").detach()) - value / 100,), (Effect(0, "gradient", value),))


def measure(args: tuple[object, ...]) -> Result:
    cap, item = capability(args[0]), record(args[1])
    prediction = float(math_bindings.measure(cap.value, as_record(item)))
    return Result(
        (
            {
                "key": item["key"],
                "epoch": cap.version.ordinal,
                "weight": float(model(cap.value).get_parameter("weight").detach()),
                "prediction": prediction,
                "loss": (prediction - item["target"]) ** 2 / 2,
            },
        )
    )


def equal(args: tuple[object, ...]) -> Result:
    return Result((args[0] == args[1],))


def select(args: tuple[object, ...]) -> Result:
    cap, key = capability(args[0]), int(numeric(args[1]))
    return Result(({"key": key, "x": key + int(numeric(cap.value)), "target": 0},))


def score(args: tuple[object, ...]) -> Result:
    return Result(((2 * numeric(record(args[0])["x"]) + 1) ** 2,))


def feedback(args: tuple[object, ...]) -> Result:
    value = int(numeric(args[1]))
    return Result((value,), (Effect(0, "assign", value % 5),))


def load_scale(initial: object):
    model = math_bindings.load_scale(numeric(initial), True).double()
    with torch.no_grad():
        model.weight.fill_(numeric(initial))
    return model


def check_scale(value: object) -> None:
    if not isinstance(value, torch.nn.Module) or next(iter(value.named_parameters()))[0] != "weight":
        raise TypeError("scale realization requires the declared scalar weight")
    if value.weight.shape != torch.Size([]) or value.weight.dtype != torch.float64:
        raise TypeError("scale realization requires a float64 scalar")


def check_policy(value: object) -> None:
    numeric(value)


def schemas():
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


def installed() -> Installed:
    return Installed(
        (
            Binding(BindingSpec("encode", ("capability", "index"), ("record",)), encode),
            Binding(BindingSpec("gradient", ("capability", "record", "scalar"), ("scalar",)), gradient),
            Binding(BindingSpec("advance", ("capability", "scalar"), ("scalar",), updates=(0,)), advance),
            Binding(BindingSpec("measure", ("capability", "record"), ("measurement",)), measure),
            Binding(BindingSpec("equal", ("index", "index"), ("bool",)), equal),
            Binding(BindingSpec("select", ("capability", "index"), ("record",)), select),
            Binding(BindingSpec("score", ("record",), ("scalar",)), score),
            Binding(BindingSpec("feedback", ("capability", "scalar"), ("scalar",), updates=(0,)), feedback),
        ),
        (
            ResourceBinding(ResourceSpec("scale", "participant", parameter="weight"), load_scale, check_scale),
            ResourceBinding(ResourceSpec("policy", "state"), numeric, check_policy),
        ),
        schemas(),
    )
