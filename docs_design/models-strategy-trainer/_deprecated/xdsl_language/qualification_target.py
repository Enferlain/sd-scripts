"""Maintained product bindings selected by the cold qualification target."""

from __future__ import annotations

from dataclasses import replace

from .prepare import BindingSpec, ResourceSpec
from .qualification_products import products
from .target import Binding, Effect, Installed, ResourceBinding, Result
from .target_bindings import capability, installed, numeric, record


def publish_measurement(args: tuple[object, ...]) -> Result:
    cap, value = capability(args[0]), record(args[1])
    return Result((value,), (Effect(0, "assign", (*products(cap.value), {"kind": "measurement", **value})),))


def publish_score(args: tuple[object, ...]) -> Result:
    cap, item, value = capability(args[0]), record(args[1]), numeric(args[2])
    product = {"kind": "score", "key": item["key"], "input": item, "value": value}
    return Result((value,), (Effect(0, "assign", (*products(cap.value), product)),))


def selected(*, wrong_score: bool = False) -> Installed:
    base = installed()

    def validate(value: object) -> None:
        products(value)

    additions = (
        Binding(BindingSpec("publish_measurement", ("capability", "measurement"), ("measurement",), updates=(0,)), publish_measurement),
        Binding(BindingSpec("publish_score", ("capability", "record", "scalar"), ("scalar",), updates=(0,)), publish_score),
    )
    selected_bindings = tuple(
        Binding(item.spec, lambda args: Result((0.0,))) if wrong_score and item.spec.name == "score" else item for item in base.bindings
    )
    return replace(
        base,
        bindings=(*selected_bindings, *additions),
        resources=(*base.resources, ResourceBinding(ResourceSpec("products", "state"), products, validate)),
    )
