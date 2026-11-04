"""Selected contracts and payload validators, independent of either executor."""

from __future__ import annotations

from .prepare import BindingSpec, Manifest, ResourceSpec


SPECS = (
    BindingSpec(
        "filter_produce", ("capability", "index"), ("record",), reads=(0,), unknown_effects=False, resource_kinds=((0, "filter_config"),)
    ),
    BindingSpec("filter_even", ("record",), ("bool",), reads=(), unknown_effects=False),
    BindingSpec("filter_extract", ("record",), ("scalar",), reads=(), unknown_effects=False),
    BindingSpec("filter_offset", ("scalar",), ("scalar",), reads=(), unknown_effects=False),
    BindingSpec("filter_double", ("scalar",), ("scalar",), reads=(), unknown_effects=False),
    BindingSpec("filter_opaque", ("scalar",), ("scalar",), reads=(), unknown_effects=True),
    BindingSpec(
        "filter_publish",
        ("capability", "record", "scalar"),
        ("scalar",),
        updates=(0,),
        reads=(0,),
        unknown_effects=False,
        resource_kinds=((0, "filter_products"),),
    ),
)
RESOURCES = (ResourceSpec("filter_config", "state"), ResourceSpec("filter_products", "state"))
MANIFEST = Manifest(SPECS, RESOURCES, ("record",))
INITIAL = {"config": 1, "products": ()}


def record(value: object) -> bool:
    return isinstance(value, dict) and set(value) == {"key", "x"} and type(value["key"]) is int and type(value["x"]) in (int, float)


def products(value: object) -> bool:
    return isinstance(value, tuple) and all(
        isinstance(item, tuple) and len(item) == 2 and type(item[0]) is int and type(item[1]) in (int, float) for item in value
    )
