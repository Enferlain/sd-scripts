"""Real owner bindings for the data-cardinality fixture; no scheduler here."""

from __future__ import annotations

from .filter_contracts import RESOURCES, SPECS, products, record
from .target import Binding, Capability, Effect, ExecutionError, Installed, ResourceBinding, Result


def installed(*, reject_all: bool = False, wrong_double: bool = False) -> Installed:
    def produce(args):
        access, ordinal = args
        if not isinstance(access, Capability):
            raise ExecutionError("expected a target capability")
        return Result(({"key": ordinal, "x": access.value * ordinal},))

    def publish(args):
        access, item, result = args
        appended = (*access.value, (item["key"], result))
        return Result((result,), (Effect(0, "assign", appended),))

    implementations = {
        "filter_produce": produce,
        "filter_even": lambda args: Result((not reject_all and not args[0]["key"] % 2,)),
        "filter_extract": lambda args: Result((args[0]["x"],)),
        "filter_offset": lambda args: Result((1 + args[0],)),
        "filter_double": lambda args: Result((0 if wrong_double else 2 * args[0],)),
        "filter_opaque": lambda args: Result(args),
        "filter_publish": publish,
    }

    def validate_config(value):
        if type(value) not in (int, float):
            raise ExecutionError("expected a numeric filter config")

    def validate_products(value):
        if not products(value):
            raise ExecutionError("expected filter products")

    return Installed(
        tuple(Binding(spec, implementations[spec.name]) for spec in SPECS),
        (
            ResourceBinding(RESOURCES[0], lambda value: value, validate_config),
            ResourceBinding(RESOURCES[1], lambda value: value, validate_products),
        ),
        {"record": record},
    )
