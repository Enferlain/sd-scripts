"""Independent numerical stand-ins; does not import the target or authoring."""

from __future__ import annotations

from .filter_contracts import SPECS, record
from .reference import CallResult, Capability, ReferenceError, StandIn, Update


def bindings(*, reject_all: bool = False) -> dict[str, StandIn]:
    def produce(args):
        cap, key = args
        if not isinstance(cap, Capability):
            raise ReferenceError("expected a reference capability")
        return CallResult(({"key": key, "x": key * cap.value},))

    def publish(args):
        cap, item, value = args
        return CallResult((value,), (Update(0, (*cap.value, (item["key"], value))),))

    functions = {
        "filter_produce": produce,
        "filter_even": lambda args: CallResult((False if reject_all else args[0]["key"] % 2 == 0,)),
        "filter_extract": lambda args: CallResult((args[0]["x"],)),
        "filter_offset": lambda args: CallResult((args[0] + 1,)),
        "filter_double": lambda args: CallResult((args[0] * 2,)),
        "filter_opaque": lambda args: CallResult(args),
        "filter_publish": publish,
    }
    return {spec.name: StandIn(spec.inputs, spec.outputs, functions[spec.name], updates=spec.updates) for spec in SPECS}


SCHEMAS = {"record": record}
