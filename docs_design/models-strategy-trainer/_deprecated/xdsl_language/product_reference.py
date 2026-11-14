"""Logical products and independent arithmetic; no Torch or actual target views."""

from __future__ import annotations

from . import product_case
from .normalize import normalize
from .reference import CallResult, Reference, StandIn, Update


def reference(kind: str = "buffer", *, derived: bool = False, history: bool = True, update: str = "none") -> Reference:
    def make(args):
        value = args[0].value
        if kind == "buffer":
            return CallResult((tuple(value[1:4]),))
        side, base, x = value
        return CallResult(((side * base * x, base * x),))

    def derive(args):
        value = args[0]
        return CallResult((value[:2] if kind == "buffer" else (2 * value[0], 2 * value[1]),))

    def consume(args):
        value = args[0]
        return CallResult((sum(value) if kind == "buffer" else (value[0] - 1) * value[1],))

    def publish(args):
        return CallResult((args[1],), (Update(0, args[1]),))

    implementations = {
        "make_" + kind: make,
        "derive_" + kind: derive,
        "consume_" + kind: consume,
        "publish": publish,
        "replace": lambda args: CallResult((), (Update(0, args[0].value),)),
    }
    bindings = {
        spec.name: StandIn(spec.inputs, spec.outputs, implementations[spec.name], spec.updates, spec.version, spec.retains)
        for spec in product_case.manifest(kind, derived=derived).bindings
    }
    return Reference(
        normalize(product_case.program(kind, derived=derived, update=update)),
        product_case.initial(kind),
        bindings,
        schemas={"view": lambda value: isinstance(value, tuple), "autograd": lambda value: isinstance(value, tuple) and len(value) == 2},
        history=history,
    )


def expected(kind: str, derived: bool = False) -> float:
    # Independent fixture expectations, never encoded as executable IR values.
    return (197.0 if derived else 297.0) if kind == "buffer" else (1128.0 if derived else 276.0)
