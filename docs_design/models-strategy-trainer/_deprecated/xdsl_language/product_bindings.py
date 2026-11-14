"""Actual storage views and retained Torch graphs on the existing owner target."""

from __future__ import annotations

import torch

from . import product_contracts
from .target import Binding, Effect, ExecutionError, Installed, ResourceBinding, Result


def installed(kind: str = "buffer", *, derived: bool = False, wrong: bool = False) -> Installed:
    def initialize(value):
        if kind == "buffer":
            return value
        side, base, x = value
        return (torch.tensor(side, dtype=torch.float64, requires_grad=True), torch.tensor(base, dtype=torch.float64), x)

    def make(args):
        value = args[0].value
        if kind == "buffer":
            return Result((memoryview(value)[1:4].toreadonly(),))
        side, base, x = value
        with torch.enable_grad():
            activation = base * (side * x)
        return Result((activation,))

    def derive(args):
        if kind == "buffer":
            return Result((args[0][:2],))
        with torch.enable_grad():
            return Result((args[0] * 2,))

    def consume(args):
        value, cap = args
        if kind == "buffer":
            if not isinstance(value, memoryview) or value.obj is not cap.value or not value.readonly:
                raise ExecutionError("buffer result lost its declared read-only storage alias")
            result = float(sum(value))
        else:
            if value.is_leaf or value.grad_fn is None:
                raise ExecutionError("autograd product lost its retained derivative graph")
            with torch.enable_grad():
                loss = (value - 1).square() / 2
                result = float(torch.autograd.grad(loss, cap.value[0], retain_graph=True)[0])
            if cap.value[1].requires_grad or cap.value[0].grad is not None:
                raise ExecutionError("frozen base or gradient ownership changed")
        return Result((result + (1 if wrong else 0),))

    def publish(args):
        return Result((args[1],), (Effect(0, "assign", args[1]),))

    def validate(value):
        if kind == "buffer" and not isinstance(value, bytearray):
            raise ExecutionError("buffer storage must be a bytearray")
        if kind != "buffer" and not (isinstance(value, tuple) and len(value) == 3 and value[0].is_leaf and value[1].is_leaf):
            raise ExecutionError("autograd storage must own leaf tensors")

    def validate_result(value):
        if type(value) not in (int, float):
            raise ExecutionError("product result must be detached numeric data")

    implementations = {
        "make_" + kind: make,
        "derive_" + kind: derive,
        "consume_" + kind: consume,
        "publish": publish,
        "replace": lambda args: Result((), (Effect(0, "assign", args[0].value),)),
    }
    manifest = product_contracts.manifest(kind, derived=derived)
    return Installed(
        tuple(Binding(spec, implementations[spec.name]) for spec in manifest.bindings),
        (
            ResourceBinding(manifest.resources[0], initialize, validate),
            ResourceBinding(manifest.resources[1], lambda value: value, validate_result),
        ),
        {"view": lambda value: isinstance(value, memoryview) and value.readonly}
        if kind == "buffer"
        else {"autograd": lambda value: isinstance(value, torch.Tensor) and value.ndim == 0 and value.grad_fn is not None},
    )
