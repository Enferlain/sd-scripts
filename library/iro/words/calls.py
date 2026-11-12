"""Call words: call, constant and yield (D16).

A call names an implementation; the strategy's ``uses`` header pins its
version. A retained result keeps a dependency on every lease and retained
input of its call until its last use. A constant prints inline where it is
used, with Python's type names for its value.
"""

import math

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, IntegerType, StringAttr, SymbolRefAttr
from xdsl.ir import Attribute
from xdsl.irdl import irdl_op_definition, opt_prop_def, prop_def, result_def, traits_def, var_operand_def, var_result_def
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException

from library.iro.operation_contracts import DecisionSemantics, RealizationSemantics, path
from library.iro.words.base import DeclaredFields, Word
from library.iro.words.types import LeaseType, RetainedType, ValueType


def literal_type(literal: Attribute) -> str | None:
    """Python's type name for a literal attribute, or None for one it cannot be."""
    if isinstance(literal, IntegerAttr) and isinstance(literal.type, IntegerType):
        return "bool" if literal.type.width.data == 1 else "int"
    if isinstance(literal, FloatAttr):
        return "float"
    if isinstance(literal, StringAttr):
        return "str"
    return None


@irdl_op_definition
class CallOp(Word):
    """Invokes an implementation by name."""

    name = "iro.call"

    inputs = var_operand_def()
    outputs = var_result_def()
    implementation = prop_def(SymbolRefAttr)
    retains = opt_prop_def(ArrayAttr[IntegerAttr])
    traits = traits_def(RealizationSemantics(), DecisionSemantics(), DeclaredFields())

    def verify_(self) -> None:
        retaining = any(isinstance(result.type, RetainedType) for result in self.outputs)
        expected = retained_positions(self) if retaining else ()
        written = tuple(item.value.data for item in self.retains.data) if self.retains is not None else ()
        if written != expected:
            raise VerifyException(f"{path(self)}: a retained result keeps every lease and retained input, positions {expected}")


def retained_positions(call: CallOp) -> tuple[int, ...]:
    """The inputs a retained result depends on: every lease and every retained input."""
    return tuple(i for i, value in enumerate(call.inputs) if isinstance(value.type, LeaseType | RetainedType))


@irdl_op_definition
class ConstantOp(Word):
    """A literal. It prints inline where it is used."""

    name = "iro.constant"

    value = prop_def(Attribute)
    result = result_def(ValueType)
    traits = traits_def(DeclaredFields())

    def verify_(self) -> None:
        natural = literal_type(self.value)
        if natural is None:
            raise VerifyException(f"{path(self)}: a literal is an int, float, bool or str")
        if isinstance(self.value, FloatAttr) and not math.isfinite(self.value.value.data):
            raise VerifyException(f"{path(self)}: a float literal is finite")
        written = self.result.type
        if isinstance(written, ValueType) and written.text in ("int", "float", "bool", "str") and written.text != natural:
            raise VerifyException(f"{path(self)}: a {natural} literal cannot be a {written.text}")


@irdl_op_definition
class YieldOp(Word):
    """Returns from a body."""

    name = "iro.yield"

    values = var_operand_def()
    traits = traits_def(IsTerminator(), DeclaredFields())


OPERATIONS = (CallOp, ConstantOp, YieldOp)

__all__ = ["OPERATIONS", "CallOp", "ConstantOp", "YieldOp", "literal_type", "retained_positions"]
