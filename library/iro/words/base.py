"""What every word shares: its authored origin, its declarations and the body shape check."""

from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.ir import Attribute, Operation
from xdsl.irdl import IRDLOperation, opt_prop_def, var_operand_def, var_result_def
from xdsl.traits import OpTrait
from xdsl.utils.exceptions import VerifyException

from library.iro.operation_contracts import (
    LifecycleProtocol,
    RegionSemantics,
    SUPPORTED_QUIESCENCE,
    decision,
    number,
    path,
    resource_uses,
)
from library.iro.words.types import (
    ClaimType,
    OptionalType,
    ResourceType,
    StreamType,
    ValueType,
    bool_type,
    int_type,
    lease,
    optional,
    written,
)


class Word(IRDLOperation):
    """Fields every Kuroiro operation may carry.

    ``origin`` is where it was authored. ``ancestry`` records what it was
    normalized from. The decision fields are its declarations, written only
    when they are not the default.
    """

    origin = opt_prop_def(StringAttr)
    ancestry = opt_prop_def(ArrayAttr[StringAttr])
    reproducibility = opt_prop_def(StringAttr)
    rank_rule = opt_prop_def(StringAttr)
    assumptions = opt_prop_def(ArrayAttr[StringAttr])


class Activity(Word):
    """An operation with bodies: its captures in, its results out."""

    sym_name = opt_prop_def(StringAttr)
    quiescence = opt_prop_def(ArrayAttr[StringAttr])
    inputs = var_operand_def()
    outputs = var_result_def()


class DeclaredFields(OpTrait):
    """Only declared fields, with readable declarations.

    xDSL already rejects undeclared properties; this also rejects its free-form
    attribute dictionary, so a written form cannot carry unchecked meaning.
    """

    def verify(self, op: Operation) -> None:
        if op.attributes:
            raise VerifyException(f"{op.name} {path(op)}: undeclared fields {sorted(op.attributes)}")
        decision(op)


class Named(OpTrait):
    """An activity that other statements refer to by its authored name."""

    def verify(self, op: Operation) -> None:
        if not isinstance(op.properties.get("sym_name"), StringAttr):
            raise VerifyException(f"{op.name} {path(op)}: needs a name")


class BodyShape(OpTrait):
    """Each body's entries and yields implement its lifecycle protocol's region contract."""

    def verify(self, op: Operation) -> None:
        semantics = op.get_trait(RegionSemantics)
        if semantics is None:
            raise VerifyException(f"{op.name} {path(op)}: a body needs a lifecycle protocol")
        contract = semantics.describe(op)
        operands = tuple(value.type for value in op.operands)
        results = tuple(value.type for value in op.results)
        for boundary in contract.quiescence:
            if boundary not in SUPPORTED_QUIESCENCE[contract.protocol]:
                raise VerifyException(f"{path(op)}: {contract.protocol} defines no cut at {boundary}")
        for role in contract.roles:
            entries, yields = _expected(op, contract.protocol, role.index, operands, results)
            if role.entries != entries:
                raise VerifyException(f"{path(op)}: body {role.index} takes {_names(role.entries)}, expected {_names(entries)}")
            if role.yields != yields:
                raise VerifyException(f"{path(op)}: body {role.index} yields {_names(role.yields)}, expected {_names(yields)}")


def _expected(
    op: Operation, protocol: LifecycleProtocol, index: int, operands: tuple[Attribute, ...], results: tuple[Attribute, ...]
) -> tuple[tuple[Attribute, ...], tuple[Attribute, ...]]:
    match protocol:
        case LifecycleProtocol.PRODUCER:
            stream = operands[0] if operands else None
            if not isinstance(stream, StreamType):
                raise VerifyException(f"{path(op)}: a producer produces into a stream")
            produced = op.regions[0].block.last_op
            yielded = tuple(value.type for value in produced.operands) if produced is not None else ()
            # A producer may yield zero or one item per production.
            accepted = (optional(stream.payload),) if yielded == (optional(stream.payload),) else (stream.payload,)
            if results:
                raise VerifyException(f"{path(op)}: a producer has no results")
            return (*operands[1:], int_type()), accepted
        case LifecycleProtocol.REPEAT:
            carried = number(op, "carry_count")
            if not 0 <= carried <= len(operands):
                raise VerifyException(f"{path(op)}: invalid carried values")
            captures, carries = operands[: len(operands) - carried], operands[len(operands) - carried :]
            if results != carries:
                raise VerifyException(f"{path(op)}: a repeat's results are its carried values")
            return (*captures, int_type(), *carries), carries
        case LifecycleProtocol.ACCESS:
            uses = resource_uses(op)
            subjects = operands[: len(uses)]
            if len(subjects) != len(uses) or not all(isinstance(s, ResourceType) for s in subjects):
                raise VerifyException(f"{path(op)}: each resource use needs a resource")
            leases = tuple(lease(s, u.mode.data) for s, u in zip(subjects, uses, strict=True) if isinstance(s, ResourceType))
            return (*leases, *operands[len(uses) :]), results
        case LifecycleProtocol.FILTER:
            taken = operands[0] if operands else None
            if not isinstance(taken, ClaimType) or results != (optional(taken.payload),):
                raise VerifyException(f"{path(op)}: a filter takes one claim and returns an optional of its value")
            if any(isinstance(t, ClaimType) for t in operands[1:]):
                raise VerifyException(f"{path(op)}: a filter takes exactly one claim")
            return (taken.payload, *operands[1:]), (bool_type(),)
        case LifecycleProtocol.CHOICE:
            condition = operands[0] if operands else None
            if len(op.regions) != 2 or not (condition == bool_type() or isinstance(condition, OptionalType)):
                raise VerifyException(f"{path(op)}: a choice needs a bool or an optional and two alternatives")
            present = (condition.payload,) if isinstance(condition, OptionalType) and index == 0 else ()
            return (*present, *operands[1:]), results
        case LifecycleProtocol.HANDLER:
            carried = op.properties.get("request")
            if not isinstance(carried, ValueType) or results:
                raise VerifyException(f"{path(op)}: a handler serves requests carrying a value and has no results")
            return (*operands, carried), ()
        case LifecycleProtocol.PARALLEL:
            if results:
                raise VerifyException(f"{path(op)}: parallel branches yield nothing")
            return operands, ()
        case _:
            return operands, results


def _names(types: tuple[Attribute, ...]) -> str:
    return "(" + ", ".join(written(t) for t in types) + ")"


__all__ = ["Activity", "BodyShape", "DeclaredFields", "Named", "Word"]
