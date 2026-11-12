"""Lifetime words: producer, handler, request and stop (D16)."""

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, SymbolRefAttr
from xdsl.irdl import irdl_op_definition, operand_def, opt_prop_def, prop_def, region_def, traits_def
from xdsl.traits import IsolatedFromAbove
from xdsl.utils.exceptions import VerifyException

from library.iro.operation_contracts import (
    Boundary,
    CompletionSemantics,
    DecisionSemantics,
    EndpointSemantics,
    EndpointUse,
    LifecycleProtocol,
    ReferenceSemantics,
    RegionSemantics,
    path,
)
from library.iro.words.base import Activity, BodyShape, DeclaredFields, Named, Word
from library.iro.words.types import ValueType


@irdl_op_definition
class ProducerOp(Activity):
    """An activity that keeps producing items into a stream.

    Operand 0 is the stream; the rest are captures. Each production receives
    its position and yields one item, or an optional for zero or one.
    """

    name = "iro.producer"

    body = region_def("single_block")
    traits = traits_def(
        RegionSemantics(LifecycleProtocol.PRODUCER),
        EndpointSemantics(EndpointUse(0, "produce")),
        BodyShape(),
        IsolatedFromAbove(),
        DeclaredFields(),
        Named(),
    )


@irdl_op_definition
class HandlerOp(Activity):
    """An activity that serves requests from declared sources.

    ``max_outstanding`` bounds the requests accepted and not yet settled; the
    first target runs one at a time.
    """

    name = "iro.handler"

    sources = prop_def(ArrayAttr[SymbolRefAttr])
    request = prop_def(ValueType)
    max_outstanding = prop_def(IntegerAttr)
    max_active = opt_prop_def(IntegerAttr)
    body = region_def("single_block")
    traits = traits_def(
        RegionSemantics(LifecycleProtocol.HANDLER),
        ReferenceSemantics((("sources", "activity"),)),
        BodyShape(),
        IsolatedFromAbove(),
        DeclaredFields(),
        Named(),
    )

    def verify_(self) -> None:
        if not self.sources.data:
            raise VerifyException(f"{path(self)}: a handler serves requests from at least one source")
        if self.max_outstanding.value.data <= 0:
            raise VerifyException(f"{path(self)}: a handler accepts at least one request")
        if self.max_active is not None and self.max_active.value.data != 1:
            raise VerifyException(f"{path(self)}: the first target runs one request at a time")


@irdl_op_definition
class RequestOp(Word):
    """An owned pending request for a handler, made when its source reaches a boundary."""

    name = "iro.request"

    payload = operand_def(ValueType)
    handler = prop_def(SymbolRefAttr)
    source = prop_def(SymbolRefAttr)
    boundary = prop_def(StringAttr)
    traits = traits_def(
        ReferenceSemantics((("handler", "handler"), ("source", "activity"))),
        CompletionSemantics("request", "handler"),
        DecisionSemantics(),
        DeclaredFields(),
    )

    def verify_(self) -> None:
        try:
            Boundary(self.boundary.data)
        except ValueError as failed:
            raise VerifyException(f"{path(self)}: unknown boundary {self.boundary.data}") from failed


@irdl_op_definition
class StopOp(Word):
    """Closes the target activity's admission once the source activity settles."""

    name = "iro.stop"

    target = prop_def(SymbolRefAttr)
    source = prop_def(SymbolRefAttr)
    traits = traits_def(
        ReferenceSemantics((("source", "activity"), ("target", "activity"))),
        CompletionSemantics("stop", "target"),
        DecisionSemantics(),
        DeclaredFields(),
    )


OPERATIONS = (ProducerOp, HandlerOp, RequestOp, StopOp)

__all__ = ["OPERATIONS", "HandlerOp", "ProducerOp", "RequestOp", "StopOp"]
