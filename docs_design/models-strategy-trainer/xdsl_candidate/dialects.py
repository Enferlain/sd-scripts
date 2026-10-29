"""Small semantic and prepared xDSL dialects for the data-only pressure case.

SSA values identify persistent endpoints, not individual runtime records.
Nested local regions use block arguments/yields; cyclic feedback is an explicit
connection between endpoint users. Region order is declaration order only.
"""

from __future__ import annotations

from io import StringIO

from xdsl.context import Context
from xdsl.dialects.builtin import ArrayAttr, Builtin, IntegerAttr, ModuleOp, StringAttr, SymbolRefAttr, i64
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import IsolatedFromAbove, IsTerminator, NoTerminator
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class FlowType(ParametrizedAttribute, TypeAttribute):
    name = "wf.flow"
    payload: StringAttr

    def verify(self) -> None:
        if self.payload.data not in ("record", "score", "feedback"):
            raise VerifyException("wf.flow: unknown payload contract")


@irdl_op_definition
class SessionOp(IRDLOperation):
    name = "wf.session"
    source_id = prop_def(StringAttr)
    contract = prop_def(StringAttr)
    version = prop_def(IntegerAttr)
    horizon = prop_def(IntegerAttr)
    stop_rule = prop_def(StringAttr)
    body = region_def("single_block")
    traits = traits_def(NoTerminator(), IsolatedFromAbove())

    def verify_(self) -> None:
        if self.horizon.value.data <= 0:
            raise VerifyException(f"{self.source_id.data}: horizon must be positive")


@irdl_op_definition
class EntityOp(IRDLOperation):
    name = "wf.entity"
    sym_name = prop_def(StringAttr)
    source_id = prop_def(StringAttr)
    kind = prop_def(StringAttr)


@irdl_op_definition
class FlowOp(IRDLOperation):
    name = "wf.endpoint"
    source_id = prop_def(StringAttr)
    endpoint = result_def(FlowType)


@irdl_op_definition
class SelectOp(IRDLOperation):
    name = "data.select"
    source_id = prop_def(StringAttr)
    cursor = prop_def(SymbolRefAttr)
    policy = prop_def(SymbolRefAttr)
    feedback = operand_def(FlowType)
    output = operand_def(FlowType)
    application = prop_def(StringAttr)


@irdl_op_definition
class ComposeOp(IRDLOperation):
    name = "wf.compose"
    source_id = prop_def(StringAttr)
    input = operand_def(FlowType)
    output = operand_def(FlowType)
    body = region_def("single_block")


@irdl_op_definition
class MapOp(IRDLOperation):
    name = "data.map"
    source_id = prop_def(StringAttr)
    implementation = prop_def(SymbolRefAttr)
    effect = prop_def(StringAttr)
    input = operand_def(FlowType)
    output = result_def(FlowType)


@irdl_op_definition
class LocalOp(IRDLOperation):
    """Recursive local composition; one record enters and one leaves."""

    name = "wf.local"
    source_id = prop_def(StringAttr)
    input = operand_def(FlowType)
    output = result_def(FlowType)
    body = region_def("single_block")


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "wf.yield"
    value = operand_def(FlowType)
    traits = traits_def(IsTerminator())


@irdl_op_definition
class ScoreOp(IRDLOperation):
    name = "data.score"
    source_id = prop_def(StringAttr)
    implementation = prop_def(SymbolRefAttr)
    input = operand_def(FlowType)
    output = operand_def(FlowType)


@irdl_op_definition
class PublishOp(IRDLOperation):
    name = "data.publish"
    source_id = prop_def(StringAttr)
    implementation = prop_def(SymbolRefAttr)
    owner = prop_def(SymbolRefAttr)
    input = operand_def(FlowType)
    feedback = operand_def(FlowType)
    acknowledgement = prop_def(StringAttr)


@irdl_op_definition
class PlanOp(IRDLOperation):
    name = "exec.plan"
    accepted_digest = prop_def(StringAttr)
    target = prop_def(StringAttr)
    contract = prop_def(StringAttr)
    horizon = prop_def(IntegerAttr)
    capacity = prop_def(IntegerAttr)
    stop_rule = prop_def(StringAttr)
    feedback = prop_def(StringAttr)
    body = region_def("single_block")
    traits = traits_def(NoTerminator(), IsolatedFromAbove())


@irdl_op_definition
class ChainOp(IRDLOperation):
    name = "exec.chain"
    role = prop_def(StringAttr)
    transport = prop_def(StringAttr)
    body = region_def("single_block")
    traits = traits_def(NoTerminator())


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "exec.call"
    origin = prop_def(StringAttr)
    ancestors = prop_def(ArrayAttr[StringAttr])
    implementation = prop_def(SymbolRefAttr)
    binding_version = prop_def(StringAttr)
    effect = prop_def(StringAttr)
    role = prop_def(StringAttr)
    completion = prop_def(StringAttr)


@irdl_op_definition
class SelectRecipeOp(IRDLOperation):
    name = "exec.select"
    origin = prop_def(StringAttr)
    cursor = prop_def(SymbolRefAttr)
    policy = prop_def(SymbolRefAttr)
    selection = prop_def(StringAttr)
    application = prop_def(StringAttr)


@irdl_op_definition
class HandoffRecipeOp(IRDLOperation):
    name = "exec.handoff"
    origin = prop_def(StringAttr)
    endpoint = prop_def(StringAttr)
    reader = prop_def(StringAttr)
    release = prop_def(StringAttr)


@irdl_op_definition
class FeedbackRecipeOp(IRDLOperation):
    name = "exec.feedback"
    origin = prop_def(StringAttr)
    publication = prop_def(SymbolRefAttr)
    policy = prop_def(SymbolRefAttr)
    update = prop_def(StringAttr)
    admission = prop_def(StringAttr)


Workflow = Dialect("wf", [SessionOp, EntityOp, FlowOp, ComposeOp, LocalOp, YieldOp], [FlowType])
Data = Dialect("data", [SelectOp, MapOp, ScoreOp, PublishOp], [])
Execution = Dialect("exec", [PlanOp, ChainOp, CallOp, SelectRecipeOp, HandoffRecipeOp, FeedbackRecipeOp], [])


def integer(value: int) -> IntegerAttr:
    return IntegerAttr(value, i64)


def dump(module: ModuleOp) -> str:
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue() + "\n"


def parse(text: str) -> ModuleOp:
    context = Context()
    for dialect in (Builtin, Workflow, Data, Execution):
        context.load_dialect(dialect)
    return Parser(context, text).parse_module()
