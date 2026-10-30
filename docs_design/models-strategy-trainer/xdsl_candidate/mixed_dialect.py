"""Installed mixed-workflow operations; no numerical or runtime dependency."""

from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, prop_def

from .dialects import FlowType, parse
from .model_dialect import ModelData


@irdl_op_definition
class DynamicFilterOp(IRDLOperation):
    name = "mixed.dynamic_filter"
    source_id = prop_def(StringAttr)
    policy = prop_def(SymbolRefAttr)
    implementation = prop_def(SymbolRefAttr)
    input = operand_def(FlowType)
    output = operand_def(FlowType)


@irdl_op_definition
class LearnedOp(IRDLOperation):
    name = "mixed.learned"
    source_id = prop_def(StringAttr)
    model = prop_def(SymbolRefAttr)
    output = operand_def(FlowType)
    observation = prop_def(StringAttr)


@irdl_op_definition
class PolicyUpdateOp(IRDLOperation):
    name = "mixed.policy_update"
    source_id = prop_def(StringAttr)
    policy = prop_def(SymbolRefAttr)
    implementation = prop_def(SymbolRefAttr)
    input = operand_def(FlowType)
    acknowledgement = operand_def(FlowType)
    protocol = prop_def(StringAttr)


@irdl_op_definition
class StopCheckpointOp(IRDLOperation):
    name = "mixed.stop_checkpoint"
    source_id = prop_def(StringAttr)
    activity = prop_def(SymbolRefAttr)
    after_windows = prop_def(IntegerAttr)
    protocol = prop_def(StringAttr)


Mixed = Dialect("mixed", [DynamicFilterOp, LearnedOp, PolicyUpdateOp, StopCheckpointOp], [])


def parse_mixed(text: str) -> ModuleOp:
    return parse(text, extensions=(ModelData, Mixed))
