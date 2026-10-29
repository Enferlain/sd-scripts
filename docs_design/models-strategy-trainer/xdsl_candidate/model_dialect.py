"""Optional data/model contract: pure IRDL definitions, with no Torch imports.

The common workflow structures remain shared with B. Model participation,
contribution windows and evaluation live in this explicitly installed extension.
"""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, prop_def, region_def, traits_def
from xdsl.traits import NoTerminator

from .dialects import FlowType, parse


@irdl_op_definition
class SourceOp(IRDLOperation):
    name = "model_data.source"
    source_id = prop_def(StringAttr)
    keys = prop_def(ArrayAttr[IntegerAttr])
    output = operand_def(FlowType)


@irdl_op_definition
class FilterOp(IRDLOperation):
    name = "model_data.filter"
    source_id = prop_def(StringAttr)
    excluded = prop_def(ArrayAttr[IntegerAttr])
    input = operand_def(FlowType)
    output = operand_def(FlowType)


@irdl_op_definition
class FanOutOp(IRDLOperation):
    name = "model_data.fanout"
    source_id = prop_def(StringAttr)
    input = operand_def(FlowType)
    left = operand_def(FlowType)
    right = operand_def(FlowType)


@irdl_op_definition
class ModelOp(IRDLOperation):
    name = "model_data.model"
    source_id = prop_def(StringAttr)
    entity = prop_def(SymbolRefAttr)
    implementation = prop_def(SymbolRefAttr)
    initial = prop_def(FloatAttr)
    treatment = prop_def(StringAttr)


@irdl_op_definition
class ModelFlowOp(IRDLOperation):
    name = "model_data.apply"
    source_id = prop_def(StringAttr)
    model = prop_def(SymbolRefAttr)
    implementation = prop_def(SymbolRefAttr)
    input = operand_def(FlowType)
    output = operand_def(FlowType)


@irdl_op_definition
class CorrelateOp(IRDLOperation):
    name = "model_data.correlate"
    source_id = prop_def(StringAttr)
    membership = prop_def(StringAttr)
    provenance = prop_def(StringAttr)
    left = operand_def(FlowType)
    right = operand_def(FlowType)
    output = operand_def(FlowType)


@irdl_op_definition
class WindowOp(IRDLOperation):
    name = "model_data.window"
    source_id = prop_def(StringAttr)
    sym_name = prop_def(StringAttr)
    count = prop_def(IntegerAttr)
    input = operand_def(FlowType)
    body = region_def("single_block")
    traits = traits_def(NoTerminator())


@irdl_op_definition
class ContributeOp(IRDLOperation):
    name = "model_data.contribute"
    source_id = prop_def(StringAttr)
    model = prop_def(SymbolRefAttr)
    unit = prop_def(SymbolRefAttr)
    implementation = prop_def(SymbolRefAttr)
    normalization = prop_def(IntegerAttr)
    input = operand_def(FlowType)


@irdl_op_definition
class AdvanceOp(IRDLOperation):
    name = "model_data.advance"
    source_id = prop_def(StringAttr)
    unit = prop_def(SymbolRefAttr)
    learning_rate = prop_def(FloatAttr)
    parameter = prop_def(StringAttr)


@irdl_op_definition
class EvaluateOp(IRDLOperation):
    name = "model_data.evaluate"
    source_id = prop_def(StringAttr)
    owner = prop_def(SymbolRefAttr)
    model = prop_def(SymbolRefAttr)
    trigger = prop_def(SymbolRefAttr)
    after_window = prop_def(IntegerAttr)
    implementation = prop_def(SymbolRefAttr)
    consistency = prop_def(StringAttr)


@irdl_op_definition
class AddressOp(IRDLOperation):
    name = "model_data.address"
    origin = prop_def(StringAttr)
    sample = prop_def(IntegerAttr)
    window = prop_def(IntegerAttr)
    slot = prop_def(IntegerAttr)
    left = prop_def(StringAttr)
    right = prop_def(StringAttr)


@irdl_op_definition
class RecipeOp(IRDLOperation):
    """Bounded owner-backend recipe with a retained semantic operation clone."""

    name = "model_data.recipe"
    origin = prop_def(StringAttr)
    role = prop_def(StringAttr)
    protocol = prop_def(StringAttr)
    body = region_def("single_block")
    traits = traits_def(NoTerminator())


ModelData = Dialect(
    "model_data",
    [
        SourceOp,
        FilterOp,
        FanOutOp,
        ModelOp,
        ModelFlowOp,
        CorrelateOp,
        WindowOp,
        ContributeOp,
        AdvanceOp,
        EvaluateOp,
        AddressOp,
        RecipeOp,
    ],
    [],
)


def parse_model(text: str) -> ModuleOp:
    return parse(text, extensions=(ModelData,))
