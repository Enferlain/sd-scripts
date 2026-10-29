"""Optional C authority exchange. No flow or numerical runtime dependency."""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
)
from xdsl.traits import NoTerminator

from .dialects import parse
from .model_dialect import ModelData


@irdl_attr_definition
class ContributionType(ParametrizedAttribute, TypeAttribute):
    name = "grant.contribution"
    unit: SymbolRefAttr


@irdl_op_definition
class PhaseOp(IRDLOperation):
    name = "grant.phase"
    source_id = prop_def(StringAttr)
    phase = prop_def(StringAttr)


@irdl_op_definition
class GrantOp(IRDLOperation):
    name = "grant.execute"
    source_id = prop_def(StringAttr)
    model = prop_def(SymbolRefAttr)
    unit = prop_def(SymbolRefAttr)
    implementation = prop_def(SymbolRefAttr)
    profile = prop_def(StringAttr)
    delegated = prop_def(ArrayAttr[StringAttr])
    retained = prop_def(ArrayAttr[StringAttr])
    body = region_def("single_block")
    handback = result_def(ContributionType)
    traits = traits_def(NoTerminator())


@irdl_op_definition
class AdvanceGrantedOp(IRDLOperation):
    name = "grant.advance"
    source_id = prop_def(StringAttr)
    unit = prop_def(SymbolRefAttr)
    parameter = prop_def(StringAttr)
    learning_rate = prop_def(FloatAttr)
    clip_norm = opt_prop_def(FloatAttr)
    handback = operand_def(ContributionType)


Grant = Dialect("grant", [PhaseOp, GrantOp, AdvanceGrantedOp], [ContributionType])


def parse_grant(text: str) -> ModuleOp:
    return parse(text, extensions=(ModelData, Grant))
