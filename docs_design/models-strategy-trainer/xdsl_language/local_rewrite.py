"""Contract-driven unary local fusion, using xDSL rewriting and SSA remapping.

The group removes local interpreter dispatch. Every selected stage still has
its own physical entry, validation and failure boundary in the target.
"""

from __future__ import annotations

import hashlib
import json

from collections.abc import Mapping

from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern
from xdsl.utils.exceptions import VerifyException

from .contracts import BindingContract, origin, strings, symbol, text
from .dialects import CallOp, FusedLocalOp, PayloadType, YieldOp, integer
from .normalize import Normalized, ancestry, source_index


def barrier(op: Operation, contracts: Mapping[str, BindingContract]) -> str | None:
    if not isinstance(op, CallOp):
        return "local operation has an authority, cardinality or completion boundary"
    selected = contracts.get(symbol(op.binding))
    if selected is None or selected.version != text(op, "binding_version"):
        return "selected binding/version is unavailable"
    if not selected.effects.pure:
        return "selected binding requires resource access or has unknown effects"
    if len(op.operands) != 1 or len(op.results) != 1:
        return "only detached unary payload stages are supported"
    input_type, output_type = op.operands[0].type, op.results[0].type
    if not isinstance(input_type, PayloadType) or not isinstance(output_type, PayloadType):
        return "only detached payload stages are supported"
    if selected.inputs != (input_type.schema.data,) or selected.outputs != (output_type.schema.data,):
        return "selected signature does not match actual operands/results"
    return None


def validate_groups(module, contracts: Mapping[str, BindingContract]) -> None:
    for group in module.walk():
        if not isinstance(group, FusedLocalOp):
            continue
        block = group.body.block
        stages = tuple(block.ops)[:-1]
        if len(stages) < 2 or len(group.operands) != 1 or len(group.results) != 1 or len(block.args) != 1:
            raise VerifyException(f"{origin(group)}: invalid local fusion arity/stage count")
        if strings(group, "members") != tuple(origin(stage) for stage in stages):
            raise VerifyException(f"{origin(group)}: fused member/source correspondence failed")
        value = block.args[0]
        for stage in stages:
            if (reason := barrier(stage, contracts)) is not None:
                raise VerifyException(f"{origin(group)}: cannot fuse {origin(stage)}: {reason}")
            if tuple(stage.operands) != (value,):
                raise VerifyException(f"{origin(group)}: local fusion does not follow its declared SSA chain")
            if stage is not stages[0] and value.uses.get_length() != 1:
                raise VerifyException(f"{origin(group)}: intermediate fused value has additional uses")
            value = stage.results[0]
        if not isinstance(block.last_op, YieldOp) or tuple(block.last_op.operands) != (value,):
            raise VerifyException(f"{origin(group)}: local fusion does not return its final stage value")


class FusePureStages(RewritePattern):
    def __init__(self, contracts: Mapping[str, BindingContract]):
        self.contracts = contracts

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if isinstance(op.parent_op(), FusedLocalOp) or barrier(op, self.contracts) is not None:
            return
        chain = [op]
        following = op.next_op
        while following is not None and barrier(following, self.contracts) is None:
            value = chain[-1].results[0]
            if tuple(following.operands) != (value,) or value.uses.get_length() != 1:
                break
            chain.append(following)
            following = following.next_op
        if len(chain) < 2:
            return
        names = tuple(origin(stage) for stage in chain)
        name = "$fused/" + hashlib.sha256(json.dumps(names).encode()).hexdigest()
        block = Block(arg_types=[op.operands[0].type])
        mapper: dict[SSAValue, SSAValue] = {op.operands[0]: block.args[0]}
        for stage in chain:
            block.add_op(stage.clone(value_mapper=mapper))
        block.add_op(
            YieldOp.create(
                operands=[mapper[chain[-1].results[0]]],
                properties={"origin": StringAttr(name + "/yield"), "version": integer(1), "ancestry": ArrayAttr([])},
            )
        )
        group = FusedLocalOp.create(
            operands=op.operands,
            result_types=[chain[-1].results[0].type],
            properties={
                "origin": StringAttr(name),
                "version": integer(1),
                "ancestry": ArrayAttr([]),
                "sym_name": StringAttr(name),
                "members": ArrayAttr(StringAttr(n) for n in names),
                "quiescence": ArrayAttr([]),
            },
            regions=[Region(block)],
        )
        rewriter.replace(chain[-1], group)
        for stage in reversed(chain[:-1]):
            rewriter.erase(stage)


def fuse(program: Normalized, contracts: Mapping[str, BindingContract]) -> Normalized:
    detached = program.module.clone()
    PatternRewriteWalker(FusePureStages(contracts), walk_regions_first=False).rewrite_module(detached)
    for op in source_index(detached).values():
        op.properties["ancestry"] = ArrayAttr(StringAttr(value) for value in ancestry(op))
    detached.verify()
    validate_groups(detached, contracts)
    return Normalized(detached, program.accepted_origins)
