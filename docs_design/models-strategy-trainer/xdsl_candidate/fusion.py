"""One conservative xDSL rewrite of prepared local binding selections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr
from xdsl.ir import Block, Operation, Region
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from .dialects import CallOp, ChainOp, FlowType, FusedOp, PlanOp, SessionOp
from .local_contract import LocalTransform, TransparentLocal

PROFILE = "ordered-validated-local-v1"


class FusionError(ValueError):
    pass


def eligible(call: CallOp, sources: Mapping[str, Operation]) -> bool:
    source = sources.get(call.origin.data)
    if source is None or call.role.data != "map" or call.effect.data != "pure":
        return False
    interface = source.get_trait(LocalTransform)
    description = interface.describe(source) if interface is not None else None
    scalar = FlowType(StringAttr("record"))
    if description is None or (
        description.input_type,
        description.output_type,
        description.cardinality,
        description.resources,
        description.outcomes,
    ) != (scalar, scalar, "one-to-one", (), ("returned", "invalid_output", "raised")):
        return False
    parent = source.parent_op()
    while parent is not None and not isinstance(parent, SessionOp):
        if parent.get_trait(TransparentLocal) is None:
            return False
        parent = parent.parent_op()
    return isinstance(parent, SessionOp)


class FuseLocal(RewritePattern):
    def __init__(self, sources: Mapping[str, Operation]):
        self.sources = sources

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PlanOp | ChainOp, rewriter: PatternRewriter) -> None:
        if isinstance(op, ChainOp) and op.role.data != "producer":
            return
        run: list[CallOp] = []
        for child in (*tuple(op.body.block.ops), None):
            if isinstance(child, CallOp) and eligible(child, self.sources):
                run.append(child)
                continue
            if len(run) >= 2:
                group = FusedOp(
                    properties={"profile": StringAttr(PROFILE), "origins": ArrayAttr(item.origin for item in run)},
                    regions=[Region(Block(item.clone() for item in run))],
                )
                rewriter.replace(run[0], group)
                for item in run[1:]:
                    rewriter.erase(item)
                return
            run.clear()


def rewrite(module: ModuleOp, sources: Mapping[str, Operation]) -> ModuleOp:
    """Clone first: the accepted document and the caller's plan remain intact."""
    result = module.clone()
    PatternRewriteWalker(FuseLocal(sources)).rewrite_module(result)
    result.verify()
    return result


def groups(operations: Sequence[Operation], sources: Mapping[str, Operation]) -> tuple[tuple[CallOp, ...], ...]:
    """Check group legality and preserved ordered stage correspondence."""
    result: list[tuple[CallOp, ...]] = []
    for op in operations:
        if isinstance(op, CallOp):
            result.append((op,))
            continue
        if not isinstance(op, FusedOp):
            raise FusionError("unsupported local dispatch operation")
        calls = tuple(op.body.block.ops)
        if (
            set(op.properties) != {"profile", "origins"}
            or op.profile.data != PROFILE
            or op.body.block.args
            or len(calls) < 2
            or any(not isinstance(call, CallOp) for call in calls)
        ):
            raise FusionError("invalid fused local profile/body")
        stages = tuple(call for call in calls if isinstance(call, CallOp))
        if tuple(item.data for item in op.origins) != tuple(call.origin.data for call in stages):
            raise FusionError("fused source correspondence differs")
        if not all(eligible(call, sources) for call in stages):
            raise FusionError("fused group crosses an unsupported effect/schema/resource boundary")
        result.append(stages)
    return tuple(result)
