"""One conservative xDSL rewrite of prepared local binding selections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr
from xdsl.ir import Attribute, Block, Operation, Region
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from .dialects import CallOp, ChainOp, FlowType, FusedOp, PlanOp, SessionOp
from .local_contract import LocalDescription, LocalTransform, ResourceUse, TransparentLocal

PROFILE = "ordered-validated-local-v1"


class FusionError(ValueError):
    pass


@dataclass(frozen=True)
class StageInput:
    """Immutable dependencies of a local legality decision, including ancestry."""

    origin: str
    source_origin: Attribute | None
    effect: str
    source_effect: Attribute | None
    input_types: tuple[Attribute, ...]
    output_types: tuple[Attribute, ...]
    description: LocalDescription | None
    ancestors: tuple[tuple[str, Attribute | None, bool], ...]
    session: Attribute | None


@dataclass(frozen=True)
class StageDecision:
    origin: str
    groupable: bool
    reason: str
    input_type: Attribute | None
    output_type: Attribute | None


@dataclass(frozen=True)
class ResourceSummary:
    reads: tuple[ResourceUse, ...]
    unknown_sources: tuple[str, ...]
    decisions: tuple[StageDecision, ...]

    def decision(self, origin: str) -> StageDecision | None:
        matches = tuple(item for item in self.decisions if item.origin == origin)
        return matches[0] if len(matches) == 1 else None

    def permits(self, origin: str) -> bool:
        decision = self.decision(origin)
        return decision is not None and decision.groupable

    def adjacent(self, left: str, right: str) -> bool:
        before, after = self.decision(left), self.decision(right)
        return before is not None and after is not None and before.output_type == after.input_type

    def describe(self) -> dict:
        return {
            "reads": [asdict(use) for use in self.reads],
            "unknown_sources": list(self.unknown_sources),
            "decisions": [
                {
                    "origin": item.origin,
                    "groupable": item.groupable,
                    "reason": item.reason,
                    "input": str(item.input_type),
                    "output": str(item.output_type),
                }
                for item in self.decisions
            ],
        }


def stage_input(call: CallOp, sources: Mapping[str, Operation]) -> StageInput:
    source = sources.get(call.origin.data)
    interface = source.get_trait(LocalTransform) if source is not None else None
    description = interface.describe(source) if interface is not None and source is not None else None
    ancestors = []
    parent = source.parent_op() if source is not None else None
    while parent is not None and not isinstance(parent, SessionOp):
        ancestors.append((parent.name, parent.properties.get("source_id"), parent.get_trait(TransparentLocal) is not None))
        parent = parent.parent_op()
    return StageInput(
        call.origin.data,
        source.properties.get("source_id") if source is not None else None,
        call.effect.data,
        source.properties.get("effect") if source is not None else None,
        tuple(value.type for value in source.operands) if source is not None else (),
        tuple(value.type for value in source.results) if source is not None else (),
        description,
        tuple(ancestors),
        parent.properties.get("source_id") if isinstance(parent, SessionOp) else None,
    )


def summarize(stages: tuple[StageInput, ...]) -> ResourceSummary:
    reads: set[ResourceUse] = set()
    unknown, decisions = [], []
    for stage in stages:
        description = stage.description
        uses = description.resources if description is not None else None
        if uses is not None:
            reads.update(use for use in uses if isinstance(use, ResourceUse))
        known = (
            stage.effect in ("pure", "read_frozen")
            and stage.source_effect == StringAttr(stage.effect)
            and stage.source_origin == StringAttr(stage.origin)
            and uses is not None
            and all(
                isinstance(use, ResourceUse) and use.access == "read_frozen" and use.symbol and use.version and use.digest for use in uses
            )
        )
        if not known:
            unknown.append(stage.origin)
        shape = description is not None and (
            stage.input_types == (description.input_type,)
            and stage.output_types == (description.output_type,)
            and isinstance(description.input_type, FlowType)
            and isinstance(description.output_type, FlowType)
            and description.cardinality == "one-to-one"
            and {"returned", "invalid_output", "raised"}.issubset(description.outcomes)
            and len(set(description.outcomes)) == len(description.outcomes)
            and all(description.outcomes)
        )
        transparent = stage.session is not None and all(item[2] for item in stage.ancestors)
        allowed = bool(known and shape and transparent)
        reason = (
            "ordered unary schemas; frozen snapshots pinned; preserve each guard/outcome/completion"
            if allowed
            else "unknown resource access/effect"
            if not known
            else "unsupported local schema/cardinality/outcomes or enclosing lifetime"
        )
        decisions.append(
            StageDecision(
                stage.origin,
                allowed,
                reason,
                description.input_type if description else None,
                description.output_type if description else None,
            )
        )
    return ResourceSummary(
        tuple(sorted(reads, key=lambda use: (use.symbol, use.access, use.version, use.digest))), tuple(unknown), tuple(decisions)
    )


def map_calls(operations: Sequence[Operation]) -> tuple[CallOp, ...]:
    return tuple(call for op in operations for call in op.walk() if isinstance(call, CallOp) and call.role.data == "map")


def dispatch_groups(module: ModuleOp) -> tuple[tuple[str, ...], ...]:
    groups = []
    for op in module.walk():
        if not isinstance(op, (PlanOp, ChainOp)) or (isinstance(op, ChainOp) and op.role.data != "producer"):
            continue
        for child in op.body.block.ops:
            if isinstance(child, CallOp) and child.role.data == "map":
                groups.append((child.origin.data,))
            elif isinstance(child, FusedOp):
                groups.append(tuple(call.origin.data for call in child.body.block.ops if isinstance(call, CallOp)))
    return tuple(groups)


@dataclass(frozen=True)
class AnalysisObservation:
    groups: tuple[tuple[str, ...], ...]
    resource_computations: int
    dispatch_computations: int


@dataclass(frozen=True)
class GroupingEvidence:
    summary: ResourceSummary
    observations: tuple[AnalysisObservation, ...]

    def describe(self) -> dict:
        return {
            "cache": "project-owned dependency snapshots",
            "resource_analysis": self.summary.describe(),
            "observations": [asdict(item) for item in self.observations],
        }


class LocalAnalyses:
    """Per-preparation project cache, not a framework AnalysisManager.

    Full immutable dependency equality protects reuse across cloned/decoded IR.
    Grouping leaves the flattened semantic dependencies unchanged. A changed
    effect, schema, resource identity, outcome or enclosing lifetime recomputes.
    Cache keys are built afresh; saved analysis observations are never inputs.
    """

    def __init__(self) -> None:
        self._resource_key: tuple[StageInput, ...] | None = None
        self._summary: ResourceSummary | None = None
        self._dispatch_key: tuple[tuple[str, ...], ...] | None = None
        self.resource_computations = 0
        self.dispatch_computations = 0
        self.observations: list[AnalysisObservation] = []

    def resources(self, operations: Sequence[Operation], sources: Mapping[str, Operation]) -> ResourceSummary:
        key = tuple(stage_input(call, sources) for call in map_calls(operations))
        if self._summary is None or key != self._resource_key:
            self._summary = summarize(key)
            self._resource_key = key
            self.resource_computations += 1
        return self._summary

    def observe(self, module: ModuleOp, sources: Mapping[str, Operation]) -> None:
        self.resources(tuple(module.body.block.ops), sources)
        key = dispatch_groups(module)
        if key != self._dispatch_key:
            self._dispatch_key = key
            self.dispatch_computations += 1
        self.observations.append(AnalysisObservation(key, self.resource_computations, self.dispatch_computations))

    def evidence(self) -> GroupingEvidence:
        if self._summary is None:
            raise FusionError("local analyses were never requested")
        return GroupingEvidence(self._summary, tuple(self.observations))


def eligible(call: CallOp, sources: Mapping[str, Operation]) -> bool:
    return call.role.data == "map" and summarize((stage_input(call, sources),)).permits(call.origin.data)


class FuseLocal(RewritePattern):
    def __init__(self, resources: ResourceSummary):
        self.resources = resources

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PlanOp | ChainOp, rewriter: PatternRewriter) -> None:
        if isinstance(op, ChainOp) and op.role.data != "producer":
            return
        run: list[CallOp] = []
        runs: list[list[CallOp]] = []
        for child in (*tuple(op.body.block.ops), None):
            allowed = isinstance(child, CallOp) and child.role.data == "map" and self.resources.permits(child.origin.data)
            if not allowed or (run and isinstance(child, CallOp) and not self.resources.adjacent(run[-1].origin.data, child.origin.data)):
                if len(run) >= 2:
                    runs.append(run)
                run = []
            if allowed:
                assert isinstance(child, CallOp)
                run.append(child)
        for run in runs:
            group = FusedOp(
                properties={"profile": StringAttr(PROFILE), "origins": ArrayAttr(item.origin for item in run)},
                regions=[Region(Block(item.clone() for item in run))],
            )
            rewriter.replace(run[0], group)
            for item in run[1:]:
                rewriter.erase(item)


def rewrite(module: ModuleOp, sources: Mapping[str, Operation], *, analyses: LocalAnalyses | None = None) -> ModuleOp:
    """Clone first: the accepted document and the caller's plan remain intact."""
    result = module.clone()
    analyses = LocalAnalyses() if analyses is None else analyses
    resources = analyses.resources(tuple(result.body.block.ops), sources)
    PatternRewriteWalker(FuseLocal(resources)).rewrite_module(result)
    result.verify()
    return result


def groups(
    operations: Sequence[Operation], sources: Mapping[str, Operation], *, analyses: LocalAnalyses | None = None
) -> tuple[tuple[CallOp, ...], ...]:
    """Check group legality and preserved ordered stage correspondence."""
    result: list[tuple[CallOp, ...]] = []
    analyses = LocalAnalyses() if analyses is None else analyses
    resources = analyses.resources(operations, sources)
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
        if not all(call.role.data == "map" and resources.permits(call.origin.data) for call in stages) or not all(
            resources.adjacent(left.origin.data, right.origin.data) for left, right in pairwise(stages)
        ):
            raise FusionError("fused group crosses an unsupported effect/schema/resource boundary")
        result.append(stages)
    return tuple(result)
