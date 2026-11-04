"""Structural normalization. Whole-lifetime conformance is milestone 7."""

from __future__ import annotations

from dataclasses import dataclass, fields

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr
from xdsl.ir import Attribute, BlockArgument, Operation, OpResult, Region, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern
from xdsl.traits import SymbolOpInterface
from xdsl.utils.exceptions import VerifyException

from .contracts import (
    CompletionSemantics,
    DecisionSemantics,
    EndpointSemantics,
    ExpandSequence,
    FlowContract,
    FlowSemantics,
    ItemSemantics,
    origin,
    Protocol,
    RealizationContract,
    RealizationSemantics,
    ReferenceSemantics,
    RegionContract,
    RegionSemantics,
    ResourceSemantics,
    strings,
    text,
)
from .dialects import ProgramOp, ResourceOp, SequenceOp, SourceShape, dump


@dataclass(frozen=True)
class Link:
    source: str
    field: str
    target: str
    kind: str


@dataclass(frozen=True)
class ValueSource:
    origin: str
    region: int | None
    index: int


@dataclass(frozen=True)
class ResourceSignature:
    origin: str
    subject: ValueSource
    fields: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class Structure:
    origins: tuple[str, ...]
    regions: tuple[RegionContract, ...]
    links: tuple[Link, ...]
    flows: tuple[FlowContract, ...]
    resources: tuple[ResourceSignature, ...]
    realizations: tuple[tuple[str, RealizationContract], ...]
    wiring: tuple[tuple[str, tuple[ValueSource, ...]], ...]
    parameters: tuple[tuple[str, tuple[tuple[str, Attribute], ...]], ...]
    leaf_contracts: tuple[tuple[str, str, object], ...]


def value_source(value: SSAValue) -> ValueSource:
    if isinstance(value, OpResult):
        return ValueSource(origin(value.owner), None, value.index)
    if isinstance(value, BlockArgument):
        block = value.owner
        parent, region = block.parent_op(), block.parent
        if parent is not None and region is not None:
            return ValueSource(origin(parent), parent.regions.index(region), value.index)
    raise VerifyException("semantic operand has no in-program SSA provider")


def source_index(module: ModuleOp) -> dict[str, Operation]:
    result: dict[str, Operation] = {}
    for op in module.walk():
        if op is module:
            continue
        if op.get_trait(SourceShape) is None:
            raise VerifyException(f"{op.name}: no registered semantic definition")
        key = origin(op)
        if key in result:
            raise VerifyException(f"{key}: duplicate source origin")
        result[key] = op
    return result


def ancestry(op: Operation) -> tuple[str, ...]:
    parents = [origin(op)]
    parent = op.parent_op()
    while parent is not None and not isinstance(parent, ModuleOp):
        parents.append(origin(parent))
        parent = parent.parent_op()
    return tuple(reversed(parents))


def symbol_index(module: ModuleOp) -> dict[str, Operation]:
    result: dict[str, Operation] = {}
    for op in module.walk():
        if op is module or op.get_trait(SymbolOpInterface) is None:
            continue
        name = text(op, "sym_name")
        if not name or name in result:
            raise VerifyException(f"{origin(op)}: empty or duplicate semantic symbol {name}")
        result[name] = op
    return result


def inspect_structure(module: ModuleOp) -> Structure:
    module.verify()
    roots = tuple(module.body.block.ops)
    if len(roots) != 1 or not isinstance(roots[0], ProgramOp) or roots[0].body.block.args:
        raise VerifyException("one closed lang.program root is required")
    index = source_index(module)
    symbols = symbol_index(module)
    regions, links, flows, resources, realizations, wiring, parameters, leaf_contracts = [], [], [], [], [], [], [], []
    for key, op in index.items():
        for kind in (ItemSemantics, EndpointSemantics, CompletionSemantics, DecisionSemantics):
            trait = op.get_trait(kind)
            if trait is not None:
                leaf_contracts.append((key, kind.__name__, trait.describe(op)))
        wiring.append((key, tuple(value_source(value) for value in op.operands)))
        parameters.append(
            (
                key,
                tuple(
                    sorted(
                        (name, value)
                        for name, value in op.properties.items()
                        if name != "ancestry" and not (isinstance(op, ProgramOp) and name == "stage")
                    )
                ),
            )
        )
        region_trait = op.get_trait(RegionSemantics)
        if region_trait is not None:
            regions.append(region_trait.describe(op))
        flow_trait = op.get_trait(FlowSemantics)
        if flow_trait is not None:
            flows.append(flow_trait.describe(op))
        resource_trait = op.get_trait(ResourceSemantics)
        if resource_trait is not None:
            use = resource_trait.describe(op)
            # SSA object identity changes on clone; its actual provider does not.
            resources.append(
                ResourceSignature(
                    key,
                    value_source(use.subject),
                    tuple((item.name, getattr(use, item.name)) for item in fields(use) if item.name != "subject"),
                )
            )
        realization_trait = op.get_trait(RealizationSemantics)
        if realization_trait is not None:
            realizations.append((key, realization_trait.describe(op)))
        reference_trait = op.get_trait(ReferenceSemantics)
        if reference_trait is None:
            continue
        for ref in reference_trait.describe(op):
            target = symbols.get(ref.target)
            if target is None:
                raise VerifyException(f"{key}: unresolved {ref.field} reference @{ref.target}")
            target_trait = target.get_trait(RegionSemantics)
            if ref.kind == "resource":
                compatible = isinstance(target, ResourceOp)
            elif ref.kind == "region":
                compatible = target_trait is not None
            else:
                compatible = target_trait is not None and target_trait.protocol == Protocol(ref.kind)
            if not compatible:
                raise VerifyException(f"{key}: {ref.field} requires {ref.kind}, got {target.name}")
            if ref.payload is not None and ref.payload != target.properties.get("request_schema"):
                raise VerifyException(f"{key}: request payload does not match handler entry schema")
            if ref.completion_source is not None and (
                target_trait is None or ref.completion_source not in target_trait.describe(target).request_sources
            ):
                raise VerifyException(f"{key}: undeclared request source @{ref.completion_source} for @{ref.target}")
            links.append(Link(key, ref.field, ref.target, ref.kind))
    return Structure(
        tuple(index),
        tuple(regions),
        tuple(links),
        tuple(flows),
        tuple(resources),
        tuple(realizations),
        tuple(wiring),
        tuple(parameters),
        tuple(leaf_contracts),
    )


class ExpandRecipe(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if op.get_trait(ExpandSequence) is None:
            return
        region = Region()
        op.regions[0].clone_into(region)
        replacement = SequenceOp.create(
            operands=op.operands,
            result_types=[x.type for x in op.results],
            properties=dict(op.properties),
            regions=[region],
        )
        rewriter.replace(op, replacement)


@dataclass(frozen=True)
class Normalized:
    module: ModuleOp
    accepted_origins: tuple[str, ...]

    @property
    def text(self) -> str:
        return dump(self.module)

    @property
    def structure(self) -> Structure:
        return inspect_structure(self.module)


def normalize(module: ModuleOp) -> Normalized:
    before = inspect_structure(module)
    root = next(iter(module.body.block.ops))
    if text(root, "stage") not in ("accepted", "normalized"):
        raise VerifyException("unknown language stage")
    if text(root, "stage") == "normalized":
        for op in source_index(module).values():
            if strings(op, "ancestry") != ancestry(op) or op.get_trait(ExpandSequence) is not None:
                raise VerifyException(f"{origin(op)}: inconsistent normalized structure/ancestry")
    detached = module.clone()
    PatternRewriteWalker(ExpandRecipe()).rewrite_module(detached)
    for op in source_index(detached).values():
        op.properties["ancestry"] = ArrayAttr(StringAttr(value) for value in ancestry(op))
    next(iter(detached.body.block.ops)).properties["stage"] = StringAttr("normalized")
    after = inspect_structure(detached)
    if before != after:
        raise VerifyException("normalization changed source coverage, semantic references, SSA wiring or contracts")
    return Normalized(detached, before.origins)
