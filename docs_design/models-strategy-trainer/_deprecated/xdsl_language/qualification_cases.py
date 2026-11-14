"""Authoring-only qualification fixtures; cold execution forbids this module."""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Block, SSAValue

from . import fixtures as f
from .contracts import AccessMode, Admission, Freshness, Protocol, RegionSemantics, number
from .dialects import CapabilityType, ProgramOp, ResourceType, integer
from .normalize import source_index


@dataclass(frozen=True)
class Case:
    name: str
    module: ModuleOp
    initial: dict[str, object]
    prefix: str = ""


def _lift(value: SSAValue, block: Block, root: Block, cache: dict[Block, SSAValue]) -> SSAValue:
    if block is root:
        return value
    if block in cache:
        return cache[block]
    owner = block.parent_op()
    outer_block = owner.parent_block() if owner is not None else None
    if owner is None or outer_block is None:
        raise ValueError("capture path has no enclosing program")
    outer = _lift(value, outer_block, root, cache)
    trait = owner.get_trait(RegionSemantics)
    if trait is None or trait.protocol not in (Protocol.ONCE, Protocol.SEQUENCE, Protocol.PARALLEL, Protocol.REPEAT, Protocol.REQUESTS):
        raise ValueError("unsupported product capture path")
    position = len(owner.operands) - (number(owner, "carry_count") if trait.protocol == Protocol.REPEAT else 0)
    owner.operands = (*owner.operands[:position], outer, *owner.operands[position:])
    for region in owner.regions:
        cache[region.block] = region.block.insert_arg(value.type, position)
    return cache[block]


def retained_products(module: ModuleOp, *, data_only: bool, prefix: str = "") -> ModuleOp:
    index = source_index(module)
    program = next(iter(module.body.block.ops))
    if not isinstance(program, ProgramOp):
        raise ValueError("expected one program")
    declaration = f.resource(prefix + "products", "products")
    activities = index[prefix + "activities"]
    program.body.block.insert_op_before(declaration, activities)
    source = index[prefix + ("feedback.access" if data_only else "measure.access")]
    block = source.parent_block()
    if block is None:
        raise ValueError("publication has no owning body")
    reference = _lift(declaration.value, block, program.body.block, {})
    payload = (
        [index[prefix + "consume.take.borrow"].results[0], index[prefix + "score.recipe"].results[0]] if data_only else [source.results[0]]
    )
    # Borrow's stable origin is defined by the maintained fixture constructor.
    inner = Block(
        arg_types=[CapabilityType(declaration.value.type, StringAttr(AccessMode.OWNER_UPDATE)), *(value.type for value in payload)]
    )
    binding = "publish_score" if data_only else "publish_measurement"
    call = f.call(inner, prefix + binding, binding, inner.args, [payload[-1].type])
    f.end(inner, prefix + binding + ".body", *call.results)
    scope = f.access(prefix + binding + ".access", [reference, *payload], inner, AccessMode.OWNER_UPDATE, [payload[-1].type])
    block.insert_op_after(scope, source)
    module.verify()
    return module


def make_case(name: str, *, windows: int = 2, horizon: int | None = None) -> Case:
    if name == "data-only":
        module = f.data_only(horizon=1 if horizon is None else horizon)
        return Case(name, retained_products(module, data_only=True), {"policy": 0, "products": ()})
    if name not in ("lifetime", "extra-scope"):
        raise ValueError("unknown maintained qualification case")
    prefix = "nested." if name == "extra-scope" else ""
    module = f.lifetime(extra_scope=name == "extra-scope", prefix=prefix, windows=windows, horizon=3 if horizon is None else horizon)
    return Case(
        name,
        retained_products(module, data_only=False, prefix=prefix),
        {prefix + "encoder": 2.0, prefix + "model": 0.8, prefix + "products": ()},
        prefix,
    )


def lag_case(mode: AccessMode, *, reject_forever: bool = False) -> Case:
    module = f.data_only(count=1, horizon=3)
    index = source_index(module)
    feedback = index["feedback.access"]
    parent = feedback.parent_block()
    if parent is None:
        raise ValueError("feedback has no body")
    parent.erase_op(feedback)
    selection = index["select.access"]
    selection.properties.update(
        mode=StringAttr(mode),
        freshness=StringAttr(Freshness.PUBLISHED if reject_forever else Freshness.CURRENT_AT_START),
        version_selector=StringAttr("initial" if reject_forever else "current"),
    )
    kind = selection.operands[0].type
    if not isinstance(kind, ResourceType):
        raise ValueError("selection must capture resource")
    selection.regions[0].block.args[0]._type = CapabilityType(kind, StringAttr(mode))
    index["records"].properties.update(admission=StringAttr(Admission.BOUNDED_LAG), max_lag=integer(0 if reject_forever else 1))
    activities = index["activities"]
    writer = f.body(activities.operands)
    for position in (1, 2):
        constant = f.node(f.ConstantOp, f"write.value.{position}", outputs=[f.SCALAR], literal=StringAttr(str(position)))
        writer.add_op(constant)
        body = Block(arg_types=[CapabilityType(kind, StringAttr(AccessMode.OWNER_UPDATE)), f.SCALAR])
        changed = f.call(body, f"write.{position}", "feedback", body.args, [f.SCALAR])
        f.end(body, f"write.body.{position}", *changed.results)
        writer.add_op(
            f.access(f"write.access.{position}", [writer.args[1], constant.results[0]], body, AccessMode.OWNER_UPDATE, [f.SCALAR])
        )
    f.end(writer, "writer")
    from xdsl.ir import Region

    activities.add_region(Region(writer))
    return Case("rejections" if reject_forever else "lag-" + str(mode), module, {"policy": 0})
