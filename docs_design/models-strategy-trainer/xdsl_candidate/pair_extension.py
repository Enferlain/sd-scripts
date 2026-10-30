"""Independent pair preprocessing dialect using only public local interfaces."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import IRDLOperation, irdl_attr_definition, irdl_op_definition, operand_def, prop_def, result_def, traits_def
from xdsl.utils.exceptions import VerifyException

from .dialects import FlowType, LocalOp, MapOp, YieldOp, integer
from .fixture import build, registry
from .local_contract import LocalDescription, LocalTransform, ResourceUse
from .payloads import DEFAULT_OUTCOMES, FrozenResource, LocalOutcomeError, PayloadCodec
from .prepare import Binding, Registry


@irdl_attr_definition
class PairSchema(ParametrizedAttribute, TypeAttribute):
    name = "pairing.vector"
    width: IntegerAttr

    def verify(self) -> None:
        if self.width.value.data <= 0:
            raise VerifyException("pairing.vector requires positive width")


class PackContract(LocalTransform):
    def describe(self, op) -> LocalDescription | None:
        if not isinstance(op, PackOp) or op.effect.data != "read_frozen":
            return None
        return LocalDescription(
            op.input.type,
            op.output.type,
            "one-to-one",
            (ResourceUse(op.resource.root_reference.data, "read_frozen", op.resource_version.data, op.resource_digest.data),),
            (*DEFAULT_OUTCOMES, "negative_input"),
        )


@irdl_op_definition
class PackOp(IRDLOperation):
    name = "pairing.pack"
    source_id = prop_def(StringAttr)
    implementation = prop_def(SymbolRefAttr)
    effect = prop_def(StringAttr)
    resource = prop_def(SymbolRefAttr)
    resource_version = prop_def(StringAttr)
    resource_digest = prop_def(StringAttr)
    input = operand_def(FlowType)
    output = result_def(FlowType)
    traits = traits_def(PackContract())

    def verify_(self) -> None:
        if self.input.type != FlowType(StringAttr("record")) or not isinstance(self.output.type.payload, PairSchema):
            raise VerifyException("pairing.pack requires record -> pairing.vector")
        if not self.resource.root_reference.data or self.resource.nested_references:
            raise VerifyException("pairing.pack requires a flat installed resource symbol")


Pairing = Dialect("pairing", [PackOp], [PairSchema])
PAIR = FlowType(PairSchema(integer(2)))
COEFFICIENTS = FrozenResource("1", (1, 2))
USE = ResourceUse("pair_coefficients", "read_frozen", "1", COEFFICIENTS.digest)


def lower_pack(resources):
    coefficients = resources["pair_coefficients"].values

    async def invoke(value):
        if value < 0:
            raise LocalOutcomeError("negative_input", "pair pack rejected negative input")
        await asyncio.sleep(0)
        return tuple(value * coefficient for coefficient in coefficients)

    return invoke


async def needs_lowering(value):
    raise AssertionError("resource-aware lowering was not installed")


async def first_lane(value):
    await asyncio.sleep(0)
    return value[0]


def extension_registry() -> Registry:
    base = registry()
    return replace(
        base,
        name="independent-pair-fixture-v1",
        bindings=dict(base.bindings)
        | {
            "pack_pair_v1": Binding(
                "1",
                "map",
                "read_frozen",
                needs_lowering,
                resources=(USE,),
                outcomes=(*DEFAULT_OUTCOMES, "negative_input"),
                lower=lower_pack,
            ),
            "first_lane_v1": Binding("1", "map", "pure", first_lane),
        },
        codecs=dict(base.codecs)
        | {
            PAIR: PayloadCodec(
                "pair-int2-v1",
                lambda value: type(value) is tuple and len(value) == 2 and all(type(item) is int for item in value),
                lambda value: list(value),
            )
        },
        resources={"pair_coefficients": COEFFICIENTS},
        dialects=(Pairing,),
    )


def build_extension() -> ModuleOp:
    module = build()
    nested = next(op for op in module.walk() if isinstance(op, LocalOp))
    yield_op = nested.body.block.last_op
    assert isinstance(yield_op, YieldOp)
    pack = PackOp(
        operands=[yield_op.value],
        result_types=[PAIR],
        properties={
            "source_id": StringAttr("pair.pack"),
            "implementation": SymbolRefAttr("pack_pair_v1"),
            "effect": StringAttr("read_frozen"),
            "resource": SymbolRefAttr(USE.symbol),
            "resource_version": StringAttr(USE.version),
            "resource_digest": StringAttr(USE.digest),
        },
    )
    unpack = MapOp(
        operands=[pack.output],
        result_types=[FlowType(StringAttr("record"))],
        properties={"source_id": StringAttr("pair.first"), "implementation": SymbolRefAttr("first_lane_v1"), "effect": StringAttr("pure")},
    )
    nested.body.block.insert_ops_before([pack, unpack], yield_op)
    yield_op.operands = [unpack.output]
    return module
