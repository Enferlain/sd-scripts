"""IRDL structure and local type/capture rules, not whole-lifetime acceptance."""

from __future__ import annotations

from io import StringIO

from xdsl.context import Context
from xdsl.ir import Dialect, Operation, ParametrizedAttribute, TypeAttribute
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from xdsl.dialects.builtin import (
    ArrayAttr,
    Builtin,
    i64,
    IntegerAttr,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)

from xdsl.irdl import (
    irdl_attr_definition,
    irdl_op_definition,
    IRDLOperation,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)

from xdsl.traits import (
    IsolatedFromAbove,
    IsTerminator,
    NoTerminator,
    OpTrait,
    SymbolOpInterface,
)

from .contracts import (
    AccessMode,
    CompletionSemantics,
    decision,
    DecisionSemantics,
    EndpointSemantics,
    EndpointUse,
    ExpandSequence,
    FlowSemantics,
    Freshness,
    ItemContract,
    ItemSemantics,
    number,
    OrderedBody,
    origin,
    ProductSemantics,
    Protocol,
    RealizationSemantics,
    ReferenceSemantics,
    RegionSemantics,
    ResourceSemantics,
)


@irdl_attr_definition
class PayloadType(ParametrizedAttribute, TypeAttribute):
    name = "lang.payload"
    schema: StringAttr


@irdl_attr_definition
class ResourceType(ParametrizedAttribute, TypeAttribute):
    name = "lang.resource"
    kind: StringAttr


@irdl_attr_definition
class EndpointType(ParametrizedAttribute, TypeAttribute):
    name = "lang.endpoint"
    payload: PayloadType


@irdl_attr_definition
class ClaimType(ParametrizedAttribute, TypeAttribute):
    name = "lang.claim"
    payload: PayloadType


@irdl_attr_definition
class OptionalPayloadType(ParametrizedAttribute, TypeAttribute):
    name = "lang.optional"
    payload: PayloadType


@irdl_attr_definition
class RetainedType(ParametrizedAttribute, TypeAttribute):
    name = "lang.retained"
    payload: PayloadType


@irdl_attr_definition
class CapabilityType(ParametrizedAttribute, TypeAttribute):
    name = "lang.capability"
    resource: ResourceType
    mode: StringAttr


@irdl_attr_definition
class RequestType(ParametrizedAttribute, TypeAttribute):
    name = "lang.request"
    payload: PayloadType


class SourceShape(OpTrait):
    def verify(self, op: Operation) -> None:
        if not origin(op) or number(op, "version") != 1:
            raise VerifyException(f"{op.name}: nonempty origin and contract version 1 required")
        if isinstance(op, IRDLOperation):
            definition = op.get_irdl_definition()
            if set(op.properties) - set(definition.properties) or op.attributes:
                raise VerifyException(f"{origin(op)}: undeclared semantic fields")


class SourcedOp(IRDLOperation):
    origin = prop_def(StringAttr)
    version = prop_def(IntegerAttr)
    ancestry = prop_def(ArrayAttr[StringAttr])
    reproducibility = opt_prop_def(StringAttr)
    rank_rule = opt_prop_def(StringAttr)
    assumptions = opt_prop_def(ArrayAttr[StringAttr])


class StructuredOp(SourcedOp):
    sym_name = prop_def(StringAttr)
    inputs = var_operand_def()
    outputs = var_result_def()
    body = region_def("single_block")
    quiescence = prop_def(ArrayAttr[StringAttr])


class BranchingOp(SourcedOp):
    sym_name = prop_def(StringAttr)
    inputs = var_operand_def()
    outputs = var_result_def()
    bodies = var_region_def("single_block")
    quiescence = prop_def(ArrayAttr[StringAttr])


class CaptureShape(OpTrait):
    def verify(self, op: Operation) -> None:
        trait = op.get_trait(RegionSemantics)
        if trait is None:
            raise VerifyException(f"{origin(op)}: missing region contract")
        contract = trait.describe(op)
        decision(op)
        for role, region in zip(contract.roles, op.regions, strict=True):
            for capture in role.captures:
                if capture.argument >= len(role.entries) or role.entries[capture.argument] != capture.type:
                    raise VerifyException(f"{origin(op)}: explicit capture type/argument mismatch")
            expected = tuple(x.type for x in op.operands)
            if contract.protocol == Protocol.PRODUCER:
                if not expected or not isinstance(expected[0], EndpointType):
                    raise VerifyException(f"{origin(op)}: producer requires a typed endpoint")
                outputs = (expected[0].payload,)
                if role.yields == (OptionalPayloadType(expected[0].payload),):
                    outputs = role.yields
                expected = (*expected[1:], index_type())
            elif contract.protocol == Protocol.FILTER:
                if not expected or not isinstance(expected[0], ClaimType):
                    raise VerifyException(f"{origin(op)}: filter requires one owned input claim")
                payload = expected[0].payload
                if tuple(value.type for value in op.results) != (OptionalPayloadType(payload),):
                    raise VerifyException(f"{origin(op)}: filter must return an explicit zero-or-one detached payload")
                if any(isinstance(value, ClaimType) for value in expected[1:]):
                    raise VerifyException(f"{origin(op)}: filter consumes exactly one claim")
                expected = (payload, *expected[1:])
                outputs = (bool_type(),)
            elif contract.protocol == Protocol.REPEAT:
                count = number(op, "carry_count")
                if not 0 <= count <= len(expected) or number(op, "limit") < 0:
                    raise VerifyException(f"{origin(op)}: invalid serial repeat bounds/carries")
                split = len(expected) - count
                expected = (*expected[:split], index_type(), *expected[split:])
                outputs = tuple(x.type for x in op.operands[split:])
                if tuple(x.type for x in op.results) != outputs:
                    raise VerifyException(f"{origin(op)}: repeat result/carry mismatch")
            elif contract.protocol == Protocol.ACCESS:
                if not expected or not isinstance(expected[0], ResourceType):
                    raise VerifyException(f"{origin(op)}: access requires a resource reference")
                access = op.get_trait(ResourceSemantics)
                if access is None:
                    raise VerifyException(f"{origin(op)}: missing resource contract")
                use = access.describe(op)
                if use.mode == AccessMode.PROTECTED_CURRENT and use.freshness != Freshness.CURRENT_AT_START:
                    raise VerifyException(f"{origin(op)}: protected live reads require current-at-start selection")
                if use.mode in (AccessMode.OWNER_UPDATE, AccessMode.GRANT) and use.freshness != Freshness.OWNER_PROTOCOL:
                    raise VerifyException(f"{origin(op)}: authority protocol must define freshness")
                expected = (CapabilityType(expected[0], StringAttr(use.mode)), *expected[1:])
                outputs = tuple(x.type for x in op.results)
            elif contract.protocol == Protocol.CHOICE:
                if not expected or expected[0] != bool_type() or len(op.regions) != 2:
                    raise VerifyException(f"{origin(op)}: choice requires a boolean and two alternatives")
                expected = expected[1:]
                outputs = tuple(x.type for x in op.results)
            elif contract.protocol == Protocol.REQUESTS:
                schema = op.properties["request_schema"]
                if not isinstance(schema, PayloadType) or number(op, "max_outstanding") <= 0 or number(op, "max_active") != 1:
                    raise VerifyException(f"{origin(op)}: bounded serial request protocol required")
                expected += (RequestType(schema),)
                outputs = ()
            else:
                outputs = tuple(x.type for x in op.results)
            if role.entries != expected or role.yields != outputs:
                raise VerifyException(f"{origin(op)}: entry/yield signature does not implement its region contract")
            if region.block.last_op is None:
                raise VerifyException(f"{origin(op)}: missing body terminator")


def structured_traits(protocol: Protocol, *extra: OpTrait):
    return traits_def(SourceShape(), CaptureShape(), RegionSemantics(protocol), SymbolOpInterface(), IsolatedFromAbove(), *extra)


@irdl_op_definition
class ProgramOp(SourcedOp):
    name = "lang.program"
    stage = prop_def(StringAttr)
    body = region_def("single_block")
    traits = traits_def(SourceShape(), OrderedBody(), IsolatedFromAbove(), NoTerminator())


@irdl_op_definition
class ResourceOp(SourcedOp):
    name = "lang.resource_decl"
    sym_name = prop_def(StringAttr)
    value = result_def(ResourceType)
    traits = traits_def(SourceShape(), SymbolOpInterface())


@irdl_op_definition
class StreamOp(SourcedOp):
    name = "lang.stream"
    sym_name = prop_def(StringAttr)
    value = result_def(EndpointType)
    horizon = prop_def(IntegerAttr)
    ready_bound = prop_def(IntegerAttr)
    admission = prop_def(StringAttr)
    owners = prop_def(ArrayAttr[SymbolRefAttr])
    metric = prop_def(StringAttr)
    comparison_boundary = prop_def(StringAttr)
    max_lag = prop_def(IntegerAttr)
    traits = traits_def(SourceShape(), SymbolOpInterface(), FlowSemantics(), ReferenceSemantics((("owners", "resource"),)))

    def verify_(self) -> None:
        FlowSemantics().describe(self)
        if number(self, "horizon") <= 0 or not 0 < number(self, "ready_bound") <= number(self, "horizon") or number(self, "max_lag") < 0:
            raise VerifyException(f"{origin(self)}: invalid flow live-work bounds")


@irdl_op_definition
class OnceOp(StructuredOp):
    name = "lang.once"
    traits = structured_traits(Protocol.ONCE)


@irdl_op_definition
class SequenceOp(StructuredOp):
    name = "lang.sequence"
    traits = structured_traits(Protocol.SEQUENCE)


@irdl_op_definition
class FusedLocalOp(StructuredOp):
    """Derived local dispatch group; it creates no new lifetime or authority."""

    name = "lang.fused_local"
    members = prop_def(ArrayAttr[StringAttr])
    traits = traits_def(SourceShape(), CaptureShape(), RegionSemantics(Protocol.SEQUENCE, transparent=True), IsolatedFromAbove())


@irdl_op_definition
class RecipeOp(StructuredOp):
    name = "lang.recipe"
    traits = traits_def(
        SourceShape(), CaptureShape(), RegionSemantics(Protocol.SEQUENCE), ExpandSequence(), SymbolOpInterface(), IsolatedFromAbove()
    )


@irdl_op_definition
class ParallelOp(BranchingOp):
    name = "lang.parallel"
    traits = structured_traits(Protocol.PARALLEL)


@irdl_op_definition
class ChoiceOp(BranchingOp):
    name = "lang.choice"
    traits = structured_traits(Protocol.CHOICE)


@irdl_op_definition
class RepeatOp(StructuredOp):
    name = "lang.repeat"
    limit = prop_def(IntegerAttr)
    carry_count = prop_def(IntegerAttr)
    policy = prop_def(SymbolRefAttr)
    traits = structured_traits(Protocol.REPEAT)


@irdl_op_definition
class ProducerOp(StructuredOp):
    name = "lang.producer"
    traits = structured_traits(Protocol.PRODUCER, EndpointSemantics(EndpointUse(0, "produce")))


@irdl_op_definition
class FilterOp(StructuredOp):
    name = "lang.filter"
    traits = structured_traits(Protocol.FILTER, ItemSemantics(ItemContract(consumes=(0,), detached_outputs=(0,), forwards=(0,))))


@irdl_op_definition
class RequestsOp(StructuredOp):
    name = "lang.requests"
    sources = prop_def(ArrayAttr[SymbolRefAttr])
    request_schema = prop_def(PayloadType)
    max_outstanding = prop_def(IntegerAttr)
    max_active = prop_def(IntegerAttr)
    traits = traits_def(
        SourceShape(),
        CaptureShape(),
        RegionSemantics(Protocol.REQUESTS),
        ReferenceSemantics((("sources", "region"),)),
        SymbolOpInterface(),
        IsolatedFromAbove(),
    )


@irdl_op_definition
class AccessOp(StructuredOp):
    name = "lang.access"
    mode = prop_def(StringAttr)
    freshness = prop_def(StringAttr)
    version_selector = prop_def(StringAttr)
    profile = prop_def(StringAttr)
    traits = traits_def(
        SourceShape(), CaptureShape(), RegionSemantics(Protocol.ACCESS), ResourceSemantics(), SymbolOpInterface(), IsolatedFromAbove()
    )


@irdl_op_definition
class CallOp(SourcedOp):
    name = "lang.call"
    inputs = var_operand_def()
    outputs = var_result_def()
    binding = prop_def(SymbolRefAttr)
    binding_version = prop_def(StringAttr)
    retains = opt_prop_def(ArrayAttr[IntegerAttr])
    traits = traits_def(SourceShape(), RealizationSemantics(), DecisionSemantics())


@irdl_op_definition
class ReleaseProductOp(SourcedOp):
    name = "lang.release_product"
    product = operand_def(RetainedType)
    traits = traits_def(SourceShape(), ProductSemantics())


@irdl_op_definition
class ConstantOp(SourcedOp):
    name = "lang.constant"
    literal = prop_def(StringAttr)
    value = result_def(PayloadType)
    traits = traits_def(SourceShape())


@irdl_op_definition
class TakeOp(SourcedOp):
    name = "lang.take"
    stream = operand_def(EndpointType)
    claim = result_def(ClaimType)
    traits = traits_def(SourceShape(), ItemSemantics(ItemContract(creates=(0,))), EndpointSemantics(EndpointUse(0, "consume")))

    def verify_(self) -> None:
        if (
            not isinstance(self.claim.type, ClaimType)
            or not isinstance(self.stream.type, EndpointType)
            or self.claim.type.payload != self.stream.type.payload
        ):
            raise VerifyException(f"{origin(self)}: stream/item schema mismatch")


@irdl_op_definition
class BorrowOp(SourcedOp):
    name = "lang.borrow"
    claim = operand_def(ClaimType)
    value = result_def(PayloadType)
    traits = traits_def(SourceShape(), ItemSemantics(ItemContract(borrows=(0,), detached_outputs=(0,))))

    def verify_(self) -> None:
        if not isinstance(self.claim.type, ClaimType) or self.value.type != self.claim.type.payload:
            raise VerifyException(f"{origin(self)}: claim/borrow schema mismatch")


@irdl_op_definition
class RetireOp(SourcedOp):
    name = "lang.retire"
    claim = operand_def(ClaimType)
    disposition = prop_def(StringAttr)
    traits = traits_def(SourceShape(), ItemSemantics(ItemContract(consumes=(0,))))


@irdl_op_definition
class RequestOp(SourcedOp):
    name = "lang.request"
    payload = operand_def(PayloadType)
    handler = prop_def(SymbolRefAttr)
    source = prop_def(SymbolRefAttr)
    boundary = prop_def(StringAttr)
    traits = traits_def(
        SourceShape(),
        ReferenceSemantics((("handler", "requests"), ("source", "region"))),
        CompletionSemantics("request", "handler"),
        DecisionSemantics(),
    )


@irdl_op_definition
class StopAfterOp(SourcedOp):
    name = "lang.stop_after"
    source = prop_def(SymbolRefAttr)
    target = prop_def(SymbolRefAttr)
    traits = traits_def(
        SourceShape(),
        ReferenceSemantics((("source", "region"), ("target", "producer"))),
        CompletionSemantics("stop", "target"),
        DecisionSemantics(),
    )


@irdl_op_definition
class YieldOp(SourcedOp):
    name = "lang.yield"
    values = var_operand_def()
    traits = traits_def(SourceShape(), IsTerminator())


Language = Dialect(
    "lang",
    [
        ProgramOp,
        ResourceOp,
        StreamOp,
        OnceOp,
        SequenceOp,
        FusedLocalOp,
        RecipeOp,
        ParallelOp,
        ChoiceOp,
        RepeatOp,
        ProducerOp,
        FilterOp,
        RequestsOp,
        AccessOp,
        CallOp,
        ReleaseProductOp,
        ConstantOp,
        TakeOp,
        BorrowOp,
        RetireOp,
        RequestOp,
        StopAfterOp,
        YieldOp,
    ],
    [PayloadType, ResourceType, EndpointType, ClaimType, OptionalPayloadType, RetainedType, CapabilityType, RequestType],
)


def index_type() -> PayloadType:
    return PayloadType(StringAttr("index"))


def bool_type() -> PayloadType:
    return PayloadType(StringAttr("bool"))


def integer(value: int) -> IntegerAttr:
    return IntegerAttr(value, i64)


def dump(module: ModuleOp) -> str:
    output = StringIO()
    Printer(stream=output, print_generic_format=True).print_op(module)
    return output.getvalue() + "\n"


def parse(document: str, *, extensions: tuple[Dialect, ...] = ()) -> ModuleOp:
    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(Language)
    for dialect in extensions:
        context.load_dialect(dialect)
    parser = Parser(context, document)
    module = parser.parse_module()
    roots = tuple(module.body.block.ops)
    if len(roots) != 1 or not isinstance(roots[0], ProgramOp):
        parser.raise_error("one lang.program document is required")
    return module
