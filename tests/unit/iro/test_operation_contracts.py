"""Operation contracts read off real IR, through a throwaway test dialect."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr, i32, i64
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, opt_prop_def, traits_def, var_operand_def, var_region_def, var_result_def
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException

from library.iro.operation_contracts import (
    ALLOWED_SELECTIONS,
    DEFAULT_SELECTION,
    AccessMode,
    Capture,
    LifecycleProtocol,
    RegionSemantics,
    ResourceSemantics,
    ResourceUseAttr,
    VersionSelection,
    path,
)


@irdl_op_definition
class Source(IRDLOperation):
    name = "test.source"
    sym_name = opt_prop_def(StringAttr)
    out = var_result_def()


@irdl_op_definition
class End(IRDLOperation):
    name = "test.end"
    values = var_operand_def()
    traits = traits_def(IsTerminator())


def structured(protocol: LifecycleProtocol, *extra) -> type[IRDLOperation]:
    @irdl_op_definition
    class Structured(IRDLOperation):
        name = f"test.{protocol}"
        sym_name = opt_prop_def(StringAttr)
        uses = opt_prop_def(ArrayAttr)
        quiescence = opt_prop_def(ArrayAttr)
        inputs = var_operand_def()
        outputs = var_result_def()
        bodies = var_region_def("single_block")
        traits = traits_def(RegionSemantics(protocol), *extra)

    return Structured


Access = structured(LifecycleProtocol.ACCESS, ResourceSemantics())
Choice = structured(LifecycleProtocol.CHOICE)
Producer = structured(LifecycleProtocol.PRODUCER)
Sequence = structured(LifecycleProtocol.SEQUENCE)


def values(*types: Attribute) -> list[SSAValue]:
    return list(Source.create(result_types=list(types)).results)


def body(*types: Attribute) -> Region:
    block = Block(arg_types=list(types))
    block.add_op(End.create())
    return Region(block)


def test_every_mode_defaults_to_an_allowed_selection() -> None:
    for mode in AccessMode:
        assert DEFAULT_SELECTION[mode] in ALLOWED_SELECTIONS[mode]
    assert VersionSelection.PUBLISHED_INITIAL == "published initial"
    assert ResourceUseAttr.of(AccessMode.PROTECTED_CURRENT).selection.data == "current_at_start"


def test_one_access_gives_its_body_a_lease_per_resource_before_its_captures() -> None:
    first, second, captured = values(i32, i32, i64)
    uses = ArrayAttr([ResourceUseAttr.of(AccessMode.PROTECTED_CURRENT), ResourceUseAttr.of(AccessMode.SNAPSHOT, route="ema")])
    op = Access.create(operands=[first, second, captured], properties={"uses": uses}, regions=[body(i32, i32, i64)])
    role = RegionSemantics(LifecycleProtocol.ACCESS).describe(op).roles[0]
    assert role.captures == (Capture(2, 2, i64),)
    assert role.invocation_arguments == ("owner_issued_lease", "owner_issued_lease")
    contract = ResourceSemantics().describe(op)
    assert [(u.subject, u.mode, u.selection, u.route) for u in contract.uses] == [
        (first, AccessMode.PROTECTED_CURRENT, VersionSelection.CURRENT_AT_START, None),
        (second, AccessMode.SNAPSHOT, VersionSelection.CURRENT_AT_START, "ema"),
    ]


def test_a_resource_use_names_a_route_or_a_view_not_both() -> None:
    (subject,) = values(i32)
    both = ResourceUseAttr(StringAttr("snapshot"), StringAttr("current_at_start"), StringAttr("ema"), StringAttr("inspection"))
    op = Access.create(operands=[subject], properties={"uses": ArrayAttr([both])}, regions=[body(i32)])
    with pytest.raises(VerifyException, match="route or a view"):
        ResourceSemantics().describe(op)


def test_a_choice_on_an_optional_gives_the_present_value_to_its_first_alternative() -> None:
    present, captured = values(i32, i64)
    op = Choice.create(operands=[present, captured], regions=[body(i32, i64), body(i64)])
    first, second = RegionSemantics(LifecycleProtocol.CHOICE).describe(op).roles
    assert first.captures == (Capture(1, 1, i64),) and first.invocation_arguments == ("present_value",)
    assert second.captures == (Capture(1, 0, i64),) and second.invocation_arguments == ()


def test_a_choice_on_a_boolean_binds_nothing() -> None:
    condition, captured = values(i32, i64)
    op = Choice.create(operands=[condition, captured], regions=[body(i64), body(i64)])
    first, _ = RegionSemantics(LifecycleProtocol.CHOICE).describe(op).roles
    assert first.captures == (Capture(1, 0, i64),) and first.invocation_arguments == ()


def test_a_producer_captures_everything_but_its_stream() -> None:
    stream, captured = values(i32, i64)
    op = Producer.create(operands=[stream, captured], regions=[body(i64, i32)])
    (role,) = RegionSemantics(LifecycleProtocol.PRODUCER).describe(op).roles
    assert role.captures == (Capture(1, 0, i64),)
    assert role.invocation_arguments == ("production_index",)


def test_paths_name_what_is_named_and_number_the_rest() -> None:
    inner = Sequence.create(properties={"sym_name": StringAttr("consumer")}, regions=[body()])
    anonymous = Source.create(result_types=[i32])
    outer_body = Block()
    outer_body.add_ops([anonymous, inner, End.create()])
    outer = Sequence.create(properties={"sym_name": StringAttr("activities")}, regions=[Region(outer_body)])
    assert path(outer) == "@activities"
    assert path(inner) == "@activities/@consumer"
    assert path(anonymous) == "@activities/0"


def test_ordered_bodies_reject_values_that_were_not_captured() -> None:
    (outside,) = values(i32)
    block = Block()
    block.add_ops([Source.create(result_types=[i32]), End.create(operands=[outside])])
    op = Sequence.create(properties={"sym_name": StringAttr("loose")}, regions=[Region(block)])
    with pytest.raises(VerifyException, match="captures are explicit"):
        op.verify()
