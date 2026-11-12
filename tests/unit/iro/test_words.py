"""Each word's own rules, on minimal operations."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr, SymbolRefAttr, f64, i1, i64
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, var_result_def
from xdsl.utils.exceptions import VerifyException

from library.iro.operation_contracts import AccessMode, ResourceUseAttr, VersionSelection
from library.iro.words import Kuroiro
from library.iro.words import types as t
from library.iro.words.access import AccessOp
from library.iro.words.calls import CallOp, ConstantOp, YieldOp
from library.iro.words.data import DisposeOp, FilterOp, ForkOp, StreamOp
from library.iro.words.lifetimes import HandlerOp, ProducerOp
from library.iro.words.model import RelationshipOp, ResourceOp
from library.iro.words.structure import ChoiceOp, RepeatOp, StrategyOp

record = t.value("record")


@irdl_op_definition
class Values(IRDLOperation):
    name = "test.values"
    out = var_result_def()


def values(*types: Attribute) -> list[SSAValue]:
    return list(Values.create(result_types=list(types)).results)


def body(*types: Attribute, yields: tuple[int, ...] = ()) -> Region:
    block = Block(arg_types=list(types))
    block.add_op(YieldOp.create(operands=[block.args[i] for i in yields]))
    return Region(block)


def stream(**overrides: Attribute) -> StreamOp:
    fields: dict[str, Attribute] = {
        "sym_name": StringAttr("records"),
        "outstanding": IntegerAttr(3, i64),
        "ready": IntegerAttr(2, i64),
        "admission": StringAttr("any_with_provenance"),
        "owners": ArrayAttr([SymbolRefAttr("policy")]),
        "metric": StringAttr("owner_update_ordinal"),
        "boundary": StringAttr("item_admission"),
        "max_lag": IntegerAttr(0, i64),
    }
    return StreamOp.create(properties=fields | overrides, result_types=[t.stream(record)])


def test_the_dialect_holds_every_core_word_under_iro() -> None:
    names = {op.name for op in Kuroiro.operations}
    assert {"iro.strategy", "iro.sequence", "iro.access", "iro.fork", "iro.relationship"} <= names
    assert "iro.once" not in names and "iro.transfer" not in names and "iro.invoke" not in names


def test_authors_cannot_write_a_system_disposition() -> None:
    (item,) = values(t.claim(record))
    DisposeOp.create(operands=[item], properties={"disposition": StringAttr("filtered_out")}).verify()
    with pytest.raises(VerifyException, match="given by the system"):
        DisposeOp.create(operands=[item], properties={"disposition": StringAttr("rejected")}).verify()


def test_a_fork_makes_at_least_two_forks_of_the_same_type() -> None:
    (item,) = values(t.claim(record))
    ForkOp.create(operands=[item], result_types=[t.claim(record)] * 2).verify()
    with pytest.raises(VerifyException, match="at least two"):
        ForkOp.create(operands=[item], result_types=[t.claim(record)]).verify()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ready": IntegerAttr(4, i64)}, "ready <= outstanding"),
        ({"admission": StringAttr("bounded_lag")}, "maximum lag"),
        ({"admission": StringAttr("exact_current"), "owners": ArrayAttr([])}, "named owners"),
    ],
)
def test_stream_bounds_and_admission_are_checked(overrides: dict[str, Attribute], message: str) -> None:
    stream().verify()
    stream(admission=StringAttr("bounded_lag"), max_lag=IntegerAttr(2, i64)).verify()
    with pytest.raises(VerifyException, match=message):
        stream(**overrides).verify()


def access(mode: AccessMode, selection: VersionSelection | None = None) -> AccessOp:
    (subject,) = values(t.participant())
    uses = ArrayAttr([ResourceUseAttr.of(mode, selection)])
    return AccessOp.create(operands=[subject], properties={"uses": uses}, regions=[body(t.lease(t.participant(), mode))])


def test_each_mode_selects_only_its_versions() -> None:
    access(AccessMode.PROTECTED_CURRENT).verify()
    access(AccessMode.SNAPSHOT, VersionSelection.PUBLISHED_INITIAL).verify()
    with pytest.raises(VerifyException, match="cannot select"):
        access(AccessMode.PROTECTED_CURRENT, VersionSelection.PUBLISHED_INITIAL).verify()


def test_an_access_gives_one_lease_per_resource_and_names_each_once() -> None:
    first, second = values(t.participant(), t.participant())
    mode = AccessMode.PROTECTED_CURRENT
    uses = ArrayAttr([ResourceUseAttr.of(mode), ResourceUseAttr.of(mode)])
    leases = (t.lease(t.participant(), mode),) * 2
    AccessOp.create(operands=[first, second], properties={"uses": uses}, regions=[body(*leases)]).verify()
    with pytest.raises(VerifyException, match="each resource once"):
        AccessOp.create(operands=[first, first], properties={"uses": uses}, regions=[body(*leases)]).verify()
    with pytest.raises(VerifyException, match="takes"):
        AccessOp.create(operands=[first, second], properties={"uses": uses}, regions=[body(leases[0])]).verify()


def repeat(**fields: Attribute) -> RepeatOp:
    properties = {"sym_name": StringAttr("loop"), "carry_count": IntegerAttr(0, i64)} | fields
    return RepeatOp.create(properties=properties, regions=[body(t.int_type())])


def test_a_repeat_runs_a_number_of_times_or_until_a_policy() -> None:
    repeat(limit=IntegerAttr(4, i64)).verify()
    repeat(policy=SymbolRefAttr("max_steps")).verify()
    with pytest.raises(VerifyException, match="exactly one"):
        repeat(limit=IntegerAttr(4, i64), policy=SymbolRefAttr("max_steps")).verify()


def test_carried_values_enter_after_the_iteration_and_come_back_as_results() -> None:
    (first,) = values(t.claim(record))
    properties = {"sym_name": StringAttr("turns"), "limit": IntegerAttr(4, i64), "carry_count": IntegerAttr(1, i64)}
    region = body(t.int_type(), t.claim(record), yields=(1,))
    RepeatOp.create(operands=[first], result_types=[t.claim(record)], properties=properties, regions=[region]).verify()
    with pytest.raises(VerifyException, match="results are its carried values"):
        RepeatOp.create(operands=[first], properties=properties, regions=[body(t.int_type(), t.claim(record), yields=(1,))]).verify()


def test_a_choice_on_an_optional_hands_its_first_alternative_the_value() -> None:
    (cached,) = values(t.optional(record))
    regions = [body(record, yields=(0,)), body(), body()]
    ChoiceOp.create(operands=[cached], result_types=[], regions=[body(record), body()]).verify()
    with pytest.raises(VerifyException, match="two alternatives"):
        ChoiceOp.create(operands=[cached], regions=regions).verify()
    (flag,) = values(t.bool_type())
    with pytest.raises(VerifyException, match="takes"):
        ChoiceOp.create(operands=[flag], regions=[body(record), body()]).verify()


def test_a_filter_keeps_or_drops_one_claim() -> None:
    (item,) = values(t.claim(record))
    keep = Block(arg_types=[record])
    (decided,) = values(t.bool_type())
    keep.add_op(YieldOp.create(operands=[decided]))
    filtered = FilterOp.create(operands=[item], result_types=[t.optional(record)], regions=[Region(keep)])
    with pytest.raises(VerifyException, match=r"IsolatedFromAbove|captures are explicit"):
        filtered.verify()  # the decision must be computed inside, not captured implicitly
    with pytest.raises(VerifyException, match="one claim"):
        FilterOp.create(operands=[item], result_types=[record], regions=[body(record)]).verify()


def test_a_producer_may_yield_an_item_or_an_optional() -> None:
    (endpoint,) = values(t.stream(record))
    for made in (record, t.optional(record)):
        production = Block(arg_types=[t.int_type()])
        item = Values.create(result_types=[made])
        production.add_ops([item, YieldOp.create(operands=[item.results[0]])])
        ProducerOp.create(operands=[endpoint], properties={"sym_name": StringAttr("p")}, regions=[Region(production)]).verify()


def test_a_handler_runs_one_request_at_a_time() -> None:
    fields = {
        "sym_name": StringAttr("evaluation"),
        "sources": ArrayAttr([SymbolRefAttr("training")]),
        "request": t.int_type(),
        "max_outstanding": IntegerAttr(1, i64),
    }
    HandlerOp.create(properties=fields, regions=[body(t.int_type())]).verify()
    with pytest.raises(VerifyException, match="one request at a time"):
        HandlerOp.create(properties=fields | {"max_active": IntegerAttr(2, i64)}, regions=[body(t.int_type())]).verify()


def test_a_retained_result_keeps_every_lease_and_retained_input() -> None:
    lease, plain = values(t.lease(t.participant(), "protected_current"), record)
    call = {"implementation": SymbolRefAttr("denoise")}
    CallOp.create(
        operands=[lease, plain], result_types=[t.retained(record)], properties=call | {"retains": ArrayAttr([IntegerAttr(0, i64)])}
    ).verify()
    with pytest.raises(VerifyException, match="positions"):
        CallOp.create(operands=[lease, plain], result_types=[t.retained(record)], properties=call).verify()


@pytest.mark.parametrize(
    ("literal", "type_name", "valid"),
    [
        (IntegerAttr(1, i64), "int", True),
        (IntegerAttr(1, i1), "bool", True),
        (FloatAttr(0.5, f64), "float", True),
        (StringAttr("name"), "str", True),
        (IntegerAttr(1, i64), "index", True),
        (FloatAttr(0.5, f64), "int", False),
    ],
)
def test_literals_take_pythons_type_names(literal: Attribute, type_name: str, valid: bool) -> None:
    op = ConstantOp.create(properties={"value": literal}, result_types=[t.value(type_name)])
    if valid:
        op.verify()
    else:
        with pytest.raises(VerifyException, match="literal"):
            op.verify()


def test_a_relationship_relates_two_participants() -> None:
    lora, unet, state = values(t.participant(), t.participant(), t.resource())
    fields = {"sym_name": StringAttr("lora_on_unet")}
    RelationshipOp.create(operands=[lora, unet], result_types=[t.relationship("attachment")], properties=fields).verify()
    with pytest.raises(VerifyException, match="relates participants"):
        RelationshipOp.create(operands=[lora, state], result_types=[t.relationship("attachment")], properties=fields).verify()


def test_a_materializer_reads_only_inspection_views() -> None:
    (unet,) = values(t.participant())
    fields = {"sym_name": StringAttr("lora"), "materialize": SymbolRefAttr("build_lora")}
    ResourceOp.create(
        operands=[unet], result_types=[t.participant()], properties=fields | {"read_views": ArrayAttr([StringAttr("inspection")])}
    ).verify()
    with pytest.raises(VerifyException, match="inspection view"):
        ResourceOp.create(
            operands=[unet], result_types=[t.participant()], properties=fields | {"read_views": ArrayAttr([StringAttr("artifact")])}
        ).verify()


def test_a_strategy_pins_each_implementation_once() -> None:
    fields = {
        "sym_name": StringAttr("s"),
        "uses": ArrayAttr([SymbolRefAttr("a"), SymbolRefAttr("a")]),
        "use_versions": ArrayAttr([StringAttr("1")] * 2),
    }
    with pytest.raises(VerifyException, match="pinned twice"):
        StrategyOp.create(properties=fields, regions=[Region(Block())]).verify()


def test_undeclared_fields_are_rejected() -> None:
    (item,) = values(t.claim(record))
    consumed = {"disposition": StringAttr("consumed")}
    with pytest.raises(VerifyException, match="not defined by the operation"):
        DisposeOp.create(operands=[item], properties=consumed | {"priority": StringAttr("high")}).verify()
    with pytest.raises(VerifyException, match="undeclared"):
        DisposeOp.create(operands=[item], properties=consumed, attributes={"priority": StringAttr("high")}).verify()
