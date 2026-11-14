"""Python construction of two domains and an alternative scope arrangement."""

from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue

from .contracts import AccessMode, Admission, Freshness, RankRule, Reproducibility
from .dialects import (
    AccessOp,
    bool_type,
    BorrowOp,
    CallOp,
    CapabilityType,
    ChoiceOp,
    ClaimType,
    ConstantOp,
    EndpointType,
    index_type,
    integer,
    OnceOp,
    ParallelOp,
    PayloadType,
    ProducerOp,
    ProgramOp,
    RecipeOp,
    RepeatOp,
    RequestOp,
    RequestsOp,
    RequestType,
    ResourceOp,
    ResourceType,
    RetireOp,
    StopAfterOp,
    StreamOp,
    TakeOp,
    YieldOp,
)


SCALAR = PayloadType(StringAttr("scalar"))
RECORD = PayloadType(StringAttr("record"))
MEASUREMENT = PayloadType(StringAttr("measurement"))


def props(name: str) -> dict[str, Attribute]:
    return {"origin": StringAttr(name), "version": integer(1), "ancestry": ArrayAttr([StringAttr(name)])}


def node[T: Operation](
    kind: type[T], name: str, *, inputs: Sequence[SSAValue] = (), outputs: Sequence[Attribute] = (), **properties: Attribute
) -> T:
    return kind.create(operands=inputs, result_types=outputs, properties={**props(name), **properties})


def body(inputs: Sequence[SSAValue], *extra: Attribute) -> Block:
    return Block(arg_types=[*(value.type for value in inputs), *extra])


def structured[T: Operation](
    kind: type[T],
    name: str,
    inputs: Sequence[SSAValue],
    blocks: Sequence[Block],
    *,
    outputs: Sequence[Attribute] = (),
    **properties: Attribute,
) -> T:
    return kind.create(
        operands=inputs,
        result_types=outputs,
        regions=[Region(block) for block in blocks],
        properties={
            **props(name),
            "sym_name": StringAttr(name),
            "reproducibility": StringAttr(Reproducibility.UNKNOWN),
            "rank_rule": StringAttr(RankRule.LOCAL),
            "assumptions": ArrayAttr([]),
            "quiescence": ArrayAttr([]),
            **properties,
        },
    )


def end(block: Block, name: str, *values: SSAValue) -> None:
    block.add_op(node(YieldOp, name + ".yield", inputs=values))


def call(block: Block, name: str, binding: str, inputs: Sequence[SSAValue], outputs: Sequence[Attribute]) -> CallOp:
    op = node(CallOp, name, inputs=inputs, outputs=outputs, binding=SymbolRefAttr(binding), binding_version=StringAttr("1"))
    block.add_op(op)
    return op


def access(name: str, inputs: Sequence[SSAValue], block: Block, mode: AccessMode, outputs: Sequence[Attribute]) -> AccessOp:
    freshness = (
        Freshness.PUBLISHED
        if mode == AccessMode.SNAPSHOT
        else Freshness.CURRENT_AT_START
        if mode == AccessMode.PROTECTED_CURRENT
        else Freshness.OWNER_PROTOCOL
    )
    return structured(
        AccessOp,
        name,
        inputs,
        [block],
        outputs=outputs,
        mode=StringAttr(mode),
        freshness=StringAttr(freshness),
        version_selector=StringAttr("initial" if mode == AccessMode.SNAPSHOT else "current"),
        profile=StringAttr("ordinary_owner"),
    )


def take(block: Block, name: str, stream: SSAValue) -> tuple[SSAValue, SSAValue]:
    item = node(TakeOp, name, inputs=[stream], outputs=[ClaimType(RECORD)])
    value = node(BorrowOp, name + ".borrow", inputs=item.results, outputs=[RECORD])
    block.add_ops([item, value])
    return item.results[0], value.results[0]


def retire(block: Block, name: str, claim: SSAValue) -> None:
    block.add_op(node(RetireOp, name, inputs=[claim], disposition=StringAttr("consumed")))


def resource(name: str, kind: str) -> ResourceOp:
    return node(ResourceOp, name, outputs=[ResourceType(StringAttr(kind))], sym_name=StringAttr(name))


def stream(name: str, owners: Sequence[str], horizon: int) -> StreamOp:
    return node(
        StreamOp,
        name,
        outputs=[EndpointType(RECORD)],
        sym_name=StringAttr(name),
        horizon=integer(horizon),
        ready_bound=integer(min(2, horizon)),
        admission=StringAttr(Admission.ANY_WITH_PROVENANCE),
        owners=ArrayAttr(SymbolRefAttr(x) for x in owners),
        metric=StringAttr("owner_update_ordinal_within_lineage"),
        comparison_boundary=StringAttr("item_admission"),
        max_lag=integer(0),
        reproducibility=StringAttr(Reproducibility.UNKNOWN),
        rank_rule=StringAttr(RankRule.LOCAL),
        assumptions=ArrayAttr([]),
    )


def repeat(
    name: str, inputs: Sequence[SSAValue], block: Block, limit: int, *, carries: int = 0, outputs: Sequence[Attribute] = ()
) -> RepeatOp:
    return structured(
        RepeatOp,
        name,
        inputs,
        [block],
        outputs=outputs,
        limit=integer(limit),
        carry_count=integer(carries),
        policy=SymbolRefAttr("bounded_count"),
    )


def lifetime(*, extra_scope: bool = False, prefix: str = "", windows: int = 2, horizon: int = 3) -> ModuleOp:
    def n(name: str) -> str:
        return prefix + name

    encoder, model = resource(n("encoder"), "scale"), resource(n("model"), "scale")
    records = stream(n("records"), [n("encoder")], horizon)
    captures = [records.value, encoder.value, model.value]
    p, consumers, controller = (body(captures) for _ in range(3))

    production = body([p.args[1]], index_type())
    encoded = Block(arg_types=[CapabilityType(encoder.value.type, StringAttr(AccessMode.SNAPSHOT)), index_type()])
    value = call(encoded, n("encode"), "encode", encoded.args, [RECORD])
    end(encoded, n("encode.access"), value.results[0])
    use = access(n("encode.access"), production.args, encoded, AccessMode.SNAPSHOT, [RECORD])
    production.add_op(use)
    end(production, n("production.body"), use.results[0])
    producer = structured(ProducerOp, n("producer"), [p.args[0], p.args[1]], [production])
    p.add_op(producer)
    end(p, n("producer.branch"))

    group_inputs = [consumers.args[0], consumers.args[2]]
    enclosing = body(group_inputs) if extra_scope else None
    actual_inputs = list(enclosing.args) if enclosing is not None else group_inputs
    driver, handler = body(actual_inputs), body(actual_inputs)
    window = body(driver.args, index_type())
    zero = node(ConstantOp, n("zero"), outputs=[SCALAR], literal=StringAttr("0.0"))
    window.add_op(zero)
    member = body([window.args[0], window.args[1]], index_type(), SCALAR)
    claim, record = take(member, n("training.take"), member.args[0])
    gradient_body = Block(arg_types=[CapabilityType(model.value.type, StringAttr(AccessMode.PROTECTED_CURRENT)), RECORD, SCALAR])
    total = call(gradient_body, n("gradient"), "gradient", gradient_body.args, [SCALAR])
    end(gradient_body, n("gradient.access"), total.results[0])
    gradient = access(n("gradient.access"), [member.args[1], record, member.args[3]], gradient_body, AccessMode.PROTECTED_CURRENT, [SCALAR])
    member.add_op(gradient)
    retire(member, n("training.retire"), claim)
    end(member, n("members.body"), gradient.results[0])
    members = repeat(n("members"), [window.args[0], window.args[1], zero.results[0]], member, 2, carries=1, outputs=[SCALAR])
    window.add_op(members)

    update_body = Block(arg_types=[CapabilityType(model.value.type, StringAttr(AccessMode.OWNER_UPDATE)), SCALAR])
    weight = call(update_body, n("advance"), "advance", update_body.args, [SCALAR])
    end(update_body, n("update.access"), weight.results[0])
    window.add_op(access(n("update.access"), [window.args[1], members.results[0]], update_body, AccessMode.OWNER_UPDATE, [SCALAR]))
    second = node(ConstantOp, n("second.window"), outputs=[index_type()], literal=StringAttr("1"))
    window.add_op(second)
    due = call(window, n("evaluation.due"), "equal", [window.args[2], second.results[0]], [bool_type()])
    yes, no = body([window.args[2]]), body([window.args[2]])
    yes.add_op(
        node(
            RequestOp,
            n("request"),
            inputs=yes.args,
            handler=SymbolRefAttr(n("evaluation")),
            source=SymbolRefAttr(n("training")),
            boundary=StringAttr("settled_success"),
        )
    )
    end(yes, n("request.true"))
    end(no, n("request.false"))
    window.add_op(structured(ChoiceOp, n("request.choice"), [due.results[0], window.args[2]], [yes, no]))
    end(window, n("window.body"))
    driver.add_op(repeat(n("training"), driver.args, window, windows))
    end(driver, n("training.branch"))

    evaluation = body(handler.args, RequestType(index_type()))
    claim, record = take(evaluation, n("evaluation.take"), evaluation.args[0])
    measure_body = Block(arg_types=[CapabilityType(model.value.type, StringAttr(AccessMode.PROTECTED_CURRENT)), RECORD])
    result = call(measure_body, n("measure"), "measure", measure_body.args, [MEASUREMENT])
    end(measure_body, n("measure.access"), result.results[0])
    evaluation.add_op(access(n("measure.access"), [evaluation.args[1], record], measure_body, AccessMode.PROTECTED_CURRENT, [MEASUREMENT]))
    retire(evaluation, n("evaluation.retire"), claim)
    end(evaluation, n("evaluation.body"))
    handler.add_op(
        structured(
            RequestsOp,
            n("evaluation"),
            handler.args,
            [evaluation],
            sources=ArrayAttr([SymbolRefAttr(n("training"))]),
            request_schema=index_type(),
            max_outstanding=integer(1),
            max_active=integer(1),
        )
    )
    end(handler, n("evaluation.branch"))
    group = structured(ParallelOp, n("consumers"), actual_inputs, [driver, handler])
    stop_source = n("consumers")
    if enclosing is not None:
        enclosing.add_op(group)
        end(enclosing, n("enclosing.body"))
        consumers.add_op(structured(OnceOp, n("consumer.scope"), group_inputs, [enclosing]))
        stop_source = n("consumer.scope")
    else:
        consumers.add_op(group)
    end(consumers, n("consumers.branch"))
    controller.add_op(node(StopAfterOp, n("stop"), source=SymbolRefAttr(stop_source), target=SymbolRefAttr(n("producer"))))
    end(controller, n("control.branch"))
    activities = structured(ParallelOp, n("activities"), captures, [p, consumers, controller])
    root = ProgramOp.create(
        properties={**props(n("run")), "stage": StringAttr("accepted")}, regions=[Region(Block([encoder, model, records, activities]))]
    )
    return ModuleOp([root])


def data_only(*, prefix: str = "", count: int = 4, horizon: int = 3) -> ModuleOp:
    def n(name: str) -> str:
        return prefix + name

    policy = resource(n("policy"), "policy")
    records = stream(n("records"), [n("policy")], horizon)
    captures = [records.value, policy.value]
    producer, consumer, controller = (body(captures) for _ in range(3))
    production = body([producer.args[1]], index_type())
    selection_body = Block(arg_types=[CapabilityType(policy.value.type, StringAttr(AccessMode.PROTECTED_CURRENT)), index_type()])
    selected = call(selection_body, n("select"), "select", selection_body.args, [RECORD])
    end(selection_body, n("select.access"), selected.results[0])
    selection = access(n("select.access"), production.args, selection_body, AccessMode.PROTECTED_CURRENT, [RECORD])
    production.add_op(selection)
    end(production, n("production.body"), selection.results[0])
    producer.add_op(structured(ProducerOp, n("producer"), producer.args, [production]))
    end(producer, n("producer.branch"))
    iteration = body(consumer.args, index_type())
    claim, record = take(iteration, n("consume.take"), iteration.args[0])
    arithmetic = Block(arg_types=[RECORD])
    score = call(arithmetic, n("score"), "score", arithmetic.args, [SCALAR])
    end(arithmetic, n("score.recipe"), score.results[0])
    recipe = structured(RecipeOp, n("score.recipe"), [record], [arithmetic], outputs=[SCALAR])
    iteration.add_op(recipe)
    feedback_body = Block(arg_types=[CapabilityType(policy.value.type, StringAttr(AccessMode.OWNER_UPDATE)), SCALAR])
    receipt = call(feedback_body, n("feedback"), "feedback", feedback_body.args, [SCALAR])
    end(feedback_body, n("feedback.access"), receipt.results[0])
    iteration.add_op(access(n("feedback.access"), [iteration.args[1], recipe.results[0]], feedback_body, AccessMode.OWNER_UPDATE, [SCALAR]))
    retire(iteration, n("consume.retire"), claim)
    end(iteration, n("consume.body"))
    consumer.add_op(repeat(n("consumer"), consumer.args, iteration, count))
    end(consumer, n("consumer.branch"))
    controller.add_op(node(StopAfterOp, n("stop"), source=SymbolRefAttr(n("consumer")), target=SymbolRefAttr(n("producer"))))
    end(controller, n("control.branch"))
    activities = structured(ParallelOp, n("activities"), captures, [producer, consumer, controller])
    root = ProgramOp.create(
        properties={**props(n("run")), "stage": StringAttr("accepted")}, regions=[Region(Block([policy, records, activities]))]
    )
    return ModuleOp([root])
