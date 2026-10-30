"""Bounded open-producer/learning contract. This is an explicit installed profile."""

from __future__ import annotations

import math

from xdsl.dialects.builtin import ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Operation, SSAValue

from .dialects import ComposeOp, EntityOp, FlowOp, FlowType, SelectOp, SessionOp, integer
from .mixed_dialect import DynamicFilterOp, LearnedOp, PolicyUpdateOp, StopCheckpointOp
from .model_contract import build_model, one
from .model_dialect import (
    AdvanceOp,
    ContributeOp,
    CorrelateOp,
    EvaluateOp,
    FanOutOp,
    FilterOp,
    ModelFlowOp,
    ModelOp,
    SourceOp,
    WindowOp,
)
from .prepare import Accepted, ContractExtension, accept, local_maps, ref, require

CONTRACT_MIXED = "continuous-learning-feedback-v1"
POLICY_PROTOCOL = "post_advance_weight; bias=int(abs(weight)*10)%5; revision+=1; retire_once"
STOP_PROTOCOL = "external_at_checkpoint; close_selection; discard_unclaimed; drain_entered"
SOURCE_PROFILE = "same_binding_protected_at_production"


def verify_mixed(session: SessionOp) -> None:
    require(session.horizon is not None, session, "explicit outstanding-ticket horizon required")
    require(session.stop_rule.data == STOP_PROTOCOL, session, "unknown mixed stop disposition")
    children = tuple(session.body.block.ops)
    allowed = (
        EntityOp,
        FlowOp,
        SelectOp,
        DynamicFilterOp,
        FanOutOp,
        ComposeOp,
        ModelOp,
        ModelFlowOp,
        CorrelateOp,
        WindowOp,
        PolicyUpdateOp,
        StopCheckpointOp,
    )
    require(all(isinstance(op, allowed) for op in children), session, "unknown mixed service")
    entities = [op for op in children if isinstance(op, EntityOp)]
    declarations = {op.sym_name.data: op.kind.data for op in entities}
    require(
        len(entities) == len(declarations) == 4
        and all(declarations)
        and set(declarations.values()) == {"cursor", "policy", "participant", "unit"},
        session,
        "one cursor, policy, participant and unit required",
    )

    def entity(op: Operation, value: SymbolRefAttr, kind: str) -> str:
        name = ref(value)
        require(declarations.get(name) == kind, op, f"unresolved {kind} use {value}")
        return name

    select, filtered, fanout = one(session, SelectOp), one(session, DynamicFilterOp), one(session, FanOutOp)
    compose, model, apply = one(session, ComposeOp), one(session, ModelOp), one(session, ModelFlowOp)
    correlate, window, update = one(session, CorrelateOp), one(session, WindowOp), one(session, PolicyUpdateOp)
    checkpoint = one(session, StopCheckpointOp)
    entity(select, select.cursor, "cursor")
    policy = entity(select, select.policy, "policy")
    require(select.application.data == "ack_before_next_selection", select, "unknown policy selection boundary")
    require(entity(filtered, filtered.policy, "policy") == entity(update, update.policy, "policy") == policy, update, "policy differs")
    require(update.protocol.data == POLICY_PROTOCOL, update, "unknown feedback update/retirement protocol")
    learner = entity(model, model.entity, "participant")
    require(model.treatment.data == "trainable" and math.isfinite(model.initial.value.data), model, "finite trainable model required")
    require(entity(apply, apply.model, "participant") == learner, apply, "preprocessing must use the shared participant")
    local_maps(compose)
    require(
        correlate.membership.data == "same_ticket_and_ordinal" and correlate.provenance.data == SOURCE_PROFILE,
        correlate,
        "unknown ticket/source admission profile",
    )
    require(
        window.count.value.data > 0 and bool(window.sym_name.data) and window.sym_name.data not in declarations, window, "invalid window"
    )
    block = window.body.block
    contribute, advance, learned = one(window, ContributeOp), one(window, AdvanceOp), one(window, LearnedOp)
    require(tuple(block.ops) == (contribute, advance, learned), window, "contribute, advance, then observed feedback required")
    require(
        len(block.args) == 1 and block.args[0].type == FlowType(StringAttr("record")) and contribute.input is block.args[0],
        window,
        "bad window input",
    )
    require(contribute.normalization.value.data == window.count.value.data, contribute, "normalization differs")
    require(
        entity(contribute, contribute.model, "participant") == entity(learned, learned.model, "participant") == learner,
        learned,
        "learning subject differs",
    )
    require(entity(contribute, contribute.unit, "unit") == entity(advance, advance.unit, "unit"), advance, "advancement unit differs")
    require(
        bool(advance.parameter.data) and math.isfinite(advance.learning_rate.value.data) and advance.learning_rate.value.data > 0,
        advance,
        "invalid update",
    )
    require(learned.observation.data == "returned_advance_current_weight_and_input_associations", learned, "unknown post-step observation")
    require(
        ref(checkpoint.activity) == window.sym_name.data and checkpoint.after_windows.value.data > 0, checkpoint, "unreachable checkpoint"
    )
    require(checkpoint.protocol.data == STOP_PROTOCOL, checkpoint, "undeclared stop checkpoint")
    for op in (model, apply, contribute, filtered, update):
        require(bool(ref(op.implementation)), op, "empty implementation symbol")
    edges = (
        (select.output, filtered.input, "record"),
        (filtered.output, fanout.input, "record"),
        (fanout.left, compose.input, "record"),
        (fanout.right, apply.input, "record"),
        (compose.output, correlate.left, "record"),
        (apply.output, correlate.right, "record"),
        (correlate.output, window.input, "record"),
        (learned.output, update.input, "feedback"),
        (update.acknowledgement, select.feedback, "feedback"),
    )
    declared = [op.endpoint for op in children if isinstance(op, FlowOp)]
    require(len(declared) == len(edges), session, "unexpected mixed endpoints")
    seen: set[SSAValue] = set()
    for writer, reader, payload in edges:
        require(writer is reader and writer in declared and writer not in seen, session, "broken or aliased mixed wiring")
        require(writer.type == FlowType(StringAttr(payload)) and len(tuple(writer.uses)) == 2, session, "wrong endpoint schema/uses")
        seen.add(writer)


EXTENSION_MIXED = ContractExtension(
    CONTRACT_MIXED,
    1,
    verify_mixed,
    (
        "resolve versioned model, data, filter, objective and policy bindings",
        "protect the actual model read through physical completion before publication",
        "check ticket membership and same-binding actual production provenance before atomic claim",
        "publish post-advance evidence through the producer-owned feedback receiver",
        "close at the explicit external-stop checkpoint and drain physical work before owner release",
    ),
)


def accept_mixed(module: ModuleOp) -> Accepted:
    return accept(module, extension=EXTENSION_MIXED)


def build_mixed(*, horizon: int = 3, after_windows: int = 3) -> ModuleOp:
    """Reuse A's authoring convenience; the resulting document has no A contract."""
    module = build_model(count=1, horizon=horizon)
    session = one(module, SessionOp)
    session.properties.update(
        contract=StringAttr(CONTRACT_MIXED), source_id=StringAttr("mixed-session"), stop_rule=StringAttr(STOP_PROTOCOL)
    )
    for op in tuple(session.body.block.ops):
        if (
            isinstance(op, SourceOp | FilterOp | EvaluateOp)
            or (isinstance(op, ModelOp) and ref(op.entity) == "encoder")
            or (isinstance(op, EntityOp) and op.sym_name.data in ("encoder", "evaluation"))
        ):
            session.body.block.erase_op(op)
    for name, kind in (("cursor", "cursor"), ("policy", "policy")):
        session.body.block.add_op(
            EntityOp(properties={"source_id": StringAttr("owner." + name), "sym_name": StringAttr(name), "kind": StringAttr(kind)})
        )
    flows = {op.source_id.data: op.endpoint for op in session.body.block.ops if isinstance(op, FlowOp)}
    controls = [
        FlowOp(properties={"source_id": StringAttr(name)}, result_types=[FlowType(StringAttr("feedback"))])
        for name in ("learned", "acknowledged")
    ]
    session.body.block.add_ops(controls)
    learned, ack = (op.endpoint for op in controls)
    session.body.block.add_ops(
        [
            SelectOp(
                operands=[ack, flows["raw"]],
                properties={
                    "source_id": StringAttr("select"),
                    "cursor": SymbolRefAttr("cursor"),
                    "policy": SymbolRefAttr("policy"),
                    "application": StringAttr("ack_before_next_selection"),
                },
            ),
            DynamicFilterOp(
                operands=[flows["raw"], flows["selected"]],
                properties={
                    "source_id": StringAttr("filter"),
                    "policy": SymbolRefAttr("policy"),
                    "implementation": SymbolRefAttr("dynamic_filter_v1"),
                },
            ),
            PolicyUpdateOp(
                operands=[learned, ack],
                properties={
                    "source_id": StringAttr("policy.update"),
                    "policy": SymbolRefAttr("policy"),
                    "implementation": SymbolRefAttr("learned_bias_v1"),
                    "protocol": StringAttr(POLICY_PROTOCOL),
                },
            ),
            StopCheckpointOp(
                properties={
                    "source_id": StringAttr("stop.checkpoint"),
                    "activity": SymbolRefAttr("learning"),
                    "after_windows": integer(after_windows),
                    "protocol": StringAttr(STOP_PROTOCOL),
                }
            ),
        ]
    )
    one(session, ModelFlowOp).properties["model"] = SymbolRefAttr("learner")
    one(session, ContributeOp).properties["implementation"] = SymbolRefAttr("mixed_loss_v1")
    one(session, CorrelateOp).properties.update(membership=StringAttr("same_ticket_and_ordinal"), provenance=StringAttr(SOURCE_PROFILE))
    one(session, AdvanceOp).properties["learning_rate"] = type(one(session, AdvanceOp).learning_rate)(
        0.001, one(session, AdvanceOp).learning_rate.type
    )
    window = one(session, WindowOp)
    window.body.block.add_op(
        LearnedOp(
            operands=[learned],
            properties={
                "source_id": StringAttr("learned.observe"),
                "model": SymbolRefAttr("learner"),
                "observation": StringAttr("returned_advance_current_weight_and_input_associations"),
            },
        )
    )
    return module
