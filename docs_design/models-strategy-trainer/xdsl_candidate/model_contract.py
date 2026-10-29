"""Semantic A contract and Python author; no model/runtime imports."""

from __future__ import annotations

import math

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, ModuleOp, StringAttr, SymbolRefAttr, f64
from xdsl.ir import Block, Operation, Region, SSAValue

from .dialects import ComposeOp, EntityOp, FlowOp, FlowType, LocalOp, MapOp, SessionOp, YieldOp, integer
from .model_dialect import AdvanceOp, ContributeOp, CorrelateOp, EvaluateOp, FanOutOp, FilterOp, ModelFlowOp, ModelOp, SourceOp, WindowOp
from .prepare import Accepted, ContractExtension, accept, local_maps, ref, require

CONTRACT_A = "correlated-model-v1"


def one[T: Operation](root: Operation, kind: type[T]) -> T:
    selected = [op for op in root.walk() if isinstance(op, kind)]
    require(len(selected) == 1, root, f"one {kind.name} required")
    return selected[0]


def numbers(values: ArrayAttr[IntegerAttr]) -> tuple[int, ...]:
    return tuple(value.value.data for value in values)


def selected_keys(session: SessionOp) -> tuple[int, ...]:
    excluded = set(numbers(one(session, FilterOp).excluded))
    return tuple(key for key in numbers(one(session, SourceOp).keys) if key not in excluded)


def verify_model(session: SessionOp) -> None:
    require(session.horizon is not None, session, "explicit finite source horizon required")
    require(session.stop_rule.data == "drain_entered_close_unadmitted", session, "unknown stop/failure disposition")
    children = tuple(session.body.block.ops)
    allowed = (EntityOp, FlowOp, SourceOp, FilterOp, FanOutOp, ComposeOp, ModelOp, ModelFlowOp, CorrelateOp, WindowOp, EvaluateOp)
    require(all(isinstance(op, allowed) for op in children), session, "unknown service operation")
    entities: dict[str, EntityOp] = {}
    for op in children:
        if isinstance(op, EntityOp):
            require(bool(op.sym_name.data) and op.sym_name.data not in entities, op, "empty/duplicate entity symbol")
            require(op.kind.data in ("participant", "unit", "observation"), op, "unknown entity kind")
            entities[op.sym_name.data] = op
    require(len(entities) == 4, session, "two participants, one unit and one observation owner required")

    def entity(op: Operation, value: SymbolRefAttr, kind: str) -> str:
        name = ref(value)
        require(name in entities and entities[name].kind.data == kind, op, f"unresolved {kind} use {value}")
        return name

    models = [op for op in children if isinstance(op, ModelOp)]
    require(
        len(models) == 2 and {op.treatment.data for op in models} == {"frozen", "trainable"},
        session,
        "frozen source and trainable subject required",
    )
    model_names: dict[str, str] = {}
    for op in models:
        name = entity(op, op.entity, "participant")
        require(name not in model_names.values(), op, "duplicate model binding")
        require(math.isfinite(op.initial.value.data) and bool(ref(op.implementation)), op, "invalid initial model or implementation")
        model_names[op.treatment.data] = name
    source, filtered, fanout = one(session, SourceOp), one(session, FilterOp), one(session, FanOutOp)
    compose, apply = one(session, ComposeOp), one(session, ModelFlowOp)
    correlate, window, evaluate = one(session, CorrelateOp), one(session, WindowOp), one(session, EvaluateOp)
    keys, excluded = numbers(source.keys), numbers(filtered.excluded)
    require(bool(keys) and len(set(keys)) == len(keys), source, "source keys must be nonempty and unique")
    require(len(set(excluded)) == len(excluded) and not set(excluded) - set(keys), filtered, "invalid authoritative exclusion set")
    require(bool(selected_keys(session)), filtered, "all records excluded")
    local_maps(compose)
    require(entity(apply, apply.model, "participant") == model_names["frozen"], apply, "model source must be frozen in this contract")
    require(bool(ref(apply.implementation)), apply, "missing model implementation")
    require(
        correlate.membership.data == "same_sample_and_slot" and correlate.provenance.data == "current_source_at_admission",
        correlate,
        "unknown admission contract",
    )
    count = window.count.value.data
    require(count > 0 and len(selected_keys(session)) % count == 0, window, "full positive contribution windows required")
    require(bool(window.sym_name.data) and window.sym_name.data not in entities, window, "invalid activity symbol")
    block = window.body.block
    require(
        len(block.args) == 1 and block.args[0].type == FlowType(StringAttr("record")), window, "one correlated endpoint argument required"
    )
    contribute, advance = one(window, ContributeOp), one(window, AdvanceOp)
    require(tuple(block.ops) == (contribute, advance) and contribute.input is block.args[0], window, "window must contribute then advance")
    require(contribute.normalization.value.data == count, contribute, "loss normalization must match the declared window")
    require(entity(contribute, contribute.model, "participant") == model_names["trainable"], contribute, "wrong training subject")
    unit = entity(contribute, contribute.unit, "unit")
    require(entity(advance, advance.unit, "unit") == unit, advance, "wrong advancement unit")
    require(
        bool(advance.parameter.data) and math.isfinite(advance.learning_rate.value.data) and advance.learning_rate.value.data > 0,
        advance,
        "invalid parameter or learning rate",
    )
    require(bool(ref(contribute.implementation)), contribute, "missing objective implementation")
    entity(evaluate, evaluate.owner, "observation")
    require(entity(evaluate, evaluate.model, "participant") == model_names["trainable"], evaluate, "wrong evaluation subject")
    require(ref(evaluate.trigger) == window.sym_name.data, evaluate, "unknown request source")
    require(evaluate.consistency.data == "current_at_start", evaluate, "unknown evaluation consistency")
    require(0 < evaluate.after_window.value.data <= len(selected_keys(session)) // count, evaluate, "unreachable evaluation trigger")
    require(bool(ref(evaluate.implementation)), evaluate, "missing evaluator implementation")
    edges = (
        (source.output, filtered.input),
        (filtered.output, fanout.input),
        (fanout.left, compose.input),
        (fanout.right, apply.input),
        (compose.output, correlate.left),
        (apply.output, correlate.right),
        (correlate.output, window.input),
    )
    declared = [op.endpoint for op in children if isinstance(op, FlowOp)]
    require(len(declared) == len(edges), session, "unexpected endpoint declarations")
    seen: set[SSAValue] = set()
    for writer, reader in edges:
        require(writer is reader and writer in declared and writer not in seen, session, "broken or aliased service wiring")
        require(
            writer.type == FlowType(StringAttr("record")) and len(tuple(writer.uses)) == 2,
            session,
            "record endpoint requires one writer/reader",
        )
        seen.add(writer)


EXTENSION_A = ContractExtension(
    CONTRACT_A,
    1,
    verify_model,
    (
        "resolve versioned model/data/objective/evaluation implementations",
        "check physical parameter and destination correspondence before publication",
        "atomically check all logical samples, slots and current source provenance before claim",
        "retain input associations through each two-contribution window and requested evaluation",
        "observe actual evaluation-start provenance and physical completion independently of its trigger",
    ),
)


def accept_model(module: ModuleOp) -> Accepted:
    return accept(module, extension=EXTENSION_A)


def build_model(*, count: int = 2, horizon: int = 4) -> ModuleOp:
    record = FlowType(StringAttr("record"))
    entities = [
        EntityOp(properties={"sym_name": StringAttr(name), "source_id": StringAttr("owner." + name), "kind": StringAttr(kind)})
        for name, kind in (("encoder", "participant"), ("learner", "participant"), ("weights", "unit"), ("evaluation", "observation"))
    ]
    names = ("raw", "selected", "left.input", "right.input", "left.output", "right.output", "correlated")
    endpoints = [FlowOp(properties={"source_id": StringAttr(name)}, result_types=[record]) for name in names]
    raw, selected, left_in, right_in, left_out, right_out, correlated = [op.endpoint for op in endpoints]
    source = SourceOp(operands=[raw], properties={"source_id": StringAttr("source"), "keys": ArrayAttr(integer(k) for k in range(1, 7))})
    filtered = FilterOp(
        operands=[raw, selected], properties={"source_id": StringAttr("filter"), "excluded": ArrayAttr([integer(3), integer(6)])}
    )
    fanout = FanOutOp(operands=[selected, left_in, right_in], properties={"source_id": StringAttr("fanout")})
    local, nested = Block(arg_types=[record]), Block(arg_types=[record])
    double = MapOp(
        operands=[local.args[0]],
        result_types=[record],
        properties={"source_id": StringAttr("data.double"), "implementation": SymbolRefAttr("double_v1"), "effect": StringAttr("pure")},
    )
    offset = MapOp(
        operands=[nested.args[0]],
        result_types=[record],
        properties={"source_id": StringAttr("data.offset"), "implementation": SymbolRefAttr("offset_v1"), "effect": StringAttr("pure")},
    )
    nested.add_ops([offset, YieldOp(operands=[offset.output])])
    group = LocalOp(
        operands=[double.output], result_types=[record], properties={"source_id": StringAttr("data.nested")}, regions=[Region(nested)]
    )
    local.add_ops([double, group, YieldOp(operands=[group.output])])
    composed = ComposeOp(operands=[left_in, left_out], properties={"source_id": StringAttr("data.composed")}, regions=[Region(local)])
    models = [
        ModelOp(
            properties={
                "source_id": StringAttr("model." + name),
                "entity": SymbolRefAttr(name),
                "implementation": SymbolRefAttr("scale_v1"),
                "initial": FloatAttr(initial, f64),
                "treatment": StringAttr(treatment),
            }
        )
        for name, initial, treatment in (("encoder", 1.5, "frozen"), ("learner", 0.8, "trainable"))
    ]
    apply = ModelFlowOp(
        operands=[right_in, right_out],
        properties={
            "source_id": StringAttr("data.encode"),
            "model": SymbolRefAttr("encoder"),
            "implementation": SymbolRefAttr("encode_v1"),
        },
    )
    correlate = CorrelateOp(
        operands=[left_out, right_out, correlated],
        properties={
            "source_id": StringAttr("admission"),
            "membership": StringAttr("same_sample_and_slot"),
            "provenance": StringAttr("current_source_at_admission"),
        },
    )
    body = Block(arg_types=[record])
    contribute = ContributeOp(
        operands=[body.args[0]],
        properties={
            "source_id": StringAttr("objective"),
            "model": SymbolRefAttr("learner"),
            "unit": SymbolRefAttr("weights"),
            "implementation": SymbolRefAttr("loss_v1"),
            "normalization": integer(count),
        },
    )
    advance = AdvanceOp(
        properties={
            "source_id": StringAttr("advance"),
            "unit": SymbolRefAttr("weights"),
            "learning_rate": FloatAttr(0.05, f64),
            "parameter": StringAttr("weight"),
        }
    )
    body.add_ops([contribute, advance])
    window = WindowOp(
        operands=[correlated],
        properties={"source_id": StringAttr("window"), "sym_name": StringAttr("learning"), "count": integer(count)},
        regions=[Region(body)],
    )
    evaluate = EvaluateOp(
        properties={
            "source_id": StringAttr("evaluation"),
            "owner": SymbolRefAttr("evaluation"),
            "model": SymbolRefAttr("learner"),
            "trigger": SymbolRefAttr("learning"),
            "after_window": integer(1),
            "implementation": SymbolRefAttr("measure_v1"),
            "consistency": StringAttr("current_at_start"),
        }
    )
    session = SessionOp(
        properties={
            "source_id": StringAttr("model-session"),
            "contract": StringAttr(CONTRACT_A),
            "version": integer(1),
            "horizon": integer(horizon),
            "stop_rule": StringAttr("drain_entered_close_unadmitted"),
        },
        regions=[Region(Block([*entities, *endpoints, source, filtered, fanout, composed, *models, apply, correlate, window, evaluate]))],
    )
    return ModuleOp([session])
