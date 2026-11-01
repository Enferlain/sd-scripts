"""Python fixture authoring; neither verification nor execution depends on it."""

from __future__ import annotations

from xdsl.dialects.builtin import DictionaryAttr, ModuleOp, StringAttr
from xdsl.ir import Attribute, Block, Region, SSAValue

from . import lifetime_ir as ir


def build(*, windows: int = 2, horizon: int = 3, after_window: int = 2, members: int = 2, prefix: str = "") -> ModuleOp:
    if any(type(n) is not int or n < 1 for n in (windows, horizon, after_window, members)):
        raise ValueError("positive integer case bounds required")

    def make(
        cls: type[ir.ActivityOp],
        source: str,
        operands: tuple[SSAValue, ...] = (),
        outputs: tuple[Attribute, ...] = (),
        body: Block | None = None,
        **policy: str,
    ) -> ir.ActivityOp:
        return cls.create(
            operands=operands,
            result_types=outputs,
            properties={"source_id": StringAttr(prefix + source), "policy": DictionaryAttr({k: StringAttr(v) for k, v in policy.items()})},
            regions=[] if body is None else [Region(body)],
        )

    encoder = make(ir.ModelOp, "encoder", outputs=(ir.ModelType(),), implementation="scale_v1", initial="2.0", treatment="frozen")
    learner = make(ir.ModelOp, "learner", outputs=(ir.ModelType(),), implementation="scale_v1", initial="0.8", treatment="trainable")
    consumers = Block()
    group = make(
        ir.GroupOp,
        "consumers",
        outputs=(ir.DoneType(),),
        body=consumers,
        completion="children_physically_settled",
        pending="include_requested",
        failure="cancel_siblings_and_drain",
    )
    producer = make(
        ir.ProducerOp,
        "producer",
        (encoder.results[0], group.results[0]),
        (ir.StreamType(), ir.DoneType()),
        implementation="encode_v1",
        horizon=str(horizon),
        ready="2",
        workers="1",
        remaining="discard_unclaimed",
        failure="close_then_drain",
    )
    window = Block(arg_types=(ir.StreamType(), ir.ModelType()))
    previous = ()
    for slot in range(members):
        take = make(
            ir.TakeOp,
            f"training.take.{slot}",
            (window.args[0],),
            (ir.SampleType(),),
            selection=f"slot:{slot}",
            provenance="actual_frozen_source",
        )
        contribution = make(
            ir.ContributeOp,
            f"training.objective.{slot}",
            (take.results[0], window.args[1], *previous),
            (ir.GradientType(),),
            implementation="objective_v1",
            normalization=str(members),
        )
        previous = (contribution.results[0],)
        window.add_ops((take, contribution))
    advance = make(
        ir.AdvanceOp,
        "training.advance",
        (*previous, window.args[1]),
        (ir.CompletionType(),),
        parameter="weight",
        learning_rate="0.01",
        authority="ordinary_owner",
    )
    window.add_ops((advance, make(ir.WindowEndOp, "training.window_end", (advance.results[0],))))
    training = make(
        ir.RepeatOp,
        "training",
        (producer.results[0], learner.results[0]),
        (ir.EventsType(), ir.DoneType()),
        window,
        max_windows=str(windows),
        admission="policy_choice_before_first_claim",
        stop="finish_selected_window",
    )
    request = make(
        ir.RequestOp,
        "request",
        (training.results[0],),
        (ir.RequestType(),),
        after_window=str(after_window),
        multiplicity="once_per_matching_completion",
        pending="keeps_group_alive",
        max_pending="1",
        cancellation="cancel_or_unfulfilled",
    )
    measurement = Block(arg_types=(ir.RequestType(), ir.StreamType(), ir.ModelType()))
    sample = make(
        ir.TakeOp,
        "evaluation.take",
        (measurement.args[1],),
        (ir.SampleType(),),
        selection="after_trigger_members",
        provenance="actual_frozen_source",
    )
    read = make(
        ir.ReadOp,
        "evaluation.read",
        (sample.results[0], measurement.args[2]),
        (ir.MeasurementType(),),
        implementation="measure_v1",
        state="current_at_start",
        mode="evaluation",
        completion="physical_validate_restore",
    )
    measurement.add_ops((sample, read, make(ir.EvaluationEndOp, "evaluation.end", (read.results[0],))))
    evaluator = make(
        ir.EvaluatorOp,
        "evaluation",
        (request.results[0], producer.results[0], learner.results[0]),
        (ir.DoneType(),),
        measurement,
        completion="physical_and_cleanup",
    )
    consumers.add_ops((training, request, evaluator, make(ir.SettleOp, "consumers.settle", (training.results[1], evaluator.results[0]))))
    body = Block((encoder, learner, producer, group, make(ir.FinishOp, "run.finish", (group.results[0], producer.results[1]))))
    return ModuleOp((make(ir.RunOp, "run", body=body, version="1", failure="cancel_siblings_and_drain"),))
