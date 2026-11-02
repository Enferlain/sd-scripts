"""Existing A owner target, entered only after independent IR verification.

Plain frozen target declarations replace frontend IR objects. Live admission,
state protection, optimization, requested work and physical completion still
belong to the existing owner engine.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType, TracebackType
from typing import Any, Literal

import torch
from torch import nn

local_runtime = importlib.import_module("docs_design.models-strategy-trainer.xdsl_candidate.local_runtime")
LocalMetrics, LocalProgram = local_runtime.LocalMetrics, local_runtime.LocalProgram
_prefix = "docs_design.models-strategy-trainer.execution_candidate"
graph = importlib.import_module(_prefix + ".graph")
backend = importlib.import_module(_prefix + ".prepare")
owners = importlib.import_module(_prefix + ".state")
physical = importlib.import_module(_prefix + ".workers")
examples = importlib.import_module(_prefix + ".examples")


class NotReady(ValueError):
    pass


@dataclass(frozen=True)
class ModelBindingSpec:
    name: str
    origin: str
    initial: float
    trainable: bool


@dataclass(frozen=True)
class ModelSpec:
    models: tuple[ModelBindingSpec, ...]
    window: str
    encoder: str
    learner: str
    unit: str
    parameter: str
    learning_rate: float
    excluded: tuple[int, ...]
    request_after: int
    normalization: int
    encode_origin: str
    objective_origin: str
    evaluate_origin: str


@dataclass(frozen=True)
class DomainBinding:
    role: str
    effect: str
    version: str
    invoke: Callable[..., Any]


@dataclass(frozen=True)
class DomainRegistry:
    name: str
    bindings: Mapping[str, DomainBinding]

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))


def scale(initial: float, trainable: bool) -> nn.Module:
    model = examples.Scale(initial)
    model.weight.requires_grad_(trainable)
    return model


def encode(model: nn.Module, sample: int) -> torch.Tensor:
    with torch.no_grad():
        return model(torch.tensor(sample / 4)).detach().clone()


def loss(model: nn.Module, sample: int, left: torch.Tensor, right: torch.Tensor, count: int) -> torch.Tensor:
    return (model(left + right) - sample / 2).square() / count


def measure(model: nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return model(torch.tensor(1.0)).detach().clone()


def model_registry() -> DomainRegistry:
    fixture = importlib.import_module("docs_design.models-strategy-trainer.xdsl_candidate.fixture")
    bindings = {
        "double_v1": DomainBinding("map", "pure", "1", fixture.double),
        "offset_v1": DomainBinding("map", "pure", "1", fixture.offset),
    }
    bindings.update(
        {
            "scale_v1": DomainBinding("model", "load", "1", scale),
            "encode_v1": DomainBinding("encode", "read_source", "1", encode),
            "loss_v1": DomainBinding("objective", "build_loss", "1", loss),
            "measure_v1": DomainBinding("measure", "current_read", "1", measure),
        }
    )
    return DomainRegistry("correlated-fixtures-v1", bindings)


@dataclass(frozen=True)
class Record:
    sample: int
    window: int
    slot: int
    branch: str
    value: torch.Tensor


@dataclass
class SourceFacts:
    excluded: list[int] = field(default_factory=list)
    products: list[dict[str, Any]] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    local_metrics: LocalMetrics = field(default_factory=LocalMetrics)


@dataclass
class ModelLocalFrame:
    owner: SourceFacts
    sample: int
    origin: str
    value: Any
    fact: dict[str, Any] = field(default_factory=dict)

    def __enter__(self) -> ModelLocalFrame:
        self.fact = {"sample": self.sample, "origin": self.origin, "input": self.value, "output": None, "outcome": "entered_failed"}
        self.owner.stages.append(self.fact)
        return self

    def physical_end(self) -> None:
        pass  # This observer records outcomes; the binding ABI owns physical completion.

    def validate(self, result: Any) -> int:
        self.fact["outcome"] = "invalid_output"
        if type(result) is not int:
            raise ValueError(f"{self.origin}: invalid returned integer after entry")
        self.fact.update(output=result, outcome="returned")
        return result

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None) -> None:
        pass


@dataclass
class LearningFacts:
    contributions: list[dict[str, Any]] = field(default_factory=list)
    windows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvaluationFacts:
    measurements: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Controls:
    """Test-only scheduling and fault controls; never serialized as meaning."""

    omit_sample: int | None = None
    corrupt_sample: int | None = None
    swap_member: int | None = None
    stale_sample: int | None = None
    history_limit: int = 256
    reverse_production: bool = True
    evaluation_requested: threading.Event | None = None


class Admission:
    """Bind physical ports once; validate member role before the shared join.

    join() holds their shared condition while calling us. Port identity is thus
    stable through all-member checking and claim. No second input owner exists.
    """

    def __init__(self, slot: int, addresses: Mapping[tuple[int, int], int], encoder: str):
        self.slot, self.addresses, self.encoder = slot, addresses, encoder
        self.ports: tuple[Any, ...] = ()

    def bind(self, left: Any, right: Any) -> None:
        if self.ports:
            raise NotReady("admission ports already bound")
        self.ports = (left, right)

    def __call__(self, product: Any, projection: Any) -> bool:
        record = product.value
        if not isinstance(record, Record):
            return False
        member = next((index for index, port in enumerate(self.ports) if port.ready.get(product.key) is product), None)
        expected = self.addresses.get((product.key, self.slot))
        if (
            member is None
            or record.sample != expected
            or record.window != product.key
            or record.slot != self.slot
            or record.branch != ("left", "right")[member]
            or not scalar_tensor(record.value)
        ):
            return False
        stamps = () if member == 0 else (projection.stamps[self.encoder],)
        return product.provenance == stamps


def scalar_tensor(value: object) -> bool:
    return (
        isinstance(value, torch.Tensor) and value.shape == torch.Size([]) and value.dtype == torch.float32 and bool(torch.isfinite(value))
    )


def handoffs(context: Any) -> list[dict[str, Any]]:
    return [
        {
            "channel": item.channel,
            "window": item.key,
            "sources": [(stamp.participant_ref, stamp.binding, stamp.numerical_epoch) for stamp in item.provenance],
        }
        for item in context.input_handoffs
    ]


@dataclass
class ModelRun:
    prepared: Any
    ready: Any
    engine: Any
    last_report: dict[str, Any] | None = None
    interrupted: bool = False
    started: bool = False

    async def run(self) -> dict[str, Any]:
        if self.started:
            raise RuntimeError("model run has one owner lifetime")
        self.started = True
        task = asyncio.current_task()
        cancellations_at_start = task.cancelling() if task is not None else 0
        failure: BaseException | None = None
        try:
            await self.engine.run()
        except BaseException as error:
            failure = error
            self.interrupted = isinstance(error, asyncio.CancelledError) or (
                task is not None and task.cancelling() > cancellations_at_start
            )
        report = self.report(failure)
        self.last_report = report
        if failure is not None and self.interrupted:
            raise failure
        return report

    def report(self, failure: BaseException | None = None) -> dict[str, Any]:
        failure = failure or self.engine.failure
        state = self.ready.state
        image = state.image
        if image is None:
            raise RuntimeError("model owner has no materialized state")
        outcome = [
            {
                "invocation": item.invocation,
                "advance_invocation": item.advancement_invocation,
                "contribution": item.contribution,
                "optimizer": item.optimizer,
                "reset": item.reset,
                "inputs": [{"channel": handoff.channel, "window": handoff.key} for handoff in item.input_handoffs],
            }
            for runtime in image.units.values()
            for item in {record.invocation: record for record in (*runtime.outcomes, *runtime.active_outcomes.values())}.values()
        ]
        result = {
            "status": "stopped"
            if state.completed
            else "cancelled"
            if self.interrupted or isinstance(failure, asyncio.CancelledError)
            else "failed",
            "error": None if failure is None else str(failure),
            "errors": [{"type": type(item).__name__, "message": str(item)} for item in causes(failure)],
            "addresses": [{"window": window, "slot": slot, "sample": sample} for (window, slot), sample in self.prepared.addresses.items()],
            "source": asdict(state.owners["a.source"]),
            "learning": asdict(state.owners["a.learning"]),
            "evaluation": asdict(state.owners["a.evaluation"]),
            "unit_outcomes": outcome,
            "pending": any(runtime.unfinished for runtime in image.units.values()),
            "steps": sum(runtime.steps for runtime in image.units.values()),
            "remaining": {name: sorted(feed.ready) for name, feed in state.feeds.items()},
            "claimed": {name: [(key, invocation) for key, invocation, _ in feed.handed_off] for name, feed in state.feeds.items()},
            "associations_on_failure": [{"channel": item.channel, "window": item.key} for item in self.engine.failure_input_handoffs],
            "released": not state.running
            and not state.readers
            and not state.writers
            and not state.state_locks
            and not state.active_producers
            and not state.active_retentions,
            "diagnostics": len(state.trace),
        }
        # Normalize opaque backend authority IDs to declared semantic names for
        # the portable observation. Actual admission still used original stamps.
        labels = {value: name for name, value in state.accepted.participant_refs.items()}

        def symbolic(value: Any) -> Any:
            if isinstance(value, str):
                return labels.get(value, value)
            if isinstance(value, dict):
                return {key: symbolic(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [symbolic(item) for item in value]
            return value

        return symbolic(result)


def causes(error: BaseException | None) -> tuple[BaseException, ...]:
    if error is None:
        return ()
    if isinstance(error, BaseExceptionGroup):
        return tuple(item for nested in error.exceptions for item in causes(nested))
    nested = getattr(error, "cause", None)
    cancellation = getattr(error, "cancellation", None)
    return (
        error,
        *(causes(nested) if isinstance(nested, BaseException) else ()),
        *(causes(cancellation) if isinstance(cancellation, BaseException) else ()),
    )


def filing(
    spec: ModelSpec,
    calls: Mapping[str, DomainBinding],
    addresses: Mapping[tuple[int, int], int],
    ports: Mapping[int, tuple[str, str]],
    controls: Controls,
    local: LocalProgram,
) -> tuple[Any, tuple[Admission, ...]]:
    encoder, learner, unit = spec.encoder, spec.learner, spec.unit
    keys = tuple(addresses[address] for address in sorted(addresses))
    by_sample = {sample: address for address, sample in addresses.items()}
    admissions = tuple(Admission(slot, addresses, encoder) for slot in range(2))
    excluded, request_after, normalization = spec.excluded, spec.request_after, spec.normalization

    async def produce(context: Any) -> None:
        facts: SourceFacts = context.states["a.source"]
        facts.excluded.extend(excluded)  # Agreement precedes either branch's fanout.
        pool = physical.Workers(2)

        async def left(sample: int) -> None:
            value = await local.run(sample, lambda source, incoming: ModelLocalFrame(facts, sample, source, incoming), facts.local_metrics)
            await publish(sample, "left", torch.tensor(float(value)), ())

        async def right(sample: int) -> None:
            if sample == controls.omit_sample:
                return
            fact = {"sample": sample, "origin": spec.encode_origin, "input": sample, "output": None, "outcome": "not_entered"}
            facts.stages.append(fact)
            async with context.use_source() as projection:
                try:
                    finished = await pool.execute(
                        lambda: calls[spec.encode_origin].invoke(projection.models[encoder], sample), threading.Event()
                    )
                except physical.NotEntered as error:
                    raise error.cause from error
                except physical.PhysicalFailed:
                    fact["outcome"] = "entered_failed"
                    raise
                result = finished.value
                fact["outcome"] = "invalid_output"
                if not scalar_tensor(result):
                    raise ValueError(f"{spec.encode_origin}: invalid returned source value after entry")
                value = result.detach().clone()
                stamps = (projection.stamps[encoder],)
                fact.update(output=float(value), outcome="returned")
                if finished.cancellation is not None:
                    fact["outcome"] = "returned_cancelled"
                    raise finished.cancellation
            await publish(sample, "right", value, stamps)

        async def publish(sample: int, branch: str, value: torch.Tensor, stamps: tuple[Any, ...]) -> None:
            index, slot = by_sample[sample]
            record_sample = sample + 100 if sample == controls.corrupt_sample and branch == "right" else sample
            record_branch = "left" if sample == controls.swap_member and branch == "right" else branch
            if sample == controls.stale_sample and branch == "right":
                stamps = tuple(replace(stamp, numerical_epoch=stamp.numerical_epoch + 1) for stamp in stamps)
            record = Record(record_sample, index, slot, record_branch, value)
            facts.products.append(
                {
                    "sample": record_sample,
                    "window": index,
                    "slot": slot,
                    "branch": record_branch,
                    "value": float(value),
                    "sources": [(stamp.participant_ref, stamp.binding, stamp.numerical_epoch) for stamp in stamps],
                }
            )
            await context.publish(ports[slot][0 if branch == "left" else 1], owners.Product(index, record, stamps))

        with pool.lifetime():
            async with asyncio.TaskGroup() as tasks:
                for sample in keys[::-1] if controls.reverse_production else keys:
                    tasks.create_task(left(sample))
                    tasks.create_task(right(sample))

    def choose(counts: Mapping[str, int], state: LearningFacts):
        index = sum(counts.values())
        return graph.Choice("window", index) if index < len(keys) // 2 else None

    def objective(context: Any, left: Record, right: Record):
        facts: LearningFacts = context.states["a.learning"]
        parameter = context.models[learner].weight
        value = calls[spec.objective_origin].invoke(context.models[learner], left.sample, left.value, right.value, normalization)
        if not scalar_tensor(value):
            raise ValueError(f"{spec.objective_origin}: invalid returned loss after entry")
        facts.contributions.append(
            {
                "sample": left.sample,
                "window": left.window,
                "slot": left.slot,
                "left": float(left.value),
                "right": float(right.value),
                "weight": float(parameter.detach()),
                "epoch": context.stamps[learner].numerical_epoch,
                "loss": float(value.detach()),
                "invocation": context.invocation,
                "inputs": handoffs(context),
            }
        )
        return value

    def completed(context: Any) -> None:
        facts: LearningFacts = context.states["a.learning"]
        facts.windows.append(
            {
                "window": context.choice.key,
                "weight": float(context.models[learner].weight.detach()),
                "epoch": context.stamps[learner].numerical_epoch,
                "inputs": handoffs(context),
            }
        )

    def request(completion: Any):
        if completion.position != request_after:
            return None
        if controls.evaluation_requested is not None:
            controls.evaluation_requested.set()
        return graph.Choice("measurement", completion.key, completion.position)

    def measurement(context: Any) -> None:
        facts: EvaluationFacts = context.states["a.evaluation"]
        fact = {
            "trigger_window": context.choice.key,
            "trigger_position": context.choice.payload,
            "epoch": context.stamps[learner].numerical_epoch,
            "binding": context.stamps[learner].binding,
            "weight": float(context.models[learner].weight.detach()),
            "value": None,
            "returned": False,
            "completed": False,
            "inputs": handoffs(context),
        }
        facts.measurements.append(fact)
        result = calls[spec.evaluate_origin].invoke(context.models[learner])
        if not scalar_tensor(result):
            raise ValueError(f"{spec.evaluate_origin}: invalid returned measurement after entry")
        fact.update(value=float(result), returned=True)

    def measurement_completed(context: Any) -> None:
        facts: EvaluationFacts = context.states["a.evaluation"]
        # This callback is reached only after numerical return, output checking
        # and mode/physical cleanup of the selected measurement have succeeded.
        facts.measurements[-1]["completed"] = True

    blocks = []
    previous = None
    for slot in range(2):
        blocks.append(
            graph.Block(
                f"contribution.{slot}",
                (
                    graph.Join(
                        f"admit.{slot}",
                        ports[slot],
                        (graph.Output(f"left{slot}", Record), graph.Output(f"right{slot}", Record)),
                        admissions[slot],
                        (graph.Use(encoder),),
                    ),
                    graph.Retain(f"retain.{slot}", (learner,), (unit,), f"old{slot}", after=(f"admit.{slot}",)),
                    graph.Call(
                        f"objective.{slot}",
                        objective,
                        (f"left{slot}", f"right{slot}"),
                        (graph.Output(f"loss{slot}", torch.Tensor),),
                        (graph.Use(learner),),
                        ("a.learning",),
                        retained=f"old{slot}",
                    ),
                    graph.Differentiate(
                        f"differentiate.{slot}", f"loss{slot}", (unit,), f"gradient{slot}", retained=f"old{slot}", accumulate=previous
                    ),
                    graph.Release(f"release.{slot}", f"old{slot}", after=(f"differentiate.{slot}",)),
                ),
            )
        )
        previous = f"gradient{slot}"
    blocks.append(
        graph.Block(
            "finish",
            (
                graph.Advance("advance", previous, (unit,)),
                graph.Call("record-window", completed, uses=(graph.Use(learner),), states=("a.learning",), after=("advance",)),
            ),
        )
    )
    participants = []
    for model in spec.models:
        name, initial, trainable = model.name, model.initial, model.trainable
        declaration = examples.scale(name, initial)
        fn = calls[model.origin].invoke
        participants.append(replace(declaration, load=lambda fn=fn, initial=initial, trainable=trainable: fn(initial, trainable)))
    filing = graph.Run(
        graph.Together(
            "run",
            (
                graph.Producer(
                    "data", tuple(name for slot in range(2) for name in ports[slot]), produce, (graph.Use(encoder),), ("a.source",)
                ),
                graph.Repeat(spec.window, (graph.Sequence("window", tuple(blocks)),), choose, "a.learning"),
                graph.Requested(
                    "requested-evaluation",
                    spec.window,
                    graph.Block(
                        "measurement",
                        (
                            graph.Call("measure", measurement, uses=(graph.Use(learner),), states=("a.evaluation",), evaluation=True),
                            graph.Call("measurement-completed", measurement_completed, states=("a.evaluation",), after=("measure",)),
                        ),
                    ),
                    request,
                ),
            ),
        ),
        tuple(participants),
        (graph.Unit(unit, ((learner, spec.parameter),), learning_rate=spec.learning_rate),),
        (
            graph.OwnedState("a.source", SourceFacts),
            graph.OwnedState("a.learning", LearningFacts),
            graph.OwnedState("a.evaluation", EvaluationFacts),
        ),
        tuple(graph.Channel(name, 2) for slot in range(2) for name in ports[slot]),
    )
    return filing, admissions


@dataclass
class PreparedTarget:
    attempt: Any
    admissions: tuple[Admission, ...]
    addresses: Mapping[tuple[int, int], int]
    controls: Controls
    local: LocalProgram

    def publish(self) -> ModelRun:
        ready = self.attempt.publish()
        return ModelRun(self, ready, backend.Engine(ready))


def lower(
    spec: ModelSpec,
    calls: Mapping[str, DomainBinding],
    addresses: Mapping[tuple[int, int], int],
    ports: Mapping[int, tuple[str, str]],
    controls: Controls,
    local: LocalProgram,
    *,
    lowering: Literal["calls", "inline"] = "calls",
    workers: bool = True,
) -> PreparedTarget:
    calls = MappingProxyType(dict(calls))
    addresses, ports = MappingProxyType(dict(addresses)), MappingProxyType(dict(ports))
    if type(controls.history_limit) is not int or controls.history_limit < 0:
        raise NotReady("nonnegative diagnostic history limit required")
    program, admissions = filing(spec, calls, addresses, ports, controls, local)
    checked = graph.Contract().accept(program)
    attempt = backend.prepare(checked, lowering=lowering, worker_calls=("measure",) if workers else (), worker_limit=2)
    for slot, admission in enumerate(admissions):
        admission.bind(*(attempt.state.feeds[name] for name in ports[slot]))
    attempt.state.trace = deque(maxlen=controls.history_limit)
    return PreparedTarget(attempt, admissions, addresses, controls, local)
