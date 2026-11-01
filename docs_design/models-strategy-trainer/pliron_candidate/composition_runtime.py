"""Optional CPU target for the composition experiment; one existing owner engine."""

from __future__ import annotations

import asyncio
import importlib
import math
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import torch

_prefix = "docs_design.models-strategy-trainer.execution_candidate"
graph = importlib.import_module(_prefix + ".graph")
backend = importlib.import_module(_prefix + ".prepare")
owners = importlib.import_module(_prefix + ".state")
physical = importlib.import_module(_prefix + ".workers")
examples = importlib.import_module(_prefix + ".examples")


@dataclass
class Controls:
    """Private test gates and faults; they are not serialized language constructs."""

    late_record: bool = False
    prefetch: bool = False
    stop_before: int | None = None
    stop_after: int | None = None
    fail_key: int | None = None
    fail_contribution: int | None = None
    fail_evaluation: bool = False
    hold_keys: dict[int, threading.Event] = field(default_factory=dict)
    evaluation_gate: threading.Event | None = None
    second_member_gate: asyncio.Event | None = None
    allow_record: asyncio.Event = field(default_factory=asyncio.Event)
    record_ready: asyncio.Event = field(default_factory=asyncio.Event)
    requested: asyncio.Event = field(default_factory=asyncio.Event)
    evaluation_entered: threading.Event = field(default_factory=threading.Event)
    key_entered: dict[int, threading.Event] = field(default_factory=dict)
    stop_signals: dict[int, threading.Event] = field(default_factory=dict)
    second_selected: asyncio.Event = field(default_factory=asyncio.Event)
    first_member: asyncio.Event = field(default_factory=asyncio.Event)
    close_training: bool = False
    erase_histories: bool = False
    erasure_points: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Record:
    key: int
    value: torch.Tensor
    target: float


@dataclass
class Journal:
    clock: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def mark(self, kind: str, **data: Any) -> int:
        self.clock += 1
        self.events.append({"clock": self.clock, "kind": kind, **data})
        return self.clock


class OwnerSignals:
    """Notify from the existing owner's actual activity exit, including failure.

    This adds notifications to the owner's existing progress contexts. It does
    not decide scope completion, cancel children, or read diagnostic histories.
    """

    def __init__(self, progress: Any, journal: Journal):
        self.events: dict[str, asyncio.Event] = {}
        self.normal: set[str] = set()
        self.clocks: dict[str, int] = {}
        original = progress.activity

        @contextmanager
        def activity(name: str):
            returned = False
            try:
                with original(name):
                    yield
                returned = True
            finally:
                if returned:
                    self.normal.add(name)
                self.clocks[name] = journal.mark("scope_settled", source=name, normal=returned)
                self.event(name).set()

        progress.activity = activity

    def event(self, name: str) -> asyncio.Event:
        return self.events.setdefault(name, asyncio.Event())


@dataclass
class SourceState:
    cursor: int = 0
    active: dict[int, str] = field(default_factory=dict)
    terminal: dict[int, str] = field(default_factory=dict)
    ready: set[int] = field(default_factory=set)
    physical: set[int] = field(default_factory=set)
    high_water: int = 0
    ready_high_water: int = 0
    products: list[dict[str, Any]] = field(default_factory=list)
    uses: list[dict[str, Any]] = field(default_factory=list)
    cleanup_complete: bool = False


@dataclass
class LearningState:
    selected: list[int] = field(default_factory=list)
    acquired: dict[int, list[int]] = field(default_factory=dict)
    windows: list[dict[str, Any]] = field(default_factory=list)
    contributions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvaluationState:
    request: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    completed: bool = False


class Transport:
    """Bounded provider notifications, separate from the owner's input claims."""

    def __init__(self, horizon: int, ready: int):
        self.messages: asyncio.Queue[tuple[str, int]] = asyncio.Queue(2 * horizon + 1)
        self.capacity = asyncio.BoundedSemaphore(horizon)
        self.ready_capacity = asyncio.BoundedSemaphore(ready)

    async def claimed(self, key: int) -> None:
        await self.messages.put(("claimed", key))

    async def settled(self, key: int) -> None:
        await self.messages.put(("consumed", key))


def scalar(value: Any) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.device.type == "cpu"
        and value.shape == torch.Size([])
        and math.isfinite(float(value.detach()))
    )


def load_scale(initial: float, trainable: bool):
    model = examples.Scale(initial)
    model.weight.requires_grad_(trainable)
    return model


def encode(model: Any, key: int):
    with torch.no_grad():
        return model(torch.tensor(float(key + 1))).detach().clone()


def objective(model: Any, record: Record, normalization: int):
    return (model(record.value) - record.target).square() / (2 * normalization)


def measure(model: Any, record: Record):
    with torch.no_grad():
        return model(record.value).detach().clone()


def functions() -> dict[str, Callable[..., Any]]:
    return {"scale_v1": load_scale, "encode_v1": encode, "objective_v1": objective, "measure_v1": measure}


def handoffs(context: Any) -> list[dict[str, Any]]:
    return [
        {"channel": item.channel, "key": item.key, "sources": [(s.participant_ref, s.binding, s.numerical_epoch) for s in item.provenance]}
        for item in context.input_handoffs
    ]


def errors(error: BaseException | None) -> list[dict[str, str]]:
    seen: set[int] = set()
    result: list[dict[str, str]] = []

    def visit(value: BaseException | None) -> None:
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        result.append({"type": type(value).__name__, "message": str(value)})
        if isinstance(value, BaseExceptionGroup):
            for child in value.exceptions:
                visit(child)
        visit(getattr(value, "cause", None))
        visit(getattr(value, "cancellation", None))
        visit(value.__cause__)

    visit(error)
    return result


@dataclass
class CompositionRun:
    spec: Mapping[str, Any]
    ready: Any
    engine: Any
    signals: OwnerSignals
    source: SourceState
    learning: LearningState
    evaluation: EvaluationState
    journal: Journal
    controls: Controls
    last_report: dict[str, Any] | None = None
    started: bool = False

    def erase_histories(self) -> None:
        self.journal.events.clear()
        self.source.products.clear()
        self.source.uses.clear()
        self.learning.windows.clear()
        self.learning.contributions.clear()
        self.ready.state.trace.clear()

    async def run(self) -> dict[str, Any]:
        if self.started:
            raise RuntimeError("composition run has one owner lifetime")
        self.started = True
        task = asyncio.current_task()
        cancellations = task.cancelling() if task is not None else 0
        failure = None
        try:
            await self.engine.run()
        except BaseException as error:
            failure = error
        interrupted = task is not None and task.cancelling() > cancellations
        self.journal.mark("run_settled", normal=self.ready.state.completed)
        if self.controls.erase_histories:
            self.erase_histories()
        self.last_report = self.report(failure, interrupted)
        if interrupted and failure is not None:
            raise failure
        return self.last_report

    def report(self, failure: BaseException | None, interrupted: bool) -> dict[str, Any]:
        state = self.ready.state
        image = state.image
        unit = image.units["unit"]
        model = image.bindings[self.spec["learner"]].model
        outcomes = [
            {
                "invocation": o.invocation,
                "contribution": o.contribution,
                "optimizer": o.optimizer,
                "reset": o.reset,
                "inputs": [{"channel": h.channel, "key": h.key} for h in o.input_handoffs],
            }
            for o in {o.invocation: o for o in (*unit.outcomes, *unit.active_outcomes.values())}.values()
        ]
        result = {
            "schema": 1,
            "status": "stopped" if state.completed else "cancelled" if interrupted else "failed",
            "errors": errors(failure),
            "weight": float(model.weight.detach()),
            "steps": unit.steps,
            "pending": unit.unfinished,
            "training_mode": model.training,
            "gradients_enabled": torch.is_grad_enabled(),
            "windows": self.learning.windows,
            "contributions": self.learning.contributions,
            "selected_windows": self.learning.selected,
            "acquired": self.learning.acquired,
            "evaluation": self.evaluation.receipt,
            "evaluation_completed": self.evaluation.completed,
            "request": None
            if self.evaluation.request is None
            else {
                "cause": self.evaluation.request,
                "disposition": "completed" if self.evaluation.completed else "cancelled_unfulfilled" if interrupted else "failed",
            },
            "source": {
                "cursor": self.source.cursor,
                "active": self.source.active,
                "physical_active": sorted(self.source.physical),
                "terminal": self.source.terminal,
                "high_water": self.source.high_water,
                "ready_high_water": self.source.ready_high_water,
                "products": self.source.products,
                "uses": self.source.uses,
                "cleanup_complete": self.source.cleanup_complete,
            },
            "scope_clocks": self.signals.clocks,
            "normal_scopes": sorted(self.signals.normal),
            "events": list(self.journal.events),
            "unit_outcomes": outcomes,
            "remaining": {name: sorted(feed.ready) for name, feed in state.feeds.items()},
            "closed": all(feed.closed for feed in state.feeds.values()),
            "released": not state.running
            and not state.readers
            and not state.writers
            and not state.state_locks
            and not state.active_producers
            and not state.active_retentions,
            "diagnostics": len(state.trace),
            "archive_erasure_points": list(self.controls.erasure_points),
            "failure_inputs": [{"channel": h.channel, "key": h.key} for h in self.engine.failure_input_handoffs],
        }
        labels = {value: name for name, value in state.accepted.participant_refs.items()}

        def symbolic(value: Any):
            if isinstance(value, str):
                return labels.get(value, value)
            if isinstance(value, dict):
                return {key: symbolic(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [symbolic(item) for item in value]
            return value

        return symbolic(result)


def lower(spec: Mapping[str, Any], implementations: Mapping[str, Callable[..., Any]], controls: Controls | None = None) -> CompositionRun:
    """Consume a verified target projection; all scheduling is existing."""

    def snapshot(value: Any) -> Any:
        if isinstance(value, Mapping):
            return MappingProxyType({key: snapshot(item) for key, item in value.items()})
        if isinstance(value, (list, tuple)):
            return tuple(snapshot(item) for item in value)
        return value

    spec = snapshot(spec)
    implementations = MappingProxyType(dict(implementations))
    controls = controls or Controls()
    source, learning, evaluation, journal = SourceState(), LearningState(), EvaluationState(), Journal()
    transport = Transport(spec["horizon"], spec["ready"])
    encoder, learner = spec["encoder"], spec["learner"]
    producer, training, requested, group = (spec[name] for name in ("producer", "training", "evaluation", "group"))
    signals: OwnerSignals
    sample_count, window_count = spec["members"], spec["windows"]
    eval_key = sample_count * spec["after_window"]
    ports = tuple(spec["ports"])
    eval_port, extra_port = spec["evaluation_port"], spec["unused_port"]
    calls = spec["calls"]

    def notify(kind: str, key: int) -> None:
        if key not in source.active:
            raise graph.OutOfBounds("notification for an unowned production ticket")
        if kind == "claimed":
            if key not in source.ready:
                raise graph.OutOfBounds("claim did not follow actual publication")
            source.ready.remove(key)
            transport.ready_capacity.release()
            source.active[key] = "claimed"
        elif kind == "consumed":
            if source.active[key] != "claimed":
                raise graph.OutOfBounds("settlement without a validated claim")
            source.terminal[key] = "consumed"
            del source.active[key]
            transport.capacity.release()
        else:
            raise graph.OutOfBounds("unknown provider notification")
        journal.mark(kind, key=key, outstanding=len(source.active))

    async def receive() -> None:
        while True:
            kind, key = await transport.messages.get()
            notify(kind, key)

    async def production_body(context: Any) -> None:
        pool = physical.Workers(spec["workers"])
        with pool.lifetime():
            while True:
                await transport.capacity.acquire()
                key = source.cursor
                source.cursor += 1
                source.active[key] = "selected"
                source.high_water = max(source.high_water, len(source.active))
                journal.mark("selected", key=key, outstanding=len(source.active))
                if controls.late_record and key == eval_key:
                    await controls.allow_record.wait()
                async with context.use_source() as projection:
                    stamp = projection.stamps[encoder]
                    source.active[key] = "physical"
                    entered = controls.key_entered.setdefault(key, threading.Event())

                    def invoke(key=key, entered=entered, projection=projection):
                        entered.set()
                        gate = controls.hold_keys.get(key)
                        if gate is not None and not gate.wait(30):
                            raise RuntimeError("production test gate expired")
                        if key == controls.fail_key:
                            raise RuntimeError(f"physical production failure {key}")
                        return implementations[calls["encode"]](projection.models[encoder], key)

                    use = {"key": key, "epoch": stamp.numerical_epoch, "physical_done": False}
                    source.uses.append(use)
                    stop_signal = controls.stop_signals.setdefault(key, threading.Event())
                    source.physical.add(key)
                    try:
                        finished = await pool.execute(invoke, stop_signal)
                    except physical.NotEntered as error:
                        raise error.cause from error
                    except physical.PhysicalFailed:
                        source.active[key] = "failed"
                        raise
                    finally:
                        source.physical.discard(key)
                        use["physical_done"] = True
                        use["entered"] = entered.is_set()
                        journal.mark("producer_physical_end", key=key, entered=entered.is_set())
                    value = finished.value
                    if not scalar(value):
                        raise ValueError("invalid encoded scalar after physical return")
                    source.active[key] = "produced"
                    if finished.cancellation is not None:
                        raise finished.cancellation
                    record = Record(key, value.detach().clone(), float(key + 1))
                    provenance = (stamp,)
                if key < sample_count * window_count:
                    window, slot = divmod(key, sample_count)
                    port, address = ports[slot], window
                elif key == eval_key:
                    port, address = eval_port, spec["after_window"] - 1
                else:
                    port, address = extra_port, key
                await transport.ready_capacity.acquire()
                await context.publish(port, owners.Product(address, record, provenance))
                source.active[key] = "ready"
                source.ready.add(key)
                source.ready_high_water = max(source.ready_high_water, len(source.ready))
                source.products.append(
                    {
                        "key": key,
                        "value": float(record.value),
                        "channel": port,
                        "address": address,
                        "sources": [(stamp.participant_ref, stamp.binding, stamp.numerical_epoch)],
                    }
                )
                journal.mark("published", key=key, ready=len(source.ready))
                if key == eval_key:
                    controls.record_ready.set()

    async def produce(context: Any) -> None:
        async with asyncio.TaskGroup() as tasks:
            receiver = tasks.create_task(receive())
            try:
                await production_body(context)
            finally:
                receiver.cancel()

    async def cleanup(context: Any) -> None:
        # Failure of the enclosing TaskGroup also settles D. D never waits for
        # this producer, so this preserves held records without a stop cycle.
        await signals.event(group).wait()
        while not transport.messages.empty():
            notify(*transport.messages.get_nowait())
        normal = group in signals.normal
        journal.mark("producer_cleanup_begin", normal=normal)
        # Unclaimed permits need no recycling: implementation/receiver joined,
        # and this terminal producer lifetime cannot select any more work.
        for key, phase in tuple(source.active.items()):
            if normal and phase == "claimed":
                raise graph.OutOfBounds("successful consumer group left an unsettled claim")
            source.terminal[key] = "failed" if phase == "failed" else "discarded" if normal else "closed_unconsumed"
            journal.mark("disposed", key=key, disposition=source.terminal[key])
            del source.active[key]
        source.ready.clear()
        source.cleanup_complete = True
        journal.mark("producer_cleanup_end")

    def choose(counts: Mapping[str, int], state: LearningState):
        index = sum(counts.values())
        if index >= window_count or controls.close_training or controls.stop_before == index:
            return None
        # Returning this accepted Choice commits the complete Sequence. Physical
        # members are acquired separately; cancellation can retain a partial one.
        state.selected.append(index)
        journal.mark("window_selected", window=index)
        if index == 1:
            controls.second_selected.set()
        if controls.stop_after == index:
            controls.close_training = True
        return graph.Choice(training + ".window", index)

    def admission(slot: int, *, evaluator: bool = False):
        def admit(product: Any, projection: Any) -> bool:
            record = product.value
            expected = eval_key if evaluator else product.key * sample_count + slot
            return (
                isinstance(record, Record)
                and record.key == expected
                and scalar(record.value)
                and product.provenance == (projection.stamps[encoder],)
            )

        return admit

    async def claimed(context: Any, record: Record) -> None:
        await transport.claimed(record.key)
        if context.choice.block == training + ".window":
            learning.acquired.setdefault(context.choice.key, []).append(record.key)
            if context.choice.key == 1 and record.key % sample_count == 0:
                controls.first_member.set()

    async def before_member(context: Any) -> None:
        if context.choice.key == 1 and controls.second_member_gate is not None:
            await controls.second_member_gate.wait()

    def loss(context: Any, record: Record):
        if record.key == controls.fail_contribution:
            raise RuntimeError(f"objective failure {record.key}")
        value = implementations[calls["objective"]](context.models[learner], record, sample_count)
        if not scalar(value) or not value.requires_grad:
            raise ValueError("invalid differentiable scalar objective")
        learning.contributions.append(
            {
                "key": record.key,
                "window": context.choice.key,
                "x": float(record.value),
                "target": record.target,
                "weight": float(context.models[learner].weight.detach()),
                "epoch": context.stamps[learner].numerical_epoch,
                "loss": float(value.detach()),
            }
        )
        return value

    async def consumed(context: Any, record: Record) -> None:
        await transport.settled(record.key)

    async def before_advance(context: Any) -> None:
        if controls.prefetch and context.choice.key == window_count - 1:
            await controls.record_ready.wait()

    def completed(context: Any) -> None:
        learning.windows.append(
            {
                "window": context.choice.key,
                "weight": float(context.models[learner].weight.detach()),
                "epoch": context.stamps[learner].numerical_epoch,
                "inputs": handoffs(context),
            }
        )
        journal.mark("window_completed", window=context.choice.key)

    def request(completion: Any):
        if completion.position != spec["after_window"]:
            return None
        cause = {"source": completion.scope, "window": completion.key, "position": completion.position}
        if evaluation.request is not None:
            raise graph.OutOfBounds("duplicate request for an exactly-once completion selection")
        evaluation.request = cause
        journal.mark("evaluation_requested", **cause)
        controls.requested.set()
        return graph.Choice(requested + ".body", completion.key, cause)

    async def measurement(context: Any, record: Record) -> None:
        receipt = {
            "cause": dict(context.choice.payload),
            "key": record.key,
            "epoch": context.stamps[learner].numerical_epoch,
            "weight": float(context.models[learner].weight.detach()),
            "inputs": handoffs(context),
            "returned": False,
            "physical_done": False,
            "prediction": None,
            "loss": None,
            "evaluation_mode": None,
            "gradients": None,
        }
        evaluation.receipt = receipt
        journal.mark("evaluation_entered", key=record.key)
        pool = physical.Workers(1)

        def invoke():
            with torch.no_grad():
                controls.evaluation_entered.set()
                if controls.evaluation_gate is not None and not controls.evaluation_gate.wait(30):
                    raise RuntimeError("evaluation test gate expired")
                if controls.fail_evaluation:
                    raise RuntimeError("physical evaluation failure")
                value = implementations[calls["measure"]](context.models[learner], record)
                return value, torch.is_grad_enabled(), not context.models[learner].training

        try:
            with pool.lifetime():
                try:
                    finished = await pool.execute(invoke, threading.Event())
                except physical.NotEntered as error:
                    raise error.cause from error
                finally:
                    receipt["physical_done"] = True
                    receipt["physical_entered"] = controls.evaluation_entered.is_set()
                    journal.mark("evaluation_physical_end")
                value, gradients, evaluation_mode = finished.value
                receipt.update(gradients=gradients, evaluation_mode=evaluation_mode)
                if not scalar(value):
                    raise ValueError("invalid evaluation scalar after physical return")
                receipt.update(prediction=float(value), loss=(float(value) - record.target) ** 2 / 2, returned=True)
                if finished.cancellation is not None:
                    raise finished.cancellation
        finally:
            # Completion of physical input use is distinct from successful
            # publication of the requested activity's result.
            if receipt.get("physical_entered", False):
                await transport.settled(record.key)

    def evaluation_completed(context: Any) -> None:
        evaluation.completed = True
        journal.mark("evaluation_completed")

    blocks = []
    previous = None
    for slot, port in enumerate(ports):
        work = []
        if slot == 1:
            work.append(graph.Call("wait-member", before_member))
        work.extend(
            (
                graph.Join(
                    f"admit.{slot}",
                    (port,),
                    (graph.Output(f"record{slot}", Record),),
                    admission(slot),
                    (graph.Use(encoder),),
                    after=("wait-member",) if slot == 1 else (),
                ),
                graph.Call(f"claim.{slot}", claimed, (f"record{slot}",)),
                graph.Retain(f"retain.{slot}", (learner,), ("unit",), f"old{slot}", after=(f"claim.{slot}",)),
                graph.Call(
                    f"loss.{slot}",
                    loss,
                    (f"record{slot}",),
                    (graph.Output(f"loss{slot}", torch.Tensor),),
                    (graph.Use(learner),),
                    ("learning",),
                    retained=f"old{slot}",
                ),
                graph.Differentiate(
                    f"backward.{slot}", f"loss{slot}", ("unit",), f"gradient{slot}", retained=f"old{slot}", accumulate=previous
                ),
                graph.Release(f"release.{slot}", f"old{slot}", after=(f"backward.{slot}",)),
                graph.Call(f"consume.{slot}", consumed, (f"record{slot}",), after=(f"release.{slot}",)),
            )
        )
        blocks.append(graph.Block(f"contribution.{slot}", tuple(work)))
        previous = f"gradient{slot}"
    blocks.append(
        graph.Block(
            "advance-window",
            (
                graph.Call("wait-prefetch", before_advance),
                graph.Advance("advance", previous, ("unit",), after=("wait-prefetch",)),
                graph.Call("record-window", completed, uses=(graph.Use(learner),), states=("learning",), after=("advance",)),
            ),
        )
    )
    participants = []
    for declaration in spec["models"]:
        participant = examples.scale(declaration["id"], declaration["initial"])
        fn = implementations[declaration["implementation"]]
        participants.append(replace(participant, load=lambda d=declaration, fn=fn: fn(d["initial"], d["trainable"])))
    filing = graph.Run(
        graph.Together(
            spec["run"],
            (
                graph.Producer(
                    producer,
                    (*ports, eval_port, extra_port),
                    produce,
                    (graph.Use(encoder),),
                    ("source",),
                    stop_when=group,
                    remaining="discard",
                    cleanup=cleanup,
                ),
                graph.Together(
                    group,
                    (
                        graph.Repeat(training, (graph.Sequence(training + ".window", tuple(blocks)),), choose, "learning"),
                        graph.Requested(
                            requested,
                            training,
                            graph.Block(
                                requested + ".body",
                                (
                                    graph.Join(
                                        "evaluation-admit",
                                        (eval_port,),
                                        (graph.Output("evaluation-record", Record),),
                                        admission(0, evaluator=True),
                                        (graph.Use(encoder),),
                                    ),
                                    graph.Call("evaluation-claim", claimed, ("evaluation-record",)),
                                    graph.Call(
                                        "measure",
                                        measurement,
                                        ("evaluation-record",),
                                        uses=(graph.Use(learner),),
                                        states=("evaluation",),
                                        after=("evaluation-claim",),
                                        evaluation=True,
                                    ),
                                    graph.Call("measurement-completed", evaluation_completed, states=("evaluation",), after=("measure",)),
                                ),
                            ),
                            request,
                        ),
                    ),
                ),
            ),
        ),
        tuple(participants),
        (graph.Unit("unit", ((learner, "weight"),), learning_rate=spec["learning_rate"]),),
        (
            graph.OwnedState("source", lambda: source, (encoder,), "reset"),
            graph.OwnedState("learning", lambda: learning),
            graph.OwnedState("evaluation", lambda: evaluation),
        ),
        tuple(graph.Channel(name, spec["ready"]) for name in (*ports, eval_port, extra_port)),
    )
    accepted = graph.Contract().accept(filing)
    attempt = backend.prepare(accepted)
    signals = OwnerSignals(attempt.state.progress, journal)
    ready = attempt.publish()
    return CompositionRun(spec, ready, backend.Engine(ready), signals, source, learning, evaluation, journal, controls)
