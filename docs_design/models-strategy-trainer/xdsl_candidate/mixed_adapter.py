"""One owner Engine for an open producer and ordinary feedback-producing learning.

The bridge carries messages, never shared mutable cursor/policy authority. This
CPU target admits detached features from the same binding protected at production;
that numerical epoch need not remain current when learning later claims them.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import torch
from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, Region

from .dialects import CallOp, ComposeOp, FusedOp, PlanOp, SessionOp, dump, integer
from .fusion import FusionError, groups as local_groups, rewrite as fuse_plan
from .local_runtime import Stage, compile_local
from .mixed_contract import accept_mixed
from .mixed_dialect import DynamicFilterOp, PolicyUpdateOp, StopCheckpointOp, parse_mixed
from .model_adapter import (
    DomainBinding,
    DomainRegistry,
    LearningFacts,
    ModelLocalFrame,
    SourceFacts,
    backend,
    causes,
    examples,
    graph,
    handoffs,
    model_registry,
    owners,
    physical,
    scalar_tensor,
)
from .model_contract import one
from .model_dialect import AdvanceOp, ContributeOp, ModelFlowOp, ModelOp, RecipeOp, WindowOp
from .prepare import Accepted, NotReady, check_calls, check_plan, check_scalar_locals, implementation, local_effect, local_maps, origin, ref

TARGET_MIXED = "continuous-owner-cpu-v1"
OWNER_PROTOCOL = "open producer; owner feedback receiver; ordered bounded ports; same-binding production snapshots; external checkpoint"
COMPLETION = "protected_physical_return_then_validate"
PORTS = ("mixed.left", "mixed.right")


def mixed_registry() -> DomainRegistry:
    bindings = dict(model_registry().bindings)
    bindings.update(
        {
            "dynamic_filter_v1": DomainBinding("filter", "read_policy", "1", lambda key, revision: not (revision > 0 and key % 4 == 0)),
            "learned_bias_v1": DomainBinding("policy", "derive_update", "1", lambda weight: int(abs(weight) * 10) % 5),
            "mixed_loss_v1": DomainBinding(
                "objective",
                "build_loss",
                "1",
                lambda model, sample, left, right, count: (model(left + right) - sample / 2).square() / (2 * count),
            ),
        }
    )
    return DomainRegistry("continuous-learning-fixture-v1", bindings)


@dataclass
class MixedControls:
    """Scheduling/fault injection only. None of these fields is serialized meaning."""

    history_limit: int = 256
    block_key: int | None = None
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    fail_physical: bool = False
    corrupt_key: int | None = None
    corrupt_stamp: bool = False
    swap_member: bool = False


@dataclass(frozen=True)
class Feedback:
    ordinal: int
    key: int
    weight: float
    epoch: int
    invocation: int
    inputs: tuple[tuple[str, int], ...]


@dataclass
class Bridge:
    """Bounded transport plus external control; no policy or model state."""

    feedback: asyncio.Queue[Feedback]
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    checkpoint: asyncio.Event = field(default_factory=asyncio.Event)
    stop: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class Ticket:
    key: int
    ordinal: int
    value: int
    revision: int
    bias: int
    stamps: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class MixedRecord:
    key: int
    ordinal: int
    selected: int
    revision: int
    bias: int
    branch: str
    value: torch.Tensor


@dataclass
class MixedSource(SourceFacts):
    cursor: int = 0
    revision: int = 0
    bias: int = 0
    active: dict[int, Ticket] = field(default_factory=dict)
    selections: list[dict[str, Any]] = field(default_factory=list)
    updates: list[dict[str, Any]] = field(default_factory=list)
    max_active: int = 0
    sequence: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    settlements: list[dict[str, Any]] = field(default_factory=list)
    lock: Any = field(default_factory=threading.Lock, repr=False)

    def mark(self, event: str, key: int | None = None) -> int:
        with self.lock:
            self.sequence += 1
            self.events.append({"event": event, "key": key, "at": self.sequence})
            return self.sequence

    def snapshot(self) -> dict[str, Any]:
        return {
            name: asdict(self.local_metrics) if name == "local_metrics" else getattr(self, name)
            for name in (
                "excluded",
                "products",
                "stages",
                "local_metrics",
                "cursor",
                "revision",
                "bias",
                "selections",
                "updates",
                "max_active",
                "events",
                "settlements",
            )
        } | {
            "active": [
                asdict(ticket) | {"stamps": None if ticket.stamps is None else [asdict(stamp) for stamp in ticket.stamps]}
                for ticket in self.active.values()
            ]
        }


class MixedAdmission:
    def __init__(self, learner: str):
        self.learner = learner

    def bind(self, state: Any) -> None:
        self.source: MixedSource = state.owners["mixed.source"]
        self.ports = tuple(state.feeds[name] for name in PORTS)

    def __call__(self, product: Any, projection: Any) -> bool:
        record = product.value
        if not isinstance(record, MixedRecord):
            return False
        ticket = self.source.active.get(product.key)
        member = next((index for index, port in enumerate(self.ports) if port.ready.get(product.key) is product), None)
        if ticket is None or member is None or not scalar_tensor(record.value):
            return False
        if (record.key, record.ordinal, record.selected, record.revision, record.bias, record.branch) != (
            ticket.key,
            ticket.ordinal,
            ticket.value,
            ticket.revision,
            ticket.bias,
            ("left", "right")[member],
        ):
            return False
        if member == 0:
            return product.provenance == ()
        expected = ticket.stamps
        current = projection.stamps[self.learner]
        return (
            expected is not None
            and len(expected) == 1
            and product.provenance == expected
            and expected[0].participant_ref == current.participant_ref
            and expected[0].binding == current.binding
            and 0 <= expected[0].numerical_epoch <= current.numerical_epoch
        )


@dataclass
class PreparedMixed:
    accepted: Accepted
    text: str
    manifest: str
    attempt: Any
    bridge: Bridge
    controls: MixedControls
    local: Any

    def publish(self) -> MixedRun:
        ready = self.attempt.publish()
        return MixedRun(self, ready, backend.Engine(ready))


@dataclass
class MixedRun:
    prepared: PreparedMixed
    ready: Any
    engine: Any
    started: bool = False
    interrupted: bool = False
    last_report: dict[str, Any] | None = None

    def stop(self) -> None:
        # This target exposes the explicit safe checkpoint in the document.
        if not self.prepared.bridge.checkpoint.is_set():
            raise NotReady("external stop requires the declared learning checkpoint")
        if not self.prepared.bridge.stop.is_set():
            self.ready.state.owners["mixed.source"].mark("stop")
            self.prepared.bridge.stop.set()
            self.prepared.bridge.wake.set()

    async def run(self) -> dict[str, Any]:
        if self.started:
            raise RuntimeError("mixed run has one owner lifetime")
        self.started = True
        task = asyncio.current_task()
        count = task.cancelling() if task is not None else 0
        failure: BaseException | None = None
        try:
            await self.engine.run()
        except BaseException as error:
            failure = error
            self.interrupted = isinstance(error, asyncio.CancelledError) or (task is not None and task.cancelling() > count)
        self.last_report = self.report(failure)
        if failure is not None and self.interrupted:
            raise failure
        return self.last_report

    def report(self, failure: BaseException | None = None) -> dict[str, Any]:
        failure = failure or self.engine.failure
        state, source = self.ready.state, self.ready.state.owners["mixed.source"]
        image = state.image
        assert image is not None
        outcomes = [
            {
                "invocation": item.invocation,
                "advance_invocation": item.advancement_invocation,
                "contribution": item.contribution,
                "optimizer": item.optimizer,
                "reset": item.reset,
                "inputs": [{"channel": fact.channel, "window": fact.key} for fact in item.input_handoffs],
            }
            for runtime in image.units.values()
            for item in {fact.invocation: fact for fact in (*runtime.outcomes, *runtime.active_outcomes.values())}.values()
        ]
        result = {
            "status": "cancelled" if self.interrupted else "stopped" if state.completed else "failed",
            "error": None if failure is None else str(failure),
            "errors": [{"type": type(error).__name__, "message": str(error)} for error in causes(failure)],
            "source": source.snapshot(),
            "learning": asdict(state.owners["mixed.learning"]),
            "unit_outcomes": outcomes,
            "steps": sum(runtime.steps for runtime in image.units.values()),
            "pending": any(runtime.unfinished for runtime in image.units.values()),
            "remaining": {name: sorted(feed.ready) for name, feed in state.feeds.items()},
            "ports_closed": {name: feed.closed for name, feed in state.feeds.items()},
            "claimed": {name: [(key, invocation) for key, invocation, _ in feed.handed_off] for name, feed in state.feeds.items()},
            "discarded": {name: [key for key, _ in feed.discarded] for name, feed in state.feeds.items()},
            "high_water": {name: feed.high_water for name, feed in state.feeds.items()},
            "released": not state.running
            and not state.readers
            and not state.writers
            and not state.state_locks
            and not state.active_producers
            and not state.active_retentions,
            "diagnostics": len(state.trace),
            "checkpoint": self.prepared.bridge.checkpoint.is_set(),
            "stop_requested": self.prepared.bridge.stop.is_set(),
        }
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


def prepare_mixed(
    accepted: Accepted,
    registry: DomainRegistry | None = None,
    *,
    controls: MixedControls | None = None,
    lowering: Literal["calls", "inline"] = "calls",
    fuse: bool = False,
    target: str = TARGET_MIXED,
) -> PreparedMixed:
    registry, controls = registry or mixed_registry(), controls or MixedControls()
    semantic = parse_mixed(accepted.text)
    if accept_mixed(semantic) != accepted:
        raise NotReady("noncanonical mixed snapshot")
    session = one(semantic, SessionOp)
    window, advance, checkpoint = one(session, WindowOp), one(session, AdvanceOp), one(session, StopCheckpointOp)
    if (
        target != TARGET_MIXED
        or session.horizon != integer(3)
        or window.count != integer(1)
        or checkpoint.after_windows != integer(3)
        or advance.parameter.data != "weight"
    ):
        raise NotReady("mixed CPU target requires horizon three, one-contribution windows and external checkpoint after three windows")
    check_scalar_locals(local_maps(one(session, ComposeOp)))
    specs: list[tuple[Any, str, str]] = [(one(session, ModelOp), "model", "load")]
    specs.extend((op, "map", local_effect(op)) for op in local_maps(one(session, ComposeOp)))
    specs.extend(
        [
            (one(session, ModelFlowOp), "encode", "read_source"),
            (one(session, ContributeOp), "objective", "build_loss"),
            (one(session, DynamicFilterOp), "filter", "read_policy"),
            (one(session, PolicyUpdateOp), "policy", "derive_update"),
        ]
    )
    sources = {origin(op): op for op, _, _ in specs}
    emitted, rows = [], []
    for op, role, effect in specs:
        symbol = ref(implementation(op))
        binding = registry.bindings.get(symbol)
        if binding is None or (binding.role, binding.effect) != (role, effect) or not binding.version:
            raise NotReady(f"{origin(op)}: missing/incompatible mixed binding @{symbol}")
        rows.append({"symbol": symbol, "version": binding.version, "role": role, "effect": effect})
        ancestors, parent = [], op.parent_op()
        while parent is not None and not isinstance(parent, ModuleOp):
            ancestors.insert(0, StringAttr(origin(parent)))
            parent = parent.parent_op()
        emitted.append(
            CallOp(
                properties={
                    "origin": StringAttr(origin(op)),
                    "ancestors": ArrayAttr(ancestors),
                    "implementation": SymbolRefAttr(symbol),
                    "binding_version": StringAttr(binding.version),
                    "role": StringAttr(role),
                    "effect": StringAttr(effect),
                    "completion": StringAttr(COMPLETION),
                }
            )
        )
    recipe = RecipeOp(
        properties={"origin": session.source_id, "role": StringAttr("mixed-owner"), "protocol": StringAttr(OWNER_PROTOCOL)},
        regions=[Region(Block([session.clone()]))],
    )
    plan = ModuleOp(
        [
            PlanOp(
                properties={
                    "accepted_digest": StringAttr(accepted.digest),
                    "target": StringAttr(target),
                    "contract": session.contract,
                    "horizon": session.horizon,
                    "capacity": integer(1),
                    "stop_rule": session.stop_rule,
                },
                regions=[Region(Block([recipe, *emitted]))],
            )
        ]
    )
    if fuse:
        plan = fuse_plan(plan, sources)
    text = dump(plan)
    try:
        loaded = parse_mixed(text)
        loaded.verify()
        root = check_plan(loaded, accepted, session, target=target, capacity=1)
        contents = tuple(root.body.block.ops)
        if any(not isinstance(op, RecipeOp | CallOp | FusedOp) for op in contents):
            raise NotReady("unknown mixed prepared operation")
        recipes = [op for op in contents if isinstance(op, RecipeOp)]
        if len(recipes) != 1:
            raise NotReady("one mixed prepared recipe required")
        recipe = recipes[0]
        if (recipe.origin.data, recipe.role.data, recipe.protocol.data) != (
            origin(session),
            "mixed-owner",
            OWNER_PROTOCOL,
        ) or recipe.body.block.args:
            raise NotReady("mixed prepared recipe protocol/source differs")
        decoded = one(recipe, SessionOp)
        if tuple(recipe.body.block.ops) != (decoded,) or accept_mixed(ModuleOp([decoded.clone()])) != accepted:
            raise NotReady("mixed prepared recipe changed accepted meaning")
        check_calls([op for op in root.walk() if isinstance(op, CallOp)], specs, registry.bindings, completion=COMPLETION)
        if not fuse and any(isinstance(op, FusedOp) for op in contents):
            raise NotReady("mixed local fusion not requested")
        groups = local_groups(
            [op for op in contents if isinstance(op, FusedOp) or (isinstance(op, CallOp) and op.role.data == "map")], sources
        )
        if tuple(call.origin.data for group in groups for call in group) != tuple(origin(op) for op in local_maps(one(decoded, ComposeOp))):
            raise NotReady("mixed prepared local stage order differs")
    except (NotReady, FusionError) as error:
        raise NotReady(f"mixed prepared plan: {error}") from error
    except Exception as error:
        raise NotReady(f"mixed prepared structure/correspondence: {error}") from error
    local = compile_local(
        tuple(tuple(Stage(call.origin.data, registry.bindings[ref(call.implementation)].invoke) for call in group) for group in groups)
    )
    calls = {call.origin.data: registry.bindings[ref(call.implementation)] for call in root.walk() if isinstance(call, CallOp)}
    bridge = Bridge(asyncio.Queue(maxsize=3))
    filing, admission = lower_mixed(decoded, calls, local, bridge, controls)
    try:
        checked = graph.Contract().accept(filing)
        attempt = backend.prepare(checked, lowering=lowering)
    except Exception as error:
        raise NotReady(f"mixed owner target realization: {error}") from error
    admission.bind(attempt.state)
    attempt.state.trace = deque(maxlen=controls.history_limit)
    manifest = (
        json.dumps(
            {
                "schema": 1,
                "abi": TARGET_MIXED,
                "registry": registry.name,
                "bindings": rows,
                "local_lowering": "fused-v1" if any(len(group) > 1 for group in groups) else "unfused-v1",
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    return PreparedMixed(accepted, text, manifest, attempt, bridge, controls, local)


def lower_mixed(session: SessionOp, calls: Mapping[str, DomainBinding], local: Any, bridge: Bridge, controls: MixedControls):
    model, apply, contribute, advance = (
        one(session, ModelOp),
        one(session, ModelFlowOp),
        one(session, ContributeOp),
        one(session, AdvanceOp),
    )
    filtered, update, checkpoint = one(session, DynamicFilterOp), one(session, PolicyUpdateOp), one(session, StopCheckpointOp)
    learner, unit = ref(model.entity), ref(advance.unit)
    admission = MixedAdmission(learner)
    assert session.horizon is not None
    horizon = session.horizon.value.data

    async def produce(context: Any) -> None:
        source: MixedSource = context.states["mixed.source"]
        pool = physical.Workers(3)
        turns = [0, 0]
        turn_changed = [asyncio.Condition(), asyncio.Condition()]

        async def publish(ticket: Ticket, branch: int, value: torch.Tensor, stamps: tuple[Any, ...]) -> None:
            condition = turn_changed[branch]
            # One publication turn per branch; no model borrow crosses this wait.
            async with condition:
                await condition.wait_for(lambda: turns[branch] == ticket.ordinal)
                name = ("left", "right")[branch]
                record = MixedRecord(ticket.key, ticket.ordinal, ticket.value, ticket.revision, ticket.bias, name, value)
                if ticket.key == controls.corrupt_key and branch == 1:
                    if controls.swap_member:
                        record = replace(record, branch="left")
                    elif not controls.corrupt_stamp:
                        record = replace(record, key=record.key + 100)
                if controls.corrupt_stamp and ticket.key == controls.corrupt_key and branch == 1:
                    stamps = tuple(replace(stamp, numerical_epoch=stamp.numerical_epoch + 100) for stamp in stamps)
                fact = asdict(record)
                fact["value"] = float(value)
                fact["sources"] = [(stamp.participant_ref, stamp.binding, stamp.numerical_epoch) for stamp in stamps]
                fact["attempt"] = source.mark("publication_attempt." + name, ticket.key)
                fact["returned"] = None
                source.products.append(fact)
                await context.publish(PORTS[branch], owners.Product(ticket.ordinal, record, stamps))
                fact["returned"] = source.mark("publication_return." + name, ticket.key)
                turns[branch] += 1
                condition.notify_all()

        async def left(ticket: Ticket) -> None:
            value = await local.run(
                ticket.value, lambda name, incoming: ModelLocalFrame(source, ticket.key, name, incoming), source.local_metrics
            )
            await publish(ticket, 0, torch.tensor(float(value)), ())

        async def right(ticket: Ticket) -> None:
            fact = {
                "sample": ticket.key,
                "origin": origin(apply),
                "input": ticket.value,
                "output": None,
                "outcome": "not_entered",
                "entered": None,
                "physical_end": None,
                "release": None,
                "sources": [],
            }
            source.stages.append(fact)
            try:
                async with context.use_source() as projection:
                    stamps = (projection.stamps[learner],)
                    fact["sources"] = [(stamp.participant_ref, stamp.binding, stamp.numerical_epoch) for stamp in stamps]

                    def invoke():
                        fact["entered"] = source.mark("source_entered", ticket.key)
                        fact["outcome"] = "entered_failed"
                        try:
                            if ticket.key == controls.block_key:
                                controls.entered.set()
                                if not controls.release.wait(15):
                                    raise TimeoutError("mixed physical test gate timed out")
                                if controls.fail_physical:
                                    raise RuntimeError("mixed physical failure after stop")
                            return calls[origin(apply)].invoke(projection.models[learner], ticket.value)
                        finally:
                            fact["physical_end"] = source.mark("source_physical_end", ticket.key)

                    try:
                        finished = await pool.execute(invoke, threading.Event())
                    except physical.NotEntered as error:
                        raise error.cause from error
                    result = finished.value
                    fact["outcome"] = "invalid_output"
                    if not scalar_tensor(result):
                        raise ValueError("mixed source returned invalid scalar after entry")
                    value = result.detach().clone()
                    ticket.stamps = stamps
                    fact.update(output=float(value), outcome="returned")
                    if finished.cancellation is not None:
                        fact["outcome"] = "returned_cancelled"
                        raise finished.cancellation
            finally:
                if fact["entered"] is not None:
                    fact["release"] = source.mark("source_release", ticket.key)
            await publish(ticket, 1, value, stamps)

        async def feedback() -> None:
            while True:
                message = await bridge.feedback.get()
                ticket = source.active.get(message.ordinal)
                if (
                    ticket is None
                    or ticket.key != message.key
                    or message.epoch != message.ordinal + 1
                    or source.revision != message.ordinal
                ):
                    raise ValueError("feedback does not name an advanced live ticket")
                if message.inputs != tuple((name, ticket.ordinal) for name in PORTS):
                    raise ValueError("feedback input membership differs")
                bias = calls[origin(update)].invoke(message.weight)
                if type(bias) is not int or not 0 <= bias < 5:
                    raise ValueError("policy binding returned invalid bias after entry")
                before = source.revision
                source.bias, source.revision = bias, before + 1
                source.updates.append(
                    asdict(message)
                    | {
                        "before_revision": before,
                        "revision": source.revision,
                        "bias": bias,
                        "at": source.mark("feedback_applied", ticket.key),
                    }
                )
                del source.active[message.ordinal]
                bridge.wake.set()

        with pool.lifetime():
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(feedback())
                ordinal = 0
                while not bridge.stop.is_set():
                    if len(source.active) >= horizon:
                        bridge.wake.clear()
                        if len(source.active) >= horizon and not bridge.stop.is_set():
                            await bridge.wake.wait()
                        continue
                    key = source.cursor
                    source.cursor += 1
                    value, revision, bias = key + source.bias, source.revision, source.bias
                    admitted = calls[origin(filtered)].invoke(key, revision)
                    if type(admitted) is not bool:
                        raise ValueError("dynamic filter returned invalid decision after entry")
                    at = source.mark("selection", key)
                    source.selections.append(
                        {"key": key, "value": value, "revision": revision, "bias": bias, "ordinal": ordinal if admitted else None, "at": at}
                    )
                    if not admitted:
                        source.excluded.append(key)
                        continue
                    ticket = Ticket(key, ordinal, value, revision, bias)
                    source.active[ordinal] = ticket
                    source.max_active = max(source.max_active, len(source.active))
                    ordinal += 1
                    tasks.create_task(left(ticket))
                    tasks.create_task(right(ticket))
                source.mark("selection_closed")
                # The owning Producer cancels and joins this private lifetime at
                # Repeat completion. Do not await a new independent scheduler.
                await asyncio.Future()

    async def cleanup(context: Any) -> None:
        source: MixedSource = context.states["mixed.source"]
        # Producer implementation and its TaskGroup/Workers have already joined.
        # Closing a ticket does not assert publication, feedback or rollback.
        for ticket in source.active.values():
            source.settlements.append(
                {
                    "key": ticket.key,
                    "ordinal": ticket.ordinal,
                    "disposition": "closed_without_feedback",
                    "at": source.mark("ticket_closed", ticket.key),
                }
            )
        source.active.clear()
        source.mark("producer_drained")

    def choose(counts: Mapping[str, int], facts: LearningFacts):
        count = counts.get("window", 0)
        if count < checkpoint.after_windows.value.data:
            return graph.Choice("window", count)
        return None if counts.get("stop-checkpoint", 0) else graph.Choice("stop-checkpoint", count)

    def objective(context: Any, left: MixedRecord, right: MixedRecord):
        source: MixedSource = admission.source
        facts: LearningFacts = context.states["mixed.learning"]
        value = calls[origin(contribute)].invoke(
            context.models[learner], left.selected, left.value, right.value, contribute.normalization.value.data
        )
        if not scalar_tensor(value):
            raise ValueError("mixed objective returned invalid loss after entry")
        facts.contributions.append(
            {
                "key": left.key,
                "ordinal": left.ordinal,
                "selected": left.selected,
                "revision": left.revision,
                "left": float(left.value),
                "right": float(right.value),
                "weight": float(context.models[learner].weight.detach()),
                "epoch": context.stamps[learner].numerical_epoch,
                "loss": float(value.detach()),
                "invocation": context.invocation,
                "inputs": handoffs(context),
                "at": source.mark("contribution", left.key),
            }
        )
        return value

    async def learned(context: Any, admitted: MixedRecord) -> None:
        facts: LearningFacts = context.states["mixed.learning"]
        message = Feedback(
            context.choice.key,
            admitted.key,
            float(context.models[learner].weight.detach()),
            context.stamps[learner].numerical_epoch,
            context.invocation,
            tuple((fact["channel"], fact["window"]) for fact in handoffs(context)),
        )
        facts.windows.append(asdict(message) | {"at": admission.source.mark("advanced", admitted.key)})
        await bridge.feedback.put(message)

    async def wait_stop(context: Any) -> None:
        bridge.checkpoint.set()
        admission.source.mark("checkpoint")
        await bridge.stop.wait()

    window = graph.Block(
        "window",
        (
            graph.Join(
                "admit", PORTS, (graph.Output("left", MixedRecord), graph.Output("right", MixedRecord)), admission, (graph.Use(learner),)
            ),
            graph.Retain("retain", (learner,), (unit,), "old", after=("admit",)),
            graph.Call(
                "objective",
                objective,
                ("left", "right"),
                (graph.Output("loss", torch.Tensor),),
                (graph.Use(learner),),
                ("mixed.learning",),
                retained="old",
            ),
            graph.Differentiate("differentiate", "loss", (unit,), "gradient", retained="old"),
            graph.Release("release", "old", after=("differentiate",)),
            graph.Advance("advance", "gradient", (unit,)),
            graph.Call("learned", learned, ("left",), uses=(graph.Use(learner),), states=("mixed.learning",), after=("advance",)),
        ),
    )
    participant = replace(
        examples.scale(learner, model.initial.value.data), load=lambda: calls[origin(model)].invoke(model.initial.value.data, True)
    )
    window = graph.Sequence("window", (replace(window, name="contribution", work=window.work[:5]), graph.Block("finish", window.work[5:])))
    filing = graph.Run(
        graph.Together(
            "mixed-run",
            (
                graph.Producer(
                    "mixed-data",
                    PORTS,
                    produce,
                    (graph.Use(learner),),
                    ("mixed.source",),
                    stop_when=one(session, WindowOp).sym_name.data,
                    remaining="discard",
                    cleanup=cleanup,
                ),
                graph.Repeat(
                    one(session, WindowOp).sym_name.data,
                    (window, graph.Block("stop-checkpoint", (graph.Call("wait-external-stop", wait_stop),))),
                    choose,
                    "mixed.learning",
                ),
            ),
        ),
        (participant,),
        (graph.Unit(unit, ((learner, advance.parameter.data),), learning_rate=advance.learning_rate.value.data),),
        (graph.OwnedState("mixed.source", MixedSource), graph.OwnedState("mixed.learning", LearningFacts)),
        tuple(graph.Channel(name, 1) for name in PORTS),
    )
    return filing, admission
