"""A's optional Torch/owner backend. One existing engine coordinates the run.

The old Run/Work records below are generated backend artifacts. Their closed
vocabulary and window-key restriction are target limits, not semantic grammar.
No author objects or serialized closures cross the accepted boundary.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from types import TracebackType
from typing import Any, Literal

import torch
from torch import nn
from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, Region

from .dialects import CallOp, ComposeOp, FusedOp, PlanOp, SessionOp, dump, integer
from .fixture import registry as scalar_registry
from .model_contract import accept_model, numbers, one, selected_keys
from .fusion import FusionError, groups as local_groups, rewrite as fuse_plan
from .local_runtime import LocalMetrics, LocalProgram, Stage, compile_local
from .model_dialect import (
    AddressOp,
    AdvanceOp,
    ContributeOp,
    CorrelateOp,
    EvaluateOp,
    FilterOp,
    ModelFlowOp,
    ModelOp,
    RecipeOp,
    WindowOp,
    parse_model,
)
from .prepare import Accepted, NotReady, check_calls, check_plan, check_scalar_locals, implementation, local_effect, local_maps, origin, ref

# The existing research directory contains a hyphen and cannot be named in a
# Python import statement. These are eager, explicit optional-backend imports.
_prefix = "docs_design.models-strategy-trainer.execution_candidate"
graph = importlib.import_module(_prefix + ".graph")
backend = importlib.import_module(_prefix + ".prepare")
owners = importlib.import_module(_prefix + ".state")
physical = importlib.import_module(_prefix + ".workers")
examples = importlib.import_module(_prefix + ".examples")

TARGET_A = "candidate-owner-cpu-v1"
OWNER_PROTOCOL = "one producer/two paths; atomic member check; two-block contribution sequence; requested current-at-start read"


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
    bindings = {
        name: DomainBinding(binding.role, binding.effect, binding.version, binding.invoke)
        for name, binding in scalar_registry().bindings.items()
        if binding.role == "map"
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
class PreparedModel:
    accepted: Accepted
    text: str
    manifest: str
    attempt: Any
    admissions: tuple[Admission, ...]
    addresses: Mapping[tuple[int, int], int]
    controls: Controls
    local: LocalProgram

    def publish(self) -> ModelRun:
        ready = self.attempt.publish()
        return ModelRun(self, ready, backend.Engine(ready))


@dataclass
class ModelRun:
    prepared: PreparedModel
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
        assert image is not None
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


def prepare_model(
    accepted: Accepted,
    registry: DomainRegistry | None = None,
    *,
    controls: Controls | None = None,
    lowering: Literal["calls", "inline"] = "calls",
    workers: bool = True,
    target: str = TARGET_A,
    fuse: bool = False,
) -> PreparedModel:
    registry, controls = registry or model_registry(), controls or Controls()
    module = parse_model(accepted.text)
    if accept_model(module) != accepted:
        raise NotReady("noncanonical A snapshot")
    session = one(module, SessionOp)
    assert session.horizon is not None
    window, advance = one(session, WindowOp), one(session, AdvanceOp)
    keys = selected_keys(session)
    if target != TARGET_A or window.count.value.data != 2 or len(keys) > 4 or session.horizon.value.data < len(keys):
        raise NotReady(
            f"{origin(window)}: owner CPU target requires two-contribution windows and at most four finite selected keys within the source horizon"
        )
    if advance.parameter.data != "weight":
        raise NotReady(f"{origin(advance)}: scalar owner target has no parameter {advance.parameter.data}")
    check_scalar_locals(local_maps(one(session, ComposeOp)))
    specs: list[tuple[Any, str, str]] = [(op, "model", "load") for op in session.body.block.ops if isinstance(op, ModelOp)]
    specs.extend((op, "map", local_effect(op)) for op in local_maps(one(session, ComposeOp)))
    specs.extend(
        [
            (one(session, ModelFlowOp), "encode", "read_source"),
            (one(session, ContributeOp), "objective", "build_loss"),
            (one(session, EvaluateOp), "measure", "current_read"),
        ]
    )
    sources = {origin(op): op for op, _, _ in specs}
    emitted: list[CallOp] = []
    manifest: list[dict[str, str]] = []
    for op, role, effect in specs:
        symbol = ref(implementation(op))
        binding = registry.bindings.get(symbol)
        if binding is None or (binding.role, binding.effect) != (role, effect) or not binding.version:
            raise NotReady(f"{origin(op)}: missing/incompatible {role} binding @{symbol}")
        manifest.append({"symbol": symbol, "version": binding.version, "role": role, "effect": effect})
        ancestors: list[StringAttr] = []
        parent = op.parent_op()
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
                    "effect": StringAttr(effect),
                    "role": StringAttr(role),
                    "completion": StringAttr("protected_physical_return_then_validate"),
                }
            )
        )
    address_ops = [
        AddressOp(
            properties={
                "origin": one(session, CorrelateOp).source_id,
                "sample": integer(sample),
                "window": integer(index // 2),
                "slot": integer(index % 2),
                "left": StringAttr(f"left.{index % 2}"),
                "right": StringAttr(f"right.{index % 2}"),
            }
        )
        for index, sample in enumerate(keys)
    ]
    recipe = RecipeOp(
        properties={
            "origin": session.source_id,
            "role": StringAttr("owner-backend"),
            "protocol": StringAttr(OWNER_PROTOCOL),
        },
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
                    "capacity": integer(2),
                    "stop_rule": session.stop_rule,
                },
                regions=[Region(Block([recipe, *address_ops, *emitted]))],
            )
        ]
    )
    plan.verify()
    if fuse:
        plan = fuse_plan(plan, sources)
    text = dump(plan)
    try:
        loaded = parse_model(text)
        loaded.verify()
    except Exception as error:
        raise NotReady(f"prepared owner plan: invalid structure: {error}") from error
    root = check_plan(loaded, accepted, session, target=target, capacity=2)
    if not fuse and any(isinstance(op, FusedOp) for op in root.walk()):
        raise NotReady("prepared owner local fusion was not selected")
    if any(not isinstance(op, (RecipeOp, AddressOp, CallOp, FusedOp)) for op in root.body.block.ops):
        raise NotReady("prepared owner recipe contains unsupported operations")
    recipes = [op for op in root.body.block.ops if isinstance(op, RecipeOp)]
    if len(recipes) != 1:
        raise NotReady("prepared owner recipe: exactly one semantic recipe required")
    recipe = recipes[0]
    if set(recipe.properties) != {"origin", "role", "protocol"}:
        raise NotReady("prepared owner recipe contains unknown protocol properties")
    semantic = tuple(recipe.body.block.ops)
    if len(semantic) != 1 or not isinstance(semantic[0], SessionOp):
        raise NotReady("prepared owner recipe: one semantic session required")
    decoded = semantic[0]
    try:
        correspondence = accept_model(ModuleOp([decoded.clone()])).digest
    except Exception as error:
        raise NotReady("prepared owner recipe invalidated the accepted meaning") from error
    if correspondence != accepted.digest:
        raise NotReady("prepared owner recipe changed the accepted meaning")
    if (
        (recipe.origin.data, recipe.role.data, recipe.protocol.data) != (origin(decoded), "owner-backend", OWNER_PROTOCOL)
        or tuple(recipe.body.block.ops) != (decoded,)
        or recipe.body.block.args
    ):
        raise NotReady("prepared owner recipe source/protocol differs")
    check_calls(
        [op for op in loaded.walk() if isinstance(op, CallOp)],
        specs,
        registry.bindings,
        completion="protected_physical_return_then_validate",
    )
    dispatch_ops = [op for op in root.body.block.ops if isinstance(op, FusedOp) or (isinstance(op, CallOp) and op.role.data == "map")]
    try:
        dispatch_groups = local_groups(dispatch_ops, sources)
    except FusionError as error:
        raise NotReady(f"prepared owner local dispatch: {error}") from error
    if tuple(call.origin.data for group in dispatch_groups for call in group) != tuple(
        origin(op) for op in local_maps(one(decoded, ComposeOp))
    ):
        raise NotReady("prepared owner local stage order differs")
    local = compile_local(
        tuple(
            tuple(Stage(call.origin.data, registry.bindings[ref(call.implementation)].invoke) for call in group)
            for group in dispatch_groups
        )
    )
    calls: dict[str, DomainBinding] = {}
    for op in loaded.walk():
        if isinstance(op, CallOp):
            binding = registry.bindings[ref(op.implementation)]
            if (op.role.data, op.effect.data, op.binding_version.data, op.completion.data) != (
                binding.role,
                binding.effect,
                binding.version,
                "protected_physical_return_then_validate",
            ):
                raise NotReady(f"{op.origin.data}: inconsistent prepared binding")
            calls[op.origin.data] = binding
    rows = [op for op in loaded.walk() if isinstance(op, AddressOp)]
    if any(set(op.properties) != {"origin", "sample", "window", "slot", "left", "right"} for op in rows):
        raise NotReady("prepared address contains unknown execution properties")
    ordered = sorted(rows, key=lambda op: (op.window.value.data, op.slot.value.data))
    expected = [(index // 2, index % 2, sample) for index, sample in enumerate(selected_keys(decoded))]
    if [(op.window.value.data, op.slot.value.data, op.sample.value.data) for op in ordered] != expected:
        raise NotReady("prepared addresses changed logical membership")
    addresses = {(op.window.value.data, op.slot.value.data): op.sample.value.data for op in rows}
    ports: dict[int, tuple[str, str]] = {}
    for op in rows:
        pair = (op.left.data, op.right.data)
        slot = op.slot.value.data
        if op.origin.data != one(decoded, CorrelateOp).source_id.data or ports.setdefault(slot, pair) != pair:
            raise NotReady("prepared address port/source correspondence differs")
    if len({name for pair in ports.values() for name in pair}) != 4:
        raise NotReady("prepared correlated ports alias")
    filing, admissions = lower_model(decoded, calls, addresses, ports, controls, local)
    try:
        checked = graph.Contract().accept(filing)
        attempt = backend.prepare(checked, lowering=lowering, worker_calls=("measure",) if workers else (), worker_limit=2)
    except Exception as error:
        raise NotReady(f"{origin(session)}: owner target realization: {error}") from error
    for slot, admission in enumerate(admissions):
        admission.bind(*(attempt.state.feeds[name] for name in ports[slot]))
    # Diagnostic history is disposable; live input links and owner facts remain.
    attempt.state.trace = deque(maxlen=controls.history_limit)
    return PreparedModel(
        accepted,
        text,
        json.dumps(
            {
                "schema": 1,
                "abi": TARGET_A,
                "registry": registry.name,
                "bindings": manifest,
                **(
                    {
                        "local_lowering": "fused-v1" if any(len(group) > 1 for group in dispatch_groups) else "unfused-v1",
                        "fused_groups": sum(len(group) > 1 for group in dispatch_groups),
                    }
                    if fuse
                    else {}
                ),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        attempt,
        admissions,
        MappingProxyType(addresses),
        controls,
        local,
    )


def lower_model(
    session: SessionOp,
    calls: Mapping[str, DomainBinding],
    addresses: Mapping[tuple[int, int], int],
    ports: Mapping[int, tuple[str, str]],
    controls: Controls,
    local: LocalProgram,
) -> tuple[Any, tuple[Admission, ...]]:
    """Compile the prepared recipe into the existing owner engine's artifact."""
    window, advance = one(session, WindowOp), one(session, AdvanceOp)
    contribute, evaluate, apply = one(session, ContributeOp), one(session, EvaluateOp), one(session, ModelFlowOp)
    encoder, learner, unit = ref(apply.model), ref(contribute.model), ref(advance.unit)
    keys = tuple(addresses[address] for address in sorted(addresses))
    by_sample = {sample: address for address, sample in addresses.items()}
    admissions = tuple(Admission(slot, addresses, encoder) for slot in range(2))
    excluded = numbers(one(session, FilterOp).excluded)
    request_after = evaluate.after_window.value.data
    normalization = contribute.normalization.value.data

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
            fact = {"sample": sample, "origin": origin(apply), "input": sample, "output": None, "outcome": "not_entered"}
            facts.stages.append(fact)
            async with context.use_source() as projection:
                try:
                    finished = await pool.execute(
                        lambda: calls[origin(apply)].invoke(projection.models[encoder], sample), threading.Event()
                    )
                except physical.NotEntered as error:
                    raise error.cause from error
                except physical.PhysicalFailed:
                    fact["outcome"] = "entered_failed"
                    raise
                result = finished.value
                fact["outcome"] = "invalid_output"
                if not scalar_tensor(result):
                    raise ValueError(f"{origin(apply)}: invalid returned source value after entry")
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
        value = calls[origin(contribute)].invoke(context.models[learner], left.sample, left.value, right.value, normalization)
        if not scalar_tensor(value):
            raise ValueError(f"{origin(contribute)}: invalid returned loss after entry")
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
        result = calls[origin(evaluate)].invoke(context.models[learner])
        if not scalar_tensor(result):
            raise ValueError(f"{origin(evaluate)}: invalid returned measurement after entry")
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
    for op in session.body.block.ops:
        if not isinstance(op, ModelOp):
            continue
        name, initial, trainable = ref(op.entity), op.initial.value.data, op.treatment.data == "trainable"
        declaration = examples.scale(name, initial)
        fn = calls[origin(op)].invoke
        participants.append(replace(declaration, load=lambda fn=fn, initial=initial, trainable=trainable: fn(initial, trainable)))
    filing = graph.Run(
        graph.Together(
            "run",
            (
                graph.Producer(
                    "data", tuple(name for slot in range(2) for name in ports[slot]), produce, (graph.Use(encoder),), ("a.source",)
                ),
                graph.Repeat(window.sym_name.data, (graph.Sequence("window", tuple(blocks)),), choose, "a.learning"),
                graph.Requested(
                    "requested-evaluation",
                    window.sym_name.data,
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
        (graph.Unit(unit, ((learner, advance.parameter.data),), learning_rate=advance.learning_rate.value.data),),
        (
            graph.OwnedState("a.source", SourceFacts),
            graph.OwnedState("a.learning", LearningFacts),
            graph.OwnedState("a.evaluation", EvaluationFacts),
        ),
        tuple(graph.Channel(name, 2) for slot in range(2) for name in ports[slot]),
    )
    return filing, admissions
