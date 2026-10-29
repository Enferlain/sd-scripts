"""Optional C lowering into the same existing grant/optimization owners as A.

The maintained two-pass numerical callable is reused unchanged. Observation
facts below have no authority and are never used to decide grant readiness.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal

import torch
from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, Region

from .dialects import CallOp, PlanOp, SessionOp, dump
from .grant_contract import PROFILE, accept_grant
from .grant_dialect import AdvanceGrantedOp, GrantOp, PhaseOp, parse_grant
from .model_adapter import DomainBinding, DomainRegistry, backend, causes, examples, graph, scale
from .model_contract import one
from .model_dialect import ModelOp, RecipeOp
from .prepare import Accepted, NotReady, origin, ref

TARGET_C = "candidate-grant-cpu-v1"
COMPLETION = "physical_return_then_owner_handback_verification"
PROTOCOL = "offered profile -> scoped execution -> restored/current/final-gradient check -> verified owner token -> retained clipping/advance/reset"


def grant_registry() -> DomainRegistry:
    # Extract only the maintained numerical callable. The old authored Run is
    # discarded; our backend topology is generated from prepared xDSL below.
    selected = examples.two_pass().root.work[0].implementation
    return DomainRegistry(
        "grant-fixtures-v1",
        {
            "scale_v1": DomainBinding("model", "load", "1", scale),
            "two_pass_v1": DomainBinding("grant", "scoped_write", "1", selected),
        },
    )


@dataclass
class GrantFacts:
    """Test observations; live grant/unit owners remain authoritative."""

    entered: bool = False
    selected_returned: bool = False
    initial_weight: float | None = None
    selected_end_weight: float | None = None
    selected_gradient: float | None = None
    restored: bool = False
    phases: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PreparedGrant:
    accepted: Accepted
    text: str
    manifest: str
    attempt: Any
    facts: GrantFacts
    model: str
    unit: str
    parameter: str

    def publish(self) -> GrantRun:
        ready = self.attempt.publish()
        return GrantRun(self, ready, backend.Engine(ready))


@dataclass
class GrantRun:
    prepared: PreparedGrant
    ready: Any
    engine: Any
    last_report: dict[str, Any] | None = None
    started: bool = False
    interrupted: bool = False

    async def run(self) -> dict[str, Any]:
        if self.started:
            raise RuntimeError("grant run has one owner lifetime")
        self.started = True
        task = asyncio.current_task()
        cancellations = task.cancelling() if task is not None else 0
        failure: BaseException | None = None
        try:
            await self.engine.run()
        except BaseException as error:
            failure = error
            self.interrupted = isinstance(error, asyncio.CancelledError) or (task is not None and task.cancelling() > cancellations)
        self.last_report = self.report(failure)
        if failure is not None and self.interrupted:
            raise failure
        return self.last_report

    def report(self, failure: BaseException | None = None) -> dict[str, Any]:
        failure = failure or self.engine.failure
        state = self.ready.state
        assert state.image is not None
        runtime = state.image.units[self.prepared.unit]
        parameter = state.image.bindings[self.prepared.model].model.get_parameter(self.prepared.parameter)
        evidence = {record.invocation: record for record in (*runtime.outcomes, *runtime.active_outcomes.values())}
        records = [
            {
                "contribution": item.contribution,
                "optimizer": item.optimizer,
                "reset": item.reset,
                "invocation": item.invocation,
                "advance_invocation": item.advancement_invocation,
                "work": item.work,
                "input_count": len(item.input_handoffs),
            }
            for item in evidence.values()
        ]
        return {
            "status": "stopped"
            if state.completed
            else "cancelled"
            if self.interrupted or isinstance(failure, asyncio.CancelledError)
            else "failed",
            "errors": [{"type": type(item).__name__, "message": str(item)} for item in causes(failure)],
            "selected": asdict(self.prepared.facts),
            "final_weight": float(parameter.detach()),
            "final_gradient": None if parameter.grad is None else float(parameter.grad.detach()),
            "epoch": state.numerical_epochs[self.prepared.model],
            "steps": runtime.steps,
            "unit_outcomes": records,
            "pending": runtime.unfinished,
            "handback_verified": bool(records) and all(item["contribution"] == "granted_ready" for item in records),
            "unusable": sorted(state.unusable),
            "released": not state.running
            and not state.readers
            and not state.writers
            and not state.state_locks
            and not state.active_producers
            and not state.active_retentions,
            "diagnostics": len(state.trace),
        }


def prepare_grant(
    accepted: Accepted,
    registry: DomainRegistry | None = None,
    *,
    offered: bool = True,
    lowering: Literal["calls", "inline"] = "calls",
    history_limit: int = 256,
    target: str = TARGET_C,
) -> PreparedGrant:
    registry = registry or grant_registry()
    module = parse_grant(accepted.text)
    if accept_grant(module) != accepted:
        raise NotReady("noncanonical C snapshot")
    session = one(module, SessionOp)
    model, grant, advance = one(session, ModelOp), one(session, GrantOp), one(session, AdvanceGrantedOp)
    if target != TARGET_C or offered is not True:
        raise NotReady(f"{origin(grant)}: target does not offer the requested grant profile")
    if advance.parameter.data != "weight":
        raise NotReady(f"{origin(advance)}: scalar target has no parameter {advance.parameter.data}")
    calls: list[CallOp] = []
    manifest: list[dict[str, str]] = []
    for op, role, effect in ((model, "model", "load"), (grant, "grant", "scoped_write")):
        symbol = ref(op.implementation)
        binding = registry.bindings.get(symbol)
        if binding is None or (binding.role, binding.effect) != (role, effect) or not binding.version:
            raise NotReady(f"{origin(op)}: missing/incompatible {role} binding @{symbol}")
        manifest.append({"symbol": symbol, "role": role, "effect": effect, "version": binding.version})
        calls.append(
            CallOp(
                properties={
                    "origin": op.source_id,
                    "ancestors": ArrayAttr([session.source_id]),
                    "implementation": SymbolRefAttr(symbol),
                    "binding_version": StringAttr(binding.version),
                    "effect": StringAttr(effect),
                    "role": StringAttr(role),
                    "completion": StringAttr(COMPLETION),
                }
            )
        )
    recipe = RecipeOp(
        properties={"origin": grant.source_id, "role": StringAttr("grant-owner"), "protocol": StringAttr(PROTOCOL)},
        regions=[Region(Block([session.clone()]))],
    )
    plan = ModuleOp(
        [
            PlanOp(
                properties={
                    "accepted_digest": StringAttr(accepted.digest),
                    "contract": session.contract,
                    "target": StringAttr(target),
                    "stop_rule": session.stop_rule,
                },
                regions=[Region(Block([recipe, *calls]))],
            )
        ]
    )
    plan.verify()
    text = dump(plan)
    decoded = parse_grant(text)
    decoded.verify()
    root = one(decoded, PlanOp)
    semantic = one(decoded, SessionOp)
    if root.accepted_digest.data != accepted.digest or accept_grant(ModuleOp([semantic.clone()])).digest != accepted.digest:
        raise NotReady("prepared grant recipe changed accepted meaning")
    if (root.contract, root.target.data, root.stop_rule) != (semantic.contract, target, semantic.stop_rule):
        raise NotReady("inconsistent prepared grant target/contract/disposition")
    if root.horizon is not None or root.capacity is not None or root.feedback is not None:
        raise NotReady("model-only grant plan cannot introduce flow coordination")
    recipe = one(decoded, RecipeOp)
    if (recipe.origin.data, recipe.role.data, recipe.protocol.data) != (origin(grant), "grant-owner", PROTOCOL) or tuple(
        recipe.body.block.ops
    ) != (semantic,):
        raise NotReady("inconsistent prepared grant owner protocol")
    if len(tuple(root.body.block.ops)) != 3 or any(not isinstance(op, (RecipeOp, CallOp)) for op in root.body.block.ops):
        raise NotReady("unexpected prepared grant operation")
    model, grant, advance = one(semantic, ModelOp), one(semantic, GrantOp), one(semantic, AdvanceGrantedOp)
    expected = {origin(op): op.implementation for op in (model, grant)}
    resolved: dict[str, DomainBinding] = {}
    for op in decoded.walk():
        if isinstance(op, CallOp):
            binding = registry.bindings.get(ref(op.implementation))
            if binding is None or (op.role.data, op.effect.data, op.binding_version.data, op.completion.data) != (
                binding.role,
                binding.effect,
                binding.version,
                COMPLETION,
            ):
                raise NotReady(f"{op.origin.data}: inconsistent prepared grant binding")
            if expected.get(op.origin.data) != op.implementation or op.ancestors != ArrayAttr([semantic.source_id]):
                raise NotReady(f"{op.origin.data}: prepared grant source correspondence differs")
            if op.origin.data in resolved:
                raise NotReady(f"{op.origin.data}: duplicate prepared binding")
            resolved[op.origin.data] = binding
    if set(resolved) != {origin(model), origin(grant)}:
        raise NotReady("prepared grant bindings lost source correspondence")
    if grant.profile.data != PROFILE:
        raise NotReady("unsupported concrete grant profile")
    model_name, unit_name = ref(model.entity), ref(grant.unit)
    phases = tuple((op.source_id.data, op.phase.data) for op in grant.body.block.ops if isinstance(op, PhaseOp))
    facts = GrantFacts()
    selected = resolved[origin(grant)].invoke

    async def observed(access: Any, models: Any) -> None:
        parameter = models[model_name].get_parameter(advance.parameter.data)
        facts.entered = True
        facts.initial_weight = float(parameter.detach())
        try:
            # "model" is the maintained numerical binding's declared role,
            # independent of the semantic participant's actual entity symbol.
            result = selected(access, MappingProxyType({"model": models[model_name]}))
            if inspect.isawaitable(result):
                await result
            facts.selected_returned = True
        finally:
            # ABI: selected execution must join all private physical work before
            # returning/raising. These observations certify no handback themselves.
            facts.selected_end_weight = float(parameter.detach())
            facts.selected_gradient = None if parameter.grad is None else float(parameter.grad.detach())
            facts.restored = access.restored
            facts.phases = [
                {"phase": phase, "origin": phases[index][0] if index < len(phases) and phase == phases[index][1] else None}
                for index, phase in enumerate(access.phases)
            ]

    base = examples.scale(model_name, model.initial.value.data)
    loader = resolved[origin(model)].invoke

    def verify(concrete: Any) -> None:
        base.verify(concrete)
        parameter = concrete.get_parameter(advance.parameter.data)
        if (
            parameter.device.type != "cpu"
            or parameter.dtype != torch.float32
            or not parameter.requires_grad
            or not bool(torch.isfinite(parameter))
        ):
            raise ValueError("C scalar target requires finite trainable CPU float32 state")

    participant = replace(base, load=lambda: loader(model.initial.value.data, True), verify=verify)
    clip = advance.clip_norm.value.data if advance.clip_norm is not None else None
    filing = graph.Run(
        graph.Block(
            session.source_id.data,
            (
                graph.TwoPass(grant.source_id.data, observed, (), unit_name, "verified-handback", (graph.Use(model_name, "write"),)),
                graph.Advance(advance.source_id.data, "verified-handback", (unit_name,)),
            ),
        ),
        (participant,),
        (graph.Unit(unit_name, ((model_name, advance.parameter.data),), learning_rate=advance.learning_rate.value.data, clip_norm=clip),),
    )
    try:
        checked = graph.Contract(two_pass=offered).accept(filing)
        attempt = backend.prepare(checked, lowering=lowering)
    except Exception as error:
        raise NotReady(f"{session.source_id.data}: grant owner realization: {error}") from error
    attempt.state.trace = deque(maxlen=history_limit)
    return PreparedGrant(
        accepted,
        text,
        json.dumps(
            {
                "schema": 1,
                "abi": TARGET_C,
                "profile": PROFILE,
                "registry": registry.name,
                "model_role": {"model": model_name},
                "bindings": manifest,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        attempt,
        facts,
        model_name,
        unit_name,
        advance.parameter.data,
    )
