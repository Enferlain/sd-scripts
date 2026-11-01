"""Optional C lowering into the same existing grant/optimization owners as A.

The maintained two-pass numerical callable is reused unchanged. Observation
facts below have no authority and are never used to decide grant readiness.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, Region

from .dialects import CallOp, PlanOp, SessionOp, dump
from .grant_contract import PROFILE, accept_grant
from .grant_dialect import AdvanceGrantedOp, GrantOp, PhaseOp, parse_grant
from .model_adapter import DomainBinding, DomainRegistry
from .model_contract import one
from .model_dialect import ModelOp, RecipeOp
from .prepare import Accepted, NotReady, origin, ref

owner_target = importlib.import_module("docs_design.models-strategy-trainer.grant_target")
backend, examples, graph, torch = owner_target.backend, owner_target.examples, owner_target.graph, owner_target.torch

TARGET_C = "candidate-grant-cpu-v1"
COMPLETION = "physical_return_then_owner_handback_verification"
PROTOCOL = "offered profile -> scoped execution -> restored/current/final-gradient check -> verified owner token -> retained clipping/advance/reset"


def grant_registry() -> DomainRegistry:
    # Extract only the maintained numerical callable. The old authored Run is
    # discarded; our backend topology is generated from prepared xDSL below.
    functions = owner_target.functions()
    return DomainRegistry(
        "grant-fixtures-v1",
        {
            "scale_v1": DomainBinding("model", "load", "1", functions["scale_v1"]),
            "two_pass_v1": DomainBinding("grant", "scoped_write", "1", functions["two_pass_v1"]),
        },
    )


GrantFacts = owner_target.GrantFacts
GrantRun = owner_target.GrantRun


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


def prepare_grant(
    accepted: Accepted,
    registry: DomainRegistry | None = None,
    *,
    offered: bool = True,
    lowering: Literal["calls", "inline"] = "calls",
    history_limit: int = 256,
    target: str = TARGET_C,
    saved_text: str | None = None,
    saved_manifest: Mapping[str, Any] | None = None,
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
    if saved_text is not None and (not isinstance(saved_text, str) or not saved_text):
        raise NotReady("prepared grant document must be nonempty text")
    text = dump(plan) if saved_text is None else saved_text
    try:
        decoded = parse_grant(text)
        decoded.verify()
    except Exception as error:
        raise NotReady(f"prepared grant plan: invalid structure: {error}") from error
    root = one(decoded, PlanOp)
    if set(root.properties) != {"accepted_digest", "contract", "target", "stop_rule"}:
        raise NotReady("prepared grant: unknown execution property or flow coordination")
    semantic = one(decoded, SessionOp)
    if root.accepted_digest.data != accepted.digest or accept_grant(ModuleOp([semantic.clone()])).digest != accepted.digest:
        raise NotReady("prepared grant recipe changed accepted meaning")
    if (root.contract, root.target.data, root.stop_rule) != (semantic.contract, target, semantic.stop_rule):
        raise NotReady("inconsistent prepared grant target/contract/disposition")
    if root.horizon is not None or root.capacity is not None or root.feedback is not None:
        raise NotReady("model-only grant plan cannot introduce flow coordination")
    recipe = one(decoded, RecipeOp)
    if set(recipe.properties) != {"origin", "role", "protocol"}:
        raise NotReady("prepared grant: unknown owner protocol property")
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
            if set(op.properties) != {"origin", "ancestors", "implementation", "binding_version", "effect", "role", "completion"}:
                raise NotReady(f"{op.origin.data}: unknown prepared grant binding property")
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
    if not plan.is_structurally_equivalent(decoded):
        raise NotReady("prepared grant document differs from the complete expected recipe")
    model_name, unit_name = ref(model.entity), ref(grant.unit)
    phases = tuple((op.source_id.data, op.phase.data) for op in grant.body.block.ops if isinstance(op, PhaseOp))
    prepared_manifest = {
        "schema": 1,
        "abi": TARGET_C,
        "profile": PROFILE,
        "registry": registry.name,
        "model_role": {"model": model_name},
        "bindings": manifest,
    }
    if saved_manifest is not None and saved_manifest != prepared_manifest:
        raise NotReady("binding manifest differs from installed grant registry")
    try:
        materialized = owner_target.lower(
            owner_target.GrantSpec(
                session.source_id.data,
                model_name,
                unit_name,
                advance.parameter.data,
                model.initial.value.data,
                advance.learning_rate.value.data,
                advance.clip_norm.value.data if advance.clip_norm is not None else None,
                grant.source_id.data,
                advance.source_id.data,
                phases,
            ),
            {"model": resolved[origin(model)].invoke, "grant": resolved[origin(grant)].invoke},
            offered=offered,
            lowering=lowering,
            history_limit=history_limit,
        )
    except Exception as error:
        raise NotReady(f"{session.source_id.data}: grant owner realization: {error}") from error
    facts, attempt = materialized.facts, materialized.attempt
    return PreparedGrant(
        accepted,
        text,
        json.dumps(prepared_manifest, sort_keys=True, indent=2) + "\n",
        attempt,
        facts,
        model_name,
        unit_name,
        advance.parameter.data,
    )
