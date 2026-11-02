"""Independent xDSL mixed verification before the shared owner realization."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, IRNode, Region, SSAValue

from .dialects import CallOp, ComposeOp, FusedOp, PlanOp, SessionOp, dump, integer
from .fusion import FusionError, groups as local_groups, rewrite as fuse_plan
from .local_runtime import Stage, compile_local
from .mixed_contract import accept_mixed
from .mixed_dialect import DynamicFilterOp, PolicyUpdateOp, StopCheckpointOp, parse_mixed
from .model_contract import one
from .model_dialect import AdvanceOp, ContributeOp, ModelFlowOp, ModelOp, RecipeOp, WindowOp
from .prepare import Accepted, NotReady, check_calls, check_plan, check_scalar_locals, implementation, local_effect, local_maps, origin, ref

owner_target = importlib.import_module("docs_design.models-strategy-trainer.mixed_target")
DomainBinding, DomainRegistry = owner_target.DomainBinding, owner_target.DomainRegistry
MixedControls, MixedSource, LearningFacts = owner_target.MixedControls, owner_target.MixedSource, owner_target.LearningFacts
Bridge, Feedback, Ticket, MixedRecord = owner_target.Bridge, owner_target.Feedback, owner_target.Ticket, owner_target.MixedRecord
MixedAdmission, mixed_registry = owner_target.MixedAdmission, owner_target.mixed_registry
backend, graph, causes = owner_target.backend, owner_target.graph, owner_target.causes
TARGET_MIXED = "continuous-owner-cpu-v1"
OWNER_PROTOCOL = "open producer; owner feedback receiver; ordered bounded ports; same-binding production snapshots; external checkpoint"
COMPLETION = "protected_physical_return_then_validate"
PORTS, SELECTION_QUANTUM = owner_target.PORTS, owner_target.SELECTION_QUANTUM


class MixedRun(owner_target.MixedRun):
    def stop(self) -> None:
        try:
            super().stop()
        except owner_target.NotReady as error:
            raise NotReady(str(error)) from error


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


def prepare_mixed(
    accepted: Accepted,
    registry: DomainRegistry | None = None,
    *,
    controls: MixedControls | None = None,
    lowering: Literal["calls", "inline"] = "calls",
    fuse: bool = False,
    target: str = TARGET_MIXED,
    saved_text: str | None = None,
    saved_manifest: Mapping[str, Any] | None = None,
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
    if saved_text is not None and (not isinstance(saved_text, str) or not saved_text):
        raise NotReady("prepared mixed document must be nonempty text")
    text = dump(plan) if saved_text is None else saved_text
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
    if not graph_equivalent(plan, loaded):
        raise NotReady("prepared mixed document differs from the complete expected recipe")
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
    if saved_manifest is not None and saved_manifest != json.loads(manifest):
        raise NotReady("saved mixed binding manifest differs from installed contracts")
    try:
        materialized = owner_target.lower(
            mixed_spec(decoded), calls, local, controls, lowering=lowering, source_factory=MixedSource, learning_factory=LearningFacts
        )
    except Exception as error:
        raise NotReady(f"mixed owner target realization: {error}") from error
    return PreparedMixed(accepted, text, manifest, materialized.attempt, materialized.bridge, controls, local)


def graph_equivalent(expected: ModuleOp, actual: ModuleOp) -> bool:
    # xDSL 0.74 compares operands before later results enter its context. Seed
    # corresponding definitions for the graph's forward feedback references.
    left, right = tuple(expected.walk()), tuple(actual.walk())
    if len(left) != len(right):
        return False
    context: dict[IRNode | SSAValue, IRNode | SSAValue] = {}
    for first, second in zip(left, right, strict=True):
        if len(first.results) != len(second.results):
            return False
        for result, other in zip(first.results, second.results, strict=True):
            if result.type != other.type:
                return False
            context[result] = other
    return expected.is_structurally_equivalent(actual, context)


def mixed_spec(session: SessionOp):
    if session.horizon is None:
        raise NotReady("explicit mixed horizon required")
    model, apply, contribute, advance = (
        one(session, ModelOp),
        one(session, ModelFlowOp),
        one(session, ContributeOp),
        one(session, AdvanceOp),
    )
    return owner_target.MixedSpec(
        ref(model.entity),
        ref(advance.unit),
        advance.parameter.data,
        model.initial.value.data,
        advance.learning_rate.value.data,
        session.horizon.value.data,
        one(session, StopCheckpointOp).after_windows.value.data,
        contribute.normalization.value.data,
        one(session, WindowOp).sym_name.data,
        origin(model),
        origin(apply),
        origin(contribute),
        origin(one(session, DynamicFilterOp)),
        origin(one(session, PolicyUpdateOp)),
    )
