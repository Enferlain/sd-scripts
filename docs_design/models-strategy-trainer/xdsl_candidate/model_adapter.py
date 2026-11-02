"""Independent xDSL A verification/preparation before the shared owner target."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, Region

from .dialects import CallOp, ComposeOp, FusedOp, PlanOp, SessionOp, dump, integer
from .model_contract import accept_model, numbers, one, selected_keys
from .fusion import FusionError, groups as local_groups, rewrite as fuse_plan
from .local_runtime import LocalProgram, Stage, compile_local
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

owner_target = importlib.import_module("docs_design.models-strategy-trainer.model_target")
backend, graph, owners, physical, examples, torch = (
    owner_target.backend,
    owner_target.graph,
    owner_target.owners,
    owner_target.physical,
    owner_target.examples,
    owner_target.torch,
)
DomainBinding, DomainRegistry = owner_target.DomainBinding, owner_target.DomainRegistry
Controls, Admission, Record = owner_target.Controls, owner_target.Admission, owner_target.Record
SourceFacts, LearningFacts, EvaluationFacts = owner_target.SourceFacts, owner_target.LearningFacts, owner_target.EvaluationFacts
ModelLocalFrame, ModelRun = owner_target.ModelLocalFrame, owner_target.ModelRun
model_registry, scale, encode, loss, measure = (
    owner_target.model_registry,
    owner_target.scale,
    owner_target.encode,
    owner_target.loss,
    owner_target.measure,
)
causes, scalar_tensor, handoffs = owner_target.causes, owner_target.scalar_tensor, owner_target.handoffs
TARGET_A = "candidate-owner-cpu-v1"
OWNER_PROTOCOL = "one producer/two paths; atomic member check; two-block contribution sequence; requested current-at-start read"


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


def prepare_model(
    accepted: Accepted,
    registry: DomainRegistry | None = None,
    *,
    controls: Controls | None = None,
    lowering: Literal["calls", "inline"] = "calls",
    workers: bool = True,
    target: str = TARGET_A,
    fuse: bool = False,
    saved_text: str | None = None,
    saved_manifest: Mapping[str, Any] | None = None,
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
    if saved_text is not None and (not isinstance(saved_text, str) or not saved_text):
        raise NotReady("prepared owner document must be nonempty text")
    text = dump(plan) if saved_text is None else saved_text
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
    if not plan.is_structurally_equivalent(loaded):
        raise NotReady("prepared owner document differs from the complete expected recipe")
    prepared_manifest = {
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
    }
    if saved_manifest is not None and saved_manifest != prepared_manifest:
        raise NotReady("saved A binding manifest differs from installed contracts")
    try:
        materialized = owner_target.lower(model_spec(decoded), calls, addresses, ports, controls, local, lowering=lowering, workers=workers)
    except Exception as error:
        raise NotReady(f"{origin(session)}: owner target realization: {error}") from error
    attempt, admissions = materialized.attempt, materialized.admissions
    return PreparedModel(
        accepted,
        text,
        json.dumps(prepared_manifest, sort_keys=True, indent=2) + "\n",
        attempt,
        admissions,
        MappingProxyType(addresses),
        controls,
        local,
    )


def model_spec(session: SessionOp):
    window, advance = one(session, WindowOp), one(session, AdvanceOp)
    contribute, evaluate, apply = one(session, ContributeOp), one(session, EvaluateOp), one(session, ModelFlowOp)
    return owner_target.ModelSpec(
        tuple(
            owner_target.ModelBindingSpec(ref(op.entity), origin(op), op.initial.value.data, op.treatment.data == "trainable")
            for op in session.body.block.ops
            if isinstance(op, ModelOp)
        ),
        window.sym_name.data,
        ref(apply.model),
        ref(contribute.model),
        ref(advance.unit),
        advance.parameter.data,
        advance.learning_rate.value.data,
        numbers(one(session, FilterOp).excluded),
        evaluate.after_window.value.data,
        contribute.normalization.value.data,
        origin(apply),
        origin(contribute),
        origin(evaluate),
    )
