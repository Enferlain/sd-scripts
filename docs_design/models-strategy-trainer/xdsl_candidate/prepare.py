"""Contract acceptance, then a deliberately small scalar CPU target.

The accepted document is a detached canonical xDSL snapshot. Target restrictions
and symbolic bindings are checked later. Runtime call tables are generated from
the prepared xDSL, rather than serving as a second semantic program.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, Operation, Region, SSAValue

from .dialects import (
    CallOp,
    ChainOp,
    ComposeOp,
    EntityOp,
    FeedbackRecipeOp,
    FlowOp,
    FlowType,
    HandoffRecipeOp,
    LocalOp,
    MapOp,
    PlanOp,
    PublishOp,
    ScoreOp,
    SelectOp,
    SelectRecipeOp,
    SessionOp,
    YieldOp,
    dump,
    integer,
    parse,
)

CONTRACT = "scalar-feedback-v1"
TARGET = "scalar-async-cpu-v1"


class Rejected(ValueError):
    """The described program violates the selected semantic contract."""


class NotReady(ValueError):
    """Accepted meaning cannot be realized by this target and binding set."""


def origin(op: Operation) -> str:
    value = op.properties.get("source_id")
    return value.data if isinstance(value, StringAttr) else op.name


def require(condition: bool, op: Operation, message: str) -> None:
    if not condition:
        raise Rejected(f"{origin(op)}: {message}")


def ref(value: SymbolRefAttr) -> str:
    # This contract uses flat module-owned entities and implementation names.
    if value.nested_references:
        raise Rejected(f"{value}: nested symbol paths are not in contract {CONTRACT}")
    return value.root_reference.data


def local_maps(op: ComposeOp | LocalOp) -> tuple[MapOp, ...]:
    """Verify recursive one-to-one composition and flatten only for lowering."""
    block = op.body.block
    require(len(block.args) == 1 and block.args[0].type == FlowType(StringAttr("record")), op, "one record endpoint argument required")
    previous: SSAValue = block.args[0]
    maps: list[MapOp] = []
    for child in block.ops:
        if isinstance(child, MapOp | LocalOp):
            require(child.input is previous and child.output.type == previous.type, child, "broken local record chain")
            if isinstance(child, MapOp):
                require(child.effect.data in ("pure", "may_effect"), child, "unknown effect contract")
                require(bool(ref(child.implementation)), child, "empty implementation reference")
                maps.append(child)
            else:
                maps.extend(local_maps(child))
            previous = child.output
        elif isinstance(child, YieldOp):
            require(child is block.last_op and child.value is previous, op, "yield must return the final endpoint")
        else:
            raise Rejected(f"{origin(child)}: unsupported local operation in {CONTRACT}")
    require(isinstance(block.last_op, YieldOp) and bool(maps), op, "nonempty local region must end in wf.yield")
    return tuple(maps)


@dataclass(frozen=True)
class Accepted:
    text: str
    obligations: tuple[str, ...] = (
        "resolve versioned implementations and effect/role compatibility",
        "validate every returned scalar after selected entry",
        "acknowledge publication before keyed feedback application",
        "enforce aggregate outstanding horizon and physical completion at runtime",
    )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


def accept(module: ModuleOp) -> Accepted:
    """xDSL structural verification plus the selected data extension's rules."""
    try:
        module.verify()
    except Exception as error:
        for op in module.walk():
            try:
                op.verify(verify_nested_ops=False)
            except Exception as detail:
                raise Rejected(f"{origin(op)}: structural verification: {detail}") from detail
        raise Rejected(str(error)) from error
    roots = tuple(module.body.block.ops)
    if len(roots) != 1 or not isinstance(roots[0], SessionOp):
        raise Rejected("module: exactly one wf.session required")
    session = roots[0]
    require(session.contract.data == CONTRACT and session.version.value.data == 1, session, "unsupported contract/version")
    require(session.stop_rule.data == "settle_selected", session, "undeclared in-flight disposition")
    ids: set[str] = set()
    for op in session.walk():
        if isinstance(op, YieldOp):
            continue
        name = origin(op)
        require(bool(name) and name not in ids, op, "empty or duplicate source identity")
        ids.add(name)

    children = tuple(session.body.block.ops)
    allowed = (EntityOp, FlowOp, SelectOp, ComposeOp, ScoreOp, PublishOp)
    require(all(isinstance(op, allowed) for op in children), session, "unsupported service operation")
    entities: dict[str, EntityOp] = {}
    for op in children:
        if isinstance(op, EntityOp):
            require(bool(op.sym_name.data) and op.sym_name.data not in entities, op, "duplicate or empty entity symbol")
            require(op.kind.data in ("cursor", "policy", "publication"), op, "unknown entity kind")
            entities[op.sym_name.data] = op

    def one[T: Operation](kind: type[T]) -> T:
        selected = [op for op in children if isinstance(op, kind)]
        require(len(selected) == 1, session, f"{CONTRACT} requires one {kind.name}")
        return selected[0]

    select, compose, score, publish = one(SelectOp), one(ComposeOp), one(ScoreOp), one(PublishOp)
    for op, reference, kind in (
        (select, select.cursor, "cursor"),
        (select, select.policy, "policy"),
        (publish, publish.owner, "publication"),
    ):
        entity = entities.get(ref(reference))
        require(entity is not None and entity.kind.data == kind, op, f"unresolved {kind} entity use {reference}")
    require(len(entities) == 3, session, "unowned/unused entity declaration")
    require(select.application.data == "ack_before_next_selection", select, "unknown feedback application boundary")
    require(publish.acknowledgement.data == "returned_receipt", publish, "unknown publication acknowledgement")
    local_maps(compose)
    for op in (score, publish):
        require(bool(ref(op.implementation)), op, "empty implementation reference")
    flows = [op for op in children if isinstance(op, FlowOp)]
    require(len(flows) == 4, session, "four declared persistent endpoints required")
    edges = (
        (select.output, compose.input, "record"),
        (compose.output, score.input, "record"),
        (score.output, publish.input, "score"),
        (publish.feedback, select.feedback, "feedback"),
    )
    seen: set[SSAValue] = set()
    for writer, reader, payload in edges:
        require(writer is reader and writer not in seen, session, "missing, aliased or cross-wired service endpoint")
        require(any(writer is flow.endpoint for flow in flows), session, "endpoint must be declared in this session")
        require(writer.type == FlowType(StringAttr(payload)), session, f"expected {payload} endpoint")
        require(len(tuple(writer.uses)) == 2, session, "endpoint requires one writer and one reader")
        seen.add(writer)
    return Accepted(dump(module))


@dataclass(frozen=True)
class Binding:
    version: str
    role: str
    effect: str
    invoke: Callable[[int], Awaitable[object]]


@dataclass(frozen=True)
class Registry:
    name: str
    bindings: Mapping[str, Binding]

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))


@dataclass(frozen=True)
class ResolvedCall:
    origin: str
    symbol: str
    binding: Binding


@dataclass(frozen=True)
class Prepared:
    accepted: Accepted
    text: str
    manifest: str
    maps: tuple[ResolvedCall, ...]
    score: ResolvedCall
    publish: ResolvedCall
    horizon: int
    capacity: int


def prepare(accepted: Accepted, registry: Registry, *, capacity: int = 1, target: str = TARGET) -> Prepared:
    # Recheck the snapshot at this public boundary; no mutable author object.
    module = parse(accepted.text)
    if accept(module).text != accepted.text:
        raise NotReady("accepted snapshot is not canonical")
    session = next(op for op in module.walk() if isinstance(op, SessionOp))
    if target != TARGET or capacity != 1 or session.horizon.value.data > 2:
        raise NotReady(f"{origin(session)}: target {target} supports capacity 1 and horizon at most 2")
    compose = next(op for op in module.walk() if isinstance(op, ComposeOp))
    score = next(op for op in module.walk() if isinstance(op, ScoreOp))
    publish = next(op for op in module.walk() if isinstance(op, PublishOp))
    select = next(op for op in module.walk() if isinstance(op, SelectOp))
    maps = local_maps(compose)
    calls = [(op, "map", op.effect.data) for op in maps] + [(score, "score", "pure"), (publish, "publish", "publication")]
    manifest: list[dict[str, str]] = []
    emitted: list[CallOp] = []
    for op, role, effect in calls:
        symbol = ref(op.implementation)
        binding = registry.bindings.get(symbol)
        if binding is None or binding.role != role or binding.effect != effect or not binding.version:
            raise NotReady(f"{origin(op)}: unresolved or incompatible {role}/{effect} binding @{symbol}")
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
                    "completion": StringAttr("await_physical_then_validate"),
                }
            )
        )
    chains = [
        ChainOp(
            properties={"role": StringAttr("producer"), "transport": StringAttr("direct_local_calls_then_bounded_mailbox")},
            regions=[Region(Block(emitted[:-2]))],
        ),
        ChainOp(
            properties={"role": StringAttr("consumer"), "transport": StringAttr("retained_mailbox_slot_through_feedback_ack")},
            regions=[Region(Block(emitted[-2:]))],
        ),
    ]
    assert isinstance(compose.output.owner, FlowOp)
    recipes: list[Operation] = [op.clone() for op in session.body.block.ops if isinstance(op, EntityOp)]
    recipes.extend(
        [
            SelectRecipeOp(
                properties={
                    "origin": select.source_id,
                    "cursor": select.cursor,
                    "policy": select.policy,
                    "selection": StringAttr("scalar-feedback-v1: cursor + current_policy.bias; source_revision=0"),
                    "application": select.application,
                }
            ),
            HandoffRecipeOp(
                properties={
                    "origin": compose.source_id,
                    "endpoint": StringAttr(origin(compose.output.owner)),
                    "reader": score.source_id,
                    "release": StringAttr("after_feedback_ack; no_protection_held_while_waiting"),
                }
            ),
            FeedbackRecipeOp(
                properties={
                    "origin": publish.source_id,
                    "publication": publish.owner,
                    "policy": select.policy,
                    "update": StringAttr("scalar-feedback-v1: policy.bias=score%5; revision+=1"),
                    "admission": StringAttr("live_ticket_with_validated_publication_receipt; once_per_key"),
                }
            ),
            *chains,
        ]
    )
    plan = ModuleOp(
        [
            PlanOp(
                properties={
                    "accepted_digest": StringAttr(accepted.digest),
                    "target": StringAttr(target),
                    "contract": session.contract,
                    "horizon": session.horizon,
                    "capacity": integer(capacity),
                    "stop_rule": session.stop_rule,
                    "feedback": StringAttr("dedicated_control_task; receipt -> policy_apply -> ticket_retire -> slot_release"),
                },
                regions=[Region(Block(recipes))],
            )
        ]
    )
    plan.verify()
    text = dump(plan)
    # Derive runtime dispatch tables from the emitted prepared xDSL itself.
    loaded = parse(text)
    loaded.verify()
    root = next(op for op in loaded.walk() if isinstance(op, PlanOp))
    resolved: dict[str, tuple[ResolvedCall, ...]] = {}
    for chain in root.body.block.ops:
        if not isinstance(chain, ChainOp):
            continue
        role = chain.role.data
        if role not in ("producer", "consumer") or role in resolved:
            raise NotReady(f"prepared plan: duplicate/unsupported chain role {role}")
        operations = tuple(chain.body.block.ops)
        if not operations or not all(isinstance(op, CallOp) for op in operations):
            raise NotReady(f"prepared {role}: nonempty call chain required")
        calls = [op for op in operations if isinstance(op, CallOp)]
        expected = ("map",) * len(calls) if role == "producer" else ("score", "publish")
        if tuple(op.role.data for op in calls) != expected:
            raise NotReady(f"prepared {role}: incompatible call roles")
        selected: list[ResolvedCall] = []
        for op in calls:
            binding = registry.bindings[ref(op.implementation)]
            if (op.binding_version.data, op.role.data, op.effect.data, op.completion.data) != (
                binding.version,
                binding.role,
                binding.effect,
                "await_physical_then_validate",
            ):
                raise NotReady(f"{op.origin.data}: prepared binding/completion contract differs")
            selected.append(ResolvedCall(op.origin.data, ref(op.implementation), binding))
        resolved[role] = tuple(selected)
    if set(resolved) != {"producer", "consumer"}:
        raise NotReady("prepared plan: producer and consumer chains required")
    return Prepared(
        accepted,
        text,
        json.dumps({"schema": 1, "abi": "scalar-async-v1", "registry": registry.name, "bindings": manifest}, sort_keys=True, indent=2)
        + "\n",
        resolved["producer"],
        resolved["consumer"][0],
        resolved["consumer"][1],
        root.horizon.value.data,
        root.capacity.value.data,
    )
