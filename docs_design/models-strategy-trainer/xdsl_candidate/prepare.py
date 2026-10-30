"""Contract acceptance, then a deliberately small scalar CPU target.

The accepted document is a detached canonical xDSL snapshot. Target restrictions
and symbolic bindings are checked later. Runtime call tables are generated from
the prepared xDSL, rather than serving as a second semantic program.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, Dialect, Operation, Region, SSAValue

from .dialects import (
    CallOp,
    ChainOp,
    ComposeOp,
    EntityOp,
    FeedbackRecipeOp,
    FlowOp,
    FlowType,
    FusedOp,
    HandoffRecipeOp,
    LocalOp,
    LocalSignatureOp,
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
from .fusion import FusionError, groups as local_groups, rewrite as fuse_plan
from .local_runtime import LocalProgram, Stage, compile_local
from .local_contract import LocalTransform, ResourceUse, TransparentLocal
from .payloads import DEFAULT_OUTCOMES, INTEGER, FrozenResource, PayloadCodec, PayloadContract, resolve_local

CONTRACT = "scalar-feedback-v1"
TARGET = "scalar-async-cpu-v1"
SCALAR_SELECTION = "scalar-feedback-v1: cursor + current_policy.bias; source_revision=0"
SCALAR_UPDATE = "scalar-feedback-v1: policy.bias=score%5; revision+=1"
SCALAR_ADMISSION = "live_ticket_with_validated_publication_receipt; once_per_key"
SCALAR_RELEASE = "after_feedback_ack; no_protection_held_while_waiting"
SCALAR_CONTROL = "dedicated_control_task; receipt -> policy_apply -> ticket_retire -> slot_release"
SCALAR_TRANSPORT = {
    "producer": "direct_local_calls_then_bounded_mailbox",
    "consumer": "retained_mailbox_slot_through_feedback_ack",
}


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
        raise Rejected(f"{value}: nested symbol paths are not supported by this flat-symbol contract")
    return value.root_reference.data


def implementation(op: Operation) -> SymbolRefAttr:
    value = op.properties.get("implementation")
    if not isinstance(value, SymbolRefAttr) or not ref(value):
        raise Rejected(f"{origin(op)}: empty or invalid implementation reference")
    return value


def local_effect(op: Operation) -> str:
    value = op.properties.get("effect")
    if not isinstance(value, StringAttr) or not value.data:
        raise Rejected(f"{origin(op)}: missing effect description")
    return value.data


def local_maps(op: ComposeOp | LocalOp) -> tuple[Operation, ...]:
    """Analyze installed unary stages through interfaces, with typed SSA links.

    The historical name is retained for callers. Local result schemas may differ;
    ordered transparent regions must return their declared final schema.
    """

    def visit(container: Operation, block: Block, input_type, output_type) -> list[Operation]:
        require(len(block.args) == 1 and block.args[0].type == input_type, container, "one typed local endpoint argument required")
        previous: SSAValue = block.args[0]
        maps: list[Operation] = []
        for child in block.ops:
            if isinstance(child, YieldOp):
                require(
                    child is block.last_op and child.value is previous and previous.type == output_type,
                    container,
                    "yield must return the final endpoint with declared schema",
                )
                continue
            interface = child.get_trait(LocalTransform)
            transparent = child.get_trait(TransparentLocal)
            if interface is None and transparent is None:
                raise Rejected(f"{origin(child)}: unsupported local operation in {CONTRACT}")
            require(
                len(child.operands) == len(child.results) == 1
                and child.operands[0] is previous
                and isinstance(child.results[0].type, FlowType),
                child,
                "broken local record chain or schema",
            )
            if transparent is not None:
                require(len(child.regions) == 1 and len(child.regions[0].blocks) == 1, child, "one ordered local block required")
                maps.extend(visit(child, child.regions[0].block, previous.type, child.results[0].type))
            else:
                assert interface is not None
                description = interface.describe(child)
                require(description is not None, child, "unknown effect or local transform contract")
                assert description is not None
                require(
                    (description.input_type, description.output_type, description.cardinality)
                    == (previous.type, child.results[0].type, "one-to-one"),
                    child,
                    "local schema/cardinality description differs",
                )
                require(
                    bool(description.outcomes)
                    and len(set(description.outcomes)) == len(description.outcomes)
                    and all(description.outcomes),
                    child,
                    "invalid local outcomes",
                )
                require(
                    set(DEFAULT_OUTCOMES).issubset(description.outcomes),
                    child,
                    "missing mandatory baseline local outcomes",
                )
                implementation(child)
                local_effect(child)
                maps.append(child)
            previous = child.results[0]
        require(isinstance(block.last_op, YieldOp) and bool(maps), container, "nonempty local region must end in wf.yield")
        return maps

    expected_output = op.output.type
    return tuple(visit(op, op.body.block, op.input.type, expected_output))


@dataclass(frozen=True)
class Accepted:
    text: str
    obligations: tuple[str, ...] = (
        "resolve versioned implementations and effect/role compatibility",
        "resolve installed local payload codecs and immutable resource snapshots before lowering",
        "validate every returned scalar after selected entry",
        "acknowledge publication before keyed feedback application",
        "enforce aggregate outstanding horizon and physical completion at runtime",
    )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


@dataclass(frozen=True)
class ContractExtension:
    """Installed contract code, selected explicitly; never retained in Accepted."""

    name: str
    version: int
    verify: Callable[[SessionOp], None]
    obligations: tuple[str, ...]


def accept(module: ModuleOp, *, extension: ContractExtension | None = None) -> Accepted:
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
    name, version = (extension.name, extension.version) if extension else (CONTRACT, 1)
    require(session.contract.data == name and session.version.value.data == version, session, "unsupported contract/version")
    ids: set[str] = set()
    for op in session.walk():
        if isinstance(op, YieldOp):
            continue
        name = origin(op)
        require(bool(name) and name not in ids, op, "empty or duplicate source identity")
        ids.add(name)

    if extension is not None:
        extension.verify(session)
        return Accepted(dump(module), extension.obligations)
    require(session.stop_rule.data == "settle_selected", session, "undeclared in-flight disposition")
    require(session.horizon is not None, session, "explicit flow horizon required")
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
    local_maps(compose)
    return Accepted(dump(module))


@dataclass(frozen=True)
class Binding:
    version: str
    role: str
    effect: str
    invoke: Callable[[Any], Awaitable[object]]
    resources: tuple[ResourceUse, ...] = ()
    outcomes: tuple[str, ...] = DEFAULT_OUTCOMES
    lower: Callable[[Mapping[str, FrozenResource]], Callable[[Any], Awaitable[object]]] | None = None


@dataclass(frozen=True)
class Registry:
    name: str
    bindings: Mapping[str, Binding]
    codecs: Mapping[FlowType, PayloadCodec] = field(default_factory=lambda: {FlowType(StringAttr("record")): INTEGER})
    resources: Mapping[str, FrozenResource] = field(default_factory=dict)
    dialects: tuple[Dialect, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))
        object.__setattr__(self, "codecs", MappingProxyType(dict(self.codecs)))
        object.__setattr__(self, "resources", MappingProxyType(dict(self.resources)))


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
    score: ResolvedCall
    publish: ResolvedCall
    horizon: int
    capacity: int
    local: LocalProgram
    local_contracts: Mapping[str, PayloadContract] = field(default_factory=dict)


class BindingDescription(Protocol):
    """Only the installed binding descriptors needed for cold verification."""

    @property
    def version(self) -> str: ...

    @property
    def role(self) -> str: ...

    @property
    def effect(self) -> str: ...


def check_plan(
    module: ModuleOp, accepted: Accepted, session: SessionOp, *, target: str, capacity: int, feedback: str | None = None
) -> PlanOp:
    """Check the decoded target envelope; none of its protocol fields are advisory."""
    roots = tuple(module.body.block.ops)
    if len(roots) != 1 or not isinstance(roots[0], PlanOp):
        raise NotReady("prepared plan: exactly one execution plan required")
    root = roots[0]
    if set(root.properties) - {"accepted_digest", "contract", "target", "stop_rule", "feedback", "horizon", "capacity"}:
        raise NotReady("prepared plan: unknown executable property")
    if (root.accepted_digest.data, root.contract, root.target.data, root.stop_rule, root.feedback) != (
        accepted.digest,
        session.contract,
        target,
        session.stop_rule,
        StringAttr(feedback) if feedback is not None else None,
    ):
        raise NotReady(f"{origin(session)}: prepared plan identity/target/protocol differs")
    if root.horizon != session.horizon or root.capacity != integer(capacity):
        raise NotReady(f"{origin(session)}: prepared plan flow bounds differ")
    if root.body.block.args:
        raise NotReady("prepared plan: closed target recipe required")
    return root


def check_scalar_locals(stages: Sequence[Operation]) -> None:
    """Older owner adapters still select their scalar-only local ABI."""
    scalar = FlowType(StringAttr("record"))
    for op in stages:
        interface = op.get_trait(LocalTransform)
        description = interface.describe(op) if interface is not None else None
        if (
            description is None
            or description.input_type != scalar
            or description.output_type != scalar
            or description.resources not in (None, ())
        ):
            raise NotReady(f"{origin(op)}: scalar owner local schema/resources unsupported")


def check_calls(
    calls: Sequence[CallOp], specs: Sequence[tuple[Operation, str, str]], bindings: Mapping[str, BindingDescription], *, completion: str
) -> None:
    """Verify source correspondence, not merely compatibility of a chosen symbol.

    This target uses one prepared call per semantic implementation selection.
    A future many-to-one rewrite must explicitly verify its stage correspondence.
    """
    expected = {origin(op): (op, role, effect) for op, role, effect in specs}
    if len(expected) != len(specs):
        raise NotReady("prepared plan: ambiguous semantic binding sources")
    seen: set[str] = set()
    for call in calls:
        source = call.origin.data
        if set(call.properties) != {"origin", "ancestors", "implementation", "binding_version", "effect", "role", "completion"}:
            raise NotReady(f"{source}: unknown prepared call property")
        if source not in expected or source in seen:
            raise NotReady(f"{source}: unknown/duplicate prepared binding source")
        seen.add(source)
        op, role, effect = expected[source]
        implementation = op.properties.get("implementation")
        if not isinstance(implementation, SymbolRefAttr) or call.implementation != implementation:
            raise NotReady(f"{source}: prepared implementation differs from accepted selection")
        binding = bindings.get(ref(implementation))
        if (
            binding is None
            or (call.role.data, call.effect.data, call.binding_version.data, call.completion.data)
            != (role, effect, binding.version, completion)
            or (binding.role, binding.effect) != (role, effect)
            or not binding.version
        ):
            raise NotReady(f"{source}: prepared binding/completion contract differs")
        ancestors: list[StringAttr] = []
        parent = op.parent_op()
        while parent is not None and not isinstance(parent, ModuleOp):
            ancestors.insert(0, StringAttr(origin(parent)))
            parent = parent.parent_op()
        if call.ancestors != ArrayAttr(ancestors):
            raise NotReady(f"{source}: prepared source ancestry differs")
    if seen != expected.keys():
        raise NotReady("prepared plan: missing semantic implementation selection")


def check_scalar_recipes(
    root: PlanOp, session: SessionOp, compose: ComposeOp, select: SelectOp, score: ScoreOp, publish: PublishOp
) -> None:
    """Match fixed scalar-owner protocols to accepted entities and endpoints."""
    children = tuple(root.body.block.ops)
    if any(not isinstance(op, (EntityOp, SelectRecipeOp, HandoffRecipeOp, FeedbackRecipeOp, ChainOp, LocalSignatureOp)) for op in children):
        raise NotReady("prepared plan: unsupported scalar recipe operation")
    declared = [op for op in session.body.block.ops if isinstance(op, EntityOp)]
    entities = [op for op in children if isinstance(op, EntityOp)]
    if len(entities) != len(declared) or {tuple(sorted(op.properties.items())) for op in entities} != {
        tuple(sorted(op.properties.items())) for op in declared
    }:
        raise NotReady("prepared plan: entity correspondence differs")
    assert isinstance(compose.output.owner, FlowOp)
    recipes = (
        (
            SelectRecipeOp,
            {
                "origin": select.source_id,
                "cursor": select.cursor,
                "policy": select.policy,
                "selection": StringAttr(SCALAR_SELECTION),
                "application": select.application,
            },
        ),
        (
            HandoffRecipeOp,
            {
                "origin": compose.source_id,
                "endpoint": StringAttr(origin(compose.output.owner)),
                "reader": score.source_id,
                "release": StringAttr(SCALAR_RELEASE),
            },
        ),
        (
            FeedbackRecipeOp,
            {
                "origin": publish.source_id,
                "publication": publish.owner,
                "policy": select.policy,
                "update": StringAttr(SCALAR_UPDATE),
                "admission": StringAttr(SCALAR_ADMISSION),
            },
        ),
    )
    for kind, properties in recipes:
        selected = [op for op in children if isinstance(op, kind)]
        if len(selected) != 1 or selected[0].properties != properties:
            raise NotReady(f"prepared plan: {kind.name} source/protocol differs")


def prepare(accepted: Accepted, registry: Registry, *, capacity: int = 1, target: str = TARGET, fuse: bool = False) -> Prepared:
    # Recheck the snapshot at this public boundary; no mutable author object.
    try:
        module = parse(accepted.text, extensions=registry.dialects) if registry.dialects else parse(accepted.text)
    except Exception as error:
        raise NotReady(f"accepted document: installed dialect support differs: {error}") from error
    if accept(module) != accepted:
        raise NotReady("accepted snapshot is not canonical")
    session = next(op for op in module.walk() if isinstance(op, SessionOp))
    horizon = session.horizon
    assert horizon is not None  # Established by this target's semantic contract.
    if target != TARGET or capacity != 1 or horizon.value.data > 2:
        raise NotReady(f"{origin(session)}: target {target} supports capacity 1 and horizon at most 2")
    compose = next(op for op in module.walk() if isinstance(op, ComposeOp))
    score = next(op for op in module.walk() if isinstance(op, ScoreOp))
    publish = next(op for op in module.walk() if isinstance(op, PublishOp))
    select = next(op for op in module.walk() if isinstance(op, SelectOp))
    maps = local_maps(compose)
    specs = [(op, "map", local_effect(op)) for op in maps] + [(score, "score", "pure"), (publish, "publish", "publication")]
    sources = {origin(op): op for op, _, _ in specs}
    manifest: list[dict[str, str]] = []
    emitted: list[CallOp] = []
    for op, role, effect in specs:
        symbol = ref(implementation(op))
        binding = registry.bindings.get(symbol)
        if binding is None or binding.role != role or binding.effect != effect or not binding.version:
            raise NotReady(f"{origin(op)}: unresolved or incompatible {role}/{effect} binding @{symbol}")
        if role != "map" and (
            binding.resources or binding.lower is not None or binding.outcomes != DEFAULT_OUTCOMES or not callable(binding.invoke)
        ):
            raise NotReady(f"{origin(op)}: nonlocal binding requires the scalar ABI")
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
    try:
        local_resolutions = {
            origin(op): resolve_local(op, registry.bindings[ref(implementation(op))], registry.codecs, registry.resources) for op in maps
        }
    except ValueError as error:
        raise NotReady(str(error)) from error
    chains = [
        ChainOp(
            properties={"role": StringAttr("producer"), "transport": StringAttr(SCALAR_TRANSPORT["producer"])},
            regions=[Region(Block(emitted[:-2]))],
        ),
        ChainOp(
            properties={"role": StringAttr("consumer"), "transport": StringAttr(SCALAR_TRANSPORT["consumer"])},
            regions=[Region(Block(emitted[-2:]))],
        ),
    ]
    assert isinstance(compose.output.owner, FlowOp)
    recipes: list[Operation] = [op.clone() for op in session.body.block.ops if isinstance(op, EntityOp)]
    recipes.extend(resolution.signature for resolution in local_resolutions.values())
    recipes.extend(
        [
            SelectRecipeOp(
                properties={
                    "origin": select.source_id,
                    "cursor": select.cursor,
                    "policy": select.policy,
                    "selection": StringAttr(SCALAR_SELECTION),
                    "application": select.application,
                }
            ),
            HandoffRecipeOp(
                properties={
                    "origin": compose.source_id,
                    "endpoint": StringAttr(origin(compose.output.owner)),
                    "reader": score.source_id,
                    "release": StringAttr(SCALAR_RELEASE),
                }
            ),
            FeedbackRecipeOp(
                properties={
                    "origin": publish.source_id,
                    "publication": publish.owner,
                    "policy": select.policy,
                    "update": StringAttr(SCALAR_UPDATE),
                    "admission": StringAttr(SCALAR_ADMISSION),
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
                    "feedback": StringAttr(SCALAR_CONTROL),
                },
                regions=[Region(Block(recipes))],
            )
        ]
    )
    plan.verify()
    if fuse:
        plan = fuse_plan(plan, sources)
    text = dump(plan)
    # Derive runtime dispatch tables from the emitted prepared xDSL itself.
    try:
        loaded = parse(text, extensions=registry.dialects) if registry.dialects else parse(text)
        loaded.verify()
    except Exception as error:
        raise NotReady(f"prepared plan: invalid structure: {error}") from error
    root = check_plan(loaded, accepted, session, target=target, capacity=capacity, feedback=SCALAR_CONTROL)
    if not fuse and any(isinstance(op, FusedOp) for op in root.walk()):
        raise NotReady("prepared local fusion was not selected for this target realization")
    assert root.horizon is not None and root.capacity is not None
    check_scalar_recipes(root, session, compose, select, score, publish)
    check_calls(
        [op for op in root.walk() if isinstance(op, CallOp)],
        specs,
        registry.bindings,
        completion="await_physical_then_validate",
    )
    signatures = [op for op in root.body.block.ops if isinstance(op, LocalSignatureOp)]
    if len(signatures) != len(local_resolutions) or {op.origin.data for op in signatures} != local_resolutions.keys():
        raise NotReady("prepared local signatures: source coverage differs")
    for signature in signatures:
        if signature.properties != local_resolutions[signature.origin.data].signature.properties:
            raise NotReady(f"{signature.origin.data}: prepared local schema/resource/outcome signature differs")
    resolved: dict[str, tuple[ResolvedCall, ...]] = {}
    selected_groups: dict[str, tuple[tuple[ResolvedCall, ...], ...]] = {}
    for chain in root.body.block.ops:
        if not isinstance(chain, ChainOp):
            continue
        role = chain.role.data
        if role not in ("producer", "consumer") or role in resolved:
            raise NotReady(f"prepared plan: duplicate/unsupported chain role {role}")
        if chain.transport.data != SCALAR_TRANSPORT[role] or chain.body.block.args:
            raise NotReady(f"prepared {role}: transport/closed-chain contract differs")
        if set(chain.properties) != {"role", "transport"}:
            raise NotReady(f"prepared {role}: unknown transport property")
        operations = tuple(chain.body.block.ops)
        if not operations or not all(isinstance(op, (CallOp, FusedOp)) for op in operations):
            raise NotReady(f"prepared {role}: nonempty call chain required")
        try:
            groups = local_groups(operations, sources)
        except FusionError as error:
            raise NotReady(f"prepared {role}: {error}") from error
        calls = [op for group in groups for op in group]
        expected = ("map",) * len(calls) if role == "producer" else ("score", "publish")
        if tuple(op.role.data for op in calls) != expected:
            raise NotReady(f"prepared {role}: incompatible call roles")
        expected_sources = tuple(origin(op) for op in maps) if role == "producer" else (origin(score), origin(publish))
        if tuple(op.origin.data for op in calls) != expected_sources:
            raise NotReady(f"prepared {role}: semantic stage order differs")
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
        by_origin = {call.origin: call for call in selected}
        selected_groups[role] = tuple(tuple(by_origin[call.origin.data] for call in group) for group in groups)
    if set(resolved) != {"producer", "consumer"}:
        raise NotReady("prepared plan: producer and consumer chains required")
    try:
        lowered = {source: resolution.lower() for source, resolution in local_resolutions.items()}
    except Exception as error:
        raise NotReady(f"local target lowering: {error}") from error
    return Prepared(
        accepted,
        text,
        json.dumps(
            {
                "schema": 2,
                "abi": "scalar-async-v1",
                "registry": registry.name,
                "bindings": manifest,
                "local_contracts": [
                    {
                        "origin": item.origin.data,
                        "input_codec": item.input_codec.data,
                        "output_codec": item.output_codec.data,
                        "resources": [
                            {name: value.data for name, value in row.data.items() if isinstance(value, StringAttr)}
                            for row in item.resources
                        ],
                        "outcomes": [name.data for name in item.outcomes],
                    }
                    for item in signatures
                ],
                **(
                    {
                        "local_lowering": "fused-v1" if any(len(group) > 1 for group in selected_groups["producer"]) else "unfused-v1",
                        "fused_groups": sum(len(group) > 1 for group in selected_groups["producer"]),
                    }
                    if fuse
                    else {}
                ),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        resolved["consumer"][0],
        resolved["consumer"][1],
        root.horizon.value.data,
        root.capacity.value.data,
        compile_local(tuple(tuple(Stage(call.origin, lowered[call.origin]) for call in group) for group in selected_groups["producer"])),
        MappingProxyType({source: resolution.contract for source, resolution in local_resolutions.items()}),
    )
