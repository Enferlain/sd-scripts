"""Contract-derived acceptance for the bounded shared language, without execution.

The report is a recomputable analysis, not readiness or an authorization token.
Input availability is deliberately absent from whole-lifetime wait edges.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import BlockArgument, Operation, OpResult, SSAValue
from xdsl.utils.exceptions import VerifyException

from .contracts import (
    AccessContract,
    AccessMode,
    Admission,
    CompletionControl,
    CompletionSemantics,
    Decision,
    DecisionSemantics,
    EndpointSemantics,
    FlowContract,
    FlowSemantics,
    Freshness,
    ItemSemantics,
    number,
    origin,
    ProductSemantics,
    Protocol,
    RankRule,
    RealizationSemantics,
    RegionContract,
    RegionSemantics,
    Reproducibility,
    ResourceAccess,
    ResourceSemantics,
    symbol,
    text,
    Version,
    VersionPolicy,
)
from .dialects import (
    CapabilityType,
    ClaimType,
    ConstantOp,
    EndpointType,
    OptionalPayloadType,
    ProgramOp,
    ResourceOp,
    ResourceType,
    RetainedType,
    YieldOp,
)
from .normalize import Normalized, Structure, normalize, source_index, symbol_index


class SemanticError(VerifyException):
    def __init__(self, code: str, source: str, detail: str):
        self.code, self.source = code, source
        super().__init__(f"{source}: [{code}] {detail}")


class TargetLimit(VerifyException):
    def __init__(self, target: str, source: str, detail: str):
        self.target, self.source = target, source
        super().__init__(f"{source}: target {target}: {detail}")


@dataclass(frozen=True)
class Comparison:
    boundary: str
    versions: tuple[Version, ...]


@dataclass(frozen=True)
class AdmissionResult:
    accepted: bool
    disposition: str
    observed: tuple[Version, ...]
    comparison: Comparison
    lags: tuple[tuple[str, int], ...] = ()


def evaluate_admission(policy: VersionPolicy, observed: Sequence[Version], comparison: Comparison) -> AdmissionResult:
    """Evaluate actual production provenance at the declared comparison boundary."""
    values = tuple(observed)

    def result(disposition: str, lags: tuple[tuple[str, int], ...] = ()) -> AdmissionResult:
        return AdmissionResult(disposition == "accepted", disposition, values, comparison, lags)

    if comparison.boundary != policy.boundary:
        return result("wrong_comparison_boundary")
    maps = []
    for versions in (values, comparison.versions):
        entries: dict[str, Version] = {}
        for version in versions:
            if (
                not version.owner
                or not version.lineage
                or type(version.incarnation) is not int
                or version.incarnation < 0
                or type(version.ordinal) is not int
                or version.ordinal < 0
                or (version.owner in entries and entries[version.owner] != version)
            ):
                return result("incomparable_provenance")
            entries[version.owner] = version
        maps.append(entries)
    actual, current = maps
    if set(actual) != set(policy.owners):
        return result("missing_or_extra_provenance")
    if policy.admission == Admission.ANY_WITH_PROVENANCE:
        return result("accepted")
    if policy.metric != "owner_update_ordinal_within_lineage":
        return result("unsupported_lag_metric")
    lags = []
    for owner in policy.owners:
        produced, now = actual[owner], current.get(owner)
        if now is None or (produced.incarnation, produced.lineage) != (now.incarnation, now.lineage):
            return result("incomparable_provenance")
        lag = now.ordinal - produced.ordinal
        if lag < 0:
            return result("incomparable_provenance")
        lags.append((owner, lag))
    allowed = 0 if policy.admission == Admission.EXACT_CURRENT else policy.max_lag
    return result("accepted" if all(lag <= allowed for _, lag in lags) else "rejected_stale", tuple(lags))


@dataclass(frozen=True)
class Obligation:
    source: str
    kind: str
    required: tuple[str, ...]


@dataclass(frozen=True)
class InputLink:
    flow: str
    producer: str
    consumers: tuple[str, ...]
    required_initial_owners: tuple[str, ...]


@dataclass(frozen=True)
class Contributor:
    source: str
    classification: Reproducibility
    reason: str
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Cut:
    region: str
    boundary: str


@dataclass(frozen=True)
class TargetProfile:
    name: str
    access_modes: frozenset[AccessMode]
    root_declarations: bool = False
    parallel_results: bool = True
    parallel_claims: bool = False
    policies: frozenset[str] = frozenset({"bounded_count"})
    snapshot_rules: frozenset[tuple[Freshness, str]] = frozenset(
        {
            (Freshness.PUBLISHED, "initial"),
            (Freshness.CURRENT_AT_START, "current"),
        }
    )


REFERENCE_TARGET = TargetProfile(
    "reference",
    frozenset({AccessMode.SNAPSHOT, AccessMode.PROTECTED_CURRENT, AccessMode.OWNER_UPDATE}),
    root_declarations=True,
    parallel_results=False,
)


Event = tuple[str, str]


class Waits:
    def __init__(self) -> None:
        self.edges: dict[Event, set[Event]] = defaultdict(set)

    def add(self, waiting: Event, required: Event) -> None:
        self.edges[waiting].add(required)

    def requires(self, waiting: Event, required: Event) -> bool:
        pending, seen = [waiting], set()
        while pending:
            node = pending.pop()
            if node == required:
                return True
            if node not in seen:
                seen.add(node)
                pending.extend(self.edges.get(node, ()))
        return False

    def check(self) -> None:
        complete: set[Event] = set()
        path: list[Event] = []

        def visit(node: Event) -> None:
            if node in path:
                cycle = [*path[path.index(node) :], node]
                raise SemanticError("settlement_cycle", node[0], " -> ".join(f"{key}.{phase}" for key, phase in cycle))
            if node in complete:
                return
            path.append(node)
            for required in sorted(self.edges.get(node, ())):
                visit(required)
            path.pop()
            complete.add(node)

        for node in sorted(self.edges):
            visit(node)


@dataclass(frozen=True)
class Verification:
    normalized: Normalized
    structure: Structure
    obligations: tuple[Obligation, ...]
    inputs: tuple[InputLink, ...]
    waits: tuple[tuple[Event, Event], ...]
    reproducibility: Reproducibility
    contributors: tuple[Contributor, ...]
    deferred: tuple[str, ...]


def _parents(op: Operation) -> tuple[Operation, ...]:
    result = []
    current: Operation | None = op
    while current is not None and not isinstance(current, ModuleOp):
        result.append(current)
        current = current.parent_op()
    return tuple(result)


class Analysis:
    def __init__(self, normalized: Normalized):
        self.normalized = normalize(normalized.module)
        self.index = source_index(self.normalized.module)
        self.symbols = symbol_index(self.normalized.module)
        self.root = next(iter(self.normalized.module.body.block.ops))
        self.regions: dict[Operation, RegionContract] = {}
        self.accesses: dict[Operation, AccessContract] = {}
        self.flows: dict[Operation, FlowContract] = {}
        self.controls: dict[Operation, CompletionControl] = {}
        self.decisions: dict[Operation, Decision] = {}
        self.obligations: list[Obligation] = []
        self.deferred = [
            "selected bindings, payload validation and owner initialization at preparation",
            "physical completion, device/collective ordering and cleanup evidence at execution",
            "resource usability and restoration require their owner protocols, including after failure",
            "declared rank distribution and coordinated quiescence are not implemented targets",
            "no proof of arbitrary selected-call termination or general deadlock freedom",
        ]
        for op in self.index.values():
            trait = op.get_trait(RegionSemantics)
            if trait is not None:
                contract = trait.describe(op)
                self.regions[op] = contract
                # A transparent realization has no new admission/decision.
                # Its original source stages remain the decision contributors.
                if not trait.transparent:
                    self.decisions[op] = contract.decision
            flow = op.get_trait(FlowSemantics)
            if flow is not None:
                self.flows[op] = flow.describe(op)
                self.decisions[op] = self.flows[op].decision
            use = op.get_trait(ResourceSemantics)
            if use is not None:
                self.accesses[op] = use.describe(op)
                self.decisions[op] = self.accesses[op].decision
            control = op.get_trait(CompletionSemantics)
            if control is not None:
                self.controls[op] = control.describe(op)
            declaration = op.get_trait(DecisionSemantics)
            if declaration is not None:
                self.decisions[op] = declaration.describe(op)
        for op, control in self.controls.items():
            source, target = self.symbols.get(control.source), self.symbols.get(control.target)
            expected = Protocol.REQUESTS if control.kind == "request" else Protocol.PRODUCER
            if (
                source is None
                or target is None
                or source not in self.regions
                or target not in self.regions
                or self.regions[target].protocol != expected
            ):
                self.error("completion_contract", op, "completion references must name an actual source region and compatible target")
            if control.kind == "request" and control.source not in self.regions[target].request_sources:
                self.error("request_source", op, "request source is absent from its actual handler contract")
        self.reachable = self.potential_entries()
        self.waits = Waits()

    def potential_entries(self) -> frozenset[Operation]:
        """Reachable templates, not fictional admitted handler invocations."""
        handlers: set[Operation] = set()
        while True:
            reachable: set[Operation] = set()

            def visit(op: Operation, found: set[Operation]) -> None:
                found.add(op)
                contract = self.regions.get(op)
                if contract is not None and (
                    (contract.protocol == Protocol.REQUESTS and op not in handlers)
                    or (contract.protocol == Protocol.REPEAT and contract.limit == 0)
                ):
                    return
                for region in op.regions:
                    for child in region.block.ops:
                        visit(child, found)

            visit(self.root, reachable)
            requested = {
                self.symbols[c.target]
                for op, c in self.controls.items()
                if c.kind == "request" and op in reachable and self.symbols[c.target] in reachable
            }
            if requested <= handlers:
                return frozenset(reachable)
            handlers.update(requested)

    def error(self, code: str, op: Operation, detail: str) -> NoReturn:
        raise SemanticError(code, origin(op), detail)

    def value_sources(self, value: SSAValue) -> frozenset[Operation]:
        """Conservative owner provenance over initialized, possibly cyclic SSA.

        A body carry can observe its initializer and prior iteration yields.
        A repeat result can also return its initializer after orderly stop.
        Visiting a carry backedge is not an uninitialized feedback error.
        """
        pending, seen, found = [value], set(), set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if isinstance(current, BlockArgument):
                parent, region = current.owner.parent_op(), current.owner.parent
                if parent is None or parent not in self.regions or region is None:
                    raise SemanticError("unbound_capture", "SSA", "reference has no registered entry mapping")
                contract = self.regions[parent]
                role = contract.roles[parent.regions.index(region)]
                capture = next((c for c in role.captures if c.argument == current.index), None)
                if capture is not None:
                    pending.append(parent.operands[capture.operand])
                elif contract.protocol == Protocol.ACCESS and current.index == 0:
                    found.add(parent)
                elif contract.protocol == Protocol.REPEAT:
                    split = len(parent.operands) - number(parent, "carry_count")
                    carry = current.index - split - 1
                    if not 0 <= carry < number(parent, "carry_count"):
                        self.error("unbound_capture", parent, "reference is not a declared loop carry")
                    pending.append(parent.operands[split + carry])
                    if contract.limit is None or contract.limit > 1:
                        terminator = parent.regions[0].block.last_op
                        if terminator is not None:
                            pending.append(terminator.operands[carry])
                else:
                    self.error("unbound_capture", parent, "reference is not a capture or issued capability")
                continue
            if isinstance(current, OpResult):
                parent = current.owner
                if parent in self.flows or isinstance(parent, ResourceOp):
                    found.add(parent)
                    continue
                endpoint = parent.get_trait(EndpointSemantics)
                item = parent.get_trait(ItemSemantics)
                if endpoint is not None and item is not None and current.index in item.describe(parent).creates:
                    pending.append(parent.operands[endpoint.describe(parent).operand])
                    continue
                if parent in self.regions:
                    contract = self.regions[parent]
                    if contract.protocol == Protocol.REPEAT:
                        split = len(parent.operands) - number(parent, "carry_count")
                        # Count zero or orderly stop before the first admitted
                        # iteration returns the original carry unchanged.
                        pending.append(parent.operands[split + current.index])
                        if contract.limit == 0:
                            continue
                    for region in parent.regions:
                        terminator = region.block.last_op
                        if terminator is not None and current.index < len(terminator.operands):
                            pending.append(terminator.operands[current.index])
                    continue
            raise SemanticError("unbound_capture", "SSA", "reference/capability has no maintained provider")
        if not found:
            raise SemanticError("unbound_capture", "SSA", "cyclic forwarding has no initialized owner/provider")
        return frozenset(found)

    def ownership(self) -> None:
        for op in self.index.values():
            if (
                isinstance(op, YieldOp)
                and isinstance(op.parent_op(), ProgramOp)
                and any(isinstance(value.type, CapabilityType | ClaimType | ResourceType | EndpointType) for value in op.operands)
            ):
                self.error("root_transfer", op, "program has no outgoing owner/claim transfer contract")
            contract = self.regions.get(op)
            item = op.get_trait(ItemSemantics)
            if (
                contract is None
                and not isinstance(op, ProgramOp | ResourceOp | ConstantOp | YieldOp)
                and not any(
                    op.get_trait(kind) is not None
                    for kind in (FlowSemantics, RealizationSemantics, ItemSemantics, ProductSemantics, CompletionSemantics)
                )
            ):
                self.error("semantic_contract", op, "registered definition lacks maintained region/leaf meaning")
            uses = item.describe(op) if item is not None else None
            if uses is not None:
                for indices, values in ((uses.borrows, op.operands), (uses.consumes, op.operands), (uses.creates, op.results)):
                    if len(indices) != len(set(indices)) or any(
                        not 0 <= index < len(values) or not isinstance(values[index].type, ClaimType) for index in indices
                    ):
                        self.error("item_contract", op, "claim positions do not match actual types")
                if any(not 0 <= index < len(op.results) for index in uses.detached_outputs):
                    self.error("item_contract", op, "detached result position is absent")
                if any(index not in uses.consumes for index in uses.forwards):
                    self.error("item_contract", op, "forwarded provenance requires a consumed claim")
                if (uses.borrows or uses.consumes) and any(
                    not isinstance(value.type, ClaimType) and index not in uses.detached_outputs for index, value in enumerate(op.results)
                ):
                    self.error("result_lifetime", op, "borrow/consume results need a declared detachment or retained-dependency contract")
            if contract is not None:
                for index, role in enumerate(contract.roles):
                    if role.index != index:
                        self.error("capture_mapping", op, "region role index does not match the actual region")
                    if len({c.argument for c in role.captures}) != len(role.captures):
                        self.error("capture_mapping", op, "an entry argument has several capture providers")
                    if any(
                        not 0 <= c.operand < len(op.operands)
                        or not 0 <= c.argument < len(role.entries)
                        or op.operands[c.operand].type != c.type
                        for c in role.captures
                    ):
                        self.error("capture_mapping", op, "capture provider/index/type does not match the actual operands")
                captured_claims = [
                    i
                    for i, value in enumerate(op.operands)
                    if isinstance(value.type, ClaimType) and (uses is None or i not in uses.consumes)
                ]
                for index in captured_claims:
                    carried = contract.protocol == Protocol.REPEAT and index >= len(op.operands) - number(op, "carry_count")
                    if any(sum(c.operand == index for c in role.captures) != (0 if carried else 1) for role in contract.roles):
                        self.error("capture_mapping", op, "each owned capture must have exactly one provider/use per selected body")
                    is_capture = any(c.operand == index for role in contract.roles for c in role.captures)
                    if is_capture and (
                        contract.protocol in (Protocol.PRODUCER, Protocol.REQUESTS)
                        or (contract.protocol == Protocol.PARALLEL and len(contract.roles) != 1)
                        or (contract.protocol == Protocol.REPEAT and contract.limit != 1)
                    ):
                        self.error(
                            "repeated_claim",
                            op,
                            "one claim cannot be copied across concurrent or repeated entries; use an explicit fork/carry",
                        )
            realization = op.get_trait(RealizationSemantics)
            if realization is not None:
                declared = realization.describe(op)
                if declared.completion != "physical_then_validate" or not {"returned", "invalid_output", "raised"}.issubset(
                    declared.outcomes
                ):
                    self.error("physical_boundary", op, "entered uses must retain physical completion and validation")
                if declared.result_lifetime not in ("detached_values", "scoped_retained") or any(
                    isinstance(v.type, CapabilityType | ResourceType | EndpointType | ClaimType | OptionalPayloadType) for v in op.results
                ):
                    self.error("result_lifetime", op, "this leaf contract returns detached values, not owner references or retained views")
            for value in (*op.operands, *op.results):
                if isinstance(value.type, ResourceType | EndpointType):
                    for provider in self.value_sources(value):
                        owner = provider.parent_op()
                        if owner is None or owner is op or not owner.is_ancestor(op):
                            self.error("owner_lifetime", op, "reference leaves its declaration's owning scope without a transfer")
                if (
                    isinstance(value.type, ResourceType)
                    and value in op.operands
                    and contract is None
                    and op not in self.accesses
                    and not isinstance(op, YieldOp)
                ):
                    self.error("unscoped_resource", op, "resource use requires an explicit access realization")
                if isinstance(value.type, EndpointType) and value in op.operands and contract is None and not isinstance(op, YieldOp):
                    endpoint = op.get_trait(EndpointSemantics)
                    if endpoint is None or op.operands[endpoint.describe(op).operand] is not value:
                        self.error("unscoped_endpoint", op, "endpoint use requires a maintained production/admission contract")
                if not isinstance(value.type, CapabilityType):
                    continue
                for scope in self.value_sources(value):
                    if scope not in self.accesses or scope is op or not scope.is_ancestor(op):
                        self.error("capability_escape", op, "scoped capability leaves its issuing access region")
                if contract is None and realization is None and not isinstance(op, YieldOp):
                    self.error("capability_use", op, "capability use has no maintained scoped-use contract")
            for region in op.regions:
                block = region.block
                positions = {child: i for i, child in enumerate(block.ops)}
                values = [*block.args, *(result for child in block.ops for result in child.results)]
                for value in values:
                    if not isinstance(value.type, ClaimType):
                        continue
                    if isinstance(value, OpResult):
                        creator = value.owner.get_trait(ItemSemantics)
                        if value.owner not in self.regions and (
                            creator is None or value.index not in creator.describe(value.owner).creates
                        ):
                            self.error("claim_provider", value.owner, "claim has no maintained admission/transfer contract")
                    terminal, borrows = [], []
                    for use in value.uses:
                        consumer = use.operation
                        if consumer.parent_block() is not block:
                            self.error("claim_escape", consumer, "claim bypasses explicit region capture")
                        if consumer in self.regions or isinstance(consumer, YieldOp):
                            terminal.append(consumer)
                            continue
                        rule = consumer.get_trait(ItemSemantics)
                        described = rule.describe(consumer) if rule is not None else None
                        if described is not None and use.index in described.consumes:
                            terminal.append(consumer)
                        elif described is not None and use.index in described.borrows:
                            borrows.append(consumer)
                        else:
                            self.error("claim_use", consumer, "claim operand has no borrow/consume/transfer contract")
                    if len(terminal) != 1:
                        self.error("affine_claim", op, f"claim requires exactly one consume/transfer per path, found {len(terminal)}")
                    if any(positions[borrow] >= positions[terminal[0]] for borrow in borrows):
                        self.error("claim_after_consume", terminal[0], "borrow follows claim consumption/transfer")
            if isinstance(op, YieldOp):
                parent = op.parent_op()
                if parent in self.accesses and any(isinstance(value.type, CapabilityType) for value in op.operands):
                    self.error("capability_escape", op, "access yield cannot release protection while returning its capability")
        self.obligations.append(
            Obligation(origin(self.root), "ownership", ("affine per invocation/path", "scoped capability extent", "detached leaf results"))
        )

    def retained_products(self) -> None:
        """One lexical profile; broader transfers need another contract."""
        for op in self.index.values():
            realization = op.get_trait(RealizationSemantics)
            declared = realization.describe(op) if realization is not None else None
            retained = [value for value in op.results if isinstance(value.type, RetainedType)]
            if (retained or (declared is not None and declared.retains)) and (
                declared is None
                or len(op.results) != 1
                or len(retained) != 1
                or not declared.retains
                or len(set(declared.retains)) != len(declared.retains)
                or any(
                    not 0 <= i < len(op.operands) or not isinstance(op.operands[i].type, CapabilityType | RetainedType)
                    for i in declared.retains
                )
            ):
                self.error("retained_dependency", op, "one retained result requires explicit capability/product dependency positions")
        # Validate every declaration before following a parent's borrower edges.
        for op in self.index.values():
            realization = op.get_trait(RealizationSemantics)
            retained = [value for value in op.results if isinstance(value.type, RetainedType)]
            for value in op.operands:
                if not isinstance(value.type, RetainedType):
                    continue
                if not isinstance(value, OpResult) or value.owner.parent_block() is not op.parent_block():
                    self.error("retained_escape", op, "scoped retained products cannot cross their defining ordered body")
                if realization is None and op.get_trait(ProductSemantics) is None:
                    self.error(
                        "retained_escape", op, "retained products require physical borrowing or explicit release; no implicit transfer"
                    )
            for product in retained:
                block = op.parent_block()
                if block is None:
                    self.error("retained_dependency", op, "retained result has no owning body")
                positions = {child: i for i, child in enumerate(block.ops)}
                releases, borrowers = [], []
                for use in product.uses:
                    consumer = use.operation
                    if consumer.parent_block() is not block:
                        self.error("retained_escape", consumer, f"product from {origin(op)} bypasses its body extent")
                    release = consumer.get_trait(ProductSemantics)
                    if release is not None and use.index in release.describe(consumer):
                        releases.append(consumer)
                    elif consumer.get_trait(RealizationSemantics) is not None:
                        borrowers.append(consumer)
                    else:
                        self.error("retained_escape", consumer, f"product from {origin(op)} has no retained borrow/release contract")
                if len(releases) != 1:
                    self.error("retained_release", op, f"retained product requires exactly one release, found {len(releases)}")
                if any(positions[borrower] >= positions[releases[0]] for borrower in borrowers):
                    self.error("retained_after_release", releases[0], f"release precedes a dependent use of {origin(op)}")
                for borrower in borrowers:
                    contract = borrower.get_trait(RealizationSemantics)
                    if contract is None:
                        continue
                    for i in contract.describe(borrower).retains:
                        if borrower.operands[i] is not product:
                            continue
                        child = borrower.results[0]
                        for use in child.uses:
                            if use.operation.get_trait(ProductSemantics) is not None and (
                                use.operation.parent_block() is not block or positions[use.operation] >= positions[releases[0]]
                            ):
                                self.error("retained_release", releases[0], f"derived product {origin(borrower)} must release first")
        if any(isinstance(value.type, RetainedType) for op in self.index.values() for value in op.results):
            self.obligations.append(
                Obligation(
                    origin(self.root),
                    "retained_products",
                    ("lexical read-only borrowing", "explicit dependencies and release", "physical-use settlement before cleanup"),
                )
            )

    def retained_effects(self, bindings) -> None:
        if not any(isinstance(value.type, RetainedType) for op in self.index.values() for value in op.results):
            return

        def owners(value: SSAValue) -> frozenset[Operation]:
            if isinstance(value.type, CapabilityType):
                return frozenset(
                    owner
                    for issuer in self.value_sources(value)
                    if self.accesses[issuer].mode != AccessMode.SNAPSHOT
                    for owner in self.value_sources(self.accesses[issuer].subject)
                )
            if isinstance(value, OpResult) and isinstance(value.type, RetainedType):
                rule = value.owner.get_trait(RealizationSemantics)
                if rule is not None:
                    return frozenset(owner for i in rule.describe(value.owner).retains for owner in owners(value.owner.operands[i]))
            self.error("retained_dependency", value.owner if isinstance(value, OpResult) else self.root, "unresolved product dependency")

        def visit(block, inherited):
            live = dict(inherited)
            for op in block.ops:
                if op.get_trait(ProductSemantics) is not None:
                    live.pop(op.operands[0], None)
                rule = op.get_trait(RealizationSemantics)
                if rule is not None:
                    binding = bindings.get(symbol(op.properties["binding"]))
                    if binding is not None:
                        for requirement in binding.effects.resources:
                            if requirement.access != ResourceAccess.UPDATE:
                                continue
                            updated = owners(op.operands[requirement.argument])
                            for product, required in live.items():
                                if updated & required:
                                    self.error(
                                        "retained_update", op, f"update conflicts with live retained product {origin(product.owner)}"
                                    )
                    if rule.describe(op).retains:
                        live[op.results[0]] = owners(op.results[0])
                for region in op.regions:
                    visit(region.block, live)

        visit(self.root.regions[0].block, {})

    def closure_source(self, waiting: Operation, source: Operation) -> Operation:
        if source.is_ancestor(waiting):
            self.error("settlement_cycle", waiting, "waiting for an enclosing source's settlement is circular")
        common = next(parent for parent in _parents(waiting) if parent in _parents(source))
        current, between = source.parent_op(), []
        while current is not None and current is not common:
            between.append(current)
            current = current.parent_op()
        for ancestor in reversed(between):
            contract = self.regions.get(ancestor)
            if contract is not None and contract.protocol in (Protocol.REPEAT, Protocol.REQUESTS, Protocol.PRODUCER, Protocol.CHOICE):
                return ancestor
        return source

    def lifecycle(self) -> None:
        requests = [(op, control) for op, control in self.controls.items() if control.kind == "request"]
        for op, contract in self.regions.items():
            region = op.get_trait(RegionSemantics)
            admission = (
                "enclosing_invocation"
                if region is not None and region.transparent and contract.protocol == Protocol.SEQUENCE
                else "explicit_invocation"
            )
            if (
                contract.admission != admission
                or contract.interruption != "close_admission_then_settle"
                or contract.settlement != "physical_cleanup_and_owned_products"
            ):
                self.error("lifecycle_contract", op, "unsupported admission, interruption or settlement relationship")
            if contract.completion != "required_children_and_sources_closed":
                self.error("pending_coverage", op, "completion must cover required children, sources and accepted pending obligations")
            supported_cuts = {"settled"}
            if contract.protocol == Protocol.REPEAT:
                supported_cuts.add("between_settled_iterations")
            if any(boundary not in supported_cuts for boundary in contract.quiescence):
                self.error("quiescence_contract", op, "boundary has no defined cut meaning for this maintained protocol")
            if contract.protocol == Protocol.REQUESTS and op in self.reachable:
                for name in contract.request_sources:
                    source = self.closure_source(op, self.symbols[name])
                    self.waits.add((origin(op), "settled"), (origin(source), "settled"))
                self.obligations.append(
                    Obligation(
                        origin(op),
                        "request_closure",
                        (*contract.request_sources, "pending/admitted dispositions", "no fictional handler invocation"),
                    )
                )
        for op in self.index.values():
            if op not in self.reachable:
                continue
            key = origin(op)
            self.waits.add((key, "settled"), (key, "entry"))
            for i, region in enumerate(op.regions):
                entered, settled = (key, f"body{i}.entry"), (key, f"body{i}.settled")
                self.waits.add(entered, (key, "entry"))
                contract = self.regions.get(op)
                entered_body = not (
                    contract is not None
                    and (
                        (contract.protocol == Protocol.REPEAT and contract.limit == 0)
                        or (contract.protocol == Protocol.REQUESTS and not any(child in self.reachable for child in region.block.ops))
                    )
                )
                if entered_body:
                    self.waits.add((key, "settled"), settled)
                previous = None
                for child in region.block.ops:
                    if child not in self.reachable:
                        continue
                    self.waits.add((origin(child), "entry"), entered)
                    if previous is not None:
                        self.waits.add((origin(child), "entry"), (origin(previous), "settled"))
                    self.waits.add(settled, (origin(child), "settled"))
                    previous = child
        for op, control in requests:
            source, target = self.symbols[control.source], self.symbols[control.target]
            if source is op or not source.is_ancestor(op):
                self.error("request_source", op, "completion request must be armed within its actual source invocation")
            if (
                control.boundary != "settled_success"
                or control.registration != "before_source_terminal"
                or control.orderly_stop != "retain_admitted_promises"
            ):
                self.error(
                    "request_registration",
                    op,
                    "completion visibility must follow owned registration/disposition, including after orderly stop",
                )
            if op not in self.reachable:
                continue
            if target not in self.reachable:
                self.error("pending_coverage", op, "request has no reachable handler service or accepted transfer/disposition")
            for parent in _parents(source):
                contract = self.regions.get(parent)
                if isinstance(parent, ProgramOp) or (contract is not None and contract.protocol == Protocol.PARALLEL):
                    self.waits.add((origin(parent), "settled"), (origin(target), "settled"))
            self.obligations.append(
                Obligation(origin(op), "completion_request", (control.source, control.target, control.registration, control.orderly_stop))
            )
        for op, control in self.controls.items():
            if op not in self.reachable:
                continue
            if control.kind == "stop":
                source = self.closure_source(op, self.symbols[control.source])
                self.waits.add((origin(op), "settled"), (origin(source), "settled"))
            elif control.kind != "request":
                self.error("completion_control", op, "unknown maintained completion-control kind")
        self.waits.check()

    def claim_dispositions(self, value: SSAValue, seen: frozenset[SSAValue] = frozenset()) -> frozenset[Operation]:
        """Follow actual ownership transfers to final disposition operations."""
        if value in seen:
            return frozenset()
        seen = seen | {value}
        ends: set[Operation] = set()
        for use in value.uses:
            op = use.operation
            if op not in self.reachable:
                continue
            item = op.get_trait(ItemSemantics)
            if item is not None and use.index in item.describe(op).consumes:
                ends.add(op)
            contract = self.regions.get(op)
            if contract is not None:
                for role in contract.roles:
                    for capture in role.captures:
                        if capture.operand == use.index:
                            ends.update(self.claim_dispositions(op.regions[role.index].block.args[capture.argument], seen))
                if contract.protocol == Protocol.REPEAT:
                    count = number(op, "carry_count")
                    split = len(op.operands) - count
                    if use.index >= split:
                        carry = use.index - split
                        forwarded = op.results[carry] if contract.limit == 0 else op.regions[0].block.args[split + 1 + carry]
                        ends.update(self.claim_dispositions(forwarded, seen))
            if isinstance(op, YieldOp):
                parent = op.parent_op()
                if parent is not None and parent in self.regions and use.index < len(parent.results):
                    ends.update(self.claim_dispositions(parent.results[use.index], seen))
                    if self.regions[parent].protocol == Protocol.REPEAT and self.regions[parent].limit != 1:
                        split = len(parent.operands) - number(parent, "carry_count")
                        ends.update(self.claim_dispositions(parent.regions[0].block.args[split + 1 + use.index], seen))
        return frozenset(ends)

    def flow_links(self) -> tuple[InputLink, ...]:
        producers: dict[Operation, list[Operation]] = defaultdict(list)
        consumers: dict[Operation, list[Operation]] = defaultdict(list)
        for op in self.index.values():
            trait = op.get_trait(EndpointSemantics)
            if trait is None:
                continue
            use = trait.describe(op)
            if (
                not 0 <= use.operand < len(op.operands)
                or not isinstance(op.operands[use.operand].type, EndpointType)
                or use.role not in ("produce", "consume")
            ):
                self.error("endpoint_contract", op, "endpoint role does not match an actual typed operand")
            if use.role == "consume":
                item = op.get_trait(ItemSemantics)
                if item is None or len(item.describe(op).creates) != 1:
                    self.error("admission_claim", op, "admission must issue the owned claim needed to check disposition")
                claimed = op.results[item.describe(op).creates[0]].type
                endpoint = op.operands[use.operand].type
                if not isinstance(claimed, ClaimType) or not isinstance(endpoint, EndpointType) or claimed.payload != endpoint.payload:
                    self.error("admission_claim", op, "issued claim must match the admitted endpoint payload")
            for flow in self.value_sources(op.operands[use.operand]):
                if flow not in self.flows:
                    self.error("endpoint_provider", op, "endpoint does not resolve to a declared flow")
                (producers if use.role == "produce" else consumers)[flow].append(op)
        dependencies_by_flow: dict[Operation, set[str]] = {flow: set() for flow in self.flows}
        forwarded: dict[Operation, set[Operation]] = {flow: set() for flow in self.flows}
        for flow, candidates in producers.items():
            for producer in candidates:
                for access, use in self.accesses.items():
                    if producer.is_ancestor(access):
                        dependencies_by_flow[flow].update(text(owner, "sym_name") for owner in self.value_sources(use.subject))
                for child in producer.walk():
                    item = child.get_trait(ItemSemantics)
                    if item is not None:
                        for position in item.describe(child).forwards:
                            forwarded[flow].update(self.value_sources(child.operands[position]))
        # Forwarding preserves actual upstream versions, rather than sampling
        # an unrelated current version when the output happens to be published.
        while True:
            changed = False
            for flow, upstream in forwarded.items():
                before = len(dependencies_by_flow[flow])
                for source in upstream:
                    if source not in dependencies_by_flow:
                        self.error("forwarding_provider", flow, f"forwarded claim resolves to non-flow provider {origin(source)}")
                    dependencies_by_flow[flow].update(dependencies_by_flow[source])
                changed |= before != len(dependencies_by_flow[flow])
            if not changed:
                break
        inputs = []
        for flow, contract in self.flows.items():
            if not producers[flow] and not consumers[flow]:
                continue
            if len(producers[flow]) != 1:
                self.error("flow_producer", flow, "this flow contract requires one actual producer")
            producer = producers[flow][0]
            active_consumers = [consumer for consumer in consumers[flow] if consumer in self.reachable]
            dependencies = dependencies_by_flow[flow]
            if dependencies != set(contract.versions.owners) or len(set(contract.versions.owners)) != len(contract.versions.owners):
                self.error("production_versions", flow, "declared owners do not match actual production access/capture dependencies")
            if not contract.versions.boundary or not contract.versions.metric:
                self.error("version_policy", flow, "admission requires an explicit comparison boundary and owner-specific lag metric")
            if contract.versions.boundary != "item_admission" or contract.versions.metric != "owner_update_ordinal_within_lineage":
                raise TargetLimit("version analysis", origin(flow), "comparison boundary or lag metric is not installed")
            if producer not in self.reachable:
                if active_consumers:
                    self.error("flow_initialization", flow, "admission has no reachable producer or declared initialized input")
                continue
            stops = [
                (op, c)
                for op, c in self.controls.items()
                if op in self.reachable and c.kind == "stop" and self.symbols[c.target] is producer
            ]
            if not stops:
                raise TargetLimit(
                    "normal settlement",
                    origin(producer),
                    "open producer has no declared completion control; external cancellation is not a normal-completion proof",
                )
            for stop, control in stops:
                source = self.closure_source(stop, self.symbols[control.source])
                for consumer in active_consumers:
                    lifetime = consumer.parent_op()
                    if lifetime is None or not self.waits.requires((origin(source), "settled"), (origin(lifetime), "settled")):
                        self.error("stop_coverage", stop, f"@{control.source} does not cover consumer/request lifetime {origin(consumer)}")
                self.obligations.append(
                    Obligation(
                        origin(stop),
                        "producer_stop",
                        (origin(source), origin(producer), "request stop without awaiting producer termination"),
                    )
                )
            progressing = [stop for stop, _ in stops if not self.waits.requires((origin(stop), "settled"), (origin(producer), "settled"))]
            if not progressing:
                self.error("settlement_cycle", producer, "every stop control waits for the producer's own termination")
            mandatory = []
            for stop in progressing:
                common = next(parent for parent in _parents(stop) if parent in _parents(producer))
                conditional = False
                for parent in _parents(stop):
                    if parent is common:
                        break
                    rule = self.regions.get(parent)
                    if rule is not None and (
                        rule.protocol in (Protocol.CHOICE, Protocol.REQUESTS, Protocol.PRODUCER)
                        or (rule.protocol == Protocol.REPEAT and (rule.limit == 0 or rule.policy != "bounded_count"))
                    ):
                        conditional = True
                if not conditional:
                    mandatory.append(stop)
            if not mandatory:
                raise TargetLimit(
                    "normal settlement", origin(producer), "conditional stop alternatives require a disjunctive progress proof"
                )
            self.waits.add((origin(producer), "settled"), (origin(mandatory[0]), "settled"))
            for consumer in active_consumers:
                item = consumer.get_trait(ItemSemantics)
                if item is None:
                    self.error("admission_claim", consumer, "missing claim contract")
                for index in item.describe(consumer).creates:
                    ends = self.claim_dispositions(consumer.results[index])
                    if not ends:
                        self.error("claim_disposition", consumer, "admitted claim has no supported final disposition path")
                    for end in ends:
                        self.waits.add((origin(producer), "settled"), (origin(end), "settled"))
                    self.obligations.append(Obligation(origin(consumer), "producer_drain", tuple(sorted(origin(end) for end in ends))))
            inputs.append(
                InputLink(origin(flow), origin(producer), tuple(origin(c) for c in active_consumers), tuple(sorted(dependencies)))
            )
        self.waits.check()
        return tuple(inputs)

    def access_rules(self) -> None:
        for op, use in self.accesses.items():
            sources = self.value_sources(use.subject)
            if any(not isinstance(owner, ResourceOp) for owner in sources):
                self.error("resource_provider", op, "access must resolve to an actual owner declaration")
            valid = (
                (use.mode == AccessMode.SNAPSHOT and use.freshness in (Freshness.PUBLISHED, Freshness.CURRENT_AT_START))
                or (use.mode == AccessMode.PROTECTED_CURRENT and use.freshness == Freshness.CURRENT_AT_START)
                or (use.mode in (AccessMode.OWNER_UPDATE, AccessMode.GRANT) and use.freshness == Freshness.OWNER_PROTOCOL)
            )
            if not valid or not use.version_selector or not use.profile or use.extent != "body_and_admitted_uses_settled":
                self.error(
                    "access_contract", op, "access realization, separate freshness, selection, profile and retained extent must agree"
                )
            if use.freshness == Freshness.CURRENT_AT_START and use.version_selector != "current":
                self.error("freshness_selector", op, "current-at-start requires an atomic current selection")
            if op in self.reachable:
                for parent in _parents(op)[1:]:
                    held = self.accesses.get(parent)
                    if held is None or parent not in self.reachable:
                        continue
                    # Published snapshots need no live-owner acquisition;
                    # snapshot scopes retain an isolated copy, not a live lock.
                    reads_live = use.mode == AccessMode.PROTECTED_CURRENT or (
                        use.mode == AccessMode.SNAPSHOT and use.freshness == Freshness.CURRENT_AT_START
                    )
                    conflicts = (held.mode == AccessMode.PROTECTED_CURRENT and use.mode == AccessMode.OWNER_UPDATE) or (
                        held.mode == AccessMode.OWNER_UPDATE and (reads_live or use.mode == AccessMode.OWNER_UPDATE)
                    )
                    outer_sources = self.value_sources(held.subject)
                    if not conflicts or not sources & outer_sources:
                        continue
                    if len(sources) == len(outer_sources) == 1:
                        self.error(
                            "nested_access_cycle",
                            op,
                            f"incompatible acquisition needs {origin(parent)} to release, but that scope retains protection through this body",
                        )
                    raise TargetLimit(
                        "nested access analysis",
                        origin(op),
                        f"possible owner aliases overlap {origin(parent)}; compatible acquisition cannot be established",
                    )
            owners = tuple(sorted(text(owner, "sym_name") for owner in sources))
            self.obligations.append(
                Obligation(origin(op), "access_evidence", (*owners, use.mode, use.freshness, use.version_selector, use.extent))
            )

    def overlap(self, first: Operation, second: Operation) -> bool:
        if first is second:
            return False
        if self.waits.requires((origin(first), "entry"), (origin(second), "settled")) or self.waits.requires(
            (origin(second), "entry"), (origin(first), "settled")
        ):
            return False
        common = next(parent for parent in _parents(first) if parent in _parents(second))
        contract = self.regions.get(common)
        return contract is not None and contract.protocol == Protocol.PARALLEL

    def reproducibility(self, inputs: tuple[InputLink, ...]) -> tuple[Reproducibility, tuple[Contributor, ...]]:
        contributors = [
            Contributor(origin(op), d.reproducibility, "declared computation/control/input/admission", d.assumptions)
            for op, d in self.decisions.items()
        ]
        writers = [
            (op, self.value_sources(use.subject))
            for op, use in self.accesses.items()
            if use.mode in (AccessMode.OWNER_UPDATE, AccessMode.GRANT)
        ]
        for op, use in self.accesses.items():
            if use.mode not in (AccessMode.SNAPSHOT, AccessMode.PROTECTED_CURRENT) or use.freshness == Freshness.PUBLISHED:
                continue
            owners = self.value_sources(use.subject)
            if any(owners & subjects and self.overlap(op, writer) for writer, subjects in writers):
                contributors.append(
                    Contributor(
                        origin(op),
                        Reproducibility.MAY_DEPEND,
                        "current version can race owner update; protection preserves safety, not version assignment",
                    )
                )
        for link in inputs:
            flow = self.index[link.flow]
            contract = self.flows[flow]
            takes = [self.index[name] for name in link.consumers]
            admission_race = contract.versions.admission != Admission.ANY_WITH_PROVENANCE and any(
                owner in self.symbols and self.symbols[owner] in subjects and self.overlap(take, writer)
                for owner in contract.versions.owners
                for take in takes
                for writer, subjects in writers
            )
            if admission_race or any(self.overlap(a, b) for i, a in enumerate(takes) for b in takes[i + 1 :]):
                contributors.append(
                    Contributor(
                        origin(flow), Reproducibility.MAY_DEPEND, "input association or version-admission outcome can depend on scheduling"
                    )
                )
        classes = {c.classification for c in contributors}
        classification = (
            Reproducibility.UNKNOWN
            if Reproducibility.UNKNOWN in classes
            else Reproducibility.MAY_DEPEND
            if Reproducibility.MAY_DEPEND in classes
            else Reproducibility.INDEPENDENT
        )
        return classification, tuple(contributors)

    def claims(self, replicated: bool, cuts: Sequence[Cut], target: TargetProfile | None) -> None:
        if replicated:
            for op, decision in self.decisions.items():
                if decision.rank not in (RankRule.REPLICATED, RankRule.DISTRIBUTED):
                    raise TargetLimit(
                        "replicated decisions",
                        origin(op),
                        "decision needs replicated determinism or distribution before dependent collective work",
                    )
                if decision.rank == RankRule.DISTRIBUTED:
                    self.obligations.append(
                        Obligation(origin(op), "rank_distribution", ("distribute before every dependent collective action",))
                    )
        for cut in cuts:
            op = self.symbols.get(cut.region)
            if op is None or op not in self.regions or cut.boundary not in self.regions[op].quiescence:
                raise TargetLimit(
                    "quiescence claim", cut.region, f"no declared boundary {cut.boundary}; exact restoration/transition unavailable"
                )
            self.obligations.append(
                Obligation(origin(op), "declared_cut", (cut.boundary, "coordinated state/pending-work capture remains target evidence"))
            )
        if target is None:
            return
        for op, contract in self.regions.items():
            if contract.protocol == Protocol.PARALLEL and op.results and not target.parallel_results:
                raise TargetLimit(target.name, origin(op), "parallel result merging is unsupported")
            if (
                contract.protocol == Protocol.PARALLEL
                and not target.parallel_claims
                and any(isinstance(value.type, ClaimType) for value in op.operands)
            ):
                raise TargetLimit(target.name, origin(op), "parallel claim transfer is unsupported")
            if contract.protocol == Protocol.REPEAT and contract.policy not in target.policies:
                raise TargetLimit(target.name, origin(op), "selected repeat policy is unsupported")
        for op, use in self.accesses.items():
            if use.mode not in target.access_modes:
                raise TargetLimit(target.name, origin(op), f"access realization {use.mode} is unsupported")
            if use.mode == AccessMode.SNAPSHOT and (use.freshness, use.version_selector) not in target.snapshot_rules:
                raise TargetLimit(target.name, origin(op), "snapshot selector is unsupported")
        if target.root_declarations:
            for op in self.index.values():
                if (isinstance(op, ResourceOp) or op in self.flows) and op.parent_op() is not self.root:
                    raise TargetLimit(target.name, origin(op), "persistent declarations require root ownership")


def verify(program: Normalized, *, replicated: bool = False, cuts: Sequence[Cut] = (), target: TargetProfile | None = None) -> Verification:
    """Recompute from detached IR; no bindings, runtime observations or factories."""
    analysis = Analysis(program)
    analysis.ownership()
    analysis.retained_products()
    analysis.access_rules()
    analysis.lifecycle()
    inputs = analysis.flow_links()
    classification, contributors = analysis.reproducibility(inputs)
    analysis.claims(replicated, cuts, target)
    return Verification(
        analysis.normalized,
        analysis.normalized.structure,
        tuple(analysis.obligations),
        inputs,
        tuple(sorted((waiting, required) for waiting, values in analysis.waits.edges.items() for required in values)),
        classification,
        contributors,
        tuple(analysis.deferred),
    )
