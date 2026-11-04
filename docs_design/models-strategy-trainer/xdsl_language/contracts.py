"""Typed contract queries over real IR, without a whole-program case registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, SymbolRefAttr
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.traits import OpTrait
from xdsl.utils.exceptions import VerifyException


class Protocol(StrEnum):
    ONCE = "once"
    SEQUENCE = "sequence"
    PARALLEL = "parallel"
    REPEAT = "repeat"
    REQUESTS = "requests"
    PRODUCER = "producer"
    ACCESS = "access"
    CHOICE = "choice"


class AccessMode(StrEnum):
    SNAPSHOT = "snapshot"
    PROTECTED_CURRENT = "protected_current"
    OWNER_UPDATE = "owner_update"
    GRANT = "grant"


class Freshness(StrEnum):
    PUBLISHED = "published"
    CURRENT_AT_START = "current_at_start"
    OWNER_PROTOCOL = "owner_protocol"


class Admission(StrEnum):
    EXACT_CURRENT = "exact_current"
    BOUNDED_LAG = "bounded_lag"
    ANY_WITH_PROVENANCE = "any_with_provenance"


class Reproducibility(StrEnum):
    INDEPENDENT = "schedule_independent"
    MAY_DEPEND = "may_depend_on_schedule"
    UNKNOWN = "unknown"


class RankRule(StrEnum):
    REPLICATED = "replicated_deterministic"
    DISTRIBUTED = "decided_once_and_distributed"
    LOCAL = "local"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Decision:
    reproducibility: Reproducibility
    rank: RankRule
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Capture:
    operand: int
    argument: int
    type: Attribute


@dataclass(frozen=True)
class RegionRole:
    index: int
    entries: tuple[Attribute, ...]
    captures: tuple[Capture, ...]
    invocation_arguments: tuple[str, ...]
    yields: tuple[Attribute, ...]


@dataclass(frozen=True)
class RegionContract:
    origin: str
    symbol: str
    version: int
    protocol: Protocol
    roles: tuple[RegionRole, ...]
    request_sources: tuple[str, ...]
    decision: Decision
    quiescence: tuple[str, ...]
    limit: int | None = None
    max_outstanding: int | None = None
    max_active: int | None = None
    policy: str | None = None
    admission: str = "explicit_invocation"
    completion: str = "required_children_and_sources_closed"
    interruption: str = "close_admission_then_settle"
    settlement: str = "physical_cleanup_and_owned_products"


@dataclass(frozen=True)
class AccessContract:
    subject: SSAValue
    mode: AccessMode
    freshness: Freshness
    version_selector: str
    profile: str
    decision: Decision
    extent: str = "body_and_admitted_uses_settled"


@dataclass(frozen=True)
class VersionPolicy:
    admission: Admission
    owners: tuple[str, ...]
    metric: str
    boundary: str
    max_lag: int
    incomparable: str = "reject_with_disposition"


@dataclass(frozen=True)
class FlowContract:
    origin: str
    payload: Attribute
    max_outstanding: int
    ready_bound: int
    versions: VersionPolicy
    decision: Decision


@dataclass(frozen=True)
class RealizationContract:
    binding: str
    version: str
    inputs: tuple[Attribute, ...]
    outputs: tuple[Attribute, ...]
    completion: str = "physical_then_validate"
    outcomes: tuple[str, ...] = ("returned", "invalid_output", "raised")
    decision: Decision = Decision(Reproducibility.UNKNOWN, RankRule.UNKNOWN)
    result_lifetime: str = "detached_values"


@dataclass(frozen=True)
class Version:
    """An observed owner version, never a request or global progress counter."""

    owner: str
    incarnation: int
    lineage: str
    ordinal: int


@dataclass(frozen=True)
class ItemContract:
    borrows: tuple[int, ...] = ()
    consumes: tuple[int, ...] = ()
    creates: tuple[int, ...] = ()
    detached_outputs: tuple[int, ...] = ()


@dataclass(frozen=True)
class ItemSemantics(OpTrait):
    contract: ItemContract

    def describe(self, op: Operation) -> ItemContract:
        return self.contract


@dataclass(frozen=True)
class EndpointUse:
    operand: int
    role: str


@dataclass(frozen=True)
class EndpointSemantics(OpTrait):
    contract: EndpointUse

    def describe(self, op: Operation) -> EndpointUse:
        return self.contract


@dataclass(frozen=True)
class CompletionControl:
    kind: str
    source: str
    target: str
    boundary: str
    registration: str = "before_source_terminal"
    orderly_stop: str = "retain_admitted_promises"


@dataclass(frozen=True)
class CompletionSemantics(OpTrait):
    kind: str
    target_field: str

    def describe(self, op: Operation) -> CompletionControl:
        source, target = op.properties.get("source"), op.properties.get(self.target_field)
        if source is None or target is None:
            raise VerifyException(f"{origin(op)}: completion control requires source and target")
        return CompletionControl(self.kind, symbol(source), symbol(target), text(op, "boundary") if self.kind == "request" else "settled")


class DecisionSemantics(OpTrait):
    """The same descriptor for leaf calls and completion decisions."""

    def describe(self, op: Operation) -> Decision:
        return decision(op)


@dataclass(frozen=True)
class Reference:
    field: str
    target: str
    kind: str
    payload: Attribute | None = None
    completion_source: str | None = None


@dataclass(frozen=True)
class ReferenceSemantics(OpTrait):
    fields: tuple[tuple[str, str], ...]

    def describe(self, op: Operation) -> tuple[Reference, ...]:
        refs = []
        for field, kind in self.fields:
            value = op.properties.get(field)
            if value is None:
                raise VerifyException(f"{origin(op)}: reference contract requires {field}")
            if kind == "requests" and (not op.operands or op.properties.get("source") is None):
                raise VerifyException(f"{origin(op)}: request reference requires a payload operand and completion source")
            values = tuple(value) if isinstance(value, ArrayAttr) else (value,)
            refs.extend(
                Reference(
                    field,
                    symbol(item),
                    kind,
                    op.operands[0].type if kind == "requests" else None,
                    symbol(op.properties["source"]) if kind == "requests" else None,
                )
                for item in values
            )
        return tuple(refs)


class ExpandSequence(OpTrait):
    """Maintained recipe expansion into the common sequence protocol."""


def text(op: Operation, key: str) -> str:
    value = op.properties.get(key)
    if not isinstance(value, StringAttr):
        raise VerifyException(f"{op.name}: {key} must be a string")
    return value.data


def number(op: Operation, key: str) -> int:
    value = op.properties.get(key)
    if not isinstance(value, IntegerAttr):
        raise VerifyException(f"{op.name}: {key} must be an integer")
    return value.value.data


def strings(op: Operation, key: str) -> tuple[str, ...]:
    value = op.properties.get(key)
    if not isinstance(value, ArrayAttr) or any(not isinstance(x, StringAttr) for x in value):
        raise VerifyException(f"{op.name}: {key} must be a string array")
    return tuple(x.data for x in value if isinstance(x, StringAttr))


def symbol(value: Attribute) -> str:
    if not isinstance(value, SymbolRefAttr) or value.nested_references:
        raise VerifyException("this slice requires a flat, typed symbol reference")
    return value.root_reference.data


def origin(op: Operation) -> str:
    return text(op, "origin")


def decision(op: Operation) -> Decision:
    try:
        return Decision(
            Reproducibility(text(op, "reproducibility")) if "reproducibility" in op.properties else Reproducibility.UNKNOWN,
            RankRule(text(op, "rank_rule")) if "rank_rule" in op.properties else RankRule.UNKNOWN,
            strings(op, "assumptions") if "assumptions" in op.properties else (),
        )
    except ValueError as failed:
        raise VerifyException(f"{origin(op)}: unsupported decision declaration") from failed


@dataclass(frozen=True)
class RegionSemantics(OpTrait):
    """An installed operation definition supplies the invocation protocol."""

    protocol: Protocol

    def describe(self, op: Operation) -> RegionContract:
        roles = []
        for index, region in enumerate(op.regions):
            block = region.block
            count = len(op.operands)
            invocation_arguments: tuple[str, ...] = ()
            if self.protocol == Protocol.PRODUCER:
                count -= 1  # The endpoint is a destination, not a body capture.
                captures = tuple(Capture(i + 1, i, op.operands[i + 1].type) for i in range(count))
                invocation_arguments = ("production_index",)
            elif self.protocol == Protocol.REPEAT:
                count -= number(op, "carry_count")
                captures = tuple(Capture(i, i, op.operands[i].type) for i in range(count))
                invocation_arguments = ("iteration_index",) + ("loop_carry",) * number(op, "carry_count")
            elif self.protocol == Protocol.ACCESS:
                captures = tuple(Capture(i, i, op.operands[i].type) for i in range(1, count))
                invocation_arguments = ("owner_issued_capability",)
            elif self.protocol == Protocol.CHOICE:
                captures = tuple(Capture(i, i - 1, op.operands[i].type) for i in range(1, count))
            else:
                captures = tuple(Capture(i, i, op.operands[i].type) for i in range(count))
                if self.protocol == Protocol.REQUESTS:
                    invocation_arguments = ("accepted_request",)
            terminator = block.last_op
            if terminator is None or terminator.name != "lang.yield":
                raise VerifyException(f"{origin(op)}: body requires lang.yield")
            roles.append(
                RegionRole(
                    index, tuple(x.type for x in block.args), captures, invocation_arguments, tuple(x.type for x in terminator.operands)
                )
            )
        refs = op.properties.get("sources")
        sources = tuple(symbol(x) for x in refs) if isinstance(refs, ArrayAttr) else ()
        return RegionContract(
            origin(op),
            text(op, "sym_name"),
            number(op, "version"),
            self.protocol,
            tuple(roles),
            sources,
            decision(op),
            strings(op, "quiescence"),
            number(op, "limit") if self.protocol == Protocol.REPEAT else None,
            number(op, "max_outstanding") if self.protocol == Protocol.REQUESTS else None,
            number(op, "max_active") if self.protocol == Protocol.REQUESTS else None,
            symbol(op.properties["policy"]) if self.protocol == Protocol.REPEAT else None,
        )


class ResourceSemantics(OpTrait):
    def describe(self, op: Operation) -> AccessContract:
        try:
            mode, freshness = AccessMode(text(op, "mode")), Freshness(text(op, "freshness"))
        except ValueError as failed:
            raise VerifyException(f"{origin(op)}: unsupported access/freshness declaration") from failed
        return AccessContract(
            op.operands[0],
            mode,
            freshness,
            text(op, "version_selector"),
            text(op, "profile"),
            decision(op),
        )


class FlowSemantics(OpTrait):
    def describe(self, op: Operation) -> FlowContract:
        owners = op.properties["owners"]
        if not isinstance(owners, ArrayAttr):
            raise VerifyException(f"{origin(op)}: owners must be symbol references")
        payload = getattr(op.results[0].type, "payload", None)
        if not isinstance(payload, Attribute):
            raise VerifyException(f"{origin(op)}: typed payload required")
        try:
            admission = Admission(text(op, "admission"))
        except ValueError as failed:
            raise VerifyException(f"{origin(op)}: unsupported admission declaration") from failed
        return FlowContract(
            origin(op),
            payload,
            number(op, "horizon"),
            number(op, "ready_bound"),
            VersionPolicy(
                admission,
                tuple(symbol(x) for x in owners),
                text(op, "metric"),
                text(op, "comparison_boundary"),
                number(op, "max_lag"),
            ),
            decision(op),
        )


class RealizationSemantics(OpTrait):
    def describe(self, op: Operation) -> RealizationContract:
        return RealizationContract(
            symbol(op.properties["binding"]),
            text(op, "binding_version"),
            tuple(x.type for x in op.operands),
            tuple(x.type for x in op.results),
            decision=decision(op),
        )
