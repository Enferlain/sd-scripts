"""Operation contracts: the parts of a word's contract entry that the IR system checks.

Every operation's definition supplies the contracts that apply to it (D14): a
region contract (how its bodies are invoked, what they capture and what they
yield), a flow contract (for streams), a resource contract (for accesses), a
realization contract (for calls to implementations) and a decision contract
(reproducibility and rank rule). Analyses read them through the traits below.
They never select behavior by operation name, fixture or case.

This module knows no word. Words declare these traits and their own shape
checks in ``library.iro.words``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, SymbolRefAttr
from xdsl.ir import Attribute, Block, BlockArgument, Operation, OpResult, ParametrizedAttribute, SSAValue
from xdsl.irdl import irdl_attr_definition
from xdsl.traits import IsTerminator, OpTrait
from xdsl.utils.exceptions import VerifyException


class LifecycleProtocol(StrEnum):
    """How an operation's bodies are invoked, wait and end (D16)."""

    SEQUENCE = "sequence"
    PARALLEL = "parallel"
    REPEAT = "repeat"
    CHOICE = "choice"
    PRODUCER = "producer"
    HANDLER = "handler"
    FILTER = "filter"
    ACCESS = "access"


class AccessMode(StrEnum):
    """The closed set of access modes (D14)."""

    SNAPSHOT = "snapshot"
    PROTECTED_CURRENT = "protected_current"
    OWNER_UPDATE = "owner_update"
    GRANT = "grant"


class VersionSelection(StrEnum):
    """Which version of owner state a use observes, separate from the mode (D14).

    ``OWNER_PROTOCOL`` is the selection for owner updates and grants, whose
    owning protocol defines freshness. It is never written.
    """

    CURRENT_AT_START = "current_at_start"
    PUBLISHED_INITIAL = "published initial"
    AT_REQUEST = "at_request"
    OWNER_PROTOCOL = "owner_protocol"


DEFAULT_SELECTION: dict[AccessMode, VersionSelection] = {
    AccessMode.SNAPSHOT: VersionSelection.CURRENT_AT_START,
    AccessMode.PROTECTED_CURRENT: VersionSelection.CURRENT_AT_START,
    AccessMode.OWNER_UPDATE: VersionSelection.OWNER_PROTOCOL,
    AccessMode.GRANT: VersionSelection.OWNER_PROTOCOL,
}

ALLOWED_SELECTIONS: dict[AccessMode, frozenset[VersionSelection]] = {
    AccessMode.SNAPSHOT: frozenset({VersionSelection.CURRENT_AT_START, VersionSelection.PUBLISHED_INITIAL, VersionSelection.AT_REQUEST}),
    AccessMode.PROTECTED_CURRENT: frozenset({VersionSelection.CURRENT_AT_START, VersionSelection.AT_REQUEST}),
    AccessMode.OWNER_UPDATE: frozenset({VersionSelection.OWNER_PROTOCOL}),
    AccessMode.GRANT: frozenset({VersionSelection.OWNER_PROTOCOL}),
}


class Admission(StrEnum):
    """Stream admission policies (D14)."""

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


class Disposition(StrEnum):
    """How a claim is settled, exactly once on every path.

    Authors write consumed, forwarded, filtered_out and cancelled (D16). The
    system also gives cancelled or failed to a claim whose holding activity
    ends early (D18), and rejected to an item that stream admission refuses
    (D14).
    """

    CONSUMED = "consumed"
    FORWARDED = "forwarded"
    FILTERED_OUT = "filtered_out"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


AUTHORED_DISPOSITIONS = frozenset({Disposition.CONSUMED, Disposition.FORWARDED, Disposition.FILTERED_OUT, Disposition.CANCELLED})


class Boundary(StrEnum):
    """Points of an activity's lifetime that requests and stops refer to."""

    COMPLETED = "completed"
    SETTLED = "settled"
    SETTLED_SUCCESS = "settled_success"


class Quiescence(StrEnum):
    """Boundaries where a protocol defines a consistent cut (D14)."""

    SETTLED = "settled"
    BETWEEN_SETTLED_ITERATIONS = "between_settled_iterations"


SUPPORTED_QUIESCENCE: dict[LifecycleProtocol, frozenset[Quiescence]] = {
    protocol: frozenset({Quiescence.SETTLED}) for protocol in LifecycleProtocol
} | {LifecycleProtocol.REPEAT: frozenset({Quiescence.SETTLED, Quiescence.BETWEEN_SETTLED_ITERATIONS})}


@dataclass(frozen=True)
class Decision:
    """What a decision declares: its reproducibility, rank rule and assumptions."""

    reproducibility: Reproducibility = Reproducibility.UNKNOWN
    rank: RankRule = RankRule.UNKNOWN
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Capture:
    """An operand passed into a body: operand position to entry-argument position."""

    operand: int
    argument: int
    type: Attribute


@dataclass(frozen=True)
class RegionRole:
    """One body of a structured operation: its entries, captures and yields."""

    index: int
    entries: tuple[Attribute, ...]
    captures: tuple[Capture, ...]
    invocation_arguments: tuple[str, ...]
    yields: tuple[Attribute, ...]


@dataclass(frozen=True)
class RegionContract:
    path: str
    symbol: str
    protocol: LifecycleProtocol
    roles: tuple[RegionRole, ...]
    request_sources: tuple[str, ...]
    decision: Decision
    quiescence: tuple[Quiescence, ...]
    limit: int | None = None
    policy: str | None = None
    max_outstanding: int | None = None
    max_active: int | None = None
    admission: str = "explicit_invocation"
    completion: str = "required_children_and_sources_closed"
    interruption: str = "close_admission_then_settle"
    settlement: str = "physical_cleanup_and_owned_products"


@dataclass(frozen=True)
class ResourceUse:
    """One resource named by an access, with how it is used."""

    subject: SSAValue
    mode: AccessMode
    selection: VersionSelection
    route: str | None = None
    view: str | None = None


@dataclass(frozen=True)
class AccessContract:
    """The resource contract of an access: one lease per named resource, taken together (D16)."""

    uses: tuple[ResourceUse, ...]
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
    path: str
    payload: Attribute
    max_outstanding: int
    ready_bound: int
    versions: VersionPolicy
    decision: Decision


@dataclass(frozen=True)
class RealizationContract:
    """What a call requires of the implementation it names."""

    implementation: str
    version: str
    inputs: tuple[Attribute, ...]
    outputs: tuple[Attribute, ...]
    decision: Decision
    retains: tuple[int, ...] = ()
    completion: str = "physical_then_validate"
    outcomes: tuple[str, ...] = ("returned", "invalid_output", "raised")

    @property
    def result_lifetime(self) -> str:
        return "scoped_retained" if self.retains else "detached_values"


@dataclass(frozen=True)
class ItemContract:
    """What an operation does with claims, by operand and result position."""

    borrows: tuple[int, ...] = ()
    consumes: tuple[int, ...] = ()
    creates: tuple[int, ...] = ()
    detached_outputs: tuple[int, ...] = ()
    forwards: tuple[int, ...] = ()
    forks: bool = False


@dataclass(frozen=True)
class EndpointUse:
    operand: int
    role: str


@dataclass(frozen=True)
class CompletionControl:
    kind: str
    source: str
    target: str
    boundary: Boundary
    registration: str = "before_source_terminal"
    orderly_stop: str = "retain_admitted_promises"


@dataclass(frozen=True)
class Reference:
    """A symbol reference from one operation to an activity or resource."""

    field: str
    target: str
    kind: str


# Reading fields off operations.


def text(op: Operation, key: str) -> str:
    value = op.properties.get(key)
    if not isinstance(value, StringAttr):
        raise VerifyException(f"{op.name} {path(op)}: {key} must be a string")
    return value.data


def optional_text(op: Operation, key: str) -> str | None:
    return text(op, key) if key in op.properties else None


def number(op: Operation, key: str) -> int:
    value = op.properties.get(key)
    if not isinstance(value, IntegerAttr):
        raise VerifyException(f"{op.name} {path(op)}: {key} must be an integer")
    return value.value.data


def strings(op: Operation, key: str) -> tuple[str, ...]:
    value = op.properties.get(key)
    if value is None:
        return ()
    if not isinstance(value, ArrayAttr) or any(not isinstance(x, StringAttr) for x in value):
        raise VerifyException(f"{op.name} {path(op)}: {key} must be a string array")
    return tuple(x.data for x in value if isinstance(x, StringAttr))


def symbol(value: Attribute) -> str:
    if not isinstance(value, SymbolRefAttr) or value.nested_references:
        raise VerifyException("a flat symbol reference is required")
    return value.root_reference.data


def symbols(op: Operation, key: str) -> tuple[str, ...]:
    value = op.properties.get(key)
    if value is None:
        return ()
    if not isinstance(value, ArrayAttr):
        raise VerifyException(f"{op.name} {path(op)}: {key} must be a symbol array")
    return tuple(symbol(item) for item in value)


def sym_name(op: Operation) -> str | None:
    """The authored name of a named operation, such as an activity, stream or resource."""
    value = op.properties.get("sym_name")
    return value.data if isinstance(value, StringAttr) else None


def in_normalized_strategy(op: Operation) -> bool:
    """Whether the operation sits in a strategy marked normalized."""
    root = op
    while (parent := root.parent_op()) is not None:
        root = parent
    return root.name == "iro.strategy" and "normalized" in root.properties


def origin(op: Operation) -> str | None:
    """Where the operation was authored, such as ``strategies/sdxl/finetune.py:88``, when known."""
    return optional_text(op, "origin")


def path(op: Operation) -> str:
    """The operation's structural identity within its strategy.

    Named operations contribute ``@name``; others their position in the
    enclosing body. The path stays the same across printing and cold
    loading, unlike Python object identity.
    """
    parts: list[str] = []
    current: Operation | None = op
    while current is not None:
        name = sym_name(current)
        block, parent = current.parent_block(), current.parent_op()
        if name is not None:
            parts.append(f"@{name}")
        elif block is not None and parent is not None:
            region = parent.regions.index(block.parent) if block.parent is not None else 0
            index = list(block.ops).index(current)
            parts.append(f"{region}.{index}" if len(parent.regions) > 1 else str(index))
        elif current is op:
            parts.append(current.name)
        current = parent
    return "/".join(reversed(parts))


def decision(op: Operation) -> Decision:
    try:
        return Decision(
            Reproducibility(text(op, "reproducibility")) if "reproducibility" in op.properties else Reproducibility.UNKNOWN,
            RankRule(text(op, "rank_rule")) if "rank_rule" in op.properties else RankRule.UNKNOWN,
            strings(op, "assumptions"),
        )
    except ValueError as failed:
        raise VerifyException(f"{op.name} {path(op)}: unsupported decision declaration") from failed


def terminator(block: Block) -> Operation:
    last = block.last_op
    if last is None or not last.has_trait(IsTerminator):
        raise VerifyException("a body must end with its yield")
    return last


# The resource use attribute, read by the resource contract.


@irdl_attr_definition
class ResourceUseAttr(ParametrizedAttribute):
    """One resource named by an access: its mode, version selection, and route or view.

    An empty route means the normal route; an empty view means none.
    """

    name = "iro.resource_use"

    mode: StringAttr
    selection: StringAttr
    route: StringAttr
    view: StringAttr

    @staticmethod
    def of(
        mode: AccessMode, selection: VersionSelection | None = None, *, route: str | None = None, view: str | None = None
    ) -> ResourceUseAttr:
        chosen = DEFAULT_SELECTION[mode] if selection is None else selection
        return ResourceUseAttr(StringAttr(mode), StringAttr(chosen), StringAttr(route or ""), StringAttr(view or ""))


def resource_uses(op: Operation) -> tuple[ResourceUseAttr, ...]:
    value = op.properties.get("uses")
    if not isinstance(value, ArrayAttr) or not value or any(not isinstance(x, ResourceUseAttr) for x in value):
        raise VerifyException(f"{op.name} {path(op)}: an access names one or more resource uses")
    return tuple(x for x in value if isinstance(x, ResourceUseAttr))


# Traits.


class OrderedBody(OpTrait):
    """Each body entry is initialized before ordered local result availability.

    Parallel roles have separate ordered bodies. Recurrence enters through
    initialized entry arguments, not forward references. Captures are explicit.
    """

    def verify(self, op: Operation) -> None:
        for region in op.regions:
            for block in region.blocks:
                available: set[SSAValue] = set(block.args)
                for child in block.ops:
                    for position, value in enumerate(child.operands):
                        if value in available:
                            continue
                        if isinstance(value, OpResult):
                            definition = f"{path(value.owner)} result {value.index}"
                        elif isinstance(value, BlockArgument):
                            parent = value.owner.parent_op()
                            definition = f"{path(parent) if parent is not None else 'an external body'} entry {value.index}"
                        else:
                            definition = "an undeclared value"
                        raise VerifyException(
                            f"{path(child)} operand {position}: {definition} is not available in {path(op)}; "
                            "use an entry argument or an earlier result (captures are explicit)"
                        )
                    available.update(child.results)


@dataclass(frozen=True)
class RegionSemantics(OrderedBody):
    """The lifecycle protocol an operation's bodies follow.

    ``transparent`` marks a derived local grouping that creates no lifetime or
    authority of its own, such as a fused group.
    """

    protocol: LifecycleProtocol
    transparent: bool = False

    def describe(self, op: Operation) -> RegionContract:
        roles: list[RegionRole] = []
        count = len(op.operands)
        for index, region in enumerate(op.regions):
            block = region.block
            invocation_arguments: tuple[str, ...] = ()
            match self.protocol:
                case LifecycleProtocol.PRODUCER:
                    # Operand 0 is the stream it produces into, not a capture.
                    captures = tuple(Capture(i, i - 1, op.operands[i].type) for i in range(1, count))
                    invocation_arguments = ("production_index",)
                case LifecycleProtocol.REPEAT:
                    carried = number(op, "carry_count")
                    captures = tuple(Capture(i, i, op.operands[i].type) for i in range(count - carried))
                    invocation_arguments = ("iteration_index",) + ("carried_value",) * carried
                case LifecycleProtocol.ACCESS:
                    leases = len(resource_uses(op))
                    captures = tuple(Capture(i, i, op.operands[i].type) for i in range(leases, count))
                    invocation_arguments = ("owner_issued_lease",) * leases
                case LifecycleProtocol.FILTER:
                    captures = tuple(Capture(i, i, op.operands[i].type) for i in range(1, count))
                    invocation_arguments = ("detached_input_payload",)
                case LifecycleProtocol.CHOICE:
                    # A choice on an optional gives its first alternative the value.
                    binds = index == 0 and len(block.args) == count
                    captures = tuple(Capture(i, i if binds else i - 1, op.operands[i].type) for i in range(1, count))
                    invocation_arguments = ("present_value",) if binds else ()
                case LifecycleProtocol.HANDLER:
                    captures = tuple(Capture(i, i, op.operands[i].type) for i in range(count))
                    invocation_arguments = ("accepted_request",)
                case _:
                    captures = tuple(Capture(i, i, op.operands[i].type) for i in range(count))
            roles.append(
                RegionRole(
                    index,
                    tuple(arg.type for arg in block.args),
                    captures,
                    invocation_arguments,
                    tuple(value.type for value in terminator(block).operands),
                )
            )
        try:
            quiescence = tuple(Quiescence(x) for x in strings(op, "quiescence"))
        except ValueError as failed:
            raise VerifyException(f"{path(op)}: unknown quiescence boundary") from failed
        name = sym_name(op)
        return RegionContract(
            path(op),
            name if name is not None else path(op),
            self.protocol,
            tuple(roles),
            symbols(op, "sources"),
            decision(op),
            quiescence,
            limit=number(op, "limit") if self.protocol == LifecycleProtocol.REPEAT and "limit" in op.properties else None,
            policy=symbol(op.properties["policy"]) if self.protocol == LifecycleProtocol.REPEAT and "policy" in op.properties else None,
            max_outstanding=number(op, "max_outstanding") if self.protocol == LifecycleProtocol.HANDLER else None,
            max_active=(number(op, "max_active") if "max_active" in op.properties else 1)
            if self.protocol == LifecycleProtocol.HANDLER
            else None,
            admission="enclosing_invocation" if self.transparent else "explicit_invocation",
        )


class ResourceSemantics(OpTrait):
    """The resource contract of an access: each named resource with its mode and selection."""

    def describe(self, op: Operation) -> AccessContract:
        attributes = resource_uses(op)
        if len(op.operands) < len(attributes):
            raise VerifyException(f"{path(op)}: each resource use needs its resource operand")
        uses: list[ResourceUse] = []
        for subject, attribute in zip(op.operands, attributes, strict=False):
            try:
                mode = AccessMode(attribute.mode.data)
                selection = VersionSelection(attribute.selection.data)
            except ValueError as failed:
                raise VerifyException(f"{path(op)}: unsupported access mode or version selection") from failed
            route, view = attribute.route.data or None, attribute.view.data or None
            if route is not None and view is not None:
                raise VerifyException(f"{path(op)}: a resource use names a route or a view, not both")
            uses.append(ResourceUse(subject, mode, selection, route, view))
        return AccessContract(tuple(uses), decision(op))


class FlowSemantics(OpTrait):
    """The flow contract of a stream: its payload, live bounds and version admission."""

    def describe(self, op: Operation) -> FlowContract:
        payload = getattr(op.results[0].type, "payload", None)
        if not isinstance(payload, Attribute):
            raise VerifyException(f"{path(op)}: a stream needs a typed payload")
        try:
            admission = Admission(text(op, "admission"))
        except ValueError as failed:
            raise VerifyException(f"{path(op)}: unsupported admission policy") from failed
        return FlowContract(
            path(op),
            payload,
            number(op, "outstanding"),
            number(op, "ready"),
            VersionPolicy(admission, symbols(op, "owners"), text(op, "metric"), text(op, "boundary"), number(op, "max_lag")),
            decision(op),
        )


class RealizationSemantics(OpTrait):
    """The realization contract of a call: the implementation it names and what it returns.

    The implementation's version is pinned once, in the strategy's ``uses``
    header (D16), not on each call.
    """

    def describe(self, op: Operation) -> RealizationContract:
        declared = op.properties.get("retains")
        if declared is not None and (not isinstance(declared, ArrayAttr) or any(not isinstance(x, IntegerAttr) for x in declared)):
            raise VerifyException(f"{path(op)}: retained positions must be integers")
        retains = tuple(x.value.data for x in declared if isinstance(x, IntegerAttr)) if isinstance(declared, ArrayAttr) else ()
        name = symbol(op.properties["implementation"])
        return RealizationContract(
            name,
            pinned_version(op, name),
            tuple(x.type for x in op.operands),
            tuple(x.type for x in op.results),
            decision(op),
            retains,
        )


def pinned_version(op: Operation, implementation: str) -> str:
    """The version the enclosing strategy's ``uses`` header pins for an implementation."""
    root: Operation | None = op
    while root is not None and root.name != "iro.strategy":
        root = root.parent_op()
    if root is None:
        raise VerifyException(f"{path(op)}: a call must be inside a strategy")
    names, versions = symbols(root, "uses"), strings(root, "use_versions")
    pins = dict(zip(names, versions, strict=False))
    if implementation not in pins:
        raise VerifyException(f"{path(op)}: @{implementation} is not pinned in the strategy's uses header")
    return pins[implementation]


@dataclass(frozen=True)
class ItemSemantics(OpTrait):
    contract: ItemContract

    def describe(self, op: Operation) -> ItemContract:
        return self.contract


@dataclass(frozen=True)
class EndpointSemantics(OpTrait):
    contract: EndpointUse

    def describe(self, op: Operation) -> EndpointUse:
        return self.contract


@dataclass(frozen=True)
class CompletionSemantics(OpTrait):
    """A completion control: a stop or a request tied to a source activity's boundary."""

    kind: str
    target_field: str

    def describe(self, op: Operation) -> CompletionControl:
        source, target = op.properties.get("source"), op.properties.get(self.target_field)
        if source is None or target is None:
            raise VerifyException(f"{path(op)}: a completion control needs a source and a target")
        try:
            boundary = Boundary(text(op, "boundary")) if "boundary" in op.properties else Boundary.SETTLED
        except ValueError as failed:
            raise VerifyException(f"{path(op)}: unknown boundary") from failed
        return CompletionControl(self.kind, symbol(source), symbol(target), boundary)


class DecisionSemantics(OpTrait):
    """The same decision contract for calls and completion decisions."""

    def describe(self, op: Operation) -> Decision:
        return decision(op)


@dataclass(frozen=True)
class ReferenceSemantics(OpTrait):
    """Symbol references an operation makes, by field and the kind of target they need."""

    fields: tuple[tuple[str, str], ...]

    def describe(self, op: Operation) -> tuple[Reference, ...]:
        refs: list[Reference] = []
        for field, kind in self.fields:
            value = op.properties.get(field)
            if value is None:
                raise VerifyException(f"{path(op)}: a reference needs {field}")
            values = tuple(value) if isinstance(value, ArrayAttr) else (value,)
            refs.extend(Reference(field, symbol(item), kind) for item in values)
        return tuple(refs)


@dataclass(frozen=True)
class ProductSemantics(OpTrait):
    """Explicit end of a retained product's obligations (D14)."""

    consumes: tuple[int, ...] = (0,)

    def describe(self, op: Operation) -> tuple[int, ...]:
        return self.consumes


def provider(value: SSAValue) -> SSAValue:
    """The value a body's entry stands for, following captures outward.

    A protocol-supplied entry, such as a lease or an iteration, is its own
    provider.
    """
    while isinstance(value, BlockArgument):
        parent, region = value.owner.parent_op(), value.owner.parent
        semantics = parent.get_trait(RegionSemantics) if parent is not None else None
        if parent is None or region is None or semantics is None:
            return value
        role = semantics.describe(parent).roles[parent.regions.index(region)]
        capture = next((c for c in role.captures if c.argument == value.index), None)
        if capture is None:
            return value
        value = parent.operands[capture.operand]
    return value


Expansion = Callable[[Operation], tuple[Sequence[Operation], Sequence[SSAValue]]]


@dataclass(frozen=True)
class DerivedWord(OpTrait):
    """A word defined by expanding into other words during normalization (D16).

    ``expansion`` builds the replacement from the word: the new operations, in
    order, and the values that replace its results. The word's authored
    bodies move into the replacement unchanged; normalization records the word
    in the ancestry of every operation the expansion creates.
    """

    expansion: Expansion | None = None

    def verify(self, op: Operation) -> None:
        if in_normalized_strategy(op):
            raise VerifyException(f"{path(op)}: a normalized strategy has its derived words expanded")

    def expand(self, op: Operation) -> tuple[Sequence[Operation], Sequence[SSAValue]]:
        if self.expansion is None:
            raise VerifyException(f"{path(op)}: {op.name} defines no expansion")
        return self.expansion(op)


__all__ = [
    "ALLOWED_SELECTIONS",
    "AUTHORED_DISPOSITIONS",
    "DEFAULT_SELECTION",
    "SUPPORTED_QUIESCENCE",
    "AccessContract",
    "AccessMode",
    "Admission",
    "Boundary",
    "Capture",
    "CompletionControl",
    "CompletionSemantics",
    "Decision",
    "DecisionSemantics",
    "DerivedWord",
    "Disposition",
    "EndpointSemantics",
    "EndpointUse",
    "Expansion",
    "FlowContract",
    "FlowSemantics",
    "ItemContract",
    "ItemSemantics",
    "LifecycleProtocol",
    "OrderedBody",
    "ProductSemantics",
    "Quiescence",
    "RankRule",
    "RealizationContract",
    "RealizationSemantics",
    "Reference",
    "ReferenceSemantics",
    "RegionContract",
    "RegionRole",
    "RegionSemantics",
    "Reproducibility",
    "ResourceSemantics",
    "ResourceUse",
    "ResourceUseAttr",
    "VersionPolicy",
    "VersionSelection",
    "decision",
    "in_normalized_strategy",
    "number",
    "optional_text",
    "origin",
    "path",
    "pinned_version",
    "provider",
    "resource_uses",
    "strings",
    "sym_name",
    "symbol",
    "symbols",
    "terminator",
    "text",
]
