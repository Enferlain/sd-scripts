"""Normalization: the accepted strategy with its implicit structure made explicit (D14, D16).

Normalization expands derived words, places each retained result's release
after its last use, and resolves the relationships that requests, stops,
handlers and streams name. It works on a copy, so the accepted strategy stays
as written, and it cannot invent input, lifecycle or authority meaning: every
authored statement except a derived word survives unchanged, and the
relationships are the same before and after. Captures need no step of their
own, since every body's captures are already its operands.

Normalizing a normalized strategy changes nothing; a normalized form that
normalization would change, such as one missing a release, is rejected.
"""

from collections import Counter
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, StringAttr, UnitAttr
from xdsl.ir import Attribute, Block, Operation, OpResult, SSAValue
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException

from library.iro.composition import Composition
from library.iro.diagnostics import Problems, Stage
from library.iro.operation_contracts import (
    DerivedWord,
    LifecycleProtocol,
    ProductSemantics,
    ReferenceSemantics,
    RegionSemantics,
    origin,
    path,
    provider,
    sym_name,
)
from library.iro.words.access import ReleaseOp
from library.iro.words.calls import CallOp, retained_positions
from library.iro.words.model import ResourceOp
from library.iro.words.structure import StrategyOp
from library.iro.text import print_strategy
from library.iro.words.types import RetainedType, payload_of, written

MAX_EXPANSION_DEPTH = 32


@dataclass(frozen=True)
class Link:
    """One statement naming another: a request's handler, a stop's target, a stream's owner."""

    path: str
    field: str
    target: str
    kind: str


@dataclass(frozen=True)
class Normalized:
    """The accepted strategy, its normalized copy and the relationships both share."""

    accepted: StrategyOp
    strategy: StrategyOp
    links: tuple[Link, ...]

    def text(self, composition: Composition | None = None, *, locations: bool = False) -> str:
        return print_strategy(self.strategy, composition, locations=locations)


def normalize(strategy: StrategyOp) -> Normalized:
    """The normalized copy of a strategy, or a rejection naming each unresolved relationship."""
    problems = Problems(Stage.NORMALIZE)
    normalized = strategy.clone()
    before = _references(normalized, problems)
    problems.reject_if_any()

    authored = {op: _signature(op) for op in normalized.walk() if not op.has_trait(DerivedWord)}
    _expand(normalized, problems)
    problems.reject_if_any()
    placed = _place_releases(normalized)
    if strategy.normalized is not None:
        for release in placed:
            problems.add(
                "iro-norm-008",
                "this normalized form lacks a release that normalization places here",
                source=origin(release),
                path=path(release),
                fix="normalize the accepted form again instead of editing the normalized one",
            )
    normalized.properties["normalized"] = UnitAttr()

    _check_unchanged(normalized, authored, problems)
    after = _references(normalized, problems)
    if Counter(after) != Counter(before):
        problems.add("iro-norm-007", "normalization changed what requests, stops, handlers or streams name", path=path(strategy))
    problems.reject_if_any()
    try:
        normalized.verify()
    except VerifyException as failed:
        problems.add("iro-norm-007", f"the normalized strategy does not verify: {failed}", path=path(strategy))
    problems.reject_if_any()
    return Normalized(strategy, normalized, tuple(Link(path(op), *rest) for op, *rest in after))


# Relationships.


def links(strategy: StrategyOp, problems: Problems) -> tuple[Link, ...]:
    """Every symbol reference, resolved against the strategy's named statements and checked for kind."""
    return tuple(Link(path(op), *rest) for op, *rest in _references(strategy, problems))


def _references(strategy: StrategyOp, problems: Problems) -> list[tuple[Operation, str, str, str]]:
    """Each resolved reference with the statement making it, so it can be followed across normalization."""
    named: dict[str, Operation] = {}
    for op in strategy.walk():
        name = sym_name(op)
        if name is None or op is strategy:
            continue
        if name in named:
            problems.add("iro-norm-001", f"@{name} names two statements", names=(f"@{name}",), source=origin(op), path=path(op))
        else:
            named[name] = op

    found: list[tuple[Operation, str, str, str]] = []
    for op in strategy.walk():
        trait = op.get_trait(ReferenceSemantics)
        if trait is None:
            continue
        for ref in trait.describe(op):
            target = named.get(ref.target)
            if target is None:
                problems.add(
                    "iro-norm-002",
                    f"{op.name} names @{ref.target} as its {ref.field}, and nothing has that name",
                    names=(f"@{ref.target}",),
                    source=origin(op),
                    path=path(op),
                )
                continue
            if not _is_kind(target, ref.kind):
                problems.add(
                    "iro-norm-003",
                    f"{op.name}'s {ref.field} @{ref.target} must be a {ref.kind}; it is {target.name}",
                    names=(f"@{ref.target}",),
                    source=origin(op),
                    path=path(op),
                )
                continue
            if ref.kind == "handler":
                _check_request(op, target, problems)
            found.append((op, ref.field, ref.target, ref.kind))
    return found


def _is_kind(target: Operation, kind: str) -> bool:
    semantics = target.get_trait(RegionSemantics)
    if kind == "resource":
        return isinstance(target, ResourceOp)
    if kind == "handler":
        return semantics is not None and semantics.protocol == LifecycleProtocol.HANDLER
    return semantics is not None


def _check_request(request: Operation, handler: Operation, problems: Problems) -> None:
    expected = handler.properties.get("request")
    carried = request.operands[0].type if request.operands else None
    if carried is not None:
        carried = payload_of(carried) or carried
    if expected is not None and carried != expected:
        problems.add(
            "iro-norm-004",
            f"the request carries {_type_name(carried)}, and @{sym_name(handler)} takes {_type_name(expected)}",
            names=(f"@{sym_name(handler)}",),
            source=origin(request),
            path=path(request),
        )
    source = request.properties.get("source")
    sources = handler.properties.get("sources")
    if source is not None and isinstance(sources, ArrayAttr) and source not in sources.data:
        name = f"@{sym_name(handler)}"
        problems.add(
            "iro-norm-005",
            f"{name} does not list {source} among its sources",
            names=(name,),
            fix=f"add {source} to {name}'s sources",
            source=origin(request),
            path=path(request),
        )


def _type_name(type_: Attribute | None) -> str:
    return written(type_) if type_ is not None else "nothing"


# Derived words.


def _expand(strategy: StrategyOp, problems: Problems) -> None:
    """Expands derived words until none remain, recording each in what its expansion creates."""
    for _ in range(MAX_EXPANSION_DEPTH):
        derived = [op for op in strategy.walk() if op.has_trait(DerivedWord)]
        if not derived:
            return
        for op in derived:
            if op.parent_op() is None:
                continue  # erased with an enclosing derived word in this round
            _expand_one(op)
    still = [op for op in strategy.walk() if op.has_trait(DerivedWord)]
    for op in still:
        problems.add(
            "iro-norm-006",
            f"{op.name} keeps expanding into derived words after {MAX_EXPANSION_DEPTH} rounds",
            source=origin(op),
            path=path(op),
        )


def _expand_one(op: Operation) -> None:
    trait = op.get_trait(DerivedWord)
    assert trait is not None
    existing = {inner for inner in op.walk() if inner is not op}
    new_ops, new_results = trait.expand(op)
    entry = f"{op.name} @{sym_name(op)}" if sym_name(op) is not None else op.name
    ancestry = op.properties.get("ancestry")
    inherited = list(ancestry.data) if isinstance(ancestry, ArrayAttr) else []
    authored_at = op.properties.get("origin")
    for top in new_ops:
        for created in top.walk():
            if created in existing:
                continue
            created.properties["ancestry"] = ArrayAttr([*inherited, StringAttr(entry)])
            if authored_at is not None and "origin" not in created.properties:
                created.properties["origin"] = authored_at
    Rewriter.replace_op(op, list(new_ops), list(new_results))


# Releases.


def _place_releases(strategy: StrategyOp) -> list[ReleaseOp]:
    """Places each retained result's release after its last use in the block that defines it.

    A retained result yielded from its body hands its dependency to the
    enclosing result, which is released instead. A result already released is
    left as it is, so normalizing twice changes nothing.
    """
    retained = [r for op in strategy.walk() for r in op.results if isinstance(r.type, RetainedType)]
    placed: list[ReleaseOp] = []
    # Later results first: a result that keeps another is released before it.
    for value in reversed(retained):
        if any(use.operation.has_trait(IsTerminator) and use.operation.parent_block() is value.owner.parent_block() for use in value.uses):
            continue
        if any(use.operation.has_trait(ProductSemantics) for use in value.uses):
            continue
        block = value.owner.parent_block()
        assert block is not None
        last = _last_use(value, block)
        for dependent in dependents(value, retained):
            for use in dependent.uses:
                released = use.operation.has_trait(ProductSemantics) and use.operation.parent_block() is block
                if released and block.get_operation_index(use.operation) > block.get_operation_index(last):
                    last = use.operation
        release = ReleaseOp.create(operands=[value])
        if "origin" in value.owner.properties:
            release.properties["origin"] = value.owner.properties["origin"]
        Rewriter.insert_op(release, InsertPoint.after(last))
        placed.append(release)
    return placed


def retained_inputs(value: SSAValue, seen: frozenset[SSAValue] = frozenset()) -> set[SSAValue]:
    """The retained results a retained value keeps, through its call's retained inputs and its body's yields."""
    found_value = provider(value)
    if found_value in seen or not isinstance(found_value, OpResult):
        return set()
    seen = seen | {found_value}
    owner = found_value.owner
    kept: set[SSAValue] = set()
    if isinstance(owner, CallOp):
        for position in retained_positions(owner):
            given = provider(owner.inputs[position])
            if isinstance(given.type, RetainedType):
                kept |= {given, *retained_inputs(given, seen)}
        return kept
    for region in owner.regions:
        last = region.block.last_op
        if last is not None and last.has_trait(IsTerminator) and found_value.index < len(last.operands):
            kept |= retained_inputs(last.operands[found_value.index], seen)
    return kept


def dependents(value: SSAValue, retained: list[OpResult]) -> list[OpResult]:
    """The retained results in the value's block that keep it."""
    block = value.owner.parent_block() if isinstance(value, OpResult) else None
    return [d for d in retained if d is not value and d.owner.parent_block() is block and value in retained_inputs(d)]


def _last_use(value: OpResult, block: Block) -> Operation:
    """The last statement of ``block`` that uses the value, directly or inside its bodies."""
    last = value.owner
    for use in value.uses:
        statement: Operation | None = use.operation
        while statement is not None and statement.parent_block() is not block:
            statement = statement.parent_op()
        if statement is not None and block.get_operation_index(statement) > block.get_operation_index(last):
            last = statement
    return last


# What normalization must not change.


def _signature(op: Operation) -> tuple:
    properties = tuple(sorted((k, v) for k, v in op.properties.items() if k != "ancestry"))
    return (op.name, properties, tuple(v.type for v in op.operands), tuple(r.type for r in op.results), len(op.regions))


def _check_unchanged(strategy: StrategyOp, authored: dict[Operation, tuple], problems: Problems) -> None:
    present = set(strategy.walk())
    changed = [op for op, signature in authored.items() if op is not strategy and (op not in present or _signature(op) != signature)]
    if changed:
        problems.add(
            "iro-norm-007",
            f"normalization dropped or changed {len(changed)} authored statement(s), first {changed[0].name}",
            path=path(strategy),
        )
    kinds = Counter(op.name for op in present if op not in authored and not isinstance(op, ReleaseOp) and "ancestry" not in op.properties)
    if kinds:
        problems.add("iro-norm-007", f"normalization created statements no derived word accounts for: {sorted(kinds)}", path=path(strategy))


__all__ = ["MAX_EXPANSION_DEPTH", "Link", "Normalized", "links", "normalize"]
