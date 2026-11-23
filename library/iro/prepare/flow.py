"""How values flow through a strategy, and reachability over named relations (D16).

The core passes values on through its own constructs: into and out of bodies
through captures, yields and repeat carries, into every alternative of a
choice, and between activities wherever it carries a value, as a stream does
from its producer to the items its consumer takes and a request does to its
handler. Implementations declare, for a named relation, which of their
inputs propagate into which outputs. A path follows both, and stops only at
an input whose implementation declares no edge; the core cuts no relation at
a construct of its own.

A domain asks which calls lie on a path into sinks it names by keys of its
own, starting from sources it names where it needs them. The core compares
relation names and never interprets them. What a path between activities
means is the asking domain's to decide, so a query may keep to one activity:
then no path leaves the activity a value lives in.
"""

from collections import defaultdict, deque
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field

from xdsl.ir import BlockArgument, Operation, SSAValue

from library.iro.operation_contracts import LifecycleProtocol, RegionSemantics, ResourceSemantics, ResourceUse, path, provider, sym_name
from library.iro.seams import Declaration
from library.iro.words.access import AccessOp
from library.iro.words.calls import CallOp, YieldOp
from library.iro.words.data import BorrowOp, FilterOp, ForkOp, TakeOp
from library.iro.words.lifetimes import HandlerOp, ProducerOp, RequestOp
from library.iro.words.model import ResourceOp
from library.iro.words.structure import StrategyOp

# Where values live.


def is_activity(op: Operation) -> bool:
    """A named operation with a lifecycle protocol: what a target can place."""
    return sym_name(op) is not None and op.get_trait(RegionSemantics) is not None


def activity_of(op: Operation) -> str:
    """The path of the nearest activity enclosing the op, or of the strategy."""
    current = op.parent_op()
    while current is not None:
        if is_activity(current) or isinstance(current, StrategyOp):
            return path(current)
        current = current.parent_op()
    return ""


def value_activity(value: SSAValue) -> str:
    """The activity a value lives in: its defining op's, or for a body's entry the activity whose body it enters."""
    if isinstance(value, BlockArgument):
        owner = value.owner.parent_op()
        if owner is None:
            return ""
        return path(owner) if is_activity(owner) or isinstance(owner, StrategyOp) else activity_of(owner)
    return activity_of(value.owner) if isinstance(value.owner, Operation) else ""


def resource_of(subject: SSAValue) -> str | None:
    """The resource an access names, following captures outward."""
    owner = provider(subject).owner
    return owner.sym_name.data if isinstance(owner, ResourceOp) else None


def lease_use(value: SSAValue) -> ResourceUse | None:
    """The use a lease was taken for: its access's use at the lease's position."""
    value = provider(value)
    if not isinstance(value, BlockArgument):
        return None
    access = value.owner.parent_op()
    if not isinstance(access, AccessOp):
        return None
    uses = ResourceSemantics().describe(access).uses
    return uses[value.index] if value.index < len(uses) else None


def leased(value: SSAValue) -> str | None:
    """The resource a lease reaches."""
    use = lease_use(value)
    return resource_of(use.subject) if use is not None else None


# How the core passes values on.


def yielded_result(yielded: YieldOp, index: int) -> SSAValue | None:
    """The result of its operation that a body's yielded value becomes, if any.

    An access, a choice, a repeat or an activity hands a yielded value out as
    its result in the same position. A filter yields whether to keep its item,
    and its result is the item; a producer yields into its stream.
    """
    parent = yielded.parent_op()
    if parent is None or index >= len(parent.results):
        return None
    semantics = parent.get_trait(RegionSemantics)
    if semantics is not None and semantics.protocol == LifecycleProtocol.FILTER:
        return None
    return parent.results[index]


def carried_entry(parent: Operation, yielded: YieldOp, index: int) -> SSAValue | None:
    """The entry of a repeat's next iteration that a yielded value becomes, if the body is a repeat's."""
    semantics = parent.get_trait(RegionSemantics)
    block = yielded.parent_block()
    if semantics is None or semantics.protocol != LifecycleProtocol.REPEAT or block is None:
        return None
    role = semantics.describe(parent).roles[0]
    captured = {capture.argument for capture in role.captures}
    supplied = [i for i in range(len(role.entries)) if i not in captured]
    # The protocol supplies the iteration index, then the carried values in yield order.
    position = 1 + index
    return block.args[supplied[position]] if position < len(supplied) else None


def entries_fed(op: Operation, operand: int) -> list[SSAValue]:
    """The body entries an operand of a structured operation becomes: a capture, a repeat's initial carried value or a choice's present value."""
    semantics = op.get_trait(RegionSemantics)
    assert semantics is not None
    fed: list[SSAValue] = []
    for role in semantics.describe(op).roles:
        block = op.regions[role.index].block
        fed += [block.args[c.argument] for c in role.captures if c.operand == operand]
        captured = {c.argument for c in role.captures}
        supplied = [i for i in range(len(role.entries)) if i not in captured]
        if semantics.protocol == LifecycleProtocol.REPEAT:
            carried = role.invocation_arguments.count("carried_value")
            first = len(op.operands) - carried
            # The protocol supplies the iteration index, then the carried values in order.
            if operand >= first and 1 + operand - first < len(supplied):
                fed.append(block.args[supplied[1 + operand - first]])
        elif semantics.protocol == LifecycleProtocol.CHOICE and operand == 0 and "present_value" in role.invocation_arguments and supplied:
            fed.append(block.args[supplied[0]])
    return fed


def supplied_entry(op: Operation, argument: str) -> SSAValue | None:
    """The body entry its protocol supplies for the named invocation argument, such as a handler's accepted request."""
    semantics = op.get_trait(RegionSemantics)
    if semantics is None:
        return None
    for role in semantics.describe(op).roles:
        if argument not in role.invocation_arguments:
            continue
        captured = {c.argument for c in role.captures}
        supplied = [i for i in range(len(role.entries)) if i not in captured]
        position = role.invocation_arguments.index(argument)
        if position < len(supplied):
            return op.regions[role.index].block.args[supplied[position]]
    return None


def passed_on(user: Operation, index: int, handlers: Mapping[str, HandlerOp]) -> list[SSAValue]:
    """Where the core passes a value used as the op's operand at ``index``; never through a call, whose implementation declares that."""
    targets: list[SSAValue] = []
    if isinstance(user, YieldOp):
        if (result := yielded_result(user, index)) is not None:
            targets.append(result)
        parent = user.parent_op()
        if parent is not None:
            if (carried := carried_entry(parent, user, index)) is not None:
                targets.append(carried)
            if isinstance(parent, ProducerOp) and parent.operands:
                # A production yields into its stream itself, whose captures reach every consumer.
                targets.append(provider(parent.operands[0]))
    elif isinstance(user, TakeOp):
        targets.append(user.item)
    elif isinstance(user, BorrowOp):
        targets.append(user.value)
    elif isinstance(user, ForkOp):
        targets.extend(user.forks)
    elif isinstance(user, RequestOp):
        handler = handlers.get(user.handler.root_reference.data)
        if handler is not None and (entry := supplied_entry(handler, "accepted_request")) is not None:
            targets.append(entry)
    elif isinstance(user, FilterOp) and index == 0:  # the item: its payload enters the body, and the item is the result
        targets.extend(user.results)
        if (entry := supplied_entry(user, "detached_input_payload")) is not None:
            targets.append(entry)
    elif not isinstance(user, CallOp) and user.get_trait(RegionSemantics) is not None:
        targets.extend(entries_fed(user, index))
    return targets


# Reachability.


@dataclass(frozen=True)
class Reached:
    """What a reachability query found.

    ``calls`` holds each call on a path into a sink, with the keys of the
    sinks it reaches; ``sources`` holds, for each sink key, the sources from
    which a path reaches that sink.
    """

    calls: Mapping[CallOp, frozenset[str]] = field(default_factory=dict)
    sources: Mapping[str, frozenset[SSAValue]] = field(default_factory=dict)


class ValueFlow:
    """A strategy's values and the edges between them: the core's own, and each call's for the relations it declares."""

    def __init__(self, strategy: StrategyOp, declaration: Callable[[CallOp], Declaration]):
        self.declaration = declaration
        handlers = {name: op for op in strategy.walk() if isinstance(op, HandlerOp) and (name := sym_name(op)) is not None}
        self._passed: dict[SSAValue, list[SSAValue]] = defaultdict(list)
        self.calls: list[CallOp] = []
        for op in strategy.walk():
            if isinstance(op, CallOp):
                self.calls.append(op)
                continue
            for index, operand in enumerate(op.operands):
                self._passed[operand].extend(passed_on(op, index, handlers))

    def _edges(self, relation: str, within: bool) -> tuple[dict[SSAValue, list[SSAValue]], dict[CallOp, tuple[tuple[int, int], ...]]]:
        forward: dict[SSAValue, list[SSAValue]] = defaultdict(list)
        for source, targets in self._passed.items():
            for target in targets:
                if not within or value_activity(source) == value_activity(target):
                    forward[source].append(target)
        through: dict[CallOp, tuple[tuple[int, int], ...]] = {}
        for call in self.calls:
            edges = tuple(
                (i, j)
                for i, j in self.declaration(call).edges(relation)
                if i < len(call.operands)
                and j < len(call.results)
                and (not within or value_activity(call.operands[i]) == value_activity(call.results[j]))
            )
            if edges:
                through[call] = edges
                for i, j in edges:
                    forward[call.operands[i]].append(call.results[j])
        return forward, through

    def reach(
        self,
        relation: str,
        sinks: Mapping[str, Collection[SSAValue]],
        sources: Collection[SSAValue] | None = None,
        *,
        within_activity: bool = False,
    ) -> Reached:
        """The calls on a path into each sink, from a source where sources are given, and which sources reach each sink.

        ``sinks`` holds, for each of the domain's keys, the values that are its
        sinks; one value may be a sink under several keys.
        """
        forward, through = self._edges(relation, within_activity)
        backward: dict[SSAValue, list[SSAValue]] = defaultdict(list)
        for source, targets in forward.items():
            for target in targets:
                backward[target].append(source)
        reaching: dict[str, set[SSAValue]] = defaultdict(set)
        for key, values in sinks.items():
            reaching[key] |= _closure(values, backward)
        started = _closure(sources, forward) if sources is not None else None
        calls: dict[CallOp, frozenset[str]] = {}
        for call, edges in through.items():
            keys = frozenset(
                key
                for key, values in reaching.items()
                for i, j in edges
                if call.results[j] in values and (started is None or call.operands[i] in started)
            )
            if keys:
                calls[call] = keys
        found = {key: frozenset(s for s in (sources or ()) if s in values) for key, values in reaching.items()}
        return Reached(calls, found)


def _closure(start: Iterable[SSAValue], edges: Mapping[SSAValue, list[SSAValue]]) -> set[SSAValue]:
    seen: set[SSAValue] = set()
    pending = deque(start)
    while pending:
        value = pending.popleft()
        if value in seen:
            continue
        seen.add(value)
        pending.extend(edges.get(value, ()))
    return seen


__all__ = [
    "Reached",
    "ValueFlow",
    "activity_of",
    "carried_entry",
    "entries_fed",
    "is_activity",
    "lease_use",
    "leased",
    "passed_on",
    "resource_of",
    "supplied_entry",
    "value_activity",
    "yielded_result",
]
