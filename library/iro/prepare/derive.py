"""What a domain derives for calls at preparation: annotations and call setups (D16).

A domain gives its words meaning the core never learns. At preparation it
reads the normalized program through :class:`Program`, asks the core's
reachability over relations it names, and attaches to calls annotations,
written data such as unit names and module paths, and call setups, such as
grad mode. It may refuse the program instead, naming why.

Derivation runs on the normalized program before fusion, and calls fuse only
where what was derived for them is the same, compared and never merged. A
deriver is recorded by its name and the evidence of its source, and since
what it derives can change while no declaration does, its outputs on small
fixture strategies are checked in the repository; the prepared form prints
the record (D16).
"""

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, runtime_checkable

from xdsl.ir import Operation, SSAValue

from library.iro.diagnostics import Problems
from library.iro.operation_contracts import ResourceUse, path, symbol
from library.iro.prepare.flow import Reached, ValueFlow, activity_of, lease_use, resource_of
from library.iro.seams import Annotations, CallSetup, Declaration, HoldNeed
from library.iro.words.access import AccessOp
from library.iro.words.calls import CallOp
from library.iro.words.structure import StrategyOp


@dataclass(frozen=True)
class Derived:
    """What a domain derived for one call: its annotations, its call setups, and the states its held setups will need."""

    annotations: Annotations = ()
    setups: tuple[CallSetup, ...] = ()
    holds: tuple[HoldNeed, ...] = ()


class Program:
    """The normalized program as a deriver reads it, with the core's reachability."""

    def __init__(self, strategy: StrategyOp, declaration: Callable[[CallOp], Declaration], problems: Problems, deriver: str = ""):
        self.strategy = strategy
        self._declaration = declaration
        self._problems = problems
        self._deriver = deriver

    @cached_property
    def flow(self) -> ValueFlow:
        return ValueFlow(self.strategy, self._declaration)

    def calls(self) -> list[CallOp]:
        return list(self.flow.calls)

    def declaration(self, call: CallOp) -> Declaration:
        return self._declaration(call)

    def implementation(self, call: CallOp) -> str:
        """The name of the implementation a call selects."""
        return symbol(call.implementation)

    def activity(self, op: Operation) -> str:
        return activity_of(op)

    def lease(self, value: SSAValue) -> tuple[str, ResourceUse] | None:
        """For a lease, the resource it reaches and the use it was taken for."""
        use = lease_use(value)
        if use is None:
            return None
        resource = resource_of(use.subject)
        return (resource, use) if resource is not None else None

    def leases(self) -> list[tuple[SSAValue, str, ResourceUse]]:
        """Every lease the program takes, with the resource it reaches and the use it was taken for."""
        found = []
        for op in self.strategy.walk():
            if isinstance(op, AccessOp):
                for value in op.regions[0].block.args:
                    if (leased := self.lease(value)) is not None:
                        found.append((value, *leased))
        return found

    def reach(
        self,
        relation: str,
        sinks: Mapping[str, Collection[SSAValue]],
        sources: Collection[SSAValue] | None = None,
        *,
        within_activity: bool = False,
    ) -> Reached:
        """The calls on a path into each sink, from a source where sources are given (:class:`ValueFlow`)."""
        return self.flow.reach(relation, sinks, sources, within_activity=within_activity)

    def refuse(self, code: str, message: str, *, at: Operation | None = None, names: Iterable[str] = (), fix: str | None = None) -> None:
        """Refuses the program for a reason the domain gives; preparation reports every reason and prepares nothing."""
        prefix = f"{self._deriver}: " if self._deriver else ""
        self._problems.add(code, prefix + message, names=tuple(names), fix=fix, path=path(at) if at is not None else None)


@runtime_checkable
class Deriver(Protocol):
    """A domain's derivation at preparation, recorded by its name and the evidence of its source (D16)."""

    @property
    def name(self) -> str: ...

    def derive(self, program: Program) -> Mapping[CallOp, Derived]: ...


__all__ = ["Derived", "Deriver", "Program"]
