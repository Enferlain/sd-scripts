"""What training derives for calls at preparation: grad mode and the units each call serves (D15, D16, D20).

Training gives the ``derivative`` relation its meaning. Implementations
declare which inputs propagate it into which outputs; training names the
sources, the leases that read the live state a unit trains, never a
snapshot's copy of it, and the sinks, its contributions' losses keyed by
unit, and asks the core's reachability for the calls on a path between them.

A derivative of trained state never crosses an activity: it travels in
results that keep that state's lease, which are retained and never leave
their activity (D16). So training counts only paths inside the activity that
contributes. A call on such a path runs with grad mode, a call setup, and is
annotated with the units whose losses it reaches, from which the route's
held setup sets module mode (D20). Any other call runs without grad mode,
such as a frozen encoder whose output reaches a loss but carries no
derivative of trained state, or the trained model sampling in another
activity whose outputs training takes as data.

Training also checks what the core does not interpret. A call on such a path
that holds a lease must pass on results declared retained, since its graph
keeps the leased state alive. Every participant a unit trains must reach one
of the unit's losses by such a path; otherwise training refuses the strategy
and names what cut the path, or that no contribution offers the unit a loss.

Its refusals: ``iro-train-001`` a contribution whose unit preparation cannot
tell, ``iro-train-002`` a trained participant that reaches none of its unit's
losses, ``iro-train-003`` a lease-holding call on a path that declares no
retained result, ``iro-train-004`` a unit no contribution offers a loss.
"""

from collections import defaultdict
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import MappingProxyType

import torch
from xdsl.ir import SSAValue

from library.iro.operation_contracts import AccessMode
from library.iro.prepare.derive import Derived, Program
from library.iro.seams import Annotations, HoldNeed, annotate
from library.iro.words.calls import CallOp
from library.iro.words.types import LeaseType
from library.training.words import CONTRIBUTE

DERIVATIVE = "derivative"
"""The relation training asks about: an input whose derivative an output carries."""

UNITS = "units"
"""The annotation naming the units whose losses a call reaches."""


@dataclass(frozen=True)
class GradMode:
    """The call setup that turns grad mode on for the calls on a path from trained state into a unit's loss."""

    name: str = "training.grad_mode"
    version: str = "1"

    def entered(self, annotations: Annotations, handles: object) -> AbstractContextManager[object]:
        return torch.enable_grad()


GRAD_MODE = GradMode()


@dataclass(frozen=True)
class TrainingDerivation:
    """Training's deriver: grad mode and unit annotations for the calls on a path from trained state into a unit's loss.

    ``trains`` names, for each unit, the participants it trains, by address.
    ``participants`` names every participant whose routes training prepares,
    each with module mode's held setup: for every call that leases one, the
    derivation declares the state that setup will need, the units the call
    serves that train it (D16). ``contributions`` names, for each
    implementation that contributes a loss to a unit, the positions of the
    unit's lease and of the loss.
    """

    trains: Mapping[str, frozenset[str]]
    participants: frozenset[str] = frozenset()
    contributions: Mapping[str, tuple[int, int]] = field(default_factory=lambda: MappingProxyType({CONTRIBUTE: (0, 1)}))
    name: str = "training.derive"
    version: str = "1"

    def derive(self, program: Program) -> Mapping[CallOp, Derived]:
        losses = self._losses(program)
        sources: dict[str, set[SSAValue]] = defaultdict(set)
        trained_by: dict[str, set[str]] = defaultdict(set)
        for unit, participants in self.trains.items():
            for participant in participants:
                trained_by[participant].add(unit)
        reads: dict[SSAValue, str] = {}
        for value, resource, use in program.leases():
            if use.mode != AccessMode.SNAPSHOT and resource in trained_by:
                reads[value] = resource
                for unit in trained_by[resource]:
                    sources[unit].add(value)
        for unit, participants in sorted(self.trains.items()):
            if participants and unit not in losses:
                program.refuse(
                    "iro-train-004",
                    f"@{unit} trains {', '.join(f'@{p}' for p in sorted(participants))}, and no contribution offers @{unit} a loss, "
                    "so it would never be trained",
                    names=(f"@{unit}",),
                    fix=f"contribute a loss to @{unit} with iro.train.contribute, or remove the unit",
                )
        units: dict[CallOp, set[str]] = defaultdict(set)
        for unit, sinks in losses.items():
            reached = program.reach(DERIVATIVE, {unit: sinks}, sources[unit], within_activity=True)
            for call in reached.calls:
                units[call].add(unit)
            self._check_reached(program, unit, sinks, sources[unit], reads, reached.sources.get(unit, frozenset()))
        for call in units:
            self._check_retained(program, call, sorted(units[call]))
        derived: dict[CallOp, Derived] = {}
        for call in program.calls():
            served = sorted(units.get(call, ()))
            holds = self._holds(program, call, served)
            if served or holds:
                derived[call] = Derived(annotate({UNITS: served}) if served else (), (GRAD_MODE,) if served else (), holds)
        return derived

    def _holds(self, program: Program, call: CallOp, served: list[str]) -> tuple[HoldNeed, ...]:
        """The module-mode state each participant the call leases through a route will need: the units it serves that train that participant."""
        needs = {}
        for operand in call.operands:
            leased = program.lease(operand) if isinstance(operand.type, LeaseType) else None
            if leased is None:
                continue
            resource, use = leased
            # A lease through a route carries its module mode; an update or a view runs nothing through one.
            if resource in self.participants and use.view is None and use.mode not in (AccessMode.OWNER_UPDATE, AccessMode.GRANT):
                needs[resource] = HoldNeed(resource, tuple(unit for unit in served if resource in self.trains.get(unit, ())))
        return tuple(needs[r] for r in sorted(needs))

    def _losses(self, program: Program) -> dict[str, list[SSAValue]]:
        """Each unit's sinks: the losses its contributions take."""
        losses: dict[str, list[SSAValue]] = defaultdict(list)
        for call in program.calls():
            positions = self.contributions.get(program.implementation(call))
            if positions is None:
                continue
            lease, loss = positions
            leased = program.lease(call.operands[lease]) if lease < len(call.operands) else None
            if leased is None or loss >= len(call.operands):
                program.refuse(
                    "iro-train-001",
                    f"@{program.implementation(call)} contributes a loss, and preparation cannot tell which unit it contributes to",
                    at=call,
                    fix="contribute a loss with iro.train.contribute",
                )
                continue
            losses[leased[0]].append(call.operands[loss])
        return losses

    def _check_reached(
        self,
        program: Program,
        unit: str,
        sinks: list[SSAValue],
        sources: set[SSAValue],
        reads: Mapping[SSAValue, str],
        reaching: frozenset[SSAValue],
    ) -> None:
        """Every participant the unit trains reaches one of its losses inside the activity that contributes (G3.6)."""
        reached = {reads[value] for value in reaching}
        missing = sorted(self.trains.get(unit, frozenset()) - reached)
        if not missing:
            return
        across = program.reach(DERIVATIVE, {unit: sinks}, sources).sources.get(unit, frozenset())
        crossing = {reads[value] for value in across}
        for participant in missing:
            if participant in crossing:
                why = (
                    "only from another activity, and a derivative never crosses an activity, since the results that carry it "
                    "are retained and never leave their activity"
                )
                fix = f"run @{participant}'s forward in the activity that contributes to @{unit}"
            else:
                why = "by no path: an input declared without an edge for the derivative, or a value taken as data, cuts it"
                fix = f"pass @{participant}'s output to the loss through inputs declared to carry its derivative"
            program.refuse(
                "iro-train-002",
                f"@{unit} trains @{participant}, which reaches @{unit}'s loss {why}, so @{participant} would never be trained",
                names=(f"@{unit}", f"@{participant}"),
                fix=fix,
            )

    def _check_retained(self, program: Program, call: CallOp, units: list[str]) -> None:
        """A call on a path from trained state that holds a lease passes on results declared retained (D16)."""
        declared = program.declaration(call)
        holds = [i for i, operand in enumerate(call.operands) if isinstance(operand.type, LeaseType)]
        if not holds or any(output.startswith("retained ") for output in declared.outputs):
            return
        leased = [found[0] for i in holds if (found := program.lease(call.operands[i])) is not None]
        names = ", ".join(f"@{name}" for name in leased)
        program.refuse(
            "iro-train-003",
            f"@{declared.name} lies on a path from trained state into the loss of {', '.join(f'@{u}' for u in units)} "
            f"while holding a lease on {names}, and declares no retained result, so the graph it records would keep "
            "that state in use after the lease ends",
            at=call,
            fix=f"declare @{declared.name}'s result retained",
        )


__all__ = ["DERIVATIVE", "GRAD_MODE", "UNITS", "GradMode", "TrainingDerivation"]
