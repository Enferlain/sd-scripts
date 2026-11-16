"""Candidate resolution: a unit's members resolved against the loaded structure inside one preparation attempt (D20).

Resolution reads the participants through the attempt's preparation
projection, pinned to the generation the attempt was derived from, and
holds the live parameters only in the candidates it returns. Parameters are
taken in module traversal order, the order the optimizer's state follows; a
parameter reached under several names, as tied weights are, is one member
that keeps its other names as aliases.

Two refusals happen here, beside verification's (D20):

- ownership does not overlap: a parameter two units both resolve to is
  refused, which catches overlap that the declared members do not show,
  such as weights tied across participants;
- a unit whose members resolve to no parameters is refused. The strategy's
  declaration that a unit may be empty comes with the second pass of
  section 16, so until then every empty unit is refused.

Resolution is a function of the participants it reads, so it runs again on
the same structure after a backend transformation such as FSDP2's sharding,
which replaces parameters in place and keeps their names (D20); the
optimizer is built afterwards, on the final parameters.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch

from library.runstate.authority import AUTHORITY
from library.runstate.owners import Dependencies
from library.runstate.projections import Preparation
from library.runstate.targets import TargetKind, TargetRef
from library.training.units.plan import OptimizationPlan, UnitDefinition


@dataclass(frozen=True)
class Member:
    """One resolved parameter: the target it resolved from, its name, the live parameter and its other names."""

    target: TargetRef
    name: str
    parameter: torch.nn.Parameter = field(repr=False, compare=False)
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnitCandidate:
    """A unit resolved inside one attempt, before its optimizer is built.

    ``participants`` holds, by address, each participant the unit trains, as
    the object its binding holds; ``modules`` the paths of the modules it
    trains in each, where the empty path is the whole participant. Module
    mode follows them (D20). ``dependencies`` name the bindings its members
    were resolved from.
    """

    definition: UnitDefinition
    members: tuple[Member, ...]
    participants: Mapping[str, torch.nn.Module]
    modules: Mapping[str, tuple[str, ...]]
    dependencies: Dependencies

    @property
    def parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(member.parameter for member in self.members)


def resolve(plan: OptimizationPlan, projection: Preparation) -> tuple[UnitCandidate, ...] | tuple[str, ...]:
    """Every unit of the plan resolved, or the reasons resolution refuses, each naming its unit."""
    reasons: list[str] = []
    candidates: list[UnitCandidate] = []
    owner: dict[int, tuple[str, str]] = {}  # each parameter's unit and name, across units
    for definition in plan.units.values():
        unit = f"@{definition.name}"
        members: list[Member] = []
        participants: dict[str, torch.nn.Module] = {}
        modules: dict[str, tuple[str, ...]] = {}
        bindings: list[tuple[str, str, int]] = []
        for target in definition.members:
            if target.kind != TargetKind.PARTICIPANT:
                reasons.append(f"{unit}: a member selected as {target.selector} is a {target.kind}; units train whole participants so far")
                continue
            try:
                participant = projection[target.participant]
            except KeyError as missing:
                reasons.append(f"{unit}: {missing.args[0]}")
                continue
            binding = participant.binding
            if binding is None:
                reasons.append(f"{unit}: {target.selector} is not bound")
                continue
            module = binding.value
            if not isinstance(module, torch.nn.Module):
                reasons.append(
                    f"{unit}: {target.selector} is bound to a {type(module).__name__}, and a whole participant trains a module's parameters"
                )
                continue
            participants[participant.address] = module
            modules[participant.address] = ("",)  # a whole participant trains its whole module
            bindings.append((AUTHORITY, str(participant.reference), binding.revision))
            named: dict[int, Member] = {}
            for path, parameter in module.named_parameters(remove_duplicate=False):
                name = f"{target.selector}.{path}"
                key = id(parameter)
                if key in named:
                    named[key] = Member(target, named[key].name, parameter, (*named[key].aliases, name))
                    continue
                named[key] = Member(target, name, parameter)
            for key, member in named.items():
                if key in owner and owner[key][0] != definition.name:
                    other, first = owner[key]
                    reasons.append(f"{unit}: {member.name} is also trained by @{other} as {first}, and standard ownership does not overlap")
                    continue
                if key in owner:  # tied across two of this unit's members: one member, with the other names as aliases
                    position = next(i for i, m in enumerate(members) if m.parameter is member.parameter)
                    first = members[position]
                    members[position] = Member(first.target, first.name, first.parameter, (*first.aliases, member.name, *member.aliases))
                    continue
                owner[key] = (definition.name, member.name)
                members.append(member)
        if not members and not any(reason.startswith(f"{unit}:") for reason in reasons):
            reasons.append(f"{unit}: its members resolve to no parameters")
        candidates.append(
            UnitCandidate(
                definition,
                tuple(members),
                MappingProxyType(participants),
                MappingProxyType(modules),
                Dependencies(exact=tuple(bindings)),
            )
        )
    if reasons:
        return tuple(reasons)
    return tuple(candidates)


__all__ = ["Member", "UnitCandidate", "resolve"]
