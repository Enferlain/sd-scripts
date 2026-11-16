"""The optimization plan: the accepted strategy's units as typed definitions (D20).

Trainer optimization derives the plan from the accepted strategy, read with
the training contract, so the units are still ``iro.train.unit`` words with
their members; normalization has already turned each into a plain resource
for the IR. The plan is never a second definition: every field comes from
the strategy, and members are named by participant reference, never by a
live object (D9).

The first pass offers what one unit needs: whole participants as members,
one optimizer with its settings, ``max_grad_norm`` and the standard
profile. Logical groups, schedulers, synchronization and averaging come with
the second pass of section 16.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from xdsl.dialects.builtin import FloatAttr, IntegerAttr, IntegerType, StringAttr
from xdsl.ir import Attribute

from library.iro.operation_contracts import origin, provider
from library.iro.words.model import ResourceOp
from library.iro.words.structure import StrategyOp
from library.runstate.authority import ParticipantRef
from library.runstate.owners import Reference
from library.runstate.targets import TargetRef, participant_target
from library.training.words import UnitOp

Setting = str | int | float | bool


class UnitRef(Reference):
    """One unit incarnation, minted by Trainer optimization; a split, merge, retirement or recreation mints a new one (D9)."""

    kind = "unit"


@dataclass(frozen=True)
class OptimizerPolicy:
    """The optimizer a unit names and its settings by name, as the strategy gives them."""

    name: str
    settings: tuple[tuple[str, Setting], ...] = ()


@dataclass(frozen=True)
class UnitDefinition:
    """A unit's accepted meaning at one definition revision (D20).

    ``name`` is the unit's resource name. Only a governed change of meaning
    revises it; new parameters, wrappers or optimizer objects never do (D9).
    """

    name: str
    revision: int
    members: tuple[TargetRef, ...]
    optimizer: OptimizerPolicy
    max_grad_norm: float | None
    profile: str
    origin: str = ""


@dataclass(frozen=True)
class OptimizationPlan:
    """The accepted strategy's units, in the order the strategy declares them."""

    units: Mapping[str, UnitDefinition]

    @classmethod
    def derive(cls, strategy: StrategyOp, incarnation: Callable[[str], ParticipantRef]) -> OptimizationPlan:
        """The plan of an accepted strategy read with the training contract.

        ``incarnation`` names the participant declared at an address by its
        current reference, as the run authority gives it.
        """
        units: dict[str, UnitDefinition] = {}
        for op in strategy.walk():
            if not isinstance(op, UnitOp):
                continue
            members = []
            for member in op.members:
                declared = provider(member).owner
                assert isinstance(declared, ResourceOp), "verification keeps a unit's members participants"
                address = declared.sym_name.data
                members.append(participant_target(incarnation(address), address))
            names = op.setting_names.data if op.setting_names is not None else ()
            values = op.setting_values.data if op.setting_values is not None else ()
            settings = tuple((name.data, _value(value)) for name, value in zip(names, values, strict=True))
            units[op.sym_name.data] = UnitDefinition(
                op.sym_name.data,
                1,
                tuple(members),
                OptimizerPolicy(op.optimizer.root_reference.data, settings),
                float(op.max_grad_norm.value.data) if op.max_grad_norm is not None else None,
                op.profile.profile.data,
                origin(op) or "",
            )
        return cls(MappingProxyType(units))


def _value(value: Attribute) -> Setting:
    """A setting as the Python value the optimizer receives."""
    if isinstance(value, IntegerAttr) and isinstance(value.type, IntegerType) and value.type.width.data == 1:
        return bool(value.value.data)
    if isinstance(value, IntegerAttr):
        return int(value.value.data)
    if isinstance(value, FloatAttr):
        return float(value.value.data)
    if isinstance(value, StringAttr):
        return value.data
    raise ValueError(f"an optimizer setting is a literal, not {value}")


__all__ = ["OptimizationPlan", "OptimizerPolicy", "Setting", "UnitDefinition", "UnitRef"]
