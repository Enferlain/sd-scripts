"""Training's words: the unit, the window and contribution (D16, D17, D20).

Each is a derived word over existing protocols, so normalization expands it
into core words and the runtime needs no new lifecycle protocol (D17):

- ``iro.train.unit`` declares an optimization unit, a resource that Trainer
  optimization holds. It expands into ``iro.resource @name : unit``. Its
  members, optimizer, settings, clipping and ownership profile stay in the
  accepted strategy, from which Trainer optimization derives its plan (D20).
- ``iro.train.contribute %loss to %unit`` offers a loss to the unit: an
  access to the unit for an owner update, calling the profile's
  ``contribute``, which runs the backward pass into the unit's gradient
  accumulation. Normalization releases the loss after it (D16).
- ``iro.train.window %unit members N { ... }`` runs its body N times, then
  advances the unit: one access that takes the unit's update lease together
  with an update lease on each participant the unit's members belong to,
  calling the profile's ``advance`` (D17, D20). Those participants' numeric
  versions advance through their leases (D16).

Normalization expands later words first, so a window, and a contribution
inside it, expand while the unit they name is still a unit and its members
can be read. The standard profile's implementations are pinned like any
other implementation the strategy selects.
"""

from collections.abc import Sequence

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr, SymbolRefAttr, i64
from xdsl.ir import Block, Operation, ParametrizedAttribute, Region, SSAValue
from xdsl.irdl import (
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import IsolatedFromAbove
from xdsl.utils.exceptions import VerifyException

from library.iro.operation_contracts import AccessMode, DerivedWord, ResourceUseAttr, path, provider
from library.iro.words import types as t
from library.iro.words.access import AccessOp
from library.iro.words.base import DeclaredFields, Named, Word
from library.iro.words.calls import CallOp, YieldOp
from library.iro.words.model import ResourceOp
from library.iro.words.structure import RepeatOp, StrategyOp
from library.iro.words.types import ResourceType, RetainedType

UNIT_KIND = "unit"
"""The resource kind of an optimization unit, which Trainer optimization holds."""

STANDARD = "standard"
CONTRIBUTE = "standard.contribute"
ADVANCE = "standard.advance"


@irdl_attr_definition
class ProfileAttr(ParametrizedAttribute):
    """An ownership profile: who performs which advancement mechanics (D18, G3.1)."""

    name = "iro.train.profile"

    profile: StringAttr


def standard() -> ProfileAttr:
    return ProfileAttr(StringAttr(STANDARD))


# The unit.


def _expand_unit(op: Operation) -> tuple[Sequence[Operation], Sequence[SSAValue]]:
    assert isinstance(op, UnitOp)
    resource = ResourceOp.create(properties={"sym_name": op.sym_name}, result_types=[t.resource(UNIT_KIND)])
    return [resource], [resource.results[0]]


@irdl_op_definition
class UnitOp(Word):
    """Declares an optimization unit: the participants it trains, its optimizer with settings, its max_grad_norm and profile.

    ``members`` are the participants whose parameters the unit trains;
    ``setting_names`` and ``setting_values`` are the optimizer's settings by
    name, such as its learning rate; ``max_grad_norm`` is the gradient norm the
    standard profile clips to.
    """

    name = "iro.train.unit"

    sym_name = prop_def(StringAttr)
    optimizer = prop_def(SymbolRefAttr)
    setting_names = opt_prop_def(ArrayAttr[StringAttr])
    setting_values = opt_prop_def(ArrayAttr)
    max_grad_norm = opt_prop_def(FloatAttr)
    profile = prop_def(ProfileAttr)
    members = var_operand_def(ResourceType)
    handle = result_def(ResourceType)
    traits = traits_def(DeclaredFields(), Named(), DerivedWord(_expand_unit))

    def verify_(self) -> None:
        if not self.members:
            raise VerifyException(f"{path(self)}: a unit trains at least one participant")
        for member in self.members:
            if not isinstance(member.type, ResourceType) or member.type.kind.data != "participant":
                raise VerifyException(f"{path(self)}: a unit's members are participants")
        declared = [provider(member) for member in self.members]
        if len(set(declared)) != len(declared):
            raise VerifyException(f"{path(self)}: a unit names each participant it trains once")
        root: Operation = self
        while not isinstance(root, StrategyOp) and (parent := root.parent_op()) is not None:
            root = parent
        for other in root.walk():
            if isinstance(other, UnitOp) and other is not self and set(map(provider, other.members)) & set(declared):
                raise VerifyException(
                    f"{path(self)}: @{self.sym_name.data} and @{other.sym_name.data} train the same participant, "
                    "and standard ownership does not overlap"
                )
        names = self.setting_names.data if self.setting_names is not None else ()
        values = self.setting_values.data if self.setting_values is not None else ()
        if len(names) != len(values):
            raise VerifyException(f"{path(self)}: each optimizer setting has a name and a value")
        if len({name.data for name in names}) != len(names):
            raise VerifyException(f"{path(self)}: an optimizer setting is named once")
        if not isinstance(self.handle.type, ResourceType) or self.handle.type.kind.data != UNIT_KIND:
            raise VerifyException(f"{path(self)}: a unit is a resource of kind {UNIT_KIND}")
        if self.profile.profile.data != STANDARD:
            raise VerifyException(f"{path(self)}: only the standard profile is offered so far")


# Contribution.


def _expand_contribute(op: Operation) -> tuple[Sequence[Operation], Sequence[SSAValue]]:
    assert isinstance(op, ContributeOp)
    unit_type = op.unit.type
    assert isinstance(unit_type, ResourceType)
    body = Block(arg_types=[t.lease(unit_type, AccessMode.OWNER_UPDATE), op.loss.type])
    body.add_ops([CallOp.create(operands=list(body.args), properties={"implementation": SymbolRefAttr(CONTRIBUTE)}), YieldOp.create()])
    access = AccessOp.create(
        operands=[op.unit, op.loss],
        properties={"uses": ArrayAttr([ResourceUseAttr.of(AccessMode.OWNER_UPDATE)])},
        regions=[Region(body)],
    )
    return [access], []


@irdl_op_definition
class ContributeOp(Word):
    """Offers a loss to a unit: its backward pass accumulates into the unit (D17)."""

    name = "iro.train.contribute"

    loss = operand_def(RetainedType)
    unit = operand_def(ResourceType)
    traits = traits_def(DeclaredFields(), DerivedWord(_expand_contribute))

    def verify_(self) -> None:
        if not isinstance(self.unit.type, ResourceType) or self.unit.type.kind.data != UNIT_KIND:
            raise VerifyException(f"{path(self)}: a loss is contributed to a unit")


# The window.


def _expand_window(op: Operation) -> tuple[Sequence[Operation], Sequence[SSAValue]]:
    assert isinstance(op, WindowOp)
    unit = _unit_of(op)
    members = _visible_members(op, unit)
    body = Region()
    op.body.move_blocks(body)
    body.block.insert_arg(t.int_type(), len(body.block.args))
    repeat = RepeatOp.create(
        operands=list(op.captures),
        properties={
            "sym_name": StringAttr(_window_name(op, unit)),
            "limit": IntegerAttr(op.members.value.data, i64),
            "carry_count": IntegerAttr(0, i64),
        },
        regions=[body],
    )
    resources = [op.unit, *members]
    types = [value.type for value in resources]
    assert all(isinstance(type_, ResourceType) for type_ in types)
    inner = Block(arg_types=[t.lease(type_, AccessMode.OWNER_UPDATE) for type_ in types if isinstance(type_, ResourceType)])
    inner.add_ops([CallOp.create(operands=list(inner.args), properties={"implementation": SymbolRefAttr(ADVANCE)}), YieldOp.create()])
    advance = AccessOp.create(
        operands=resources,
        properties={"uses": ArrayAttr([ResourceUseAttr.of(AccessMode.OWNER_UPDATE)] * len(resources))},
        regions=[Region(inner)],
    )
    return [repeat, advance], []


def _unit_of(op: WindowOp) -> UnitOp:
    found = provider(op.unit)
    owner = found.owner if not isinstance(found.owner, Block) else None
    if not isinstance(owner, UnitOp):
        raise VerifyException("a window advances a unit the strategy declares with iro.train.unit")
    return owner


def _visible_members(op: WindowOp, unit: UnitOp) -> list[SSAValue]:
    """For each participant the unit trains, a value that stands for it where the window advances the unit."""
    block = op.parent_block()
    assert block is not None
    visible: list[SSAValue] = list(block.args)
    for earlier in block.ops:
        if earlier is op:
            break
        visible.extend(earlier.results)
    found = []
    for member in unit.members:
        declaration = provider(member)
        value = next((v for v in visible if provider(v) is declaration), None)
        if value is None:
            declared = declaration.owner.properties.get("sym_name") if not isinstance(declaration.owner, Block) else None
            name = declared.data if isinstance(declared, StringAttr) else "a member"
            raise VerifyException(f"@{unit.sym_name.data} trains @{name}, which is not captured where its window advances it")
        found.append(value)
    return found


def _window_name(op: WindowOp, unit: UnitOp) -> str:
    """The repeat's name: the unit's name, numbered by the window's place in the strategy among the unit's windows.

    Later words expand first, so every earlier window on the unit is still
    a window here.
    """
    root: Operation = op
    while not isinstance(root, StrategyOp) and (parent := root.parent_op()) is not None:
        root = parent
    earlier = 0
    for inner in root.walk():
        if inner is op:
            break
        if isinstance(inner, WindowOp) and _unit_of(inner) is unit:
            earlier += 1
    taken = {named.data for inner in root.walk() if isinstance(named := inner.properties.get("sym_name"), StringAttr)}
    base = f"{unit.sym_name.data}.window"
    count = earlier + 1
    name = base if count == 1 else f"{base}{count}"
    while name in taken:
        count += 1
        name = f"{base}{count}"
    return name


@irdl_op_definition
class WindowOp(Word):
    """Runs its body for each member contribution, then advances the unit under its profile (D17).

    The first operand is the unit; the rest are the body's captures, which
    the body receives in the same order.
    """

    name = "iro.train.window"

    members = prop_def(IntegerAttr)
    unit = operand_def(ResourceType)
    captures = var_operand_def()
    body = region_def("single_block")
    traits = traits_def(DeclaredFields(), IsolatedFromAbove(), DerivedWord(_expand_window))

    def verify_(self) -> None:
        if self.members.value.data < 1:
            raise VerifyException(f"{path(self)}: a window has at least one member")
        if not isinstance(self.unit.type, ResourceType) or self.unit.type.kind.data != UNIT_KIND:
            raise VerifyException(f"{path(self)}: a window advances a unit")
        if tuple(arg.type for arg in self.body.block.args) != tuple(value.type for value in self.captures):
            raise VerifyException(f"{path(self)}: a window's body receives its captures")


__all__ = [
    "ADVANCE",
    "CONTRIBUTE",
    "STANDARD",
    "UNIT_KIND",
    "ContributeOp",
    "ProfileAttr",
    "UnitOp",
    "WindowOp",
    "standard",
]
