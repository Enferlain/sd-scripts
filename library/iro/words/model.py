"""Model-use words: resource and relationship (D6, D16).

A participant is a resource of kind ``participant``. Its routes and views are
named on the accesses that use it, and a call reaches the route through its
lease. A resource may name the implementation that materializes it; resources
naming the same materializer are materialized in one attempt, and a
materializer that needs another participant's structure reads its inspection
view (D17).
"""

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr, SymbolRefAttr, UnitAttr
from xdsl.ir import Attribute
from xdsl.irdl import irdl_op_definition, operand_def, opt_prop_def, prop_def, result_def, traits_def, var_operand_def
from xdsl.utils.exceptions import VerifyException

from library.iro.operation_contracts import path
from library.iro.words.base import DeclaredFields, Named, Word
from library.iro.words.types import RelationshipType, ResourceType

INSPECTION = "inspection"


@irdl_op_definition
class ResourceOp(Word):
    """Declares something an owner holds, such as a participant.

    ``reads`` are the participants whose views the materializer reads, with
    the view kind of each in ``read_views``. ``settings`` are the values the
    materializer receives after those views, such as the checkpoint it
    loads; they come from the strategy's parameters (D16, D18).
    ``may_leave_device`` permits a participant's device storage to leave the
    device between uses (D15). ``realize`` lists the realizations of a
    participant's routes, each a name and its settings, such as
    ``realize(@mixed_precision, "bfloat16")``: what the calls through its
    routes run under, which preparation installs as the routes' call setups
    (D15, D16).
    """

    name = "iro.resource"

    sym_name = prop_def(StringAttr)
    materialize = opt_prop_def(SymbolRefAttr)
    read_views = opt_prop_def(ArrayAttr[StringAttr])
    settings = opt_prop_def(ArrayAttr)
    may_leave_device = opt_prop_def(UnitAttr)
    realize = opt_prop_def(ArrayAttr[ArrayAttr])
    reads = var_operand_def(ResourceType)
    handle = result_def(ResourceType)
    traits = traits_def(DeclaredFields(), Named())

    def verify_(self) -> None:
        views = tuple(view.data for view in self.read_views.data) if self.read_views is not None else ()
        if len(views) != len(self.reads):
            raise VerifyException(f"{path(self)}: each participant a materializer reads names its view")
        if self.reads and self.materialize is None:
            raise VerifyException(f"{path(self)}: only a materializer reads other participants")
        if self.settings is not None and self.materialize is None:
            raise VerifyException(f"{path(self)}: only a materializer takes settings")
        if self.settings is not None and not all(_literal(value) for value in self.settings.data):
            raise VerifyException(f"{path(self)}: a materializer's settings are literals")
        if any(view != INSPECTION for view in views):
            raise VerifyException(f"{path(self)}: a materializer reads a participant's inspection view")
        kind = self.handle.type.kind.data if isinstance(self.handle.type, ResourceType) else ""
        if self.may_leave_device is not None and kind != "participant":
            raise VerifyException(f"{path(self)}: only a participant's device storage may leave the device")
        if self.realize is not None:
            if kind != "participant":
                raise VerifyException(f"{path(self)}: only a participant's routes are realized")
            names = []
            for entry in self.realize.data:
                if not entry.data or not isinstance(entry.data[0], SymbolRefAttr):
                    raise VerifyException(f"{path(self)}: a realization names its implementation first")
                if not all(_literal(value) for value in entry.data[1:]):
                    raise VerifyException(f"{path(self)}: a realization's settings are literals")
                names.append(entry.data[0].root_reference.data)
            if len(set(names)) != len(names):
                raise VerifyException(f"{path(self)}: a realization is named once")


def _literal(value: Attribute) -> bool:
    """Whether a setting is a literal its implementation can receive: an integer, a float or a string."""
    return isinstance(value, IntegerAttr | FloatAttr | StringAttr)


@irdl_op_definition
class RelationshipOp(Word):
    """A declared relation between participants, such as an attachment, with its own identity and state."""

    name = "iro.relationship"

    sym_name = prop_def(StringAttr)
    source = operand_def(ResourceType)
    target = operand_def(ResourceType)
    handle = result_def(RelationshipType)
    traits = traits_def(DeclaredFields(), Named())

    def verify_(self) -> None:
        for end in (self.source, self.target):
            if not isinstance(end.type, ResourceType) or end.type.kind.data != "participant":
                raise VerifyException(f"{path(self)}: a relationship relates participants")
        if self.source == self.target:
            raise VerifyException(f"{path(self)}: a relationship relates two different participants")


OPERATIONS = (ResourceOp, RelationshipOp)

__all__ = ["INSPECTION", "OPERATIONS", "RelationshipOp", "ResourceOp"]
