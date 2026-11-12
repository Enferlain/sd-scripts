"""Model-use words: resource and relationship (D6, D16).

A participant is a resource of kind ``participant``. Its routes and views are
named on the accesses that use it, and a call reaches the route through its
lease. A resource may name the implementation that materializes it; resources
naming the same materializer are materialized in one attempt, and a
materializer that needs another participant's structure reads its inspection
view (D17).
"""

from xdsl.dialects.builtin import ArrayAttr, StringAttr, SymbolRefAttr
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
    the view kind of each in ``read_views``.
    """

    name = "iro.resource"

    sym_name = prop_def(StringAttr)
    materialize = opt_prop_def(SymbolRefAttr)
    read_views = opt_prop_def(ArrayAttr[StringAttr])
    reads = var_operand_def(ResourceType)
    handle = result_def(ResourceType)
    traits = traits_def(DeclaredFields(), Named())

    def verify_(self) -> None:
        views = tuple(view.data for view in self.read_views.data) if self.read_views is not None else ()
        if len(views) != len(self.reads):
            raise VerifyException(f"{path(self)}: each participant a materializer reads names its view")
        if self.reads and self.materialize is None:
            raise VerifyException(f"{path(self)}: only a materializer reads other participants")
        if any(view != INSPECTION for view in views):
            raise VerifyException(f"{path(self)}: a materializer reads a participant's inspection view")


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
