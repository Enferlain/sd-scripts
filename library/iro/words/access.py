"""Access words: access and release (D14, D16).

An access names one or more resources, each with its mode, version selection
and route or view, and gives its body one lease per resource. The leases are
taken together. A result that keeps a dependency on leased state is retained
until its last use, where normalization places its release.
"""

from xdsl.dialects.builtin import ArrayAttr
from xdsl.irdl import irdl_op_definition, operand_def, prop_def, region_def, traits_def
from xdsl.traits import IsolatedFromAbove
from xdsl.utils.exceptions import VerifyException

from library.iro.operation_contracts import (
    ALLOWED_SELECTIONS,
    AccessMode,
    LifecycleProtocol,
    ProductSemantics,
    RegionSemantics,
    ResourceSemantics,
    ResourceUseAttr,
    VersionSelection,
    path,
)
from library.iro.words.base import Activity, BodyShape, DeclaredFields, Word
from library.iro.words.types import RetainedType


@irdl_op_definition
class AccessOp(Activity):
    """Uses one or more resources within a scope.

    Its first operands are the resources, one per entry of ``uses``; the rest
    are captures.
    """

    name = "iro.access"

    uses = prop_def(ArrayAttr[ResourceUseAttr])
    body = region_def("single_block")
    traits = traits_def(
        RegionSemantics(LifecycleProtocol.ACCESS),
        ResourceSemantics(),
        BodyShape(),
        IsolatedFromAbove(),
        DeclaredFields(),
    )

    def verify_(self) -> None:
        subjects = self.operands[: len(self.uses.data)]
        if len(set(subjects)) != len(subjects):
            raise VerifyException(f"{path(self)}: an access names each resource once")
        for use in self.uses.data:
            try:
                mode, selection = AccessMode(use.mode.data), VersionSelection(use.selection.data)
            except ValueError as failed:
                raise VerifyException(f"{path(self)}: unknown access mode or version selection") from failed
            if selection not in ALLOWED_SELECTIONS[mode]:
                raise VerifyException(f"{path(self)}: {mode} cannot select {selection}")


@irdl_op_definition
class ReleaseOp(Word):
    """Ends a retained result's dependency on leased state, after its last use.

    Normalization places it; the normalized form shows it.
    """

    name = "iro.release"

    product = operand_def(RetainedType)
    traits = traits_def(ProductSemantics(), DeclaredFields())


OPERATIONS = (AccessOp, ReleaseOp)

__all__ = ["OPERATIONS", "AccessOp", "ReleaseOp"]
