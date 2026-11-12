"""Data words: stream, take, borrow, dispose, filter and fork (D16).

A claim is affine: on every path it is consumed, transferred by a capture or
a yield, forked or disposed exactly once. Verification checks that; these
definitions state what each word does with claims.
"""

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, SymbolRefAttr
from xdsl.irdl import irdl_op_definition, operand_def, prop_def, region_def, result_def, traits_def, var_result_def
from xdsl.traits import IsolatedFromAbove
from xdsl.utils.exceptions import VerifyException

from library.iro.operation_contracts import (
    AUTHORED_DISPOSITIONS,
    Admission,
    Disposition,
    EndpointSemantics,
    EndpointUse,
    FlowSemantics,
    ItemContract,
    ItemSemantics,
    LifecycleProtocol,
    RegionSemantics,
    path,
)
from library.iro.words.base import Activity, BodyShape, DeclaredFields, Word
from library.iro.words.types import ClaimType, StreamType, ValueType

DEFAULT_METRIC = "owner_update_ordinal"
DEFAULT_COMPARISON = "item_admission"


@irdl_op_definition
class StreamOp(Word):
    """A bounded flow of items with an admission policy; the spec's "flow".

    ``outstanding`` bounds items produced and not yet settled, ``ready``
    those produced and not yet taken. Admission compares the versions of the
    named owners that production observed.
    """

    name = "iro.stream"

    sym_name = prop_def(StringAttr)
    outstanding = prop_def(IntegerAttr)
    ready = prop_def(IntegerAttr)
    admission = prop_def(StringAttr)
    owners = prop_def(ArrayAttr[SymbolRefAttr])
    metric = prop_def(StringAttr)
    boundary = prop_def(StringAttr)
    max_lag = prop_def(IntegerAttr)
    endpoint = result_def(StreamType)
    traits = traits_def(FlowSemantics(), DeclaredFields())

    def verify_(self) -> None:
        outstanding, ready = self.outstanding.value.data, self.ready.value.data
        if outstanding <= 0 or not 0 < ready <= outstanding:
            raise VerifyException(f"{path(self)}: bounds need 0 < ready <= outstanding")
        try:
            admission = Admission(self.admission.data)
        except ValueError as failed:
            raise VerifyException(f"{path(self)}: unknown admission policy {self.admission.data}") from failed
        lag = self.max_lag.value.data
        if lag < 0 or (lag > 0) != (admission == Admission.BOUNDED_LAG):
            raise VerifyException(f"{path(self)}: only bounded_lag admission has a maximum lag, and it is positive")
        if admission != Admission.ANY_WITH_PROVENANCE and not self.owners.data:
            raise VerifyException(f"{path(self)}: {admission} admission compares the versions of named owners")


@irdl_op_definition
class TakeOp(Word):
    """Takes the next admitted item from a stream as a claim."""

    name = "iro.take"

    source = operand_def(StreamType)
    item = result_def(ClaimType)
    traits = traits_def(ItemSemantics(ItemContract(creates=(0,))), EndpointSemantics(EndpointUse(0, "consume")), DeclaredFields())

    def verify_(self) -> None:
        stream, taken = self.source.type, self.item.type
        if not isinstance(stream, StreamType) or not isinstance(taken, ClaimType) or stream.payload != taken.payload:
            raise VerifyException(f"{path(self)}: a claim carries its stream's value type")


@irdl_op_definition
class BorrowOp(Word):
    """Reads a claim's value without owning the claim."""

    name = "iro.borrow"

    item = operand_def(ClaimType)
    value = result_def(ValueType)
    traits = traits_def(ItemSemantics(ItemContract(borrows=(0,), detached_outputs=(0,))), DeclaredFields())

    def verify_(self) -> None:
        taken = self.item.type
        if not isinstance(taken, ClaimType) or self.value.type != taken.payload:
            raise VerifyException(f"{path(self)}: a borrow returns the claim's value type")


@irdl_op_definition
class DisposeOp(Word):
    """Settles a claim with a disposition."""

    name = "iro.dispose"

    item = operand_def(ClaimType)
    disposition = prop_def(StringAttr)
    traits = traits_def(ItemSemantics(ItemContract(consumes=(0,))), DeclaredFields())

    def verify_(self) -> None:
        try:
            written = Disposition(self.disposition.data)
        except ValueError as failed:
            raise VerifyException(f"{path(self)}: unknown disposition {self.disposition.data}") from failed
        if written not in AUTHORED_DISPOSITIONS:
            raise VerifyException(f"{path(self)}: {written} is given by the system, not written")


@irdl_op_definition
class FilterOp(Activity):
    """Turns one claim into zero or one items.

    Its body receives the item's value and yields whether to keep it. The
    claim is settled forwarded or filtered_out; the result is an optional.
    """

    name = "iro.filter"

    body = region_def("single_block")
    traits = traits_def(
        RegionSemantics(LifecycleProtocol.FILTER),
        ItemSemantics(ItemContract(consumes=(0,), detached_outputs=(0,), forwards=(0,))),
        BodyShape(),
        IsolatedFromAbove(),
        DeclaredFields(),
    )


@irdl_op_definition
class ForkOp(Word):
    """Splits a claim into forks that each settle exactly once; the item settles when all have."""

    name = "iro.fork"

    item = operand_def(ClaimType)
    forks = var_result_def(ClaimType)
    traits = traits_def(ItemSemantics(ItemContract(consumes=(0,), forks=True)), DeclaredFields())

    def verify_(self) -> None:
        if len(self.forks) < 2:
            raise VerifyException(f"{path(self)}: a fork makes at least two forks")
        if any(fork.type != self.item.type for fork in self.forks):
            raise VerifyException(f"{path(self)}: forks carry the claim's value type")


OPERATIONS = (StreamOp, TakeOp, BorrowOp, DisposeOp, FilterOp, ForkOp)

__all__ = ["DEFAULT_COMPARISON", "DEFAULT_METRIC", "OPERATIONS", "BorrowOp", "DisposeOp", "FilterOp", "ForkOp", "StreamOp", "TakeOp"]
