"""The reference semantics (D14, D16).

A deterministic, Torch-free interpreter that executes a normalized
strategy's lifecycle protocols with declared stand-ins in place of selected
computations, and records the owner-visible facts every runtime realization
is checked against. The conformance suite and the doctor's dry run use it.
"""

from library.iro.reference.interpreter import (
    Cancelled,
    Fact,
    Lease,
    Outcome,
    Product,
    Reference,
    ReferenceFailure,
    ReferenceLimit,
    Request,
    State,
    Ticket,
    Version,
)
from library.iro.reference.standins import CallResult, StandIn, Update, returning

__all__ = [
    "CallResult",
    "Cancelled",
    "Fact",
    "Lease",
    "Outcome",
    "Product",
    "Reference",
    "ReferenceFailure",
    "ReferenceLimit",
    "Request",
    "StandIn",
    "State",
    "Ticket",
    "Update",
    "Version",
    "returning",
]
