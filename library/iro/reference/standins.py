"""Declared stand-ins: what replaces selected computations in the reference semantics.

A stand-in honors the contract of the implementation it replaces: its input
and output types as the written form names them, the lease inputs it updates
and the inputs its result retains. It supplies fixture values, never real
computation.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Update:
    """A new value for the state behind the lease at input position ``lease``."""

    lease: int
    value: object


@dataclass(frozen=True)
class CallResult:
    """What a stand-in returns: its result values and the updates it makes through owner_update leases."""

    values: tuple[object, ...]
    updates: tuple[Update, ...] = ()


@dataclass(frozen=True)
class StandIn:
    """A stand-in for one implementation at one version.

    ``inputs`` and ``outputs`` are type names as the written form prints
    them, such as ``int``, ``latents``, ``protected_current lease`` or
    ``retained loss``. ``invoke`` receives the input values, with leases
    replaced by the state they give access to and retained results by their
    values.
    """

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    invoke: Callable[[tuple[object, ...]], CallResult]
    version: str = "1"
    updates: tuple[int, ...] = ()
    retains: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("a stand-in names its version")
        for position in (*self.updates, *self.retains):
            if not 0 <= position < len(self.inputs):
                raise ValueError(f"stand-in position {position} is not one of its {len(self.inputs)} inputs")
        for position in self.updates:
            if not self.inputs[position].endswith("lease"):
                raise ValueError(f"a stand-in updates state only through a lease, not input {position}")


def returning(*values: object) -> Callable[[tuple[object, ...]], CallResult]:
    """An invoke that ignores its inputs and returns fixed values."""
    return lambda _: CallResult(values)


__all__ = ["CallResult", "StandIn", "Update", "returning"]
