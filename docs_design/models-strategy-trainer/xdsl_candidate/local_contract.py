"""Interfaces queried by ordered local analysis and conservative rewriting."""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute, Operation
from xdsl.traits import OpTrait


@dataclass(frozen=True)
class ResourceUse:
    symbol: str
    access: str
    version: str
    digest: str


@dataclass(frozen=True)
class LocalDescription:
    """Selected-entry outcomes include returned/invalid_output/raised.

    Additional names describe domain failures, not replacements for this
    mandatory baseline supplied by the guarded execution ABI.
    """

    input_type: Attribute
    output_type: Attribute
    cardinality: str
    resources: tuple[ResourceUse, ...] | None
    outcomes: tuple[str, ...]


class LocalTransform(OpTrait):
    def describe(self, op: Operation) -> LocalDescription | None:
        return None  # Unknown semantics do not authorize a rewrite.


class ScalarTransform(LocalTransform):
    """Historical name: unary bindings now describe their actual flow schemas."""

    def describe(self, op: Operation) -> LocalDescription | None:
        if (
            len(op.operands) != 1
            or len(op.results) != 1
            or op.properties.get("effect") not in (StringAttr("pure"), StringAttr("may_effect"))
        ):
            return None
        return LocalDescription(
            op.operands[0].type,
            op.results[0].type,
            "one-to-one",
            () if op.properties.get("effect") == StringAttr("pure") else None,
            ("returned", "invalid_output", "raised"),
        )


class TransparentLocal(OpTrait):
    """Ordered local composition with no separate live resource lifetime.

    Source ancestry remains observable on its children. It does not authorize
    dropping child validation, failure, entry or completion boundaries.
    """
