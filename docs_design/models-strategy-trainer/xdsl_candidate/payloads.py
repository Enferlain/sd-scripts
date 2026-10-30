"""Small installed payload/resource ABI for local preprocessing.

This is an explicit registry of trusted code, not discovery or a Python sandbox.
Resources here are immutable snapshots, not leases on models or mutable state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
from xdsl.ir import Operation

from .dialects import FlowType, LocalSignatureOp
from .local_contract import LocalTransform, ResourceUse

DEFAULT_OUTCOMES = ("returned", "invalid_output", "raised")
RESERVED_OUTCOMES = frozenset((*DEFAULT_OUTCOMES, "not_entered", "entered_failed", "invalid_input", "undeclared_outcome"))


@dataclass(frozen=True)
class PayloadCodec:
    version: str
    validate: Callable[[Any], bool]
    observe: Callable[[Any], Any]


INTEGER = PayloadCodec("exact-integer-v1", lambda value: type(value) is int, lambda value: value)


@dataclass(frozen=True)
class FrozenResource:
    version: str
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.version) is not str
            or not self.version
            or type(self.values) is not tuple
            or not self.values
            or any(type(value) is not int for value in self.values)
        ):
            raise ValueError("frozen local resource needs a version and nonempty immutable integer tuple")

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.values, separators=(",", ":")).encode()).hexdigest()


class LocalOutcomeError(ValueError):
    def __init__(self, outcome: str, message: str):
        super().__init__(message)
        self.outcome = outcome


@dataclass(frozen=True)
class PayloadContract:
    input: PayloadCodec
    output: PayloadCodec
    outcomes: tuple[str, ...]


class LocalBinding(Protocol):
    @property
    def resources(self) -> tuple[ResourceUse, ...]: ...

    @property
    def outcomes(self) -> tuple[str, ...]: ...

    @property
    def invoke(self) -> Callable[[Any], Awaitable[object]]: ...

    @property
    def lower(self) -> Callable[[Mapping[str, FrozenResource]], Callable[[Any], Awaitable[object]]] | None: ...


@dataclass(frozen=True)
class LocalResolution:
    signature: LocalSignatureOp
    contract: PayloadContract
    binding: LocalBinding
    resources: Mapping[str, FrozenResource]

    def lower(self) -> Callable[[Any], Awaitable[Any]]:
        invoke = self.binding.lower(self.resources) if self.binding.lower is not None else self.binding.invoke
        if not callable(invoke):
            raise ValueError(f"{self.signature.origin.data}: lowering did not provide a callable")
        return invoke


def resolve_local(
    op: Operation, binding: LocalBinding, codecs: Mapping[FlowType, PayloadCodec], resources: Mapping[str, FrozenResource]
) -> LocalResolution:
    source = op.properties.get("source_id")
    if not isinstance(source, StringAttr) or not source.data:
        raise ValueError(f"{op.name}: explicit local source identity required")
    interface = op.get_trait(LocalTransform)
    description = interface.describe(op) if interface is not None else None
    if description is None:
        raise ValueError(f"{source.data}: missing local transform description")
    if not isinstance(description.input_type, FlowType) or not isinstance(description.output_type, FlowType):
        raise ValueError(f"{source.data}: local stage schemas must be wf.flow types")
    if not set(DEFAULT_OUTCOMES).issubset(description.outcomes):
        raise ValueError(f"{source.data}: missing mandatory baseline local outcomes")
    input_codec = codecs.get(description.input_type)
    output_codec = codecs.get(description.output_type)
    if (
        input_codec is None
        or output_codec is None
        or any(not codec.version or not callable(codec.validate) or not callable(codec.observe) for codec in (input_codec, output_codec))
    ):
        raise ValueError(f"{source.data}: missing installed payload codec/version")
    if binding.outcomes != description.outcomes:
        raise ValueError(f"{source.data}: unsupported declared outcomes")
    if any(name in RESERVED_OUTCOMES - set(DEFAULT_OUTCOMES) for name in description.outcomes):
        raise ValueError(f"{source.data}: reserved runtime outcome declared as a domain outcome")
    uses = description.resources
    if uses is not None and binding.resources != uses:
        raise ValueError(f"{source.data}: resource declaration differs from lowering")
    if uses is None and binding.resources:
        raise ValueError(f"{source.data}: unknown resource access cannot grant a snapshot")
    snapshots: dict[str, FrozenResource] = {}
    rows = []
    for use in uses or ():
        if not isinstance(use, ResourceUse) or use.access != "read_frozen":
            raise ValueError(f"{source.data}: unsupported mutable/local resource access")
        resource = resources.get(use.symbol)
        if resource is None or (resource.version, resource.digest) != (use.version, use.digest):
            raise ValueError(f"{source.data}: missing/incompatible frozen resource @{use.symbol}")
        if use.symbol in snapshots:
            raise ValueError(f"{source.data}: duplicate resource use")
        snapshots[use.symbol] = resource
        rows.append(
            DictionaryAttr(
                {
                    "symbol": StringAttr(use.symbol),
                    "access": StringAttr(use.access),
                    "version": StringAttr(resource.version),
                    "digest": StringAttr(resource.digest),
                }
            )
        )
    if snapshots and binding.lower is None:
        raise ValueError(f"{source.data}: missing resource-aware lowering")
    signature = LocalSignatureOp(
        properties={
            "origin": source,
            "input_type": description.input_type,
            "output_type": description.output_type,
            "input_codec": StringAttr(input_codec.version),
            "output_codec": StringAttr(output_codec.version),
            "resources": ArrayAttr(rows),
            "outcomes": ArrayAttr(StringAttr(name) for name in description.outcomes),
        }
    )
    return LocalResolution(
        signature, PayloadContract(input_codec, output_codec, description.outcomes), binding, MappingProxyType(snapshots)
    )
