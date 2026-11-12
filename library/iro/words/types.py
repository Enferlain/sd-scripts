"""Kuroiro's types. Types print by name (D16).

Value types are named after what they carry, such as ``record`` or ``loss``;
the literal and position types take Python's names: ``int``, ``float``,
``bool`` and ``str``. The other types wrap a value type or name a kind:
streams, claims, optionals, retained results, leases, requests, resources and
relationships. Claim and lease types never print: a take's type follows from
its stream, and captured, carried and leased values are named by their
clauses.
"""

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition

PYTHON_TYPES = ("int", "float", "bool", "str")
DEFAULT_RESOURCE_KIND = "state"


@irdl_attr_definition
class ValueType(ParametrizedAttribute, TypeAttribute):
    """A value a body computes, passes or yields, named by what it carries."""

    name = "iro.value"

    value_name: StringAttr

    @property
    def text(self) -> str:
        return self.value_name.data


@irdl_attr_definition
class StreamType(ParametrizedAttribute, TypeAttribute):
    """A bounded flow of items with an admission policy."""

    name = "iro.stream_type"

    payload: ValueType


@irdl_attr_definition
class ClaimType(ParametrizedAttribute, TypeAttribute):
    """An item taken from a stream, settled exactly once."""

    name = "iro.claim"

    payload: ValueType


@irdl_attr_definition
class OptionalType(ParametrizedAttribute, TypeAttribute):
    """Zero or one value."""

    name = "iro.optional"

    payload: ValueType


@irdl_attr_definition
class RetainedType(ParametrizedAttribute, TypeAttribute):
    """A result that keeps a dependency on leased state until its last use (D16)."""

    name = "iro.retained"

    payload: ValueType


@irdl_attr_definition
class ResourceType(ParametrizedAttribute, TypeAttribute):
    """Something an owner holds, such as a participant."""

    name = "iro.resource_type"

    kind: StringAttr


@irdl_attr_definition
class LeaseType(ParametrizedAttribute, TypeAttribute):
    """The handle an access gives its body for one resource; it cannot leave the scope."""

    name = "iro.lease"

    kind: StringAttr
    mode: StringAttr


@irdl_attr_definition
class RequestType(ParametrizedAttribute, TypeAttribute):
    """An owned pending request for a handler, carrying a value."""

    name = "iro.request_type"

    payload: ValueType


@irdl_attr_definition
class RelationshipType(ParametrizedAttribute, TypeAttribute):
    """A declared relation between participants, such as an attachment."""

    name = "iro.relationship_type"

    kind: StringAttr


def value(name: str) -> ValueType:
    return ValueType(StringAttr(name))


def int_type() -> ValueType:
    return value("int")


def bool_type() -> ValueType:
    return value("bool")


def stream(payload: ValueType) -> StreamType:
    return StreamType(payload)


def claim(payload: ValueType) -> ClaimType:
    return ClaimType(payload)


def optional(payload: ValueType) -> OptionalType:
    return OptionalType(payload)


def retained(payload: ValueType) -> RetainedType:
    return RetainedType(payload)


def resource(kind: str = DEFAULT_RESOURCE_KIND) -> ResourceType:
    return ResourceType(StringAttr(kind))


def participant() -> ResourceType:
    return resource("participant")


def lease(resource_type: ResourceType, mode: str) -> LeaseType:
    return LeaseType(resource_type.kind, StringAttr(mode))


def request(payload: ValueType) -> RequestType:
    return RequestType(payload)


def relationship(kind: str) -> RelationshipType:
    return RelationshipType(StringAttr(kind))


def payload_of(type_: Attribute) -> ValueType | None:
    """The value a stream, claim, optional, retained result or request carries."""
    carried = getattr(type_, "payload", None)
    return carried if isinstance(carried, ValueType) else None


def written(type_: Attribute) -> str:
    """A type as the written form prints it, such as ``optional latents`` or ``participant``."""
    if isinstance(type_, ValueType):
        return type_.text
    if isinstance(type_, OptionalType):
        return f"optional {type_.payload.text}"
    if isinstance(type_, RetainedType):
        return f"retained {type_.payload.text}"
    if isinstance(type_, StreamType):
        return f"stream {type_.payload.text}"
    if isinstance(type_, ClaimType):
        return f"claim {type_.payload.text}"
    if isinstance(type_, RequestType):
        return f"request {type_.payload.text}"
    if isinstance(type_, ResourceType | RelationshipType):
        return type_.kind.data
    if isinstance(type_, LeaseType):
        return f"{type_.mode.data} lease"
    return str(type_)


TYPES = (ValueType, StreamType, ClaimType, OptionalType, RetainedType, ResourceType, LeaseType, RequestType, RelationshipType)

__all__ = [
    "DEFAULT_RESOURCE_KIND",
    "PYTHON_TYPES",
    "TYPES",
    "ClaimType",
    "LeaseType",
    "OptionalType",
    "RelationshipType",
    "RequestType",
    "ResourceType",
    "RetainedType",
    "StreamType",
    "ValueType",
    "bool_type",
    "claim",
    "int_type",
    "lease",
    "optional",
    "participant",
    "payload_of",
    "relationship",
    "request",
    "resource",
    "retained",
    "stream",
    "value",
    "written",
]
