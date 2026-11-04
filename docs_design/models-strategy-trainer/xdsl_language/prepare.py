"""Torch-free preparation and cold correspondence for the bounded owner target."""

from __future__ import annotations

import json

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from enum import Enum

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Attribute, Operation, SSAValue

from .contracts import (
    CompletionSemantics,
    FlowSemantics,
    origin,
    Protocol,
    RealizationSemantics,
    RegionSemantics,
    ResourceSemantics,
    symbol,
    text,
)
from .dialects import (
    BorrowOp,
    CallOp,
    ConstantOp,
    dump,
    parse,
    PayloadType,
    ProgramOp,
    RequestOp,
    ResourceOp,
    RetireOp,
    StopAfterOp,
    StreamOp,
    TakeOp,
    YieldOp,
)
from .normalize import ancestry, normalize, source_index, symbol_index, value_source
from .verify import Analysis, REFERENCE_TARGET, TargetLimit, Verification, verify


class CorrespondenceError(ValueError):
    pass


@dataclass(frozen=True)
class BindingSpec:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    version: str = "1"
    updates: tuple[int, ...] = ()


@dataclass(frozen=True)
class ResourceSpec:
    kind: str
    storage: str
    version: str = "1"
    parameter: str = ""
    learning_rate: float = 0.01


@dataclass(frozen=True)
class Policy:
    target: str = "owner_cpu_v1"
    physical: str = "host_workers"
    workers: int = 2


@dataclass(frozen=True)
class Manifest:
    bindings: tuple[BindingSpec, ...]
    resources: tuple[ResourceSpec, ...]
    schemas: tuple[str, ...]


def schema(value: SSAValue) -> str:
    return value.type.schema.data if isinstance(value.type, PayloadType) else value.type.name.removeprefix("lang.")


def data(value: object) -> object:
    if isinstance(value, SSAValue):
        return data(value_source(value))
    if isinstance(value, Attribute):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: data(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): data(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [data(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"not a prepared contract value: {type(value)}")


def document(value: object) -> str:
    return json.dumps(data(value), sort_keys=True, indent=2, allow_nan=False) + "\n"


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CorrespondenceError(f"duplicate artifact field {key}")
        result[key] = value
    return result


def decode(value: str) -> object:
    try:
        return json.loads(value, object_pairs_hook=_pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except ValueError as failed:
        raise CorrespondenceError(f"invalid artifact JSON: {failed}") from failed


@dataclass(frozen=True)
class Artifact:
    accepted: str
    normalized: str
    prepared: str

    def to_json(self) -> str:
        return document(asdict(self))

    @classmethod
    def from_json(cls, value: str) -> Artifact:
        decoded = decode(value)
        if (
            not isinstance(decoded, dict)
            or set(decoded) != {"accepted", "normalized", "prepared"}
            or any(type(item) is not str for item in decoded.values())
        ):
            raise CorrespondenceError("artifact requires exactly three text documents")
        return cls(**decoded)


@dataclass(frozen=True)
class Prepared:
    artifact: Artifact
    verification: Verification
    manifest: Manifest
    policy: Policy


_PRIMITIVES = {
    Protocol.ONCE: "local_sequence",
    Protocol.SEQUENCE: "local_sequence",
    Protocol.PARALLEL: "owner.Together",
    Protocol.REPEAT: "owner.Repeat",
    Protocol.REQUESTS: "owner.Requested",
    Protocol.PRODUCER: "owner.Producer",
    Protocol.ACCESS: "owner.use_or_isolated_retain",
    Protocol.CHOICE: "local_finite_choice",
}


def _common(left: Operation, right: Operation) -> Operation:
    ancestors = []
    current: Operation | None = left
    while current is not None:
        ancestors.append(current)
        current = current.parent_op()
    current = right
    while current not in ancestors:
        if current is None:
            raise CorrespondenceError("relationship crosses programs")
        current = current.parent_op()
    if current is None:
        raise CorrespondenceError("relationship crosses programs")
    return current


def _derive(module: ModuleOp, manifest: Manifest, policy: Policy) -> tuple[Verification, dict]:
    if policy.target != "owner_cpu_v1" or policy.physical != "host_workers" or type(policy.workers) is not int or policy.workers < 1:
        raise TargetLimit(policy.target, "target", "unsupported explicit physical realization or worker bound")
    checked = verify(normalize(module), target=replace(REFERENCE_TARGET, name=policy.target))
    normalized = checked.normalized.module
    index, symbols = source_index(normalized), symbol_index(normalized)
    root = next(iter(normalized.body.block.ops))
    bindings = {item.name: item for item in manifest.bindings}
    resources = {item.kind: item for item in manifest.resources}
    if (
        len(bindings) != len(manifest.bindings)
        or len(resources) != len(manifest.resources)
        or len(set(manifest.schemas)) != len(manifest.schemas)
    ):
        raise CorrespondenceError("duplicate installed manifest entry")
    if any(
        not item.version or any(i < 0 or i >= len(item.inputs) or item.inputs[i] != "capability" for i in item.updates)
        for item in bindings.values()
    ):
        raise CorrespondenceError("invalid installed binding update/version contract")
    if any(
        item.storage not in ("participant", "state") or not item.version or (item.storage == "participant" and not item.parameter)
        for item in resources.values()
    ):
        raise CorrespondenceError("invalid installed resource realization")
    fragments = []
    for key, op in index.items():
        if isinstance(op, ResourceOp | StreamOp) and op.parent_op() is not root:
            raise TargetLimit(policy.target, key, "persistent declarations must be root-owned")
        if isinstance(op, ResourceOp) and op.value.type.kind.data not in resources:
            raise TargetLimit(policy.target, key, "resource kind has no installed realization")
        region = op.get_trait(RegionSemantics)
        flow, use, realization, control = (
            op.get_trait(kind) for kind in (FlowSemantics, ResourceSemantics, RealizationSemantics, CompletionSemantics)
        )
        contract = region.describe(op) if region else None
        if contract is not None:
            if contract.protocol not in _PRIMITIVES:
                raise TargetLimit(policy.target, key, "invocation protocol is not installed")
            if contract.protocol == Protocol.REQUESTS and (contract.max_active != 1 or contract.max_outstanding != 1):
                raise TargetLimit(policy.target, key, "owner adapter currently supports one live request per handler")
            if contract.protocol == Protocol.PRODUCER:
                stops = [
                    candidate
                    for candidate in index.values()
                    if isinstance(candidate, StopAfterOp) and symbol(candidate.properties["target"]) == contract.symbol
                ]
                if len(stops) != 1:
                    raise TargetLimit(policy.target, key, "owner producer adapter requires one declared completion stop")
        elif not isinstance(
            op, ProgramOp | ResourceOp | StreamOp | ConstantOp | CallOp | TakeOp | BorrowOp | RetireOp | RequestOp | StopAfterOp | YieldOp
        ):
            raise TargetLimit(policy.target, key, "leaf realization is not installed")
        if realization:
            required = realization.describe(op)
            installed = bindings.get(required.binding)
            if installed is None or (installed.version, installed.inputs, installed.outputs) != (
                required.version,
                tuple(schema(x) for x in op.operands),
                tuple(schema(x) for x in op.results),
            ):
                raise CorrespondenceError(f"{key}: installed binding signature/version mismatch")
            for name in (*installed.inputs, *installed.outputs):
                if name not in ("capability", "index", "bool", "scalar") and name not in manifest.schemas:
                    raise CorrespondenceError(f"{key}: payload schema {name} has no installed validator")
        if control:
            description = control.describe(op)
            source = symbols[description.source]
            common = _common(symbols[description.target] if description.kind == "request" else op, source)
            current = source.parent_op()
            while current is not None and current is not common:
                parent_contract = current.get_trait(RegionSemantics)
                if parent_contract and parent_contract.protocol in (Protocol.REPEAT, Protocol.PRODUCER, Protocol.REQUESTS):
                    raise TargetLimit(
                        policy.target, key, "completion source can have several service invocations in one relationship scope"
                    )
                current = current.parent_op()
        fragments.append(
            {
                "source": key,
                "operation": op.name,
                "ancestry": ancestry(op),
                "operands": tuple(value_source(value) for value in op.operands),
                "results": tuple(value.type for value in op.results),
                "properties": dict(op.properties),
                "region": contract,
                "primitive": _PRIMITIVES[contract.protocol] if contract else "local_leaf",
                "flow": flow.describe(op) if flow else None,
                "access": use.describe(op) if use else None,
                "realization": realization.describe(op) if realization else None,
                "completion": control.describe(op) if control else None,
            }
        )
    analysis = Analysis(checked.normalized)
    for op, access in analysis.accesses.items():
        inner = analysis.value_sources(access.subject)
        parent = op.parent_op()
        while parent is not None:
            outer = analysis.accesses.get(parent)
            if outer is not None and outer.mode.value != "snapshot":
                shared = inner & analysis.value_sources(outer.subject)
                if any(isinstance(owner, ResourceOp) and resources[owner.value.type.kind.data].storage == "state" for owner in shared):
                    raise TargetLimit(
                        policy.target, origin(op), "state-cell protection currently excludes nested acquisition of the same owner"
                    )
            parent = parent.parent_op()
    return checked, {
        "format": "shared-owner-fragments-v1",
        "target": policy,
        "manifest": manifest,
        "fragments": fragments,
        "obligations": checked.obligations,
        "inputs": checked.inputs,
        "waits": checked.waits,
        "reproducibility": checked.reproducibility,
        "contributors": checked.contributors,
    }


def prepare(module: ModuleOp, manifest: Manifest, policy: Policy) -> Prepared:
    checked, recipe = _derive(module, manifest, policy)
    return Prepared(Artifact(dump(module), checked.normalized.text, document(recipe)), checked, manifest, policy)


def load(artifact: Artifact, manifest: Manifest, policy: Policy) -> Prepared:
    """Recompute correspondence; saved summaries never grant runtime authority."""
    accepted, normalized = parse(artifact.accepted), parse(artifact.normalized)
    if text(next(iter(normalized.body.block.ops)), "stage") != "normalized":
        raise CorrespondenceError("normalized artifact has the wrong compiler stage")
    checked, expected = _derive(accepted, manifest, policy)
    cold = normalize(normalized)
    if cold.structure != checked.normalized.structure:
        raise CorrespondenceError("normalized document does not correspond to accepted IR")
    if decode(artifact.prepared) != data(expected):
        raise CorrespondenceError("prepared source/contract/manifest/policy correspondence failed")
    return Prepared(artifact, checked, manifest, policy)
