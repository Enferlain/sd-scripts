"""Native verification boundary to B's existing loop-confined Python target.

Only native-verified source/binding/group projections create runtime artifacts.
The original runtime, stage guards, bindings and independent oracle are reused;
this module does not use xDSL IR construction, parsing, verification or passes.
"""

from __future__ import annotations

import json
import importlib
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

# The research parent contains a hyphen, so use its explicit import path.
_prefix = "docs_design.models-strategy-trainer.xdsl_candidate"
local_runtime = importlib.import_module(_prefix + ".local_runtime")
payloads = importlib.import_module(_prefix + ".payloads")
preparation = importlib.import_module(_prefix + ".prepare")


class Rejected(ValueError):
    """Native parsing, acceptance, preparation or correspondence rejected work."""


@dataclass(frozen=True)
class TargetRegistry:
    """Installed target contracts; schema names are data, not operation dispatch."""

    name: str
    bindings: Mapping[str, Any]
    codecs: Mapping[str, Any] = field(default_factory=lambda: {"record": payloads.INTEGER})
    resources: Mapping[str, Any] = field(default_factory=dict)
    signatures: Mapping[str, tuple[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("bindings", "codecs", "resources", "signatures"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def installed(registry: Any) -> TargetRegistry:
    if isinstance(registry, TargetRegistry):
        return registry
    default = len(registry.codecs) == 1 and all(
        type(schema) is preparation.FlowType
        and type(schema.payload) is preparation.StringAttr
        and schema.payload.data == "record"
        and codec is payloads.INTEGER
        for schema, codec in registry.codecs.items()
    )
    if not default:
        raise Rejected("legacy registry installs non-default codecs; use TargetRegistry with explicit native schema signatures")
    if registry.dialects:
        raise Rejected("legacy registry installs dialects; use TargetRegistry and native extension registration")
    return TargetRegistry(registry.name, registry.bindings, resources=registry.resources)


@dataclass(frozen=True)
class Native:
    binary: Path

    def request(self, **request: Any) -> dict[str, Any]:
        completed = subprocess.run([str(self.binary)], input=json.dumps(request), text=True, capture_output=True, timeout=30, check=False)
        if completed.returncode:
            raise Rejected(completed.stderr.strip())
        return json.loads(completed.stdout)

    def accept(self, text: str) -> str:
        return self.request(action="accept", text=text)["accepted"]

    def prepare(
        self, accepted: str, registry: Any, *, fuse: bool = False, capacity: int = 1
    ) -> tuple[preparation.Prepared, dict[str, Any]]:
        if type(capacity) is not int or capacity < 1:
            raise Rejected("positive integer capacity required")
        manifest = describe(registry)
        artifact = self.request(action="prepare", accepted=accepted, manifest=manifest, capacity=capacity, fuse=fuse)
        return _materialize(artifact, registry), artifact

    def load(self, accepted: str, prepared: str, manifest: Mapping[str, Any], registry: Any) -> tuple[preparation.Prepared, dict[str, Any]]:
        current = describe(registry)
        if not isinstance(manifest, Mapping):
            raise Rejected("cold-load target: saved binding manifest must be a mapping")
        # Native verification can attribute an incompatible installed contract to
        # its semantic source. No dispatch/lowering happens until both checks pass.
        artifact = self.request(action="load", accepted=accepted, prepared=prepared, manifest=current)
        if manifest != current:
            for section in ("resources", "codecs", "bindings"):
                expected = current.get(section, {})
                provided = manifest.get(section, {})
                if not isinstance(provided, Mapping):
                    raise Rejected(f"cold-load target {section}: expected mapping, provided {provided!r}")
                if expected != provided:
                    for name in sorted(set(expected) | set(provided)):
                        if expected.get(name) != provided.get(name):
                            raise Rejected(
                                f"cold-load target {section} @{name}: saved binding manifest differs; expected installed {expected.get(name)!r}, provided saved {provided.get(name)!r}"
                            )
            raise Rejected("cold-load target: saved binding manifest differs from installed registry/schema")
        return _materialize(artifact, registry), artifact


def describe(registry: Any) -> dict[str, Any]:
    registry = installed(registry)
    for schema, codec in registry.codecs.items():
        if not codec.version or not callable(codec.validate) or not callable(codec.observe):
            raise Rejected(f"target payload schema @{schema}: versioned callable validator and observer required")
    bindings = {}
    for symbol, binding in registry.bindings.items():
        if (
            not callable(binding.invoke)
            or (binding.resources and binding.lower is None)
            or (binding.lower is not None and not callable(binding.lower))
        ):
            raise Rejected(f"{symbol}: missing invocation/resource-aware lowering")
        input_schema, output_schema = registry.signatures.get(symbol, ("record", "record"))
        output_codec = registry.codecs.get(output_schema)
        uses = (
            None
            if binding.effect == "may_effect" and not binding.resources
            else [{"symbol": use.symbol, "access": use.access, "version": use.version, "digest": use.digest} for use in binding.resources]
        )
        bindings[symbol] = {
            "version": binding.version,
            "role": binding.role,
            "effect": binding.effect,
            "outcomes": list(binding.outcomes),
            "codec": output_codec.version if output_codec is not None else "",
            "input": input_schema,
            "output": output_schema,
            "resources": uses,
        }
    for symbol, resource in registry.resources.items():
        if not isinstance(resource, payloads.FrozenResource):
            raise Rejected(f"resource @{symbol}: immutable FrozenResource required")
    return {
        "schema": 2,
        "registry": registry.name,
        "bindings": bindings,
        "codecs": {schema: codec.version for schema, codec in registry.codecs.items()},
        "resources": {symbol: {"version": r.version, "digest": r.digest} for symbol, r in registry.resources.items()},
        "runtime_outcomes": sorted(payloads.RESERVED_OUTCOMES),
    }


def _materialize(artifact: Mapping[str, Any], registry: Any) -> preparation.Prepared:
    """Private target projection; callers must enter through Native preparation/load."""

    registry = installed(registry)

    def resolve(call: Mapping[str, Any]) -> preparation.ResolvedCall:
        return preparation.ResolvedCall(call["origin"], call["symbol"], registry.bindings[call["symbol"]])

    calls = tuple(call for group in artifact["producer"] for call in group)
    lowered = {}
    contracts = {}
    for call in calls:
        selected = registry.bindings[call["symbol"]]
        snapshots = MappingProxyType({use["symbol"]: registry.resources[use["symbol"]] for use in call["resources"] or ()})
        invoke = selected.lower(snapshots) if selected.lower is not None else selected.invoke
        if not callable(invoke):
            raise Rejected(f"{call['origin']}: lowering did not provide a callable")
        lowered[call["origin"]] = local_runtime.Stage(call["origin"], invoke)
        contracts[call["origin"]] = payloads.PayloadContract(
            registry.codecs[call["input"]], registry.codecs[call["output"]], tuple(call["outcomes"])
        )
    groups = tuple(tuple(lowered[call["origin"]] for call in group) for group in artifact["producer"])
    score, publish = (resolve(call) for call in artifact["consumer"])
    return preparation.Prepared(
        preparation.Accepted(artifact["accepted"]),
        artifact["prepared"],
        json.dumps(artifact["manifest"], sort_keys=True, indent=2) + "\n",
        score,
        publish,
        artifact["horizon"],
        artifact["capacity"],
        local_runtime.compile_local(groups),
        MappingProxyType(contracts),
    )
