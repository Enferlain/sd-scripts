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
from dataclasses import dataclass
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
        self, accepted: str, registry: preparation.Registry, *, fuse: bool = False, capacity: int = 1
    ) -> tuple[preparation.Prepared, dict[str, Any]]:
        if type(capacity) is not int or capacity < 1:
            raise Rejected("positive integer capacity required")
        manifest = describe(registry)
        artifact = self.request(action="prepare", accepted=accepted, manifest=manifest, capacity=capacity, fuse=fuse)
        return _materialize(artifact, registry), artifact

    def load(
        self, accepted: str, prepared: str, manifest: Mapping[str, Any], registry: preparation.Registry
    ) -> tuple[preparation.Prepared, dict[str, Any]]:
        if manifest != describe(registry):
            raise Rejected("saved binding manifest differs from installed bindings")
        artifact = self.request(action="load", accepted=accepted, prepared=prepared, manifest=manifest)
        return _materialize(artifact, registry), artifact


def describe(registry: preparation.Registry) -> dict[str, Any]:
    bindings = {}
    for symbol, binding in registry.bindings.items():
        if binding.resources or binding.lower is not None or binding.outcomes != payloads.DEFAULT_OUTCOMES or not callable(binding.invoke):
            raise Rejected(f"{symbol}: unsupported resources/lowering/outcomes/invocation")
        bindings[symbol] = {
            "version": binding.version,
            "role": binding.role,
            "effect": binding.effect,
            "outcomes": list(binding.outcomes),
            "codec": payloads.INTEGER.version,
            "resources": [],
        }
    return {"schema": 1, "registry": registry.name, "bindings": bindings}


def _materialize(artifact: Mapping[str, Any], registry: preparation.Registry) -> preparation.Prepared:
    """Private target projection; callers must enter through Native preparation/load."""

    def resolve(call: Mapping[str, Any]) -> preparation.ResolvedCall:
        return preparation.ResolvedCall(call["origin"], call["symbol"], registry.bindings[call["symbol"]])

    groups = tuple(tuple(resolve(call) for call in group) for group in artifact["producer"])
    score, publish = (resolve(call) for call in artifact["consumer"])
    return preparation.Prepared(
        preparation.Accepted(artifact["accepted"]),
        artifact["prepared"],
        json.dumps(artifact["manifest"], sort_keys=True, indent=2) + "\n",
        score,
        publish,
        artifact["horizon"],
        artifact["capacity"],
        local_runtime.compile_local(
            tuple(tuple(local_runtime.Stage(call.origin, call.binding.invoke) for call in group) for group in groups)
        ),
        MappingProxyType(
            {
                call.origin: payloads.PayloadContract(payloads.INTEGER, payloads.INTEGER, payloads.DEFAULT_OUTCOMES)
                for group in groups
                for call in group
            }
        ),
    )
