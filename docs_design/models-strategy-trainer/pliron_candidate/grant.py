"""Torch-free C documents and native verification before shared owner lowering."""

from __future__ import annotations

import importlib
import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

TARGET = "candidate-grant-cpu-v1"
PROFILE = "restored-two-pass-v1"
COMPLETION = "physical_return_then_owner_handback_verification"
_CONTRACTS = {"scale_v1": ("model", "load", "before_install"), "two_pass_v1": ("grant", "scoped_write", COMPLETION)}
_FIELDS = frozenset({"schema", "pliron", "target", "accepted", "prepared", "manifest", "accepted_sha256", "profile"})


class Rejected(ValueError):
    pass


def contracts(*, offered: bool = True, lowering: Literal["calls", "inline"] = "calls") -> dict[str, Any]:
    return {
        "schema": 1,
        "target": TARGET,
        "offered": [PROFILE] if offered else [],
        "lowering": lowering,
        "bindings": {
            name: {"version": "1", "role": role, "effect": effect, "completion": completion}
            for name, (role, effect, completion) in _CONTRACTS.items()
        },
    }


@dataclass(frozen=True)
class Binding:
    version: str
    role: str
    effect: str
    completion: str
    invoke: Callable[..., Any]


@dataclass(frozen=True)
class Registry:
    bindings: Mapping[str, Binding]
    offered: tuple[str, ...] = (PROFILE,)
    lowering: Literal["calls", "inline"] = "calls"

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))
        object.__setattr__(self, "offered", tuple(self.offered))

    def describe(self) -> dict[str, Any]:
        described = {}
        for name, binding in self.bindings.items():
            if not callable(binding.invoke):
                raise Rejected(f"@{name}: callable implementation required")
            described[name] = {key: getattr(binding, key) for key in ("version", "role", "effect", "completion")}
        return {"schema": 1, "target": TARGET, "offered": list(self.offered), "lowering": self.lowering, "bindings": described}


def registry(*, offered: bool = True, lowering: Literal["calls", "inline"] = "calls") -> Registry:
    target = importlib.import_module("docs_design.models-strategy-trainer.grant_target")
    functions = target.functions()
    return Registry(
        {name: Binding("1", *descriptor, functions[name]) for name, descriptor in _CONTRACTS.items()},
        (PROFILE,) if offered else (),
        lowering,
    )


@dataclass(frozen=True)
class GrantNative:
    binary: Path

    def request(self, **request: Any) -> dict[str, Any]:
        try:
            result = subprocess.run([str(self.binary)], input=json.dumps(request), capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise Rejected(f"native authority verification: {error}") from error
        if result.returncode:
            raise Rejected(result.stderr.strip())
        try:
            return json.loads(result.stdout)
        except ValueError as error:
            raise Rejected("native authority verification returned invalid JSON") from error

    def accept(self, text: str) -> str:
        return self.request(action="accept", text=text)["accepted"]

    def prepare(self, accepted: str, manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self.request(action="prepare", accepted=accepted, manifest=contracts() if manifest is None else manifest)

    def verify_saved(self, artifact: Mapping[str, Any], installed: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(artifact, Mapping):
            raise Rejected("saved authority artifact: expected mapping")
        unknown = set(artifact) - _FIELDS
        if unknown:
            raise Rejected(f"saved authority artifact: unknown fields {unknown}")
        if (
            type(artifact.get("schema")) is not int
            or artifact["schema"] != 1
            or artifact.get("pliron") != "0.20.0"
            or artifact.get("target") != TARGET
        ):
            raise Rejected("unsupported saved authority artifact/target")
        for name in ("accepted", "prepared"):
            if not isinstance(artifact.get(name), str) or not artifact[name]:
                raise Rejected(f"saved authority artifact {name}: expected nonempty document text")
        if not isinstance(artifact.get("manifest"), Mapping):
            raise Rejected("saved authority artifact manifest: expected binding mapping")
        verified = self.request(action="load", accepted=artifact["accepted"], prepared=artifact["prepared"], manifest=installed)
        if artifact["manifest"] != installed:
            raise Rejected("saved authority binding manifest differs from installed contracts")
        return verified

    def materialize(self, artifact: Mapping[str, Any], selected: Registry | None = None, *, history_limit: int = 256):
        if type(history_limit) is not int or history_limit < 0:
            raise Rejected("nonnegative diagnostic history limit required")
        selected = registry() if selected is None else selected
        verified = self.verify_saved(artifact, selected.describe())
        target = importlib.import_module("docs_design.models-strategy-trainer.grant_target")
        profile = verified["profile"]  # Saved inspection projection never authorizes execution.
        spec = target.GrantSpec(
            *(
                profile[name]
                for name in ("session", "model", "unit", "parameter", "initial", "learning_rate", "clip_norm", "grant", "advance")
            ),
            tuple(tuple(item) for item in profile["phases"]),
        )
        functions = {
            "model": selected.bindings[profile["model_implementation"]].invoke,
            "grant": selected.bindings[profile["grant_implementation"]].invoke,
        }
        try:
            return target.lower(spec, functions, lowering=profile["lowering"], history_limit=history_limit)
        except Exception as error:
            raise Rejected(f"{profile['session']}: grant owner realization: {error}") from error
