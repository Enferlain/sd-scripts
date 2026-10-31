"""Torch-free composition documents and verified handoff to the optional owner target."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .bridge import Native, Rejected

TARGET = "existing-owner-composition-cpu-v1"
_CONTRACTS = {
    "scale_v1": ("model", "load", "before_install"),
    "encode_v1": ("produce", "read_source", "physical_return_before_release"),
    "objective_v1": ("objective", "build_loss", "return_before_backward"),
    "measure_v1": ("evaluate", "protected_read", "physical_validate_restore"),
}


def contracts() -> dict[str, Any]:
    return {
        "schema": 1,
        "target": TARGET,
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))

    def describe(self) -> dict[str, Any]:
        bindings = {}
        for name, binding in self.bindings.items():
            if not callable(binding.invoke):
                raise Rejected(f"@{name}: callable implementation required")
            bindings[name] = {key: getattr(binding, key) for key in ("version", "role", "effect", "completion")}
        return {"schema": 1, "target": TARGET, "bindings": bindings}


def registry() -> Registry:
    runtime = importlib.import_module(__name__.rsplit(".", 1)[0] + ".composition_runtime")
    functions = runtime.functions()
    return Registry({name: Binding("1", *descriptor, functions[name]) for name, descriptor in _CONTRACTS.items()})


@dataclass(frozen=True)
class CompositionNative:
    binary: Path

    def accept(self, text: str) -> str:
        return Native(self.binary).accept(text)

    def prepare(self, accepted: str, manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return Native(self.binary).request(action="prepare", accepted=accepted, manifest=contracts() if manifest is None else manifest)

    def verify_saved(self, artifact: Mapping[str, Any], installed: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(artifact, Mapping):
            raise Rejected("saved composition artifact: expected a mapping")
        if type(artifact.get("schema")) is not int or artifact["schema"] != 1 or artifact.get("target") != TARGET:
            raise Rejected("unsupported saved composition artifact/target")
        accepted, prepared, saved = (artifact.get(name) for name in ("accepted", "prepared", "manifest"))
        for name, text in (("accepted", accepted), ("prepared", prepared)):
            if not isinstance(text, str) or not text:
                raise Rejected(f"saved composition artifact {name}: expected nonempty Pliron document text")
        if not isinstance(saved, Mapping):
            raise Rejected("saved composition artifact manifest: expected a binding mapping")
        verified = Native(self.binary).request(action="load", accepted=accepted, prepared=prepared, manifest=installed)
        if saved != installed:
            raise Rejected("saved composition binding manifest differs from installed contracts")
        return verified

    def materialize(self, artifact: Mapping[str, Any], selected: Registry | None = None, controls: Any = None):
        selected = registry() if selected is None else selected
        # Ignore the saved inspection projection. Native verification recreates
        # it from both raw documents and the installed manifest before factories.
        verified = self.verify_saved(artifact, selected.describe())
        runtime = importlib.import_module(__name__.rsplit(".", 1)[0] + ".composition_runtime")
        return runtime.lower(verified["profile"], {name: binding.invoke for name, binding in selected.bindings.items()}, controls)
