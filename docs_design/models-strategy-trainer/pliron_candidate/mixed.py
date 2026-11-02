"""Torch-free mixed documents; verification precedes shared owner realization."""

from __future__ import annotations

import importlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TARGET = "continuous-owner-cpu-v1"
_BINDINGS = {
    "scale_v1": ("model", "load"),
    "double_v1": ("map", "pure"),
    "offset_v1": ("map", "pure"),
    "encode_v1": ("encode", "read_source"),
    "loss_v1": ("objective", "build_loss"),
    "measure_v1": ("measure", "current_read"),
    "dynamic_filter_v1": ("filter", "read_policy"),
    "learned_bias_v1": ("policy", "derive_update"),
    "mixed_loss_v1": ("objective", "build_loss"),
}
_FIELDS = {"schema", "pliron", "target", "accepted", "prepared", "manifest", "accepted_sha256", "profile"}


class Rejected(ValueError):
    pass


def contracts(*, lowering: Literal["calls", "inline"] = "calls", fuse: bool = False) -> dict[str, Any]:
    return {
        "schema": 1,
        "target": TARGET,
        "registry": "continuous-learning-fixture-v1",
        "lowering": lowering,
        "fuse": fuse,
        "bindings": {name: {"version": "1", "role": role, "effect": effect} for name, (role, effect) in _BINDINGS.items()},
    }


def target_module():
    return importlib.import_module("docs_design.models-strategy-trainer.mixed_target")


def registry():
    return target_module().mixed_registry()


def describe(selected, *, lowering="calls", fuse=False):
    target = target_module()
    for name, binding in selected.bindings.items():
        if type(binding) is not target.DomainBinding or not callable(binding.invoke):
            raise Rejected(f"@{name}: mixed target requires an explicit callable domain binding")
    return {
        "schema": 1,
        "target": TARGET,
        "registry": selected.name,
        "lowering": lowering,
        "fuse": fuse,
        "bindings": {
            name: {key: getattr(binding, key) for key in ("version", "role", "effect")} for name, binding in selected.bindings.items()
        },
    }


@dataclass(frozen=True)
class MixedNative:
    binary: Path

    def request(self, **request):
        try:
            result = subprocess.run([str(self.binary)], input=json.dumps(request), capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise Rejected(f"native mixed verification: {error}") from error
        if result.returncode:
            raise Rejected(result.stderr.strip())
        try:
            return json.loads(result.stdout)
        except ValueError as error:
            raise Rejected("native mixed verification returned invalid JSON") from error

    def accept(self, text: str) -> str:
        return self.request(action="accept", text=text)["accepted"]

    def prepare(self, accepted: str, manifest: Mapping[str, Any] | None = None):
        return self.request(action="prepare", accepted=accepted, manifest=contracts() if manifest is None else manifest)

    def verify_saved(self, artifact: Mapping[str, Any], installed: Mapping[str, Any]):
        if not isinstance(artifact, Mapping):
            raise Rejected("saved mixed artifact: expected mapping")
        if set(artifact) - _FIELDS:
            raise Rejected("saved mixed artifact: unknown fields")
        if (
            type(artifact.get("schema")) is not int
            or artifact["schema"] != 1
            or artifact.get("pliron") != "0.20.0"
            or artifact.get("target") != TARGET
        ):
            raise Rejected("unsupported saved mixed artifact/target")
        for key in ("accepted", "prepared"):
            if not isinstance(artifact.get(key), str) or not artifact[key]:
                raise Rejected(f"saved mixed artifact {key}: nonempty document required")
        if not isinstance(artifact.get("manifest"), Mapping):
            raise Rejected("saved mixed artifact manifest: expected binding mapping")
        verified = self.request(action="load", accepted=artifact["accepted"], prepared=artifact["prepared"], manifest=installed)
        if artifact["manifest"] != installed:
            raise Rejected("saved mixed binding manifest differs from installed contracts")
        return verified

    def materialize(self, artifact: Mapping[str, Any], selected=None, *, controls=None, lowering="calls", fuse=False):
        target = target_module()
        selected = target.mixed_registry() if selected is None else target.DomainRegistry(selected.name, selected.bindings)
        verified = self.verify_saved(artifact, describe(selected, lowering=lowering, fuse=fuse))
        p = verified["profile"]  # Never use a saved inspection projection to authorize effects.
        fields = target.MixedSpec.__dataclass_fields__
        spec = target.MixedSpec(**{key: p[key] for key in fields})
        calls = {origin: selected.bindings[symbol] for origin, symbol in p["bindings"].items()}
        local = target.base.local_runtime.compile_local(
            tuple(tuple(target.base.local_runtime.Stage(origin, calls[origin].invoke) for origin in group) for group in p["local_groups"])
        )
        try:
            return target.lower(
                spec,
                calls,
                local,
                target.MixedControls() if controls is None else controls,
                lowering=p["lowering"],
            )
        except Exception as error:
            raise Rejected(f"mixed owner realization: {error}") from error
