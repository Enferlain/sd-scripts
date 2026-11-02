"""Torch-free A documents; native verification precedes shared owner lowering."""

from __future__ import annotations

import importlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TARGET = "candidate-owner-cpu-v1"
_BINDINGS = {
    "scale_v1": ("model", "load"),
    "double_v1": ("map", "pure"),
    "offset_v1": ("map", "pure"),
    "encode_v1": ("encode", "read_source"),
    "loss_v1": ("objective", "build_loss"),
    "measure_v1": ("measure", "current_read"),
}
_FIELDS = {"schema", "pliron", "target", "accepted", "prepared", "manifest", "accepted_sha256", "profile"}


class Rejected(ValueError):
    pass


def contracts(*, lowering: Literal["calls", "inline"] = "calls", workers: bool = True, fuse: bool = False) -> dict[str, Any]:
    return {
        "schema": 1,
        "target": TARGET,
        "registry": "correlated-fixtures-v1",
        "lowering": lowering,
        "workers": workers,
        "fuse": fuse,
        "bindings": {name: {"version": "1", "role": role, "effect": effect} for name, (role, effect) in _BINDINGS.items()},
    }


def target_module():
    return importlib.import_module("docs_design.models-strategy-trainer.model_target")


def registry():
    return target_module().model_registry()


def describe(selected, *, lowering="calls", workers=True, fuse=False):
    target = target_module()
    for name, binding in selected.bindings.items():
        if type(binding) is not target.DomainBinding or not callable(binding.invoke):
            raise Rejected(f"@{name}: A target requires an explicit callable domain binding")
    return {
        "schema": 1,
        "target": TARGET,
        "registry": selected.name,
        "lowering": lowering,
        "workers": workers,
        "fuse": fuse,
        "bindings": {
            name: {key: getattr(binding, key) for key in ("version", "role", "effect")} for name, binding in selected.bindings.items()
        },
    }


@dataclass(frozen=True)
class CorrelationNative:
    binary: Path

    def request(self, **request):
        try:
            result = subprocess.run([str(self.binary)], input=json.dumps(request), capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise Rejected(f"native A verification: {error}") from error
        if result.returncode:
            raise Rejected(result.stderr.strip())
        try:
            return json.loads(result.stdout)
        except ValueError as error:
            raise Rejected("native A verification returned invalid JSON") from error

    def accept(self, text: str) -> str:
        return self.request(action="accept", text=text)["accepted"]

    def prepare(self, accepted: str, manifest: Mapping[str, Any] | None = None):
        return self.request(action="prepare", accepted=accepted, manifest=contracts() if manifest is None else manifest)

    def verify_saved(self, artifact: Mapping[str, Any], installed: Mapping[str, Any]):
        if not isinstance(artifact, Mapping):
            raise Rejected("saved A artifact: expected mapping")
        if set(artifact) - _FIELDS:
            raise Rejected("saved A artifact: unknown fields")
        if (
            type(artifact.get("schema")) is not int
            or artifact["schema"] != 1
            or artifact.get("pliron") != "0.20.0"
            or artifact.get("target") != TARGET
        ):
            raise Rejected("unsupported saved A artifact/target")
        for key in ("accepted", "prepared"):
            if not isinstance(artifact.get(key), str) or not artifact[key]:
                raise Rejected(f"saved A artifact {key}: nonempty document required")
        if not isinstance(artifact.get("manifest"), Mapping):
            raise Rejected("saved A artifact manifest: expected binding mapping")
        verified = self.request(action="load", accepted=artifact["accepted"], prepared=artifact["prepared"], manifest=installed)
        if artifact["manifest"] != installed:
            raise Rejected("saved A binding manifest differs from installed contracts")
        return verified

    def materialize(self, artifact: Mapping[str, Any], selected=None, *, controls=None, lowering="calls", workers=True, fuse=False):
        target = target_module()
        selected = target.model_registry() if selected is None else target.DomainRegistry(selected.name, selected.bindings)
        verified = self.verify_saved(artifact, describe(selected, lowering=lowering, workers=workers, fuse=fuse))
        p = verified["profile"]  # Never use a saved inspection projection to authorize effects.
        spec = target.ModelSpec(
            tuple(target.ModelBindingSpec(**model) for model in p["models"]),
            *(p[key] for key in ("window", "encoder", "learner", "unit", "parameter", "learning_rate")),
            tuple(p["excluded"]),
            *(p[key] for key in ("request_after", "normalization", "encode_origin", "objective_origin", "evaluate_origin")),
        )
        calls = {origin: selected.bindings[symbol] for origin, symbol in p["bindings"].items()}
        addresses = {(row["window"], row["slot"]): row["sample"] for row in p["addresses"]}
        ports = {row["slot"]: (row["left"], row["right"]) for row in p["addresses"]}
        local = target.local_runtime.compile_local(
            tuple(tuple(target.local_runtime.Stage(origin, calls[origin].invoke) for origin in group) for group in p["local_groups"])
        )
        try:
            return target.lower(
                spec,
                calls,
                addresses,
                ports,
                target.Controls() if controls is None else controls,
                local,
                lowering=p["lowering"],
                workers=p["workers"],
            )
        except Exception as error:
            raise Rejected(f"A owner realization: {error}") from error
