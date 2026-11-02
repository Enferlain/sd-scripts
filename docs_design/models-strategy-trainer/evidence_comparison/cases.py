"""Fixture-only adapters to each independent frontend and its existing target."""

from __future__ import annotations

import asyncio
import importlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns
from typing import Any

PREFIX = "docs_design.models-strategy-trainer"
NATIVE_EVENTS: list[dict] = []


def module(name):
    return importlib.import_module(PREFIX + "." + name)


def measured_request(self, **request):
    # Profiling normalizes bridge failures; diagnostics use the original methods.
    start = perf_counter_ns()
    encoded = json.dumps(request)
    encode_ns = perf_counter_ns() - start
    start = perf_counter_ns()
    result = subprocess.run([str(self.binary), "--measure"], input=encoded, capture_output=True, text=True, timeout=30, check=False)
    wall_ns = perf_counter_ns() - start
    event = {
        "action": request.get("action"),
        "returncode": result.returncode,
        "parent_json_encode_ns": encode_ns,
        "subprocess_wall_ns": wall_ns,
        "input_bytes": len(encoded.encode()),
        "output_bytes": len(result.stdout.encode()),
    }
    if result.returncode:
        event["stderr"] = result.stderr
        NATIVE_EVENTS.append(event)
        raise RuntimeError(result.stderr)
    start = perf_counter_ns()
    payload = json.loads(result.stdout)
    event["parent_json_decode_ns"] = perf_counter_ns() - start
    event.update(payload["timing"])
    NATIVE_EVENTS.append(event)
    return payload["outcome"]["Ok"]


class Case:
    # Frontends deliberately expose different private fixture APIs.
    ir: Any
    b: Any
    pair: Any
    api: Any
    native: Any
    contract: Any
    author_module: Any
    selected: Any

    def __init__(self, name: str, framework: str, binaries: Path, *, measured: bool = False):
        self.name, self.framework = name, framework
        x, p = "xdsl_candidate.", "pliron_candidate."
        self.fixture = module(x + "fixture")
        if name in ("B", "extension"):
            self.ir = module(x + "prepare")
            if framework == "pliron":
                self.b = module(p + "bridge")
                self.api = self.b.Native(binaries / "rmst-pliron-b")
            if name == "extension":
                self.pair = module((x if framework == "xdsl" else p) + "pair_extension")
                self.selected = self.pair.extension_registry() if framework == "xdsl" else self.pair.registry()
                self.build = self.pair.build_extension if framework == "xdsl" else self.pair.build
            else:
                self.selected = self.fixture.registry()
                self.build = module(p + "author").build if framework == "pliron" else self.fixture.build
        elif name == "lifetime":
            self.build = module((x + "lifetime_author") if framework == "xdsl" else p + "composition_author").build
            if framework == "xdsl":
                self.ir = module(x + "lifetime")
                self.api, self.selected = self.ir.LifetimeCandidate(), self.ir.registry()
            else:
                binding_api = module(p + "composition")
                self.api, self.selected = binding_api.CompositionNative(binaries / "rmst-pliron-composition"), binding_api.registry()
        else:
            root = {"A": ("model", "correlation"), "C": ("grant", "grant"), "mixed": ("mixed", "mixed")}[name]
            if framework == "xdsl":
                self.ir, self.contract = module(x + root[0] + "_adapter"), module(x + root[0] + "_contract")
                self.selected = getattr(self.ir, {"A": "model_registry", "C": "grant_registry", "mixed": "mixed_registry"}[name])()
                self.build = getattr(self.contract, "build_" + root[0])
            else:
                self.native = module(p + root[1])
                self.api = getattr(self.native, {"A": "CorrelationNative", "C": "GrantNative", "mixed": "MixedNative"}[name])(
                    binaries / ("rmst-pliron-" + root[1])
                )
                self.selected, self.build = self.native.registry(), module(p + root[1] + "_author").build
        if measured and framework == "pliron":
            cls = type(self.api) if hasattr(self.api, "request") else module(p + "bridge").Native
            cls.request = measured_request
        self.accepted = self.accept(self.build())
        self.artifacts = {}

    def accept(self, authored):
        if self.name == "lifetime":
            return self.api.accept(self.ir.ir.printed(authored) if self.framework == "xdsl" else authored)
        if self.framework == "pliron":
            return self.api.accept(authored)
        if self.name in ("B", "extension"):
            return self.ir.accept(authored)
        return getattr(self.contract, {"A": "accept_model", "C": "accept_grant", "mixed": "accept_mixed"}[self.name])(authored)

    def options(self, fuse):
        if self.name == "A":
            return {"fuse": fuse, "lowering": "calls", "workers": True}
        if self.name == "mixed":
            return {"fuse": fuse, "lowering": "calls"}
        return {}

    def prepare(self, fuse=False):
        if self.name == "lifetime":
            return self.api.prepare(self.accepted, self.selected.describe())
        if self.framework == "pliron":
            if self.name in ("B", "extension"):
                prepared, artifact = self.api.prepare(self.accepted, self.selected, fuse=fuse)
                return artifact
            manifest = self.selected.describe() if self.name == "C" else self.native.describe(self.selected, **self.options(fuse))
            return self.api.prepare(self.accepted, manifest)
        if self.name in ("B", "extension"):
            prepared = self.ir.prepare(self.accepted, self.selected, fuse=fuse)
        else:
            options = self.options(fuse)
            if self.name == "mixed":
                options["controls"] = module("mixed_target").MixedControls(block_key=6, history_limit=0)
            prepared = getattr(self.ir, {"A": "prepare_model", "C": "prepare_grant", "mixed": "prepare_mixed"}[self.name])(
                self.accepted, self.selected, **options
            )
        return {"framework": "xdsl", "accepted": self.accepted.text, "prepared": prepared.text, "manifest": json.loads(prepared.manifest)}

    def cold(self, artifact, fuse=False):
        if self.name == "lifetime":
            return self.api.verify_saved(artifact, self.selected.describe())
        if self.framework == "pliron":
            if self.name in ("B", "extension"):
                return self.api.load(artifact["accepted"], artifact["prepared"], artifact["manifest"], self.selected)[1]
            return self.api.verify_saved(
                artifact, self.selected.describe() if self.name == "C" else self.native.describe(self.selected, **self.options(fuse))
            )
        return self.materialize(artifact, fuse)

    def materialize(self, artifact, fuse=False):
        if self.name == "lifetime":
            return self.api.materialize(artifact, self.selected)
        if self.framework == "pliron":
            if self.name in ("B", "extension"):
                return self.api.load(artifact["accepted"], artifact["prepared"], artifact["manifest"], self.selected)[0]
            options = self.options(fuse)
            if self.name == "mixed":
                options["controls"] = module("mixed_target").MixedControls(block_key=6, history_limit=0)
            return self.api.materialize(artifact, self.selected, **options)
        options = {"saved_text": artifact["prepared"], "saved_manifest": artifact["manifest"]}
        if self.name in ("B", "extension"):
            return self.ir.prepare(self.accepted, self.selected, fuse=fuse, **options)
        options.update(self.options(fuse))
        if self.name == "mixed":
            options["controls"] = module("mixed_target").MixedControls(block_key=6, history_limit=0)
        return getattr(self.ir, {"A": "prepare_model", "C": "prepare_grant", "mixed": "prepare_mixed"}[self.name])(
            self.accepted, self.selected, **options
        )

    def execute(self, artifact, fuse=False):
        if self.name == "lifetime":
            return asyncio.run(module("pliron_candidate.composition_demo").pressure(self.api, artifact, mode="late"))
        prepared = self.materialize(artifact, fuse)
        if self.name in ("B", "extension"):
            return asdict(asyncio.run(self.fixture.execute(prepared, stop_after=4, diagnostic_limit=0)).report)
        if self.name == "mixed":
            return asyncio.run(module("mixed_target").exercise(prepared))
        return asyncio.run(prepared.publish().run())

    def check(self, report):
        if self.name == "lifetime":
            return module("pliron_candidate.composition_conformance").check(report)
        names = {
            "B": ("conformance", "check_b"),
            "extension": ("extension_conformance", "check_extension"),
            "A": ("model_conformance", "check_a"),
            "C": ("grant_conformance", "check_c"),
            "mixed": ("mixed_conformance", "check_mixed"),
        }
        name, fn = names[self.name]
        return getattr(module("xdsl_candidate." + name), fn)(report)

    def invocation(self, fuse=False):
        authored = self.build()
        self.accepted = self.accept(authored)
        artifact = self.prepare(fuse)
        report = self.execute(artifact, fuse)
        return report

    def natural_invocation(self, fuse=False):
        """Execute the prepared object directly when the public API returns one."""
        self.accepted = self.accept(self.build())
        if self.name == "lifetime":
            return self.execute(self.prepare(fuse), fuse)
        if self.framework == "pliron":
            if self.name in ("B", "extension"):
                prepared, _ = self.api.prepare(self.accepted, self.selected, fuse=fuse)
            else:
                artifact = self.prepare(fuse)
                options = self.options(fuse)
                if self.name == "mixed":
                    options["controls"] = module("mixed_target").MixedControls(block_key=6, history_limit=0)
                prepared = self.api.materialize(artifact, self.selected, **options)
        else:
            if self.name in ("B", "extension"):
                prepared = self.ir.prepare(self.accepted, self.selected, fuse=fuse)
            else:
                options = self.options(fuse)
                if self.name == "mixed":
                    options["controls"] = module("mixed_target").MixedControls(block_key=6, history_limit=0)
                prepared = getattr(self.ir, {"A": "prepare_model", "C": "prepare_grant", "mixed": "prepare_mixed"}[self.name])(
                    self.accepted, self.selected, **options
                )
        if self.name in ("B", "extension"):
            return asdict(asyncio.run(self.fixture.execute(prepared, stop_after=4, diagnostic_limit=0)).report)
        if self.name == "mixed":
            return asyncio.run(module("mixed_target").exercise(prepared))
        return asyncio.run(prepared.publish().run())
