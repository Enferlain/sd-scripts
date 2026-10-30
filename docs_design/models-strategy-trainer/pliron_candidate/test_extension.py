"""Independent extension, analysis decisions, cold contracts and live failures."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import re
import subprocess
import sys
import threading
from dataclasses import asdict, replace
from types import MappingProxyType

import pytest

from . import pair_extension as extension
from .bridge import Native, Rejected, payloads
from . import test_b as base_tests

native = base_tests.native
fixture, preparation, runtime = base_tests.fixture, base_tests.preparation, base_tests.runtime

oracle = importlib.import_module("docs_design.models-strategy-trainer.xdsl_candidate.extension_conformance")


def ready(native: Native, *, fuse=False, opaque=False, registry=None, horizon=2):
    return native.prepare(
        native.accept(extension.build(opaque=opaque, horizon=horizon)), registry or extension.registry(opaque=opaque), fuse=fuse
    )


def with_pack(lower):
    registry = extension.registry()
    bindings = dict(registry.bindings)
    bindings["pack_pair_v1"] = replace(bindings["pack_pair_v1"], lower=lower)
    return replace(registry, bindings=bindings)


@pytest.mark.asyncio
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("opaque", [False, True])
@pytest.mark.parametrize("horizon", [1, 2])
async def test_typed_resource_extension_and_real_legality_decision(native: Native, fuse: bool, opaque: bool, horizon: int) -> None:
    effects = []
    prepared, artifact = ready(
        native, fuse=fuse, opaque=opaque, horizon=horizon, registry=extension.registry(opaque=opaque, effects=effects)
    )
    result = asdict((await fixture.execute(prepared, diagnostic_limit=0)).report)
    oracle.check_extension(result)
    groups = 3 if fuse and opaque else 1 if fuse else 4
    assert len(prepared.local.source_groups) == groups
    assert result["local_dispatches"] == len(result["tickets"]) * groups
    summary = artifact["resource_analysis"]
    assert summary["computations"] == 1
    assert summary["reads"] == [asdict(extension.USE)]
    assert summary["unknown_sources"] == (["data.offset"] if opaque else [])
    decisions = {d["origin"]: d for d in summary["decisions"]}
    assert decisions["pair.pack"]["groupable"] and decisions["pair.first"]["groupable"]
    if opaque:
        assert not decisions["data.offset"]["groupable"] and "unknown resource access" in decisions["data.offset"]["reason"]
        assert effects == [ticket["calls"][1]["input"] for ticket in result["tickets"]]
        assert all("data.offset" not in g or len(g) == 1 for g in prepared.local.source_groups)
    assert artifact["analysis"] == [[4, 1, 1], [groups, 2 if fuse else 1, 1]]
    assert "torch" not in sys.modules and not sys._is_gil_enabled()


@pytest.mark.parametrize("fuse", [False, True])
def test_cold_load_resolves_only_declared_frozen_snapshots(native: Native, fuse: bool, monkeypatch) -> None:
    received = []

    def lower(resources):
        assert isinstance(resources, MappingProxyType)
        assert dict(resources) == {extension.USE.symbol: extension.COEFFICIENTS}
        received.append(resources)
        return extension.lower_pack(resources)

    registry = with_pack(lower)
    registry = replace(registry, resources=dict(registry.resources) | {"unrelated": payloads.FrozenResource("9", (99,))})
    _, artifact = ready(native, fuse=fuse, registry=registry)
    assert len(received) == 1

    def forbidden(*a, **k):
        raise AssertionError("author/xDSL IR machinery used in cold load")

    monkeypatch.setattr(extension, "build", forbidden)
    monkeypatch.setattr(preparation, "prepare", forbidden)
    monkeypatch.setattr(preparation, "parse", forbidden)
    prepared, _ = native.load(artifact["accepted"], artifact["prepared"], artifact["manifest"], registry)
    assert len(received) == 2
    pack = prepared.local_contracts["pair.pack"]
    assert pack.input.version == "exact-integer-v1" and pack.output.version == "pair-int2-v1"
    assert pack.outcomes == (*payloads.DEFAULT_OUTCOMES, "negative_input")
    assert prepared.local_contracts["pair.first"].input is extension.PAIR


@pytest.mark.parametrize("boundary", ["prepare", "load"])
@pytest.mark.parametrize(
    "case", ["missing_resource", "version", "digest", "missing_codec", "codec_version", "signature", "outcomes", "access"]
)
def test_invalid_installed_contracts_reject_before_lowering(native: Native, boundary: str, case: str) -> None:
    registry = extension.registry()
    _, artifact = ready(native, registry=registry)
    lowered = []

    def lower(resources):
        lowered.append(True)
        return extension.lower_pack(resources)

    bindings = dict(registry.bindings)
    bindings["pack_pair_v1"] = replace(bindings["pack_pair_v1"], lower=lower)
    registry = replace(registry, bindings=bindings)
    if case == "missing_resource":
        registry = replace(registry, resources={})
    elif case == "version":
        registry = replace(registry, resources={extension.USE.symbol: payloads.FrozenResource("2", (1, 2))})
    elif case == "digest":
        registry = replace(registry, resources={extension.USE.symbol: payloads.FrozenResource("1", (1, 3))})
    elif case == "missing_codec":
        registry = replace(registry, codecs={"record": payloads.INTEGER})
    elif case == "codec_version":
        registry = replace(registry, codecs=dict(registry.codecs) | {"pair-int2": replace(extension.PAIR, version="pair-v99")})
    elif case == "signature":
        registry = replace(registry, signatures=dict(registry.signatures) | {"pack_pair_v1": ("record", "record")})
    elif case == "outcomes":
        bindings["pack_pair_v1"] = replace(bindings["pack_pair_v1"], outcomes=("negative_input",))
        registry = replace(registry, bindings=bindings)
    else:
        bindings["pack_pair_v1"] = replace(bindings["pack_pair_v1"], resources=(replace(extension.USE, access="write"),))
        registry = replace(registry, bindings=bindings)
    with pytest.raises(Rejected) as error:
        if boundary == "prepare":
            native.prepare(artifact["accepted"], registry, fuse=True)
        else:
            native.load(artifact["accepted"], artifact["prepared"], artifact["manifest"], registry)
    text = str(error.value)
    assert "pair.pack" in text and "scalar-async-cpu-v1" in text and "provided" in text, text
    if case in ("missing_resource", "version", "digest"):
        assert "pair_coefficients" in text and "version" in text and "digest" in text
    else:
        assert "expected" in text and ("schema" in text or "resources" in text)
    assert lowered == []


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("field", ["input", "output", "input_codec", "codec", "resources", "outcomes", "ancestors", "implementation"])
def test_cold_signature_corruption_rejected_before_resource_factory(native: Native, fuse: bool, field: str) -> None:
    lowered = []

    def lower(resources):
        lowered.append(True)
        return extension.lower_pack(resources)

    registry = with_pack(lower)
    _, artifact = ready(native, fuse=fuse, registry=registry)
    lowered.clear()
    text = artifact["prepared"]
    match = next(m for m in re.finditer(r"(?m)^\s+exec.call[^\n]+", text) if 'origin: builtin.string "pair.pack"' in m.group())
    line = match.group()
    if field == "input":
        bad = line.replace('input: builtin.string "record"', 'input: builtin.string "pair-int2"')
    elif field == "output":
        bad = line.replace('output: builtin.string "pair-int2"', 'output: builtin.string "record"')
    elif field == "input_codec":
        bad = line.replace('input_codec: builtin.string "exact-integer-v1"', 'input_codec: builtin.string "unchecked"')
    elif field == "codec":
        bad = line.replace('codec: builtin.string "pair-int2-v1"', 'codec: builtin.string "unchecked"')
    elif field == "resources":
        bad = line.replace(extension.USE.digest, "0" * 64)
    elif field == "outcomes":
        bad = line.replace("negative_input", "lost_outcome")
    elif field == "ancestors":
        bad = line.replace("data.nested", "unrelated")
    else:
        bad = line.replace("pack_pair_v1", "first_lane_v1")
    assert bad != line
    corrupted = text[: match.start()] + bad + text[match.end() :]
    with pytest.raises(Rejected) as error:
        native.load(artifact["accepted"], corrupted, artifact["manifest"], registry)
    diagnostic = str(error.value)
    assert "pair.pack" in diagnostic and "expected" in diagnostic and "provided" in diagnostic and field in diagnostic, diagnostic
    assert lowered == []


def test_forged_group_crossing_opaque_stage_is_rejected(native: Native) -> None:
    _, artifact = ready(native, opaque=True)
    text = artifact["prepared"]
    # Move all valid calls into a well-formed group. The source claims and
    # binding signatures stay valid; resource analysis must decline this group.
    lines = list(re.finditer(r"(?m)^\s+exec.call[^\n]+", text))[:4]
    assert len(lines) == 4
    calls = "\n".join(m.group().rstrip(";") + (";" if i < 3 else "") for i, m in enumerate(lines))
    wrapper = (
        'exec.fused () [] [protocol: builtin.string "preserve_each_stage_guard_and_completion"] : <() -> ()> {\n^forged():\n'
        + calls
        + "\n}"
    )
    corrupted = text[: lines[0].start()] + "\n" + wrapper + text[lines[-1].end() :]
    with pytest.raises(Rejected, match=r"data.offset.*ineligible fused stage.*unknown|data.offset.*ineligible fused stage.*may_effect"):
        native.load(artifact["accepted"], corrupted, artifact["manifest"], extension.registry(opaque=True))


@pytest.mark.parametrize("case", ["pack_schema", "first_schema", "access", "attributes"])
def test_invalid_extension_diagnostics(native: Native, case: str) -> None:
    text = extension.build()
    if case == "pack_schema":
        text = text.replace("-> (pairing.pair)", "-> (wf.record)").replace(
            "<(pairing.pair) -> (wf.record)>", "<(wf.record) -> (wf.record)>"
        )
    elif case == "first_schema":
        text = text.replace("<(pairing.pair) -> (wf.record)>", "<(pairing.pair) -> (pairing.pair)>").replace(
            "wf.yield (first) [] [] : <(wf.record) -> ()>", "wf.yield (first) [] [] : <(pairing.pair) -> ()>"
        )
    elif case == "access":
        text = text.replace('effect: builtin.string "read_frozen"', 'effect: builtin.string "write"')
    else:
        text = text.replace(
            'source_id: builtin.string "pair.pack"', 'extra: builtin.string "unknown", source_id: builtin.string "pair.pack"'
        )
    with pytest.raises(Rejected) as error:
        native.accept(text)
    diagnostic = str(error.value)
    assert ("pair.first" if case == "first_schema" else "pair.pack") in diagnostic
    assert "provided" in diagnostic and ("expected" in diagnostic or "requires" in diagnostic)
    if case == "access":
        assert "pair_coefficients" in diagnostic and "read_frozen" in diagnostic


def test_unsupported_target_diagnostic(native: Native) -> None:
    accepted = native.accept(extension.build(horizon=3))
    with pytest.raises(Rejected) as error:
        native.prepare(accepted, extension.registry())
    diagnostic = str(error.value)
    assert all(word in diagnostic for word in ["data-session", "scalar-async-cpu-v1", "provided horizon=3", "expected horizon 1..=2"])


@pytest.mark.parametrize("field", ["validate", "observe"])
def test_unusable_codec_rejected_before_lowering(native: Native, field: str) -> None:
    lowered = []

    def lower(resources):
        lowered.append(True)
        return extension.lower_pack(resources)

    registry = with_pack(lower)
    codec = replace(extension.PAIR, **{field: None})
    registry = replace(registry, codecs=dict(registry.codecs) | {"pair-int2": codec})
    with pytest.raises(Rejected, match=r"pair-int2.*callable validator and observer"):
        native.prepare(native.accept(extension.build()), registry)
    assert lowered == []


@pytest.mark.parametrize("malformed", [False, True])
def test_saved_manifest_mismatch_keeps_resource_factory_cold(native: Native, malformed: bool) -> None:
    lowered = []

    def lower(resources):
        lowered.append(True)
        return extension.lower_pack(resources)

    registry = with_pack(lower)
    _, artifact = ready(native, registry=registry)
    lowered.clear()
    manifest = copy.deepcopy(artifact["manifest"])
    if malformed:
        manifest["resources"] = None
    else:
        manifest["resources"][extension.USE.symbol]["version"] = "fake"
    with pytest.raises(Rejected, match="cold-load target resources"):
        native.load(artifact["accepted"], artifact["prepared"], manifest, registry)
    assert lowered == []


@pytest.mark.parametrize("case", ["codec", "schema", "dialect"])
def test_legacy_registry_adaptation_does_not_drop_contracts(native: Native, case: str) -> None:
    legacy = fixture.registry()
    if case == "codec":
        legacy = replace(legacy, codecs={next(iter(legacy.codecs)): replace(payloads.INTEGER, version="custom")})
    elif case == "schema":
        legacy = replace(legacy, codecs={preparation.FlowType(preparation.StringAttr("custom")): payloads.INTEGER})
    else:
        legacy = replace(
            legacy, dialects=(importlib.import_module("docs_design.models-strategy-trainer.xdsl_candidate.pair_extension").Pairing,)
        )
    with pytest.raises(Rejected, match=r"legacy registry installs.*use TargetRegistry"):
        native.prepare(native.accept(base_tests.author.build()), legacy)


@pytest.mark.parametrize("change", ["add", "remove"])
def test_runtime_reserved_outcome_drift_rejects_before_factories(native: Native, monkeypatch, change: str) -> None:
    lowered = []

    def lower(resources):
        lowered.append(True)
        return extension.lower_pack(resources)

    reserved = set(payloads.RESERVED_OUTCOMES)
    if change == "add":
        reserved.add("future_reserved")
    else:
        reserved.remove("invalid_input")
    monkeypatch.setattr(payloads, "RESERVED_OUTCOMES", frozenset(reserved))
    monkeypatch.setattr(runtime, "RESERVED_OUTCOMES", frozenset(reserved))
    with pytest.raises(Rejected, match=r"runtime outcome ABI mismatch.*expected.*provided"):
        native.prepare(native.accept(extension.build()), with_pack(lower))
    assert lowered == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("fault", ["invalid_pair", "hidden_lane", "declared", "undeclared"])
async def test_payload_validation_and_domain_outcomes(native: Native, fuse: bool, fault: str) -> None:
    def lower(resources):
        original = extension.lower_pack(resources)

        async def invoke(value):
            if fault in ("declared", "undeclared"):
                raise payloads.LocalOutcomeError("negative_input" if fault == "declared" else "unknown_outcome", "injected domain failure")
            result = await original(value)
            return (result[0], True) if fault == "invalid_pair" else (result[0], result[1] + 9)

        return invoke

    prepared, _ = ready(native, fuse=fuse, registry=with_pack(lower))
    report = asdict((await fixture.execute(prepared, diagnostic_limit=0)).report)
    if fault == "hidden_lane":
        assert report["status"] == "stopped"
        with pytest.raises(AssertionError, match="pair resource/schema arithmetic"):
            oracle.check_extension(report)
        # Final scalar arithmetic can still be correct while the hidden lane is wrong.
        projected = copy.deepcopy(report)
        for ticket in projected["tickets"]:
            ticket["calls"] = [c for c in ticket["calls"] if c["origin"] not in ("pair.pack", "pair.first")]
        oracle.check_b(projected)
        return
    assert report["status"] == "failed"
    ticket = report["tickets"][0]
    assert [c["origin"] for c in ticket["calls"]] == ["data.double", "data.offset", "pair.pack"]
    call = ticket["calls"][-1]
    assert call["outcome"] == {"invalid_pair": "invalid_output", "declared": "negative_input", "undeclared": "undeclared_outcome"}[fault]
    assert call["entered"] < call["physical_end"] < call["released"] < ticket["retired"] < report["released"]
    assert ticket["publication_ack"] is None and ticket["feedback_ack"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("interrupt", ["stop", "cancel"])
@pytest.mark.parametrize("fail", [False, True])
async def test_resource_use_survives_physical_interruption(native: Native, fuse: bool, interrupt: str, fail: bool) -> None:
    entered, release, finished = (threading.Event() for _ in range(3))

    def lower(resources):
        snapshot = resources[extension.USE.symbol]

        async def invoke(value):
            def physical():
                entered.set()
                try:
                    if not release.wait(5):
                        raise TimeoutError("test did not release physical resource read")
                    assert snapshot is extension.COEFFICIENTS and snapshot.values == (1, 2)
                    if fail:
                        raise RuntimeError("physical frozen-resource failure")
                    return tuple(value * c for c in snapshot.values)
                finally:
                    finished.set()

            return await asyncio.to_thread(physical)

        return invoke

    prepared, _ = ready(native, fuse=fuse, registry=with_pack(lower))
    session = runtime.ScalarSession(prepared, diagnostic_limit=0)
    running = asyncio.create_task(session.run())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        session.stop()
        if interrupt == "cancel":
            running.cancel()
            await asyncio.sleep(0)
            running.cancel()
        await asyncio.sleep(0)
        assert not finished.is_set() and not running.done() and session.active_calls == 1 and session.report is None
    finally:
        session.stop()
        release.set()
    if interrupt == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 3)
        report = session.report
        assert report is not None and report.status == "cancelled"
    else:
        report = await asyncio.wait_for(running, 3)
    assert finished.is_set() and session.active_calls == session.outstanding == 0
    if not fail:
        oracle.check_extension(asdict(report))
    else:
        assert any("physical frozen-resource failure" in e for e in report.errors)
        call = report.tickets[0].calls[-1]
        assert call.origin == "pair.pack" and call.outcome == "entered_failed"
        assert call.entered < call.physical_end < call.released < report.tickets[0].retired < report.released


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
def test_fresh_process_extension_reconstruction(native: Native, optimized: bool, fuse: bool) -> None:
    _, artifact = ready(native, fuse=fuse)
    code = """
import asyncio,importlib,json,sys
from pathlib import Path
from dataclasses import asdict
p="docs_design.models-strategy-trainer."
b=importlib.import_module(p+"pliron_candidate.bridge")
e=importlib.import_module(p+"pliron_candidate.pair_extension")
f=importlib.import_module(p+"xdsl_candidate.fixture")
def forbidden(*a,**k):raise AssertionError("author/xDSL IR machinery called")
registry=e.registry()
e.build=forbidden
f.build=forbidden
b.preparation.parse=forbidden
b.preparation.prepare=forbidden
data=json.load(sys.stdin)
prepared,_=b.Native(Path(sys.argv[1])).load(data["accepted"],data["prepared"],data["manifest"],registry)
report=asdict(asyncio.run(f.execute(prepared,diagnostic_limit=0)).report)
print(json.dumps({"report":report,"gil":sys._is_gil_enabled(),"torch":"torch" in sys.modules}))
"""
    child = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), "-c", code, str(native.binary)],
        input=json.dumps(artifact),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    assert not result["gil"] and not result["torch"]
    oracle.check_extension(result["report"])
