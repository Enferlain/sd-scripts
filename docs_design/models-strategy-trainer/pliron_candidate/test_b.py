"""Native representation/verification with independently checked live B facts."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from . import author
from .bridge import Native, Rejected, describe

_prefix = "docs_design.models-strategy-trainer.xdsl_candidate"
fixture = importlib.import_module(_prefix + ".fixture")
oracle = importlib.import_module(_prefix + ".conformance")
preparation = importlib.import_module(_prefix + ".prepare")
runtime = importlib.import_module(_prefix + ".runtime")


@pytest.fixture(scope="session")
def native() -> Native:
    target = Path(os.environ.get("CARGO_TARGET_DIR", "/tmp/sd-scripts-pliron-b-target"))
    manifest = Path(__file__).parent / "native" / "Cargo.toml"
    completed = subprocess.run(
        ["cargo", "build", "--locked", "--manifest-path", str(manifest)],
        env={**os.environ, "CARGO_TARGET_DIR": str(target)},
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return Native(target / "debug" / "rmst-pliron-b")


def ready(native: Native, *, fuse: bool = False, horizon: int = 2, registry=None):
    return native.prepare(native.accept(author.build(horizon=horizon)), registry or fixture.registry(), fuse=fuse)


def change_binding(symbol: str, invoke):
    bindings = dict(fixture.registry().bindings)
    bindings[symbol] = replace(bindings[symbol], invoke=invoke)
    return preparation.Registry("probe", bindings)


def test_actual_rewrite_and_analysis_invalidation(native: Native) -> None:
    accepted = native.accept(author.build())
    plain, before = native.prepare(accepted, fixture.registry())
    fused, after = native.prepare(accepted, fixture.registry(), fuse=True)
    assert before["pliron"] == after["pliron"] == "0.20.0"
    assert before["accepted"] == after["accepted"] == accepted
    assert before["accepted_sha256"] == after["accepted_sha256"]
    assert before["analysis"] == [[2, 1, 1], [2, 1, 1]]
    assert after["analysis"] == [[2, 1, 1], [1, 2, 1]]
    assert "exec.fused" not in plain.text and "exec.fused" in fused.text
    assert plain.local.source_groups == (("data.double",), ("data.offset",))
    assert fused.local.source_groups == (("data.double", "data.offset"),)
    assert len(fused.local.generated) == 1 and "frame.validate" in fused.local.generated[0]
    assert after["producer"][0][1]["ancestors"] == ["data-session", "data.composed", "data.nested"]


def test_graph_forward_reference_and_ordered_ssa_verification(native: Native) -> None:
    text = author.build()
    flow = next(line for line in text.splitlines() if "feedback = wf.flow" in line)
    # A definition after its graph use is valid; local SSA forward use is not.
    moved = text.replace(flow + "\n", "").replace("    wf.score", flow + "\n    wf.score")
    assert native.accept(moved)
    with pytest.raises(Rejected, match="not dominated by definition"):
        native.accept(text.replace("wf.map (input)", "wf.map (nested)"))


@pytest.mark.parametrize("case", ["horizon", "wiring", "entity", "receipt", "yield", "source", "contract", "effect", "unknown", "type"])
def test_bad_semantics_rejected(native: Native, case: str) -> None:
    substitutions = {
        "horizon": ('horizon: builtin.string "2"', 'horizon: builtin.string "0"'),
        "wiring": ("wf.score (mapped, scored)", "wf.score (selected, scored)"),
        "entity": ("wf.select (cursor, policy", "wf.select (policy, cursor"),
        "receipt": ('"returned_receipt"', '"ignore_receipt"'),
        "yield": ("wf.yield (nested)", "wf.yield (doubled)"),
        "source": ('"data.offset"', '"data.double"'),
        "contract": ('"scalar-feedback-v1"', '"unknown-v1"'),
        "effect": ('effect: builtin.string "pure"', 'effect: builtin.string "unknown"'),
        "unknown": ('source_id: builtin.string "scorer"', 'extra: builtin.string "bad", source_id: builtin.string "scorer"'),
        "type": ("wf.score)>", "wf.record)>"),
    }
    old, new = substitutions[case]
    text = author.build()
    assert old in text
    with pytest.raises(Rejected):
        native.accept(text.replace(old, new, 1))


def test_accepted_is_separate_from_target_readiness(native: Native) -> None:
    accepted = native.accept(author.build(horizon=3))
    with pytest.raises(Rejected, match="unsupported target"):
        native.prepare(accepted, fixture.registry())
    with pytest.raises(Rejected, match="unsupported target"):
        native.prepare(native.accept(author.build()), fixture.registry(), capacity=2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "score"),
        ("effect", "publication"),
        ("version", ""),
        ("codec", "unknown"),
        ("outcomes", ["negative_input"]),
        ("resources", ["mutable"]),
    ],
)
def test_binding_contract_rejected_before_projection(native: Native, field, value) -> None:
    manifest = describe(fixture.registry())
    manifest["bindings"]["double_v1"][field] = value
    with pytest.raises(Rejected, match="incompatible binding"):
        native.request(action="prepare", accepted=native.accept(author.build()), manifest=manifest, capacity=1, fuse=False)


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "target",
        "bound",
        "stop",
        "snapshot",
        "ancestry",
        "binding",
        "version",
        "origin",
        "role",
        "completion",
        "codec",
        "outcomes",
        "control",
        "missing",
        "duplicate",
        "fusion",
    ],
)
def test_cold_correspondence_rejects_before_dispatch(native: Native, monkeypatch, fuse: bool, case: str) -> None:
    _, artifact = ready(native, fuse=fuse)
    text = artifact["prepared"]
    replacements = {
        "target": ('target: builtin.string "scalar-async-cpu-v1"', 'target: builtin.string "unknown"'),
        "bound": ('horizon: builtin.string "2"', 'horizon: builtin.string "1"'),
        "stop": ('stop_rule: builtin.string "settle_selected"', 'stop_rule: builtin.string "discard"'),
        "snapshot": ('kind: builtin.string "cursor"', 'kind: builtin.string "policy"'),
        "ancestry": ('data.nested\\"]', 'wrong.parent\\"]'),
        "binding": ('implementation: builtin.string "double_v1"', 'implementation: builtin.string "offset_v1"'),
        "version": ('binding_version: builtin.string "1"', 'binding_version: builtin.string "2"'),
        "origin": ('origin: builtin.string "data.double"', 'origin: builtin.string "data.offset"'),
        "role": ('role: builtin.string "producer"', 'role: builtin.string "consumer"'),
        "completion": ('"await_physical_then_validate"', '"release_early"'),
        "codec": ('"exact-integer-v1"', '"unchecked"'),
        "outcomes": ('\\"invalid_output\\"', '\\"omitted\\"'),
        "control": ('"dedicated_task;receipt_then_apply_then_retire_then_release"', '"ignore_feedback"'),
        "fusion": ('"preserve_each_stage_guard_and_completion"', '"skip_validation"')
        if fuse
        else ('role: builtin.string "producer"', 'role: builtin.string "fused"'),
    }
    if case in ("missing", "duplicate"):
        match = re.search(r"(?m)^\s+exec.call[^\n]+data.double[^\n]+", text)
        assert match
        line = match.group()
        if case == "missing":
            # Keep the remaining statement separator valid.
            text = text[: match.start()] + text[match.end() :]
        else:
            text = text[: match.start()] + line + "\n" + text[match.start() :]
    else:
        old, new = replacements[case]
        if case == "binding":
            # Alter only the prepared claim, retaining the original semantic symbol.
            start = text.index("exec.call")
            assert old in text[start:]
            text = text[:start] + text[start:].replace(old, new, 1)
        else:
            assert old in text, (case, text)
            text = text.replace(old, new, 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid correspondence reached executable projection")

    monkeypatch.setattr(importlib.import_module(Native.__module__).local_runtime, "compile_local", forbidden)
    with pytest.raises(Rejected):
        native.load(artifact["accepted"], text, artifact["manifest"], fixture.registry())


@pytest.mark.asyncio
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("horizon", [1, 2])
@pytest.mark.parametrize("history", [0, 32])
async def test_live_b_from_cold_plan(native: Native, fuse: bool, horizon: int, history: int, monkeypatch) -> None:
    _, artifact = ready(native, fuse=fuse, horizon=horizon)

    def forbidden(*args, **kwargs):
        raise AssertionError("authoring/xDSL IR machinery used during cold load")

    monkeypatch.setattr(author, "build", forbidden)
    monkeypatch.setattr(preparation, "parse", forbidden)
    monkeypatch.setattr(preparation, "prepare", forbidden)
    prepared, _ = native.load(artifact["accepted"], artifact["prepared"], artifact["manifest"], fixture.registry())
    report = asdict((await asyncio.wait_for(fixture.execute(prepared, diagnostic_limit=history), 3)).report)
    oracle.check_b(report, horizon=horizon)
    assert report["status"] == "stopped" and len(report["tickets"]) >= 4
    assert report["local_dispatches"] == len(report["tickets"]) * (1 if fuse else 2)
    assert len(report["diagnostics"]) <= history
    assert "torch" not in sys.modules
    assert not sys._is_gil_enabled()


@pytest.mark.asyncio
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("symbol", ["double_v1", "offset_v1", "square_v1", "receipt_v1"])
@pytest.mark.parametrize("failure", ["invalid", "raise", "receipt"])
async def test_stage_failure_preserves_attribution(native: Native, fuse: bool, symbol: str, failure: str) -> None:
    async def bad(value):
        await asyncio.sleep(0)
        if failure == "raise":
            raise RuntimeError("selected failure")
        return value + 123 if failure == "receipt" else True

    prepared, _ = ready(native, fuse=fuse, horizon=1, registry=change_binding(symbol, bad))
    report = asdict((await asyncio.wait_for(fixture.execute(prepared, diagnostic_limit=0), 3)).report)
    if failure == "receipt" and symbol != "receipt_v1":
        # A numerically wrong returning map/score can satisfy its type contract;
        # the independent oracle, not framework verification, must catch it.
        with pytest.raises(AssertionError):
            oracle.check_b(report, horizon=1)
        return
    oracle.check_b(report, horizon=1)
    assert report["status"] == "failed"
    failed = report["tickets"][0]
    stage_index = ["double_v1", "offset_v1", "square_v1", "receipt_v1"].index(symbol)
    assert len(failed["calls"]) == stage_index + 1
    assert failed["calls"][-1]["outcome"] == ("entered_failed" if failure == "raise" else "invalid_output")
    assert failed["publication_ack"] is None and failed["feedback_ack"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("interrupt", ["stop", "cancel"])
@pytest.mark.parametrize("fail", [False, True])
async def test_physical_completion_survives_interruption(native: Native, fuse: bool, interrupt: str, fail: bool) -> None:
    entered, release, finished = (threading.Event() for _ in range(3))
    original = fixture.registry().bindings["double_v1"].invoke

    def physical():
        entered.set()
        try:
            if not release.wait(5):
                raise TimeoutError("test did not release worker")
            if fail:
                raise RuntimeError("physical failure after interruption")
        finally:
            finished.set()

    async def selected(value):
        await asyncio.to_thread(physical)
        return await original(value)

    prepared, _ = ready(native, fuse=fuse, registry=change_binding("double_v1", selected))
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
        assert report.status == ("failed" if fail else "stopped")
    assert finished.is_set() and session.active_calls == session.outstanding == 0
    oracle.check_b(asdict(report))
    if fail:
        assert any("physical failure" in error for error in report.errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("fuse", [False, True])
async def test_feedback_progress_during_blocked_publication(native: Native, fuse: bool, monkeypatch) -> None:
    gate = asyncio.Event()

    async def receipt(value):
        await gate.wait()
        return value

    prepared, _ = ready(native, fuse=fuse, registry=change_binding("receipt_v1", receipt))
    session = runtime.ScalarSession(prepared)
    original = session._apply_feedback
    progressed = []

    def apply(ticket, score):
        if not progressed:
            assert session.publication_waiting == {1} and session.buffered == 1
            original(ticket, score)
            assert session.publication_waiting == {1} and session.feedback_count == 1
            progressed.append(True)
            session.stop()
        else:
            original(ticket, score)

    monkeypatch.setattr(session, "_apply_feedback", apply)
    running = asyncio.create_task(session.run())
    try:
        await asyncio.wait_for(session.wait_for(lambda: session.publication_waiting == {1}), 3)
        assert session.outstanding == 2
    finally:
        gate.set()
    report = await asyncio.wait_for(running, 3)
    assert progressed == [True]
    oracle.check_b(asdict(report))


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["number", "policy", "association", "feedback", "release", "bound"])
async def test_oracle_corruption_controls(native: Native, corruption: str) -> None:
    prepared, _ = ready(native, fuse=True)
    report = asdict((await fixture.execute(prepared)).report)
    oracle.check_b(report)
    bad = copy.deepcopy(report)
    ticket = bad["tickets"][0]
    if corruption == "number":
        ticket["calls"][0]["output"] += 1
    elif corruption == "policy":
        bad["tickets"][1]["policy_revision"] += 1
    elif corruption == "association":
        ticket["calls"][0]["origin"] = "data.offset"
    elif corruption == "feedback":
        ticket["feedback_ack"] = ticket["publication_ack"]
    elif corruption == "release":
        ticket["calls"][0]["released"] = ticket["calls"][0]["entered"]
    else:
        bad["high_water"] = 3
    with pytest.raises(AssertionError):
        oracle.check_b(bad)


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_fresh_process_reconstructs_without_author_or_xdsl_passes(native: Native, fuse: bool, optimized: bool) -> None:
    _, artifact = ready(native, fuse=fuse)
    code = """
import asyncio, importlib, json, sys
from pathlib import Path
from dataclasses import asdict
p = "docs_design.models-strategy-trainer."
b = importlib.import_module(p + "pliron_candidate.bridge")
f = importlib.import_module(p + "xdsl_candidate.fixture")
def forbidden(*a, **k): raise AssertionError("author/xDSL machinery called")
importlib.import_module(p + "pliron_candidate.author").build = forbidden
f.build = forbidden
b.preparation.parse = forbidden
b.preparation.prepare = forbidden
artifact = json.load(sys.stdin)
ready, _ = b.Native(Path(sys.argv[1])).load(artifact["accepted"], artifact["prepared"], artifact["manifest"], f.registry())
report = asdict(asyncio.run(f.execute(ready, diagnostic_limit=0)).report)
print(json.dumps({"report": report, "gil": sys._is_gil_enabled(), "torch_imported": "torch" in sys.modules, "optimized": sys.flags.optimize}))
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
    assert result["report"]["status"] == "stopped" and not result["gil"] and not result["torch_imported"]
    assert result["optimized"] == int(optimized)
    # The independent assertion-based checker always runs in this normal parent.
    oracle.check_b(result["report"])
