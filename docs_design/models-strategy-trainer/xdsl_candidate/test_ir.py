"""Serialization, acceptance/readiness separation and target diagnostics."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
from xdsl.ir import Operation

from .dialects import (
    CallOp,
    ChainOp,
    ComposeOp,
    EntityOp,
    FlowOp,
    FlowType,
    LocalOp,
    MapOp,
    SelectOp,
    SessionOp,
    YieldOp,
    dump,
    integer,
    parse,
)
from .fixture import build, registry
from . import prepare as preparation_module
from .prepare import NotReady, Registry, Rejected, accept, prepare


def find[T: Operation](module: Operation, kind: type[T]) -> T:
    return next(op for op in module.walk() if isinstance(op, kind))


def test_snapshot_round_trip_and_detachment() -> None:
    author = build()
    accepted = accept(author)
    assert accept(parse(accepted.text)) == accepted
    prepared = prepare(accepted, registry())
    assert dump(parse(prepared.text)) == prepared.text
    assert accepted.digest in prepared.text
    assert "wf.local" in accepted.text and '!wf.flow<"feedback">' in accepted.text
    assert "await_physical_then_validate" in prepared.text
    assert [op.origin.data for op in parse(prepared.text).walk() if isinstance(op, CallOp)] == [
        "data.double",
        "data.offset",
        "scorer",
        "publisher",
    ]
    find(author, MapOp).properties["implementation"] = SymbolRefAttr("not_available")
    assert prepare(accepted, registry()).manifest == prepared.manifest
    assert accepted.obligations and "not_available" not in accepted.text


@pytest.mark.parametrize("horizon", [1, 2, 3, 99])
def test_semantic_horizon_is_independent_of_cpu_target(horizon: int) -> None:
    accepted = accept(build(horizon=horizon))
    if horizon <= 2:
        assert prepare(accepted, registry()).horizon == horizon
    else:
        with pytest.raises(NotReady, match=r"data-session.*target"):
            prepare(accepted, registry())


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("horizon", "data-session.*positive"),
        ("version", "data-session.*version"),
        ("stop", "data-session.*disposition"),
        ("identity", "data.double.*identity"),
        ("entity", "selector.*unresolved policy"),
        ("entity_kind", "selector.*unresolved policy"),
        ("endpoint", "one writer and one reader"),
        ("payload", "expected record"),
        ("local_chain", "data.offset.*chain"),
        ("yield", "data.nested.*yield"),
        ("effect", "data.double.*effect"),
        ("nested_symbol", "nested symbol"),
    ],
)
def test_semantic_rejections_have_source_diagnostics(case: str, diagnostic: str) -> None:
    module = build()
    session, select, stage = find(module, SessionOp), find(module, SelectOp), find(module, MapOp)
    if case == "horizon":
        session.properties["horizon"] = integer(0)
    elif case == "version":
        session.properties["version"] = integer(2)
    elif case == "stop":
        session.properties["stop_rule"] = StringAttr("drop_everything")
    elif case == "identity":
        find(module, LocalOp).properties["source_id"] = stage.source_id
    elif case == "entity":
        select.properties["policy"] = SymbolRefAttr("missing")
    elif case == "entity_kind":
        for op in module.walk():
            if isinstance(op, EntityOp) and op.kind.data == "policy":
                op.properties["kind"] = StringAttr("cursor")
    elif case == "endpoint":
        select.operands = [select.output, select.output]
    elif case == "payload":
        find(module, FlowOp).endpoint._type = FlowType(StringAttr("score"))
    elif case == "local_chain":
        local = find(module, LocalOp)
        find(local, MapOp).operands = [find(module, ComposeOp).body.block.args[0]]
    elif case == "yield":
        local = find(module, LocalOp)
        find(local, YieldOp).operands = [local.body.block.args[0]]
    elif case == "effect":
        stage.properties["effect"] = StringAttr("memory_read_therefore_removable")
    else:
        select.properties["policy"] = SymbolRefAttr("root", [StringAttr("policy")])
    with pytest.raises(Rejected, match=diagnostic):
        accept(module)


def test_irdl_structural_verifier_precedes_contract_acceptance() -> None:
    module = build()
    find(module, MapOp).properties["implementation"] = integer(42)
    with pytest.raises(Rejected, match="symbol_ref"):
        accept(module)


@pytest.mark.parametrize("mismatch", ["missing", "role", "effect", "version"])
def test_binding_failure_is_preparation_without_selected_entry(mismatch: str) -> None:
    bindings = dict(registry().bindings)
    binding = bindings["double_v1"]
    if mismatch == "missing":
        del bindings["double_v1"]
    else:
        bindings["double_v1"] = replace(binding, **{mismatch: "" if mismatch == "version" else "incompatible"})
    accepted = accept(build())
    with pytest.raises(NotReady, match=r"data.double.*binding @double_v1"):
        prepare(accepted, Registry("incomplete", bindings))


@pytest.mark.parametrize("capacity,target", [(2, "scalar-async-cpu-v1"), (1, "gpu_unqualified")])
def test_target_failures_do_not_reject_semantics(capacity: int, target: str) -> None:
    accepted = accept(build())
    with pytest.raises(NotReady, match="target"):
        prepare(accepted, registry(), capacity=capacity, target=target)


@pytest.mark.parametrize("horizon", [1, 2])
def test_loaded_document_executes_in_fresh_process_without_torch(tmp_path: Path, horizon: int) -> None:
    prepared = prepare(accept(build(horizon=horizon)), registry())
    semantic = tmp_path / "input.mlir"
    manifest = tmp_path / "bindings.json"
    semantic.write_text(prepared.accepted.text)
    manifest.write_text(prepared.manifest)
    # The fresh process replaces the authoring builder with a failure before
    # loading, proving it is unnecessary for reconstruction or execution.
    script = f"""
import sys, xdsl_candidate.fixture as fixture
fixture.build = lambda: (_ for _ in ()).throw(AssertionError('author object used'))
import xdsl_candidate.__main__ as cli
original_check = cli.check_b
def check(report, *, horizon=2):
    assert horizon == {horizon}, 'CLI lost the declared horizon'
    original_check(report, horizon=horizon)
cli.check_b = check
cli.main()
assert 'torch' not in sys.modules
"""
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
    completed = subprocess.run(
        [sys.executable, "-c", script, "--load", str(semantic), "--manifest", str(manifest), "--dump-dir", str(tmp_path / "loaded")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["report"]["status"] == "stopped" and result["accepted_digest"] == prepared.accepted.digest
    assert type(result["gil_enabled"]) is bool
    assert (tmp_path / "loaded" / "prepared.mlir").read_text() == prepared.text
    # Version mismatch must reject before execution in another fresh process.
    wrong = json.loads(prepared.manifest)
    wrong["bindings"][0]["version"] = "other-version"
    manifest.write_text(json.dumps(wrong))
    failed = subprocess.run(
        [sys.executable, "-c", script, "--load", str(semantic), "--manifest", str(manifest)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert failed.returncode != 0 and "binding manifest differs" in failed.stderr


@pytest.mark.parametrize("corruption", ["duplicate_chain", "wrong_call_role"])
def test_prepared_dispatch_checks_chain_roles(corruption: str, monkeypatch: pytest.MonkeyPatch) -> None:
    original = preparation_module.parse

    def corrupted(text: str):
        module = original(text)
        if "exec.chain" in text:
            if corruption == "duplicate_chain":
                chains = [op for op in module.walk() if isinstance(op, ChainOp)]
                chains[-1].properties["role"] = StringAttr("producer")
            else:
                find(module, CallOp).properties["role"] = StringAttr("publish")
        return module

    monkeypatch.setattr(preparation_module, "parse", corrupted)
    with pytest.raises(NotReady, match=r"prepared.*(chain role|call roles)"):
        prepare(accept(build()), registry())


@pytest.mark.parametrize("count", [0, -1])
def test_cli_rejects_nonpositive_stop_count_before_execution(count: int) -> None:
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
    completed = subprocess.run(
        [sys.executable, "-m", "xdsl_candidate", "--stop-after", str(count)], env=environment, capture_output=True, text=True, timeout=15
    )
    assert completed.returncode == 2 and "--stop-after must be positive" in completed.stderr
    assert not completed.stdout and "Traceback" not in completed.stderr
