"""Shared owner witnesses exercised through independently verified xDSL IR."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from .lifetime import LifetimeCandidate, Registry, Rejected, registry
from .lifetime_author import build
from .lifetime_ir import printed

pytest.importorskip("torch", reason="lifetime owner case requires optional Torch")
shared = importlib.import_module("docs_design.models-strategy-trainer.pliron_candidate.test_composition_runtime")


@pytest.fixture
def candidate():
    return LifetimeCandidate()


@pytest.fixture
def artifact(candidate):
    return candidate.prepare(candidate.accept(printed(build())))


# These are owner-service procedures, not another framework's IR constructors,
# verifier or prepared plan. Both candidates must meet the same obligations.
@pytest.mark.parametrize(
    "name,parameters",
    [
        ("test_lifetimes_bounds_and_both_input_availabilities", {"mode": "late"}),
        ("test_lifetimes_bounds_and_both_input_availabilities", {"mode": "prefetched"}),
        ("test_orderly_stop_respects_window_admission", {"stop": "before"}),
        ("test_orderly_stop_respects_window_admission", {"stop": "after"}),
        ("test_partial_window_claim_survives_root_cancellation", {}),
        ("test_orderly_child_stop_finishes_a_partly_acquired_window", {}),
        ("test_root_cancel_can_leave_a_requested_evaluation_unfulfilled", {}),
        ("test_group_failure_shuts_down_producer", {"phase": "training"}),
        ("test_group_failure_shuts_down_producer", {"phase": "evaluation"}),
        ("test_completed_evaluation_and_later_physical_failure_both_survive", {"cancel": False}),
        ("test_completed_evaluation_and_later_physical_failure_both_survive", {"cancel": True}),
        ("test_repeated_cancellation_drains_entered_evaluation", {}),
        ("test_cold_reconstruction_ignores_inspection_projection", {}),
        ("test_independent_oracle_catches_shared_wrong_objective", {}),
        ("test_histories_are_not_execution_authority", {}),
        ("test_verified_projection_is_snapshotted", {}),
        ("test_orderly_shutdown_can_cancel_a_call_before_physical_entry", {"patch": True}),
        ("test_unentered_evaluation_input_is_not_reported_consumed", {"patch": True}),
    ],
)
def test_shared_owner_obligation(candidate, artifact, monkeypatch, name, parameters):
    arguments = dict(parameters)
    if arguments.pop("patch", False):
        arguments["monkeypatch"] = monkeypatch
    getattr(shared, name)(candidate, artifact, **arguments)


@pytest.mark.parametrize("field", ["accepted", "prepared", "manifest"])
@pytest.mark.parametrize("missing", [False, True])
def test_rejection_precedes_model_factories(candidate, artifact, field, missing):
    calls = []
    bindings = dict(registry().bindings)
    original = bindings["scale_v1"]
    bindings["scale_v1"] = replace(original, invoke=lambda *args: calls.append(args) or original.invoke(*args))
    saved = copy.deepcopy(artifact)
    if missing:
        saved.pop(field)
    else:
        saved[field] = [] if field == "manifest" else 7
    with pytest.raises(Rejected, match=field):
        candidate.materialize(saved, Registry(bindings))
    assert not calls


def test_valid_structure_with_forged_lifetime_never_reaches_factory(candidate, artifact):
    calls = []
    bindings = dict(registry().bindings)
    original = bindings["scale_v1"]
    bindings["scale_v1"] = replace(original, invoke=lambda *args: calls.append(args) or original.invoke(*args))
    saved = copy.deepcopy(artifact)
    saved["prepared"] = saved["prepared"].replace("normal_group_end_requests_stop", "training_end_requests_stop")
    with pytest.raises(Rejected, match="correspondence"):
        candidate.materialize(saved, Registry(bindings))
    assert not calls


def test_unknown_artifact_claim_precedes_factories(candidate, artifact):
    calls = []
    bindings = dict(registry().bindings)
    original = bindings["scale_v1"]
    bindings["scale_v1"] = replace(original, invoke=lambda *args: calls.append(args) or original.invoke(*args))
    saved = copy.deepcopy(artifact)
    saved["realization"] = {"mode": "unprotected"}
    with pytest.raises(Rejected, match="unknown fields"):
        candidate.materialize(saved, Registry(bindings))
    assert not calls


def test_renamed_representation_runs_through_shared_services(candidate):
    saved = candidate.prepare(candidate.accept(printed(build(prefix="other."))))
    shared.check(asyncio.run(shared.pressure(candidate, saved)), prefix="other.")


@pytest.mark.parametrize("optimized", [False, True])
def test_fresh_cold_process_disables_authors_and_pliron_machinery(candidate, artifact, tmp_path, optimized):
    saved = tmp_path / "artifact.json"
    saved.write_text(json.dumps(artifact))
    script = r"""
import asyncio, importlib, json, sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]); before=sys._is_gil_enabled()
p='docs_design.models-strategy-trainer'
def forbidden(*args,**kwargs): raise RuntimeError('authoring or Pliron machinery entered xDSL cold execution')
for name in ('xdsl_candidate.lifetime_author','pliron_candidate.composition_author'):
    importlib.import_module(p+'.'+name).build=forbidden
native=importlib.import_module(p+'.pliron_candidate.composition')
native.CompositionNative.accept=forbidden; native.CompositionNative.prepare=forbidden
native.CompositionNative.verify_saved=forbidden
bridge=importlib.import_module(p+'.pliron_candidate.bridge'); bridge.Native.request=forbidden
api=importlib.import_module(p+'.xdsl_candidate.lifetime'); api.LifetimeCandidate.accept=forbidden
driver=importlib.import_module(p+'.pliron_candidate.composition_demo')
report=asyncio.run(driver.pressure(api.LifetimeCandidate(),json.loads(Path(sys.argv[2]).read_text())))
print(json.dumps({'report':report,'gil_before':before,'gil_after':sys._is_gil_enabled(),'optimized':not __debug__}))
"""
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), "-c", script, str(Path(__file__).resolve().parents[3]), str(saved)],
        cwd="/tmp",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["optimized"] is optimized
    assert observed["gil_before"] is False and observed["gil_after"] is False
    shared.check(observed["report"])
