"""Torch-free representation, lifetime contracts and cold correspondence."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from .composition import CompositionNative, contracts
from .composition_author import build
from .bridge import Rejected


@pytest.fixture(scope="session")
def composition_native() -> CompositionNative:
    target = Path(os.environ.get("CARGO_TARGET_DIR", "/tmp/sd-scripts-pliron-b-target"))
    manifest = Path(__file__).parent / "native" / "Cargo.toml"
    completed = subprocess.run(
        ["cargo", "build", "--locked", "--manifest-path", str(manifest), "--bin", "rmst-pliron-composition"],
        env={**os.environ, "CARGO_TARGET_DIR": str(target)},
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return CompositionNative(target / "debug" / "rmst-pliron-composition")


@pytest.fixture
def composition_artifact(composition_native):
    return composition_native.prepare(composition_native.accept(build()))


def test_composition_round_trip_and_inspectable_obligations(composition_native, composition_artifact):
    artifact = composition_artifact
    restored = composition_native.verify_saved(json.loads(json.dumps(artifact)), contracts())
    assert restored["accepted"] == artifact["accepted"]
    assert restored["prepared"] == artifact["prepared"]
    assert restored["profile"] == artifact["profile"]
    nodes = {node["id"]: node for node in artifact["nodes"]}
    assert nodes["producer"]["inputs"][1]["source"] == "consumers"
    assert {edge["source"] for edge in nodes["consumers.settle"]["inputs"]} == {"training", "evaluation"}
    assert {edge["source"] for edge in nodes["run.finish"]["inputs"]} == {"producer", "consumers"}
    for requirement in (
        "include_requested",
        "keeps_group_alive",
        "policy_choice_before_first_claim",
        "finish_selected_window",
        "physical_and_restore",
        "normal_group_end_requests_stop",
        "producer_and_group_settled",
        "aggregate_provider_budget",
    ):
        assert requirement in artifact["prepared"]
    assert artifact["profile"]["ports"] == ["producer.slot.0", "producer.slot.1"]


@pytest.mark.parametrize(
    "old,new",
    [
        ("lc.producer (enc, group_done)", "lc.producer (enc, training_done)"),
        ("lc.settle (training_done, evaluation_done)", "lc.settle (training_done, producer_done)"),
        ("lc.finish (group_done, producer_done)", "lc.finish (group_done, group_done)"),
        ('pending: builtin.string "include_requested"', 'pending: builtin.string "executing_only"'),
        ('pending: builtin.string "keeps_group_alive"', 'pending: builtin.string "ignore_until_ready"'),
        ('admission: builtin.string "policy_choice_before_first_claim"', 'admission: builtin.string "after_first_member"'),
        ('stop: builtin.string "finish_selected_window"', 'stop: builtin.string "drop_selected_window"'),
        ('multiplicity: builtin.string "once_per_matching_completion"', 'multiplicity: builtin.string "each_poll"'),
        ('after_window: builtin.string "2"', 'after_window: builtin.string "3"'),
        ('state: builtin.string "current_at_start"', 'state: builtin.string "request_time_guess"'),
        ('completion: builtin.string "physical_validate_restore"', 'completion: builtin.string "logical_return"'),
        ('failure: builtin.string "cancel_siblings_and_drain"', 'failure: builtin.string "ignore_children"'),
        ('remaining: builtin.string "discard_unclaimed"', 'remaining: builtin.string "report_consumed"'),
        ("lc.advance (g1, window_model)", "lc.advance (g0, window_model)"),
        ('normalization: builtin.string "2"', 'normalization: builtin.string "1"'),
        ('selection: builtin.string "after_trigger_members"', 'selection: builtin.string "training_member"'),
        ('source_id: builtin.string "evaluation.read"', 'source_id: builtin.string "training.advance"'),
        ("lc.evaluator (due, records, learn)", "lc.evaluator (due, records, enc)"),
    ],
)
def test_invalid_composition_contracts_reject(composition_native, old, new):
    text = build()
    assert old in text
    with pytest.raises(Rejected):
        composition_native.accept(text.replace(old, new))


@pytest.mark.parametrize("boundary", ["accept", "prepare", "accepted_load", "prepared_load"])
@pytest.mark.parametrize("suffix", ["garbage", "second_root", "comment_like"])
def test_full_composition_documents(composition_native, composition_artifact, boundary, suffix):
    artifact = copy.deepcopy(composition_artifact)
    tail = {"garbage": "unverified", "second_root": artifact["accepted"], "comment_like": "// not grammar trivia"}[suffix]
    with pytest.raises(Rejected):
        if boundary == "accept":
            composition_native.accept(artifact["accepted"] + tail)
        elif boundary == "prepare":
            composition_native.prepare(artifact["accepted"] + tail)
        else:
            name = "accepted" if boundary == "accepted_load" else "prepared"
            artifact[name] += tail
            composition_native.verify_saved(artifact, contracts())


@pytest.mark.parametrize("bounds", [{"windows": 3}, {"horizon": 4}, {"windows": 3, "after_window": 3}])
def test_semantically_valid_but_unsupported_profile(composition_native, bounds):
    accepted = composition_native.accept(build(**bounds))
    with pytest.raises(Rejected, match=r"unsupported.*bounds"):
        composition_native.prepare(accepted)


@pytest.mark.parametrize(
    "old,new",
    [
        ('through: builtin.string "physical_and_restore"', 'through: builtin.string "request"'),
        ('pending: builtin.string "include_requested"', 'pending: builtin.string "executing_only"'),
        ('relation: builtin.string "normal_group_end_requests_stop"', 'relation: builtin.string "training_end_requests_stop"'),
        ('participant: builtin.string "learner"', 'participant: builtin.string "encoder"'),
        ('role: builtin.string "objective"', 'role: builtin.string "evaluate"'),
        ('channel: builtin.string "producer.evaluation"', 'channel: builtin.string "producer.slot.0"'),
        ('parent: builtin.string "consumers"', 'parent: builtin.string "run"'),
        ('target: builtin.string "existing-owner-composition-cpu-v1"', 'target: builtin.string "scalar-async-cpu-v1"'),
    ],
)
def test_cold_prepared_obligation_corruption(composition_native, composition_artifact, old, new):
    artifact = copy.deepcopy(composition_artifact)
    assert old in artifact["prepared"]
    artifact["prepared"] = artifact["prepared"].replace(old, new)
    with pytest.raises(Rejected, match=r"correspondence|structural"):
        composition_native.verify_saved(artifact, contracts())


@pytest.mark.parametrize("field", ["version", "role", "effect", "completion"])
def test_installed_contract_drift_rejects(composition_native, composition_artifact, field):
    manifest = contracts()
    manifest["bindings"]["measure_v1"][field] = "incompatible"
    with pytest.raises(Rejected, match=r"evaluation.read.*measure_v1.*expected"):
        composition_native.verify_saved(composition_artifact, manifest)


def test_names_are_identities_not_case_dispatch(composition_native):
    artifact = composition_native.prepare(composition_native.accept(build(prefix="renamed.")))
    assert artifact["profile"]["producer"] == "renamed.producer"
    assert artifact["profile"]["training"] == "renamed.training"
    assert artifact["profile"]["ports"] == ["renamed.producer.slot.0", "renamed.producer.slot.1"]
    composition_native.verify_saved(artifact, contracts())
