"""Native A semantic regions, typed associations and cold correspondence."""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path

import pytest

from .correlation import CorrelationNative, Rejected, contracts
from .correlation_author import build


@pytest.fixture(scope="session")
def correlation_native():
    target = Path(os.environ.get("CARGO_TARGET_DIR", "/tmp/sd-scripts-pliron-b-target"))
    manifest = Path(__file__).parent / "native/Cargo.toml"
    result = subprocess.run(
        ["cargo", "build", "--locked", "--manifest-path", str(manifest), "--bin", "rmst-pliron-correlation"],
        env={**os.environ, "CARGO_TARGET_DIR": str(target)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return CorrelationNative(target / "debug/rmst-pliron-correlation")


@pytest.fixture
def correlation_artifact(correlation_native):
    return correlation_native.prepare(correlation_native.accept(build()))


def test_actual_typed_regions_and_native_local_vocabulary(correlation_native, correlation_artifact):
    saved = correlation_artifact
    assert all(name in saved["accepted"] for name in ("ac.correlated", "ac.gradient", "ac.events", "wf.local", "ac.window"))
    assert saved["profile"]["local_groups"] == [["data.double"], ["data.offset"]]
    assert [row["sample"] for row in saved["profile"]["addresses"]] == [1, 2, 4, 5]
    assert correlation_native.verify_saved(saved, contracts())["prepared"] == saved["prepared"]


@pytest.mark.parametrize(
    "old,new",
    [
        ('version: builtin.string "1"', 'version: builtin.string "2"'),
        ('stop_rule: builtin.string "drain_entered_close_unadmitted"', 'stop_rule: builtin.string "discard"'),
        ('initial: builtin.string "1.5"', 'initial: builtin.string "nan"'),
        ('treatment: builtin.string "frozen"', 'treatment: builtin.string "trainable"'),
        ('effect: builtin.string "pure"', 'effect: builtin.string "write_live"'),
        ('implementation: builtin.string "double_v1"', 'implementation: builtin.string ""'),
        ('keys: builtin.string "[1,2,3,4,5,6]"', 'keys: builtin.string "[1,1,2]"'),
        ('excluded: builtin.string "[3,6]"', 'excluded: builtin.string "[99]"'),
        ("ac.unit (learner)", "ac.unit (encoder)"),
        ("ac.encode (right_in, encoder)", "ac.encode (right_in, learner)"),
        ("ac.correlate (left, right, encoder)", "ac.correlate (left, left, encoder)"),
        ("ac.window (correlated, learner, unit)", "ac.window (correlated, encoder, unit)"),
        ('membership: builtin.string "same_sample_and_slot_before_any_claim"', 'membership: builtin.string "claim_then_check"'),
        ('provenance: builtin.string "current_source_at_admission"', 'provenance: builtin.string "any_epoch"'),
        ('accumulation: builtin.string "sum"', 'accumulation: builtin.string "mean"'),
        ('normalization: builtin.string "2"', 'normalization: builtin.string "1"'),
        ("ac.contribute (input, subject, weights)", "ac.contribute (input, subject, unit)"),
        ("ac.advance (gradient, weights)", "ac.advance (gradient, unit)"),
        ('authority: builtin.string "retained_owner_only_after_all_contributions"', 'authority: builtin.string "selected_callback"'),
        ('learning_rate: builtin.string "0.05"', 'learning_rate: builtin.string "0"'),
        ("ac.window_end (advanced)", "ac.window_end (evaluated)"),
        ("ac.evaluate (events, learner)", "ac.evaluate (events, encoder)"),
        ('after_window: builtin.string "1"', 'after_window: builtin.string "3"'),
        ('consistency: builtin.string "current_at_start"', 'consistency: builtin.string "trigger_snapshot"'),
        ("current_read_through_physical_return_validation_and_mode_restore", "release_before_physical_return"),
        ("ac.finish (learned, evaluated)", "ac.finish (learned, learned)"),
        ('source_id: builtin.string "data.offset"', 'source_id: builtin.string "data.double"'),
    ],
)
def test_invalid_semantic_claims_reject(correlation_native, old, new):
    text = build()
    assert old in text
    with pytest.raises(Rejected):
        correlation_native.accept(text.replace(old, new))


def test_region_kind_distinguishes_graph_forward_use_and_ordered_local_dominance(correlation_native):
    text = build()
    declaration = next(line for line in text.splitlines() if "unit = ac.unit" in line)
    reordered = text.replace(declaration + "\n", "").replace("  learner =", declaration + "\n  learner =")
    correlation_native.accept(reordered)
    with pytest.raises(Rejected, match=r"not dominated|dominance"):
        correlation_native.accept(text.replace("wf.map (record)", "wf.map (mapped)"))


@pytest.mark.parametrize("boundary", ["accept", "prepare", "accepted_load", "prepared_load"])
@pytest.mark.parametrize("tail", ["garbage", "second_root"])
def test_complete_native_documents(correlation_native, correlation_artifact, boundary, tail):
    saved = copy.deepcopy(correlation_artifact)
    suffix = " unchecked garbage" if tail == "garbage" else "\n" + build()
    with pytest.raises(Rejected):
        if boundary == "accept":
            correlation_native.accept(saved["accepted"] + suffix)
        elif boundary == "prepare":
            correlation_native.prepare(saved["accepted"] + suffix)
        else:
            saved["accepted" if boundary == "accepted_load" else "prepared"] += suffix
            correlation_native.verify_saved(saved, contracts())


@pytest.mark.parametrize(
    "old,new",
    [
        ('target: builtin.string "candidate-owner-cpu-v1"', 'target: builtin.string "other"'),
        ("validate_all_members_and_current_source_before_any_claim", "claim_one_member_before_validation"),
        ("two_separate_atomic_member_joins_then_sum_retained_contributions", "discard_first_contribution"),
        ("consume_complete_window_gradient_then_owner_step_and_reset", "step_each_contribution"),
        ("request_cause_retained_separately_from_current_at_start_state", "trigger_state_is_current_state"),
        ('sample: builtin.string "1"', 'sample: builtin.string "99"'),
        ('left: builtin.string "left.0"', 'left: builtin.string "right.0"'),
        ("protected_physical_return_then_validate", "early_release"),
    ],
)
def test_cold_claims_reject(correlation_native, correlation_artifact, old, new):
    saved = copy.deepcopy(correlation_artifact)
    assert old in saved["prepared"]
    saved["prepared"] = saved["prepared"].replace(old, new)
    with pytest.raises(Rejected):
        correlation_native.verify_saved(saved, contracts())


@pytest.mark.parametrize("field", ["version", "role", "effect"])
def test_changed_installed_binding_rejects_saved_calls(correlation_native, correlation_artifact, field):
    installed = contracts()
    installed["bindings"]["encode_v1"][field] = "changed"
    with pytest.raises(Rejected):
        correlation_native.verify_saved(correlation_artifact, installed)


def test_compatible_binding_substitution_does_not_preserve_source_correspondence(correlation_native):
    installed = contracts()
    installed["bindings"]["another_encoder"] = dict(installed["bindings"]["encode_v1"])
    saved = correlation_native.prepare(correlation_native.accept(build()), installed)
    lines = saved["prepared"].splitlines()
    index = next(index for index, line in enumerate(lines) if "ac_exec.declaration" in line and 'kind: builtin.string "encode"' in line)
    assert 'implementation: builtin.string "encode_v1"' in lines[index]
    lines[index] = lines[index].replace('implementation: builtin.string "encode_v1"', 'implementation: builtin.string "another_encoder"')
    saved["prepared"] = "\n".join(lines)
    with pytest.raises(Rejected, match="correspondence"):
        correlation_native.verify_saved(saved, installed)


@pytest.mark.parametrize("case", ["count", "horizon", "parameter", "lowering", "binding"])
def test_accepted_but_unsupported_target(correlation_native, case):
    text = build(count=1 if case == "count" else 2, horizon=2 if case == "horizon" else 4)
    if case == "parameter":
        text = text.replace('parameter: builtin.string "weight"', 'parameter: builtin.string "unknown"')
    accepted = correlation_native.accept(text)
    installed = contracts()
    if case == "lowering":
        installed["lowering"] = "unknown"
    elif case == "binding":
        del installed["bindings"]["loss_v1"]
    with pytest.raises(Rejected):
        correlation_native.prepare(accepted, installed)


def test_source_names_and_saved_projection_are_not_dispatch(correlation_native, correlation_artifact):
    renamed = correlation_native.prepare(correlation_native.accept(build(prefix="other.")))
    assert renamed["profile"]["objective_origin"] == "other.objective"
    saved = copy.deepcopy(correlation_artifact)
    saved["profile"] = {"ready": True, "addresses": [999]}
    saved["accepted_sha256"] = "forged observation"
    assert correlation_native.verify_saved(saved, contracts())["profile"]["addresses"][0]["sample"] == 1


@pytest.mark.parametrize("field", ["accepted", "prepared", "manifest"])
@pytest.mark.parametrize("value", [None, 42])
def test_malformed_envelope_is_rejected(correlation_native, correlation_artifact, field, value):
    saved = dict(correlation_artifact)
    saved[field] = value
    with pytest.raises(Rejected, match=field):
        correlation_native.verify_saved(saved, contracts())
