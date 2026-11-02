"""Mixed graph/SSA contracts and native saved-document correspondence."""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path

import pytest

from .mixed import MixedNative, Rejected, contracts
from .mixed_author import build


@pytest.fixture(scope="session")
def mixed_native():
    target = Path(os.environ.get("CARGO_TARGET_DIR", "/tmp/sd-scripts-pliron-b-target"))
    result = subprocess.run(
        ["cargo", "build", "--locked", "--manifest-path", str(Path(__file__).parent / "native/Cargo.toml"), "--bin", "rmst-pliron-mixed"],
        env=os.environ | {"CARGO_TARGET_DIR": str(target)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return MixedNative(target / "debug/rmst-pliron-mixed")


@pytest.fixture
def mixed_artifact(mixed_native):
    return mixed_native.prepare(mixed_native.accept(build()))


def test_real_feedback_cycle_regions_and_reused_operations(mixed_native, mixed_artifact):
    text = mixed_artifact["accepted"]
    assert all(
        name in text for name in ("mixed.producer", "mixed.repeat", "mixed.feedback", "mixed.ack", "mixed.stop", "ac.gradient", "wf.local")
    )
    assert text.index("mixed.producer") < text.index("mixed.repeat")
    assert text.index("mixed.select") < text.index("mixed.update")
    assert mixed_native.verify_saved(mixed_artifact, contracts())["profile"]["local_groups"] == [["data.double"], ["data.offset"]]
    grouped = mixed_native.prepare(text, contracts(fuse=True))
    assert grouped["profile"]["local_groups"] == [["data.double", "data.offset"]]
    assert "ac_exec.fused" in grouped["prepared"]


@pytest.mark.parametrize(
    "old,new",
    [
        ('version: builtin.string "1"', 'version: builtin.string "2"'),
        ('horizon: builtin.string "3"', 'horizon: builtin.string "0"'),
        ('initial: builtin.string "0.8"', 'initial: builtin.string "nan"'),
        ('treatment: builtin.string "trainable"', 'treatment: builtin.string "frozen"'),
        ('owner: builtin.string "policy"', 'owner: builtin.string "learner"'),
        ('start: builtin.string "0"', 'start: builtin.string "1"'),
        ('bias: builtin.string "0"', 'bias: builtin.string "2"'),
        ('capacity: builtin.string "1"', 'capacity: builtin.string "0"'),
        ("ordered_bounded_ports_after_model_release", "retain_read_through_publication"),
        ("close_selection_discard_unclaimed_join_private_physical_work", "return_before_draining"),
        ("mixed.producer (learner, feedback, stop)", "mixed.producer (learner, feedback, settled)"),
        ("mixed.select (cursor, policy, acknowledged, closing)", "mixed.select (cursor, policy, acknowledged, stop)"),
        ("ack_before_next_selection", "use_unacknowledged_policy"),
        ("mixed.filter (raw, policy)", "mixed.filter (selected, policy)"),
        ("ac.fanout (selected)", "ac.fanout (raw)"),
        ("ac.encode (right_in, subject)", "ac.encode (left_in, subject)"),
        ("same_binding_read_through_physical_completion", "release_before_physical_return"),
        ("ac.correlate (left, right, subject)", "ac.correlate (left, left, subject)"),
        ("same_ticket_and_ordinal", "claim_one_before_checking_other"),
        ("same_binding_protected_at_production", "require_latest_at_admission"),
        ("mixed.publish (correlated)", "mixed.publish (records)"),
        ('count: builtin.string "1"', 'count: builtin.string "0"'),
        ('accumulation: builtin.string "sum"', 'accumulation: builtin.string "mean"'),
        ('normalization: builtin.string "1"', 'normalization: builtin.string "2"'),
        ("ac.contribute (input, subject, weights)", "ac.contribute (input, subject, unit)"),
        ("ac.advance (gradient, weights)", "ac.advance (gradient, unit)"),
        ('learning_rate: builtin.string "0.001"', 'learning_rate: builtin.string "nan"'),
        ("retained_owner_only_after_all_contributions", "arbitrary_callback"),
        ("mixed.learned (advanced, input, subject)", "mixed.learned (advanced, records, subject)"),
        ("returned_advance_current_weight_and_input_associations", "last_diagnostic_record"),
        ("post_advance_weight; bias=int(abs(weight)*10)%5; revision+=1; retire_once", "unverified_feedback"),
        ('after_windows: builtin.string "3"', 'after_windows: builtin.string "0"'),
        ("mixed.finish (settled, drained)", "mixed.finish (settled, settled)"),
        ('effect: builtin.string "pure"', 'effect: builtin.string "write_live"'),
        ('implementation: builtin.string "double_v1"', 'implementation: builtin.string ""'),
        ('source_id: builtin.string "data.offset"', 'source_id: builtin.string "data.double"'),
    ],
)
def test_invalid_semantic_contracts(mixed_native, old, new):
    text = build()
    assert old in text
    with pytest.raises(Rejected):
        mixed_native.accept(text.replace(old, new))


def test_ordered_local_region_does_not_allow_future_ssa_use(mixed_native):
    with pytest.raises(Rejected, match="dominat"):
        mixed_native.accept(build().replace("wf.map (record)", "wf.map (mapped)"))


@pytest.mark.parametrize("boundary", ["accept", "prepare", "accepted_load", "prepared_load"])
@pytest.mark.parametrize("tail", ["garbage", "second_root"])
def test_full_document_consumption(mixed_native, mixed_artifact, boundary, tail):
    saved = copy.deepcopy(mixed_artifact)
    suffix = " unchecked garbage" if tail == "garbage" else "\n" + build()
    with pytest.raises(Rejected):
        if boundary == "accept":
            mixed_native.accept(saved["accepted"] + suffix)
        elif boundary == "prepare":
            mixed_native.prepare(saved["accepted"] + suffix)
        else:
            saved["accepted" if boundary == "accepted_load" else "prepared"] += suffix
            mixed_native.verify_saved(saved, contracts())


@pytest.mark.parametrize(
    "old,new",
    [
        ('target: builtin.string "continuous-owner-cpu-v1"', 'target: builtin.string "other"'),
        ("aggregate3_ports1_ordered_publish_after_read_release_stop_then_drain", "ports3_no_aggregate_limit"),
        ("same_ticket_both_members_and_actual_production_binding_before_any_claim", "latest_epoch_and_partial_claim"),
        ("one_retained_contribution_then_advance_reset_then_feedback", "feedback_before_advance"),
        ("actual_admitted_input_value_link_not_diagnostic_archive", "use_latest_observation"),
        ("producer_owned_policy_validate_ack_retire_once", "retire_twice"),
        ("three_returned_updates_then_explicit_external_stop", "implicit_early_stop"),
        ("checkpoint_settlement_and_independent_producer_physical_drain", "skip_producer_drain"),
        ("protected_physical_return_then_validate", "early_release"),
        ('parent_origin: builtin.string "producer"', 'parent_origin: builtin.string "window"'),
    ],
)
def test_forged_prepared_claims(mixed_native, mixed_artifact, old, new):
    saved = copy.deepcopy(mixed_artifact)
    assert old in saved["prepared"]
    saved["prepared"] = saved["prepared"].replace(old, new)
    with pytest.raises(Rejected, match="correspondence"):
        mixed_native.verify_saved(saved, contracts())


@pytest.mark.parametrize("field", ["version", "role", "effect"])
def test_installed_binding_drift(mixed_native, mixed_artifact, field):
    installed = contracts()
    installed["bindings"]["learned_bias_v1"][field] = "changed"
    with pytest.raises(Rejected):
        mixed_native.verify_saved(mixed_artifact, installed)


def test_compatible_binding_substitution_is_not_source_correspondence(mixed_native):
    installed = contracts()
    installed["bindings"]["other_policy"] = dict(installed["bindings"]["learned_bias_v1"])
    saved = mixed_native.prepare(mixed_native.accept(build()), installed)
    lines = saved["prepared"].splitlines()
    index = next(i for i, line in enumerate(lines) if "ac_exec.declaration" in line and 'kind: builtin.string "policy_update"' in line)
    lines[index] = lines[index].replace('implementation: builtin.string "learned_bias_v1"', 'implementation: builtin.string "other_policy"')
    saved["prepared"] = "\n".join(lines)
    with pytest.raises(Rejected, match="correspondence"):
        mixed_native.verify_saved(saved, installed)


@pytest.mark.parametrize("options", [{"horizon": 4}, {"count": 2}, {"after_windows": 4}, {"capacity": 2}])
def test_valid_but_unsupported_target(mixed_native, options):
    accepted = mixed_native.accept(build(**options))
    with pytest.raises(Rejected, match="accepted but unsupported"):
        mixed_native.prepare(accepted)


def test_renamed_sources_and_ignored_saved_projections(mixed_native, mixed_artifact):
    renamed = mixed_native.prepare(mixed_native.accept(build(prefix="other.")))
    assert renamed["profile"]["policy_origin"] == "other.policy.update"
    saved = copy.deepcopy(mixed_artifact)
    saved["profile"] = {"horizon": 999, "ready": True}
    saved["accepted_sha256"] = "forged observation"
    assert mixed_native.verify_saved(saved, contracts())["profile"]["horizon"] == 3


@pytest.mark.parametrize("field", ["accepted", "prepared", "manifest"])
@pytest.mark.parametrize("value", [None, 42])
def test_malformed_envelope(mixed_native, mixed_artifact, field, value):
    saved = dict(mixed_artifact)
    saved[field] = value
    with pytest.raises(Rejected, match=field):
        mixed_native.verify_saved(saved, contracts())
