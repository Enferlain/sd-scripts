"""Native C contracts, real SSA authority links and cold correspondence."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from .grant import GrantNative, Rejected, contracts
from .grant_author import build


@pytest.fixture(scope="session")
def grant_native():
    target = Path(os.environ.get("CARGO_TARGET_DIR", "/tmp/sd-scripts-pliron-b-target"))
    manifest = Path(__file__).parent / "native/Cargo.toml"
    completed = subprocess.run(
        ["cargo", "build", "--locked", "--manifest-path", str(manifest), "--bin", "rmst-pliron-grant"],
        env={**os.environ, "CARGO_TARGET_DIR": str(target)},
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return GrantNative(target / "debug/rmst-pliron-grant")


@pytest.fixture
def grant_artifact(grant_native):
    return grant_native.prepare(grant_native.accept(build()))


def change_declaration(text, kind, key, old, new):
    lines = text.splitlines()
    selected = [index for index, line in enumerate(lines) if "ga_exec.declaration" in line and f'kind: builtin.string "{kind}"' in line]
    assert len(selected) == 1
    index = selected[0]
    needle = f"{key}: builtin.string {json.dumps(old)}"
    assert needle in lines[index]
    lines[index] = lines[index].replace(needle, f"{key}: builtin.string {json.dumps(new)}")
    return "\n".join(lines)


def test_real_typed_authority_and_prepared_boundaries(grant_native, grant_artifact):
    assert "ga.handback" in grant_artifact["accepted"] and "ga.access" in grant_artifact["accepted"]
    assert "owner_verified_handback_not_restoration" in grant_artifact["prepared"]
    assert "not_an_owner_verified_token" in grant_artifact["prepared"]
    verified = grant_native.verify_saved(grant_artifact, contracts())
    assert verified["profile"]["initial"] == 0.8 and verified["profile"]["unit"] == "weights"


@pytest.mark.parametrize(
    "old,new",
    [
        ('version: builtin.string "1"', 'version: builtin.string "2"'),
        ('failure: builtin.string "drain_entered_preserve_uncertain"', 'failure: builtin.string "drop_entered"'),
        ('treatment: builtin.string "trainable"', 'treatment: builtin.string "frozen"'),
        ('initial: builtin.string "0.8"', 'initial: builtin.string "nan"'),
        ('profile: builtin.string "restored-two-pass-v1"', 'profile: builtin.string "unknown"'),
        (
            'delegated: builtin.string "backward,temporary_edit,intermediate_reset,restore"',
            'delegated: builtin.string "backward,temporary_edit,intermediate_reset,restore,final_advancement"',
        ),
        ('retained: builtin.string "clipping,final_advancement,final_reset"', 'retained: builtin.string "final_advancement"'),
        ('phase: builtin.string "temporary_edit"', 'phase: builtin.string "final_advancement"'),
        (
            'completion: builtin.string "physical_return_then_owner_handback_verification"',
            'completion: builtin.string "restoration_is_handback"',
        ),
        ('handback: builtin.string "restored_current_final_gradient_then_owner_verifies"', 'handback: builtin.string "claim_verified"'),
        ('authority: builtin.string "retained_owner_only"', 'authority: builtin.string "selected_grant"'),
        ('learning_rate: builtin.string "0.05"', 'learning_rate: builtin.string "0"'),
        ('clip_norm: builtin.string "none"', 'clip_norm: builtin.string "-1"'),
        ("ga.phase (p0)", "ga.phase (access)"),
        ("ga.phase (access)", "ga.phase (p1)"),
        ("ga.yield (p4)", "ga.yield (p3)"),
        ('source_id: builtin.string "phase.4.restore"', 'source_id: builtin.string "selected-two-pass"'),
    ],
)
def test_invalid_authority_claims_reject(grant_native, old, new):
    text = build()
    assert old in text
    with pytest.raises(Rejected):
        grant_native.accept(text.replace(old, new))


def test_ssa_cfg_forward_phase_use_rejected_by_framework(grant_native):
    with pytest.raises(Rejected, match=r"not dominated|dominance"):
        grant_native.accept(build().replace("ga.phase (access)", "ga.phase (p1)"))


def test_unique_handback_and_unit_associations(grant_native):
    text = build()
    extra = text.split("unit = ga.unit (model)", 1)[1].split(";", 1)[0]
    extra = (
        "other = ga.unit (model)"
        + extra.replace('"owner.weights"', '"owner.other"').replace('unit: builtin.string "weights"', 'unit: builtin.string "other"')
        + ";"
    )
    text = text.replace("    profile =", "    " + extra + "\n    profile =")
    with pytest.raises(Rejected, match="unit association"):
        grant_native.accept(text.replace("ga.advance (handback, unit)", "ga.advance (handback, other)"))
    # A valid handback cannot be consumed twice, even if both consumers are typed.
    source = build()
    advance = next(line for line in source.splitlines() if "advanced = ga.advance" in line)
    duplicate = advance.replace("advanced =", "again =").replace('"retained-final-step"', '"other-step"')
    with pytest.raises(Rejected, match="exactly one retained consumer"):
        grant_native.accept(source.replace(advance, advance + "\n" + duplicate))


@pytest.mark.parametrize("boundary", ["accept", "prepare", "accepted_load", "prepared_load"])
@pytest.mark.parametrize("tail", ["garbage", "second_root"])
def test_complete_authority_documents(grant_native, grant_artifact, boundary, tail):
    suffix = "\nunchecked garbage" if tail == "garbage" else "\n" + build()
    saved = copy.deepcopy(grant_artifact)
    with pytest.raises(Rejected):
        if boundary == "accept":
            grant_native.accept(saved["accepted"] + suffix)
        elif boundary == "prepare":
            grant_native.prepare(saved["accepted"] + suffix)
        else:
            saved["accepted" if boundary == "accepted_load" else "prepared"] += suffix
            grant_native.verify_saved(saved, contracts())


@pytest.mark.parametrize(
    "kind,key,old,new",
    [
        ("execute", "protection", "current_write_through_physical_return_and_handback", "release_on_restore"),
        ("execute", "call_completion", "physical_return_then_owner_handback_verification", "restore_is_handback"),
        ("execute", "unit", "weights", "other"),
        ("advance", "requires", "owner_verified_handback_not_restoration", "restored_boolean"),
        ("advance", "authority", "retained_owner_only", "selected_grant"),
        ("yield", "authority", "not_an_owner_verified_token", "already_verified"),
    ],
)
def test_cold_authority_correspondence(grant_native, grant_artifact, kind, key, old, new):
    saved = copy.deepcopy(grant_artifact)
    saved["prepared"] = change_declaration(saved["prepared"], kind, key, old, new)
    with pytest.raises(Rejected, match="correspondence"):
        grant_native.verify_saved(saved, contracts())


@pytest.mark.parametrize("field", ["version", "role", "effect", "completion"])
def test_installed_contract_drift_rejects(grant_native, grant_artifact, field):
    manifest = contracts()
    manifest["bindings"]["two_pass_v1"][field] = "different"
    with pytest.raises(Rejected, match=r"expected|correspondence"):
        grant_native.verify_saved(grant_artifact, manifest)


def test_compatible_binding_substitution_is_not_correspondence(grant_native):
    manifest = contracts()
    manifest["bindings"]["alternate"] = dict(manifest["bindings"]["two_pass_v1"])
    saved = grant_native.prepare(grant_native.accept(build()), manifest)
    saved["prepared"] = change_declaration(saved["prepared"], "execute", "implementation", "two_pass_v1", "alternate")
    with pytest.raises(Rejected, match="correspondence"):
        grant_native.verify_saved(saved, manifest)


@pytest.mark.parametrize("case", ["unoffered", "lowering", "parameter"])
def test_semantics_and_selected_target_support_are_separate(grant_native, case):
    source = build()
    manifest = contracts()
    if case == "unoffered":
        manifest["offered"] = []
    elif case == "lowering":
        manifest["lowering"] = "unknown"
    else:
        source = source.replace('parameter: builtin.string "weight"', 'parameter: builtin.string "unsupported"')
    accepted = grant_native.accept(source)
    with pytest.raises(Rejected, match=r"offer|lowering|parameter"):
        grant_native.prepare(accepted, manifest)


@pytest.mark.parametrize("field", ["accepted", "prepared", "manifest"])
@pytest.mark.parametrize("missing", [False, True])
def test_malformed_envelopes_are_rejected(grant_native, grant_artifact, field, missing):
    saved = copy.deepcopy(grant_artifact)
    if missing:
        saved.pop(field)
    else:
        saved[field] = [] if field == "manifest" else 7
    with pytest.raises(Rejected, match=field):
        grant_native.verify_saved(saved, contracts())


def test_source_identity_renaming_and_non_authoritative_inspection(grant_native):
    saved = grant_native.prepare(grant_native.accept(build(prefix="other.")))
    saved["profile"] = {"initial": 999, "owner_handback_verified": True}
    saved["accepted_sha256"] = "not proof"
    actual = grant_native.verify_saved(saved, contracts())
    assert actual["profile"]["session"] == "other.grant-session" and actual["profile"]["initial"] == 0.8


def test_unknown_claim_and_unused_manifest_growth_reject(grant_native, grant_artifact):
    saved = copy.deepcopy(grant_artifact)
    saved["verified_handback"] = True
    with pytest.raises(Rejected, match="unknown fields"):
        grant_native.verify_saved(saved, contracts())
    manifest = contracts()
    manifest["bindings"]["unused"] = dict(manifest["bindings"]["scale_v1"])
    with pytest.raises(Rejected):
        grant_native.verify_saved(grant_artifact, manifest)
