"""Cold correspondence and explicit target limits without importing Torch."""

from __future__ import annotations

import json

from dataclasses import replace

import pytest

from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
from xdsl.utils.exceptions import ParseError, VerifyException

from . import fixtures, standins
from .contracts import AccessMode
from .dialects import integer
from .normalize import source_index
from .prepare import Artifact, BindingSpec, CorrespondenceError, Manifest, Policy, ResourceSpec, load, prepare
from .verify import SemanticError, TargetLimit


def manifest() -> Manifest:
    return Manifest(
        tuple(
            BindingSpec(name, binding.inputs, binding.outputs, binding.version, binding.updates)
            for name, binding in standins.bindings().items()
        ),
        (ResourceSpec("scale", "participant", parameter="weight"), ResourceSpec("policy", "state")),
        ("measurement", "record"),
    )


@pytest.mark.parametrize("case", [fixtures.lifetime, fixtures.data_only, lambda: fixtures.lifetime(extra_scope=True, prefix="alternate.")])
def test_derived_fragments_roundtrip_and_detach(case):
    authored = case()
    prepared = prepare(authored, manifest(), Policy())
    artifact = Artifact.from_json(prepared.artifact.to_json())
    root = next(iter(authored.body.block.ops))
    root.properties["origin"] = StringAttr("changed-authoring")
    cold = load(artifact, manifest(), Policy())
    assert cold.verification.structure == prepared.verification.structure
    tree = json.loads(artifact.prepared)
    assert {item["source"] for item in tree["fragments"]} == set(source_index(cold.verification.normalized.module))
    assert {"owner.Producer", "owner.Repeat", "owner.Together"} <= {item["primitive"] for item in tree["fragments"]}
    assert tree["obligations"] and tree["inputs"]


@pytest.mark.parametrize(
    "field",
    ["source", "ancestry", "operands", "results", "properties", "region", "primitive", "flow", "access", "realization", "completion"],
)
def test_forged_prepared_fragment_rejects(field):
    prepared = prepare(fixtures.lifetime(), manifest(), Policy())
    tree = json.loads(prepared.artifact.prepared)
    tree["fragments"][0][field] = "forged"
    artifact = replace(prepared.artifact, prepared=json.dumps(tree))
    with pytest.raises(CorrespondenceError, match="correspondence"):
        load(artifact, manifest(), Policy())


@pytest.mark.parametrize("field", ["manifest", "target", "obligations", "inputs", "waits", "reproducibility", "contributors"])
def test_forged_prepared_claim_rejects(field):
    prepared = prepare(fixtures.data_only(), manifest(), Policy())
    tree = json.loads(prepared.artifact.prepared)
    tree[field] = []
    with pytest.raises(CorrespondenceError, match="correspondence"):
        load(replace(prepared.artifact, prepared=json.dumps(tree)), manifest(), Policy())


@pytest.mark.parametrize("target", ["accepted", "normalized"])
@pytest.mark.parametrize("mutation", ["source", "stop", "binding", "access", "capture", "stage"])
def test_forged_document_rejects(target, mutation):
    prepared = prepare(fixtures.lifetime(), manifest(), Policy())
    from .dialects import dump, parse

    module = parse(getattr(prepared.artifact, target))
    index = source_index(module)
    if mutation == "source":
        index["encode"].properties["origin"] = StringAttr("forged-source")
    elif mutation == "stop":
        index["stop"].properties["source"] = SymbolRefAttr("training")
    elif mutation == "binding":
        index["encode"].properties["binding_version"] = StringAttr("wrong")
    elif mutation == "access":
        index["encode.access"].properties["mode"] = StringAttr(AccessMode.PROTECTED_CURRENT)
    elif mutation == "capture":
        op = index["activities"]
        op.operands = (op.operands[0], op.operands[2], op.operands[1])
    else:
        index["run"].properties["stage"] = StringAttr("forged")
    changed = replace(prepared.artifact, **{target: dump(module)})
    with pytest.raises((CorrespondenceError, VerifyException)):
        load(changed, manifest(), Policy())


@pytest.mark.parametrize("change", ["version", "signature", "updates", "resource", "schema", "workers"])
def test_cold_installed_manifest_or_policy_change_rejects(change):
    installed = manifest()
    prepared = prepare(fixtures.data_only(), installed, Policy())
    policy = Policy()
    if change in ("version", "signature", "updates"):
        first = installed.bindings[0]
        changed = replace(
            first, **{"version": "2"} if change == "version" else {"inputs": ()} if change == "signature" else {"updates": (0,)}
        )
        installed = replace(installed, bindings=(changed, *installed.bindings[1:]))
    elif change == "resource":
        installed = replace(installed, resources=(installed.resources[0], replace(installed.resources[1], version="2")))
    elif change == "schema":
        installed = replace(installed, schemas=())
    else:
        policy = replace(policy, workers=3)
    with pytest.raises(CorrespondenceError):
        load(prepared.artifact, installed, policy)


@pytest.mark.parametrize("policy", [Policy(target="missing"), Policy(physical="implicit"), Policy(workers=0), Policy(workers=True)])
def test_unsupported_realization_rejects_explicitly(policy):
    with pytest.raises(TargetLimit):
        prepare(fixtures.data_only(), manifest(), policy)


def test_unsupported_request_bound_is_target_limit():
    module = fixtures.lifetime()
    source_index(module)["evaluation"].properties["max_outstanding"] = integer(2)
    with pytest.raises(TargetLimit, match="one live request"):
        prepare(module, manifest(), Policy())


def test_broken_lifetime_is_semantic_error():
    module = fixtures.lifetime()
    source_index(module)["stop"].properties["source"] = SymbolRefAttr("training")
    with pytest.raises(SemanticError):
        prepare(module, manifest(), Policy())


@pytest.mark.parametrize("suffix", ["garbage", '"lang.program"() : () -> ()'])
@pytest.mark.parametrize("field", ["accepted", "normalized", "prepared"])
def test_complete_artifact_consumption(field, suffix):
    prepared = prepare(fixtures.data_only(), manifest(), Policy())
    artifact = replace(prepared.artifact, **{field: getattr(prepared.artifact, field) + suffix})
    with pytest.raises((CorrespondenceError, ParseError)):
        load(artifact, manifest(), Policy())


def test_duplicate_and_unknown_artifact_fields_reject():
    with pytest.raises(CorrespondenceError):
        Artifact.from_json('{"accepted":"a","accepted":"b","normalized":"","prepared":""}')
    with pytest.raises(CorrespondenceError):
        Artifact.from_json('{"accepted":"","normalized":"","prepared":"","trusted":true}')
