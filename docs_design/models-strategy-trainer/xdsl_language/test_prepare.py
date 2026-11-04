"""Cold correspondence and explicit target limits without importing Torch."""

from __future__ import annotations

import json

from dataclasses import replace

import pytest

from xdsl.dialects.builtin import ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import ParseError, VerifyException

from . import fixtures, standins
from .contracts import AccessMode, Freshness
from .dialects import CapabilityType, ProgramOp, ResourceType, dump, integer
from .normalize import source_index
from .prepare import Artifact, BindingSpec, CorrespondenceError, Manifest, Policy, ResourceSpec, data, load, prepare
from .verify import SemanticError, TargetLimit
from .test_structure import ordered_values_program


def manifest() -> Manifest:
    return Manifest(
        tuple(
            BindingSpec(name, binding.inputs, binding.outputs, binding.version, binding.updates)
            for name, binding in standins.bindings().items()
        ),
        (ResourceSpec("scale", "participant", parameter="weight"), ResourceSpec("policy", "state")),
        ("measurement", "record"),
    )


@pytest.mark.parametrize("shape", ["direct", "nested_capture", "recipe_capture", "recipe_body", "independent_recipe"])
@pytest.mark.parametrize("normalized", [False, True])
def test_preparation_rejects_unavailable_ordered_values(shape, normalized):
    module = ordered_values_program(shape, normalized=normalized)
    installed = Manifest((BindingSpec("echo", ("scalar",), ("scalar",)),), (), ())
    with pytest.raises(VerifyException, match=r"later.definition.*unavailable in ordered body"):
        prepare(module, installed, Policy())


@pytest.mark.parametrize("shape", ["direct", "nested_capture", "recipe_capture", "recipe_body", "independent_recipe"])
def test_preparation_accepts_earlier_local_values_and_explicit_captures(shape):
    installed = Manifest((BindingSpec("echo", ("scalar",), ("scalar",)),), (), ())
    prepare(ordered_values_program(shape, invalid=False), installed, Policy())


@pytest.mark.parametrize("mode", [AccessMode.PROTECTED_CURRENT, AccessMode.SNAPSHOT])
def test_declared_update_rejects_read_authority_during_preparation(mode):
    module = fixtures.lifetime()
    scope = source_index(module)["update.access"]
    scope.properties.update(
        mode=StringAttr(mode),
        freshness=StringAttr(Freshness.PUBLISHED if mode == AccessMode.SNAPSHOT else Freshness.CURRENT_AT_START),
        version_selector=StringAttr("initial" if mode == AccessMode.SNAPSHOT else "current"),
    )
    kind = scope.operands[0].type
    assert isinstance(kind, ResourceType)
    # Deliberate type forgery keeps the access's local entry shape consistent
    # while its selected binding still requires update authority (xDSL 0.74).
    scope.regions[0].block.args[0]._type = CapabilityType(kind, StringAttr(mode))
    with pytest.raises(CorrespondenceError, match=r"advance.*update operand 0 requires owner_update.*model"):
        prepare(module, manifest(), Policy())


def two_capability_program(*, reverse: bool = False) -> ModuleOp:
    first, second = fixtures.resource("first", "counter"), fixtures.resource("second", "counter")
    read = CapabilityType(first.value.type, StringAttr(AccessMode.PROTECTED_CURRENT))
    update = CapabilityType(second.value.type, StringAttr(AccessMode.OWNER_UPDATE))
    outer = Block(arg_types=[read, second.value.type])
    inner = Block(arg_types=[update, read])
    args = [inner.args[0], inner.args[1]] if reverse else [inner.args[1], inner.args[0]]
    value = fixtures.call(inner, "mutate", "mutate", args, [fixtures.SCALAR])
    fixtures.end(inner, "write.body", *value.results)
    write = fixtures.access("write.second", [outer.args[1], outer.args[0]], inner, AccessMode.OWNER_UPDATE, [fixtures.SCALAR])
    outer.add_op(write)
    fixtures.end(outer, "read.body", *write.results)
    access = fixtures.access("read.first", [first.value, second.value], outer, AccessMode.PROTECTED_CURRENT, [fixtures.SCALAR])
    root = ProgramOp.create(
        properties={**fixtures.props("run"), "stage": StringAttr("accepted")}, regions=[Region(Block([first, second, access]))]
    )
    return ModuleOp([root])


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("position", [0, 1])
def test_binding_update_follows_actual_operand_issuer_and_owner(reverse, position):
    installed = Manifest(
        (BindingSpec("mutate", ("capability", "capability"), ("scalar",), updates=(position,)),), (ResourceSpec("counter", "state"),), ()
    )
    if position == (0 if reverse else 1):
        prepare(two_capability_program(reverse=reverse), installed, Policy())
    else:
        with pytest.raises(CorrespondenceError, match=rf"mutate.*operand {position}.*protected_current from read.first for first"):
            prepare(two_capability_program(reverse=reverse), installed, Policy())


def test_payload_schema_name_cannot_impersonate_binding_capability_authority():
    value = fixtures.node(
        fixtures.ConstantOp, "fake.capability", outputs=[fixtures.PayloadType(StringAttr("capability"))], literal=StringAttr("0")
    )
    block = Block([value])
    fixtures.call(block, "mutate", "mutate", value.results, [fixtures.SCALAR])
    root = ProgramOp.create(properties={**fixtures.props("run"), "stage": StringAttr("accepted")}, regions=[Region(block)])
    installed = Manifest((BindingSpec("mutate", ("capability",), ("scalar",), updates=(0,)),), (), ())
    with pytest.raises(CorrespondenceError, match=r"mutate.*operand 0 requires an issued capability type"):
        prepare(ModuleOp([root]), installed, Policy())


def test_effect_contract_is_explicit_and_legacy_absence_is_opaque():
    old = BindingSpec("plain", ("scalar",), ("scalar",))
    assert old.contract.effects.unknown and not old.contract.effects.pure
    before = data(old)
    assert isinstance(before, dict)
    assert set(before) == {"name", "inputs", "outputs", "version", "updates"}
    pure = replace(old, reads=(), unknown_effects=False)
    assert pure.contract.effects.pure and not pure.contract.effects.unknown
    after = data(pure)
    assert isinstance(after, dict)
    assert after["reads"] == [] and after["unknown_effects"] is False
    with pytest.raises(ValueError, match="effect/version contract"):
        _ = replace(old, unknown_effects=False).contract


def test_read_update_requirements_share_the_same_selected_contract():
    binding = BindingSpec(
        "mutate",
        ("capability", "capability"),
        ("scalar",),
        updates=(1,),
        reads=(0, 1),
        unknown_effects=False,
        resource_kinds=((0, "counter"), (1, "counter")),
    )
    assert [(entry.argument, entry.access, entry.kind) for entry in binding.contract.effects.resources] == [
        (0, "read", "counter"),
        (1, "update", "counter"),
    ]
    installed = Manifest((binding,), (ResourceSpec("counter", "state"),), ())
    prepare(two_capability_program(), installed, Policy())
    wrong = replace(binding, resource_kinds=((0, "scale"), (1, "counter")))
    with pytest.raises(CorrespondenceError, match=r"operand 0 requires resource kind scale.*counter.*first"):
        prepare(two_capability_program(), replace(installed, bindings=(wrong,)), Policy())


@pytest.mark.parametrize(
    "change",
    [{"reads": (True,)}, {"reads": (2,)}, {"reads": (0, 0)}, {"unknown_effects": 0}, {"resource_kinds": ((0, "counter"), (0, "counter"))}],
)
def test_invalid_effect_requirement_rejects_before_readiness(change):
    binding = BindingSpec("mutate", ("capability", "capability"), ("scalar",), updates=(1,), reads=(0,), unknown_effects=False)
    installed = Manifest((replace(binding, **change),), (ResourceSpec("counter", "state"),), ())
    with pytest.raises(CorrespondenceError, match=r"binding.*contract|binding resource-kind"):
        prepare(two_capability_program(), installed, Policy())


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
    from .dialects import parse

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
