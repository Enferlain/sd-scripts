"""Static ownership, detached correspondence and independent logical products."""

from __future__ import annotations

import json

from dataclasses import replace

import pytest

from xdsl.dialects.builtin import ArrayAttr
from xdsl.utils.exceptions import VerifyException

from . import product_case, product_reference
from .dialects import ReleaseProductOp, integer
from .fixtures import node
from .normalize import normalize, source_index
from .prepare import BindingSpec, Artifact, CorrespondenceError, Policy, data, load, prepare
from .reference import ReferenceError
from .verify import SemanticError, verify


@pytest.mark.parametrize("kind", ["buffer", "autograd"])
@pytest.mark.parametrize("derived", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_products_reference_and_cold_correspondence(kind, derived, fused):
    manifest = product_case.manifest(kind, derived=derived)
    plan = prepare(product_case.program(kind, derived=derived), manifest, Policy(fuse_local=fused))
    cold = load(Artifact.from_json(plan.artifact.to_json()), manifest, plan.policy)
    assert cold.artifact == plan.artifact
    ref = product_reference.reference(kind, derived=derived, history=False)
    ref.run()
    assert ref.settled and not ref.retained and not ref.facts
    assert ref.resources["result"].value == product_reference.expected(kind, derived)
    assert len(ref.product_receipts) == (2 if derived else 1)
    assert all(
        status == "released" and versions[0].owner == "storage" and versions[0].ordinal == 0
        for status, versions in ref.product_receipts.values()
    )


@pytest.mark.parametrize(
    "mutation", ["missing_release", "duplicate_release", "early_release", "escape", "missing_dependency", "bad_dependency", "early_parent"]
)
@pytest.mark.parametrize("normalized", [False, True])
def test_invalid_product_wiring_rejects_before_preparation(mutation, normalized):
    module = product_case.program(derived=mutation == "early_parent")
    index = source_index(module)
    block = index["make"].parent_block()
    assert block is not None
    if mutation == "missing_release":
        block.erase_op(index["release"])
    elif mutation == "duplicate_release":
        block.insert_op_before(node(ReleaseProductOp, "duplicate", inputs=index["make"].results), index["release"])
    elif mutation == "early_release":
        block.detach_op(index["release"])
        block.insert_op_before(index["release"], index["consume"])
    elif mutation == "escape":
        index["use.yield"].operands = index["make"].results
        index["use"].results[0]._type = index["make"].results[0].type
    elif mutation == "missing_dependency":
        del index["make"].properties["retains"]
    elif mutation == "bad_dependency":
        index["make"].properties["retains"] = ArrayAttr([integer(9)])
    elif mutation == "early_parent":
        block.detach_op(index["release"])
        block.insert_op_before(index["release"], index["release.child"])
    with pytest.raises(VerifyException):
        verified = normalize(module)
        verify(verified if normalized else normalize(verified.module))


def test_known_owner_replacement_rejects_while_product_is_live():
    with pytest.raises(SemanticError, match=r"replace.*retained_update.*make"):
        prepare(product_case.program(update="before"), product_case.manifest(), Policy())
    prepare(product_case.program(update="after"), product_case.manifest(), Policy())
    ref = product_reference.reference(update="after")
    ref.run()
    assert ref.resources["storage"].ordinal == 1


def test_installed_retention_mismatch_rejects_and_legacy_spelling_remains():
    manifest = product_case.manifest()
    changed = replace(manifest, bindings=tuple(replace(b, retains=()) if b.name == "make_buffer" else b for b in manifest.bindings))
    with pytest.raises(CorrespondenceError, match="invalid installed"):
        prepare(product_case.program(), changed, Policy())

    old = data(BindingSpec("old", ("scalar",), ("scalar",)))
    assert isinstance(old, dict) and "retains" not in old


def test_cold_forgery_does_not_authorize_a_retained_dependency_change():
    manifest = product_case.manifest(derived=True)
    plan = prepare(product_case.program(derived=True), manifest, Policy())

    forged = json.loads(plan.artifact.prepared)
    call = next(fragment for fragment in forged["fragments"] if fragment["source"] == "derive")
    call["realization"]["retains"] = []
    with pytest.raises(CorrespondenceError):
        load(replace(plan.artifact, prepared=json.dumps(forged)), manifest, plan.policy)


@pytest.mark.parametrize("kind", ["buffer", "autograd"])
@pytest.mark.parametrize("late_failure", [False, True])
def test_reference_retains_dependencies_through_cancelled_physical_consumer(kind, late_failure):
    ref = product_reference.reference(kind, derived=True, history=False)
    ref.held.add(("consume", 0))
    for _ in range(100):
        if any(call.command.op is ref.index["consume"] for call in ref.calls.values()):
            break
        assert ref.step()
    else:
        pytest.fail("consumer did not enter")
    product = ref.retained["derive", 0]
    assert product.borrowers == 1 and ref.resources["storage"].readers
    ref.cancelled = True
    with pytest.raises(ReferenceError, match="dependent physical"):
        ref._release_product(product, "forged")
    assert product.active and not ref.product_receipts
    if late_failure:
        ref.failures["consume", 0] = "late product failure"
    ref.held.clear()
    ref.run()
    assert ref.settled and not ref.retained and not ref.resources["storage"].readers
    assert all(status == "cancelled" for status, _ in ref.product_receipts.values())
    assert ref.resources["result"].value == 0


@pytest.mark.parametrize("kind", ["buffer", "autograd"])
def test_cancelled_producing_call_settles_without_admitting_a_retained_product(kind):
    ref = product_reference.reference(kind, history=False)
    ref.held.add(("make", 0))
    for _ in range(100):
        if ref.calls:
            break
        assert ref.step()
    assert ref.calls and ref.resources["storage"].readers and not ref.retained
    ref.cancelled = True
    ref.held.clear()
    ref.run()
    assert ref.settled and not ref.retained and not ref.product_receipts
    assert not ref.resources["storage"].readers and ref.resources["result"].value == 0


@pytest.mark.parametrize("positions", [(9,), (-1,), (0, 0), ()])
def test_malformed_derived_positions_have_attributed_rejection(positions):
    module = product_case.program(derived=True)
    source_index(module)["derive"].properties["retains"] = ArrayAttr(integer(i) for i in positions)
    with pytest.raises(SemanticError, match=r"derive.*retained_dependency"):
        prepare(module, product_case.manifest(derived=True), Policy())
