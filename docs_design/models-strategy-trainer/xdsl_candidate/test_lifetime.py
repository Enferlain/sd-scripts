"""Lifetime contracts checked independently in actual xDSL operations."""

from __future__ import annotations

import copy
import json

import pytest
from xdsl.dialects.builtin import DictionaryAttr, StringAttr

from . import lifetime_ir as ir
from .lifetime import LifetimeCandidate, Rejected, contracts
from .lifetime_author import build


@pytest.fixture
def candidate():
    return LifetimeCandidate()


@pytest.fixture
def artifact(candidate):
    return candidate.prepare(candidate.accept(ir.printed(build())))


def selected(module, source):
    return next(op for op in module.walk() if isinstance(op, ir.ActivityOp) and op.source_id.data == source)


def set_policy(op, key, value):
    op.properties["policy"] = DictionaryAttr(dict(op.policy.data) | {key: StringAttr(value)})


def test_real_typed_regions_and_graph_roundtrip(candidate, artifact):
    module = ir.parse(artifact["accepted"])
    producer, group = selected(module, "producer"), selected(module, "consumers")
    assert producer.stop is group.done
    assert len(selected(module, "training").body.block.args) == 2
    assert len(selected(module, "evaluation").body.block.args) == 3
    assert ir.equivalent(module, ir.parse(ir.printed(module)))
    assert candidate.verify_saved(artifact, contracts())["profile"]["horizon"] == 3


@pytest.mark.parametrize("change", ["property", "operand"])
def test_equivalence_dependency_contract(artifact, change):
    before = ir.parse(artifact["prepared"])
    after = ir.parse(ir.printed(before))
    assert ir.equivalent(before, after)
    if change == "property":
        op = next(item for item in after.walk() if isinstance(item, ir.RecipeOp) and item.role.data == "access")
        op.properties["values"] = DictionaryAttr(dict(op.values.data) | {"through": StringAttr("request_only")})
    else:
        producer = selected(after, "producer")
        producer.operands = [producer.model, selected(after, "training").done]
    after.verify()
    assert not ir.equivalent(before, after)


def test_unknown_artifact_fields_are_not_silently_verified(candidate, artifact):
    saved = copy.deepcopy(artifact)
    saved["executable"] = {"stop_after": "training"}
    with pytest.raises(Rejected, match="unknown fields"):
        candidate.verify_saved(saved, contracts())


@pytest.mark.parametrize(
    "source,key,value",
    [
        ("run", "failure", "ignore_children"),
        ("consumers", "pending", "executing_only"),
        ("consumers", "completion", "logical_return"),
        ("request", "pending", "ignore_until_ready"),
        ("request", "multiplicity", "each_poll"),
        ("request", "after_window", "3"),
        ("request", "cancellation", "discard_receipt"),
        ("training", "admission", "after_first_claim"),
        ("training", "stop", "drop_selected_window"),
        ("training.objective.1", "normalization", "1"),
        ("training.advance", "authority", "selected_objective"),
        ("training.advance", "learning_rate", "nan"),
        ("evaluation.read", "state", "request_time_guess"),
        ("evaluation.read", "mode", "training"),
        ("evaluation.read", "completion", "logical_return"),
        ("evaluation.take", "selection", "training_member"),
        ("producer", "remaining", "report_consumed"),
        ("producer", "horizon", "0"),
    ],
)
def test_semantic_obligations_reject(candidate, source, key, value):
    module = build()
    set_policy(selected(module, source), key, value)
    with pytest.raises(Rejected):
        candidate.accept(ir.printed(module))


@pytest.mark.parametrize("case", ["premature_stop", "circular_settlement", "duplicate_root", "gradient", "model", "source_id"])
def test_typed_but_wrong_relationships_reject(candidate, case):
    module = build()
    if case == "premature_stop":
        selected(module, "producer").operands = [selected(module, "encoder").model, selected(module, "training").done]
    elif case == "circular_settlement":
        selected(module, "consumers.settle").operands = [selected(module, "training").done, selected(module, "producer").done]
    elif case == "duplicate_root":
        selected(module, "run.finish").operands = [selected(module, "consumers").done] * 2
    elif case == "gradient":
        advance = selected(module, "training.advance")
        advance.operands = [selected(module, "training.objective.0").gradient, advance.model]
    elif case == "model":
        evaluator = selected(module, "evaluation")
        evaluator.operands = [evaluator.due, evaluator.records, selected(module, "encoder").model]
    else:
        selected(module, "evaluation.read").properties["source_id"] = StringAttr("training.advance")
    with pytest.raises(Rejected):
        candidate.accept(ir.printed(module))


@pytest.mark.parametrize("bounds", [{"windows": 3}, {"horizon": 4}, {"members": 3}])
def test_acceptance_does_not_imply_target_support(candidate, bounds):
    text = candidate.accept(ir.printed(build(**bounds)))
    with pytest.raises(Rejected, match=r"unsupported.*bounds"):
        candidate.prepare(text)


@pytest.mark.parametrize("boundary", ["accept", "prepare", "accepted", "prepared"])
@pytest.mark.parametrize("suffix", ["garbage", "second_root"])
def test_complete_documents_required(candidate, artifact, boundary, suffix):
    tail = "\nunchecked garbage" if suffix == "garbage" else "\n" + artifact["accepted"]
    with pytest.raises(Rejected):
        if boundary == "accept":
            candidate.accept(artifact["accepted"] + tail)
        elif boundary == "prepare":
            candidate.prepare(artifact["accepted"] + tail)
        else:
            saved = copy.deepcopy(artifact)
            saved[boundary] += tail
            candidate.verify_saved(saved, contracts())


@pytest.mark.parametrize(
    "role,key,value",
    [
        ("scope", "pending", "executing_only"),
        ("access", "through", "logical_return"),
        ("access", "participant", "encoder"),
        ("call", "binding_role", "unknown"),
        ("request", "pending", "ignore_pending"),
        ("stop_link", "relation", "training_end_requests_stop"),
        ("port", "channel", "other.input"),
    ],
)
def test_cold_recipe_correspondence(candidate, artifact, role, key, value):
    saved = copy.deepcopy(artifact)
    module = ir.parse(saved["prepared"])
    operation = next(
        op
        for op in module.walk()
        if isinstance(op, ir.RecipeOp)
        and op.role.data == role
        and key in op.values.data
        and (role != "access" or op.values.data.get("mode") == StringAttr("read_evaluation"))
    )
    operation.properties["values"] = DictionaryAttr(dict(operation.values.data) | {key: StringAttr(value)})
    saved["prepared"] = ir.printed(module)
    with pytest.raises(Rejected, match="correspondence"):
        candidate.verify_saved(saved, contracts())


@pytest.mark.parametrize("field", ["version", "role", "effect", "completion"])
def test_installed_binding_drift(candidate, artifact, field):
    manifest = contracts()
    manifest["bindings"]["measure_v1"][field] = "wrong"
    with pytest.raises(Rejected, match=r"evaluation.read.*measure_v1.*expected"):
        candidate.verify_saved(artifact, manifest)


def test_saved_semantic_substitution_is_not_authorized_by_hash(candidate, artifact):
    saved = copy.deepcopy(artifact)
    module = ir.parse(saved["accepted"])
    set_policy(selected(module, "learner"), "initial", "0.9")
    saved["accepted"] = ir.printed(module)
    with pytest.raises(Rejected, match="correspondence"):
        candidate.verify_saved(saved, contracts())
    artifact["accepted_sha256"] = "not an authorization"
    candidate.verify_saved(artifact, contracts())


@pytest.mark.parametrize("case", ["stop", "model", "gradient"])
def test_framework_equivalence_detects_forward_and_ordered_use_changes(candidate, artifact, case):
    saved = copy.deepcopy(artifact)
    module = ir.parse(saved["prepared"])
    if case == "stop":
        producer = selected(module, "producer")
        producer.operands = [producer.model, selected(module, "training").done]
    elif case == "model":
        evaluator = selected(module, "evaluation")
        evaluator.operands = [evaluator.due, evaluator.records, selected(module, "encoder").model]
    else:
        advance = selected(module, "training.advance")
        advance.operands = [selected(module, "training.objective.0").gradient, advance.model]
    module.verify()  # The corruption is typed and structurally valid.
    saved["prepared"] = ir.printed(module)
    with pytest.raises(Rejected, match="correspondence"):
        candidate.verify_saved(saved, contracts())


def test_source_renaming_and_untrusted_inspection(candidate):
    saved = candidate.prepare(candidate.accept(ir.printed(build(prefix="other."))))
    saved["profile"] = {"windows": 999}
    verified = candidate.verify_saved(json.loads(json.dumps(saved)), contracts())
    assert verified["profile"]["training"] == "other.training"
    assert verified["profile"]["windows"] == 2


def test_unused_binding_growth_is_an_explicit_compatibility_boundary(candidate, artifact):
    installed = contracts()
    installed["bindings"]["unused_v1"] = dict(installed["bindings"]["scale_v1"])
    with pytest.raises(Rejected, match="manifest"):
        candidate.verify_saved(artifact, installed)
