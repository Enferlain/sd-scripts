"""A faulty emission/pass cannot redefine an accepted scalar workflow."""

from __future__ import annotations

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr, SymbolRefAttr

from . import prepare as preparation
from .dialects import CallOp, ChainOp, EntityOp, FeedbackRecipeOp, HandoffRecipeOp, PlanOp, SelectRecipeOp, integer
from .fixture import build, registry
from .prepare import NotReady, accept, prepare


def test_scalar_snapshot_cannot_weaken_installed_contract_obligations() -> None:
    with pytest.raises(NotReady, match="canonical"):
        prepare(replace(accept(build()), obligations=()), registry())


@pytest.mark.parametrize("field", ["accepted_digest", "target", "contract", "horizon", "capacity", "stop_rule", "feedback"])
@pytest.mark.parametrize("missing", [False, True])
def test_decoded_scalar_envelope_matches_accepted_target(field: str, missing: bool, monkeypatch) -> None:
    original = preparation.parse

    def corrupt(text):
        module = original(text)
        for root in module.walk():
            if isinstance(root, PlanOp):
                if missing:
                    root.properties.pop(field)
                else:
                    root.properties[field] = integer(0) if field in ("horizon", "capacity") else StringAttr("changed")
        return module

    monkeypatch.setattr(preparation, "parse", corrupt)
    with pytest.raises(NotReady, match="prepared"):
        prepare(accept(build()), registry())


@pytest.mark.parametrize("case", ["swapped_symbols", "missing", "duplicate", "reordered", "origin", "ancestry", "completion", "version"])
def test_scalar_calls_preserve_semantic_correspondence_and_order(case: str, monkeypatch) -> None:
    original = preparation.parse

    def corrupt(text):
        module = original(text)
        roots = [op for op in module.walk() if isinstance(op, PlanOp)]
        if not roots:
            return module
        producer = next(op for op in module.walk() if isinstance(op, ChainOp) and op.role.data == "producer")
        left, right = tuple(producer.body.block.ops)
        assert isinstance(left, CallOp) and isinstance(right, CallOp)
        if case == "swapped_symbols":
            left.properties["implementation"], right.properties["implementation"] = right.implementation, left.implementation
        elif case == "missing":
            producer.body.block.erase_op(right)
        elif case == "duplicate":
            producer.body.block.add_op(left.clone())
        elif case == "reordered":
            producer.body.block.detach_op(left)
            producer.body.block.add_op(left)
        elif case == "origin":
            left.properties["origin"] = StringAttr("unrelated-stage")
        elif case == "ancestry":
            left.properties["ancestors"] = ArrayAttr([StringAttr("unrelated-region")])
        elif case == "completion":
            left.properties["completion"] = StringAttr("enqueue_is_complete")
        else:
            left.properties["binding_version"] = StringAttr("other")
        return module

    monkeypatch.setattr(preparation, "parse", corrupt)
    with pytest.raises(NotReady, match=r"prepared|semantic"):
        prepare(accept(build()), registry())


@pytest.mark.parametrize(
    "kind,field",
    [
        (SelectRecipeOp, "selection"),
        (SelectRecipeOp, "application"),
        (SelectRecipeOp, "policy"),
        (HandoffRecipeOp, "release"),
        (HandoffRecipeOp, "reader"),
        (HandoffRecipeOp, "endpoint"),
        (FeedbackRecipeOp, "update"),
        (FeedbackRecipeOp, "admission"),
        (FeedbackRecipeOp, "publication"),
        (ChainOp, "transport"),
        (EntityOp, "kind"),
    ],
)
def test_fixed_protocol_descriptors_are_checked_not_interpreted(kind, field: str, monkeypatch) -> None:
    original = preparation.parse

    def corrupt(text):
        module = original(text)
        if any(isinstance(op, PlanOp) for op in module.walk()):
            op = next(op for op in module.walk() if isinstance(op, kind))
            op.properties[field] = SymbolRefAttr("other") if field in ("policy", "publication") else StringAttr("changed")
        return module

    monkeypatch.setattr(preparation, "parse", corrupt)
    with pytest.raises(NotReady, match=r"prepared.*(protocol|correspondence|transport)"):
        prepare(accept(build()), registry())


@pytest.mark.parametrize("kind", [SelectRecipeOp, HandoffRecipeOp, FeedbackRecipeOp, EntityOp])
@pytest.mark.parametrize("missing", [False, True])
def test_scalar_owner_recipe_completeness(kind, missing: bool, monkeypatch) -> None:
    original = preparation.parse

    def corrupt(text):
        module = original(text)
        for root in module.walk():
            if isinstance(root, PlanOp):
                item = next(op for op in root.body.block.ops if isinstance(op, kind))
                if missing:
                    root.body.block.erase_op(item)
                else:
                    root.body.block.add_op(item.clone())
                break
        return module

    monkeypatch.setattr(preparation, "parse", corrupt)
    with pytest.raises(NotReady, match=r"prepared.*(protocol|correspondence)"):
        prepare(accept(build()), registry())
