"""A's structural/semantic contract is usable without a numerical backend."""

from __future__ import annotations

import subprocess
import sys

import pytest
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, StringAttr, SymbolRefAttr, f64

from .dialects import ComposeOp, SessionOp, integer
from .model_contract import accept_model, build_model, one
from .model_dialect import AdvanceOp, ContributeOp, CorrelateOp, EvaluateOp, FilterOp, ModelFlowOp, WindowOp, parse_model
from .prepare import Rejected, accept


def test_model_contract_detaches_author_and_round_trips() -> None:
    author = build_model()
    accepted = accept_model(author)
    assert accept_model(parse_model(accepted.text)) == accepted
    one(author, ContributeOp).properties["implementation"] = SymbolRefAttr("different")
    assert "different" not in accepted.text and accepted.obligations
    with pytest.raises(Rejected, match="unsupported contract"):
        accept(parse_model(accepted.text))  # Explicit installed extension required.


def test_model_semantic_authoring_and_acceptance_do_not_import_torch() -> None:
    script = """
import importlib, sys
contract=importlib.import_module('docs_design.models-strategy-trainer.xdsl_candidate.model_contract')
dialect=importlib.import_module('docs_design.models-strategy-trainer.xdsl_candidate.model_dialect')
accepted=contract.accept_model(contract.build_model())
assert contract.accept_model(dialect.parse_model(accepted.text)) == accepted
assert 'torch' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("count", [1, 2, 4])
def test_window_semantics_do_not_impose_the_cpu_two_contribution_limit(count: int) -> None:
    accepted = accept_model(build_model(count=count))
    assert one(parse_model(accepted.text), WindowOp).count.value.data == count


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("count", "normalization"),
        ("unit", "unresolved unit"),
        ("model", "frozen"),
        ("filter", "exclusion"),
        ("correlation", "admission contract"),
        ("wire", "wiring"),
        ("request", "request source"),
        ("consistency", "consistency"),
        ("trigger", "trigger"),
        ("rate", "learning rate"),
        ("identity", "identity"),
        ("stop", "disposition"),
    ],
)
def test_rejections_precede_backend_loading(case: str, diagnostic: str) -> None:
    author = build_model()
    if case == "count":
        one(author, ContributeOp).properties["normalization"] = integer(1)
    elif case == "unit":
        one(author, AdvanceOp).properties["unit"] = SymbolRefAttr("missing")
    elif case == "model":
        one(author, ModelFlowOp).properties["model"] = SymbolRefAttr("learner")
    elif case == "filter":
        one(author, FilterOp).properties["excluded"] = ArrayAttr([integer(99)])
    elif case == "correlation":
        one(author, CorrelateOp).properties["membership"] = StringAttr("take_anything")
    elif case == "wire":
        correlate = one(author, CorrelateOp)
        correlate.operands = [correlate.right, correlate.left, correlate.output]
    elif case == "request":
        one(author, EvaluateOp).properties["trigger"] = SymbolRefAttr("unknown")
    elif case == "consistency":
        one(author, EvaluateOp).properties["consistency"] = StringAttr("trigger_snapshot")
    elif case == "trigger":
        one(author, EvaluateOp).properties["after_window"] = integer(3)
    elif case == "rate":
        one(author, AdvanceOp).properties["learning_rate"] = FloatAttr(float("nan"), f64)
    elif case == "identity":
        one(author, ComposeOp).properties["source_id"] = one(author, WindowOp).source_id
    else:
        one(author, SessionOp).properties["stop_rule"] = StringAttr("discard_physical_work")
    with pytest.raises(Rejected, match=diagnostic):
        accept_model(author)
