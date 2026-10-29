"""C authoring and authority verification stay independent of Torch."""

from __future__ import annotations

import subprocess
import sys

import pytest
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, StringAttr, SymbolRefAttr, f64

from .dialects import SessionOp, integer
from .fixture import build
from .grant_contract import DELEGATED, RETAINED, accept_grant, build_grant
from .grant_dialect import AdvanceGrantedOp, ContributionType, GrantOp, PhaseOp, parse_grant
from .model_contract import accept_model, build_model, one
from .model_dialect import ModelOp
from .prepare import Rejected, accept


def test_grant_round_trip_is_detached_and_has_a_real_typed_edge() -> None:
    author = build_grant()
    accepted = accept_grant(author)
    loaded = parse_grant(accepted.text)
    assert accept_grant(loaded) == accepted
    grant, advance = one(loaded, GrantOp), one(loaded, AdvanceGrantedOp)
    assert advance.handback is grant.handback and len(tuple(grant.handback.uses)) == 1
    assert grant.handback.type == ContributionType(SymbolRefAttr("weights"))
    assert one(loaded, SessionOp).horizon is None
    assert "wf.flow" not in accepted.text and "feedback" not in accepted.text
    one(author, GrantOp).properties["implementation"] = SymbolRefAttr("changed")
    assert "changed" not in accepted.text and accepted.obligations
    with pytest.raises(Rejected, match="unsupported contract"):
        accept(loaded)


def test_grant_acceptance_does_not_import_torch() -> None:
    script = """
import importlib, sys
contract=importlib.import_module('docs_design.models-strategy-trainer.xdsl_candidate.grant_contract')
dialect=importlib.import_module('docs_design.models-strategy-trainer.xdsl_candidate.grant_dialect')
accepted=contract.accept_grant(contract.build_grant())
assert contract.accept_grant(dialect.parse_grant(accepted.text)) == accepted
assert 'torch' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("phase", range(5))
def test_each_phase_is_part_of_the_offered_protocol(phase: int) -> None:
    author = build_grant()
    phases = [op for op in author.walk() if isinstance(op, PhaseOp)]
    phases[phase].properties["phase"] = StringAttr("final_advancement")
    with pytest.raises(Rejected, match="phase protocol"):
        accept_grant(author)


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("delegated", "authority split"),
        ("retained", "authority split"),
        ("profile", "profile"),
        ("model", "subject/unit"),
        ("unit", "subject/unit"),
        ("type", "unit type"),
        ("advance_unit", "typed handback"),
        ("initial", "initial state"),
        ("rate", "learning rate"),
        ("clip", "clipping bound"),
        ("stop", "disposition"),
        ("horizon", "no flow horizon"),
        ("identity", "source identity"),
        ("implementation", "profile"),
    ],
)
def test_authority_and_structure_reject_before_numerical_loading(case: str, diagnostic: str) -> None:
    author = build_grant()
    grant, advance = one(author, GrantOp), one(author, AdvanceGrantedOp)
    if case == "delegated":
        grant.properties["delegated"] = ArrayAttr(StringAttr(name) for name in (*DELEGATED, "final_advancement"))
    elif case == "retained":
        grant.properties["retained"] = ArrayAttr(StringAttr(name) for name in RETAINED[:-1])
    elif case == "profile":
        grant.properties["profile"] = StringAttr("anything")
    elif case == "model":
        grant.properties["model"] = SymbolRefAttr("other")
    elif case == "unit":
        grant.properties["unit"] = SymbolRefAttr("other")
    elif case == "type":
        grant.handback._type = ContributionType(SymbolRefAttr("other"))
    elif case == "advance_unit":
        advance.properties["unit"] = SymbolRefAttr("other")
    elif case == "initial":
        one(author, ModelOp).properties["initial"] = FloatAttr(float("nan"), f64)
    elif case == "rate":
        advance.properties["learning_rate"] = FloatAttr(0.0, f64)
    elif case == "clip":
        advance.properties["clip_norm"] = FloatAttr(float("inf"), f64)
    elif case == "stop":
        one(author, SessionOp).properties["stop_rule"] = StringAttr("discard_uncertain")
    elif case == "horizon":
        one(author, SessionOp).properties["horizon"] = integer(1)
    elif case == "identity":
        advance.properties["source_id"] = grant.source_id
    else:
        grant.properties["implementation"] = SymbolRefAttr("")
    with pytest.raises(Rejected, match=diagnostic):
        accept_grant(author)


@pytest.mark.parametrize("fixture,acceptor", [(build, accept), (build_model, accept_model)])
def test_data_contracts_still_require_their_own_horizon(fixture, acceptor) -> None:
    author = fixture()
    one(author, SessionOp).properties.pop("horizon")
    with pytest.raises(Rejected, match="horizon"):
        acceptor(author)
