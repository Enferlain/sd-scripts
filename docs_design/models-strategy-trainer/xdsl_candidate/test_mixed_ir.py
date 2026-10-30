"""Mixed-profile structure remains author/runtime independent."""

from __future__ import annotations

import pytest

from xdsl.dialects.builtin import StringAttr, SymbolRefAttr

from .dialects import SelectOp, dump, integer
from .mixed_contract import accept_mixed, build_mixed
from .mixed_dialect import DynamicFilterOp, LearnedOp, PolicyUpdateOp, StopCheckpointOp, parse_mixed
from .model_contract import one
from .model_dialect import CorrelateOp
from .prepare import Rejected


def test_mixed_roundtrip_detaches_author() -> None:
    module = build_mixed()
    accepted = accept_mixed(module)
    one(module, DynamicFilterOp).properties["policy"] = SymbolRefAttr("missing")
    assert accept_mixed(parse_mixed(accepted.text)) == accepted
    assert "mixed.learned" in accepted.text and "mixed.dynamic_filter" in accepted.text
    assert "model_data.source" not in accepted.text and "model_data.evaluate" not in accepted.text


@pytest.mark.parametrize("case", ["policy", "wire", "provenance", "update", "observation", "checkpoint", "version", "learned_model"])
def test_mixed_invalid_semantics_reject(case: str) -> None:
    module = build_mixed()
    if case == "policy":
        one(module, DynamicFilterOp).properties["policy"] = SymbolRefAttr("learner")
    elif case == "wire":
        one(module, PolicyUpdateOp).operands = [one(module, SelectOp).output, one(module, SelectOp).feedback]
    elif case == "provenance":
        one(module, CorrelateOp).properties["provenance"] = StringAttr("assume_current_forever")
    elif case == "update":
        one(module, PolicyUpdateOp).properties["protocol"] = StringAttr("guess_from_history")
    elif case == "observation":
        one(module, LearnedOp).properties["observation"] = StringAttr("before_advance")
    elif case == "checkpoint":
        one(module, StopCheckpointOp).properties["after_windows"] = integer(0)
    elif case == "learned_model":
        one(module, LearnedOp).properties["model"] = SymbolRefAttr("policy")
    else:
        from .dialects import SessionOp

        one(module, SessionOp).properties["version"] = integer(2)
    with pytest.raises(Rejected):
        accept_mixed(parse_mixed(dump(module)))
