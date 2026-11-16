"""Training's words: the unit, the window and contribution, read, printed, expanded and accepted (D16, D17, D20)."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from library.iro.accepted import cold_load
from library.iro.diagnostics import Rejection
from library.iro.frontend import implementation
from library.iro.normalize import normalize
from library.iro.text import parse, print_strategy, round_trips
from library.iro.verify import ObligationKind, verify
from library.iro.words.access import AccessOp
from library.iro.words.calls import CallOp
from library.iro.words.model import ResourceOp
from library.iro.words.structure import RepeatOp
from library.training.words import ADVANCE, CONTRIBUTE, UnitOp, contract
from tests.unit.iro.accepted_runs import accepted_run

STEP = """iro.strategy @step uses(@batch v1, @predict v1, @objective v1, @standard.contribute v1, @standard.advance v1) {
  %unet = iro.resource @unet : participant
  %unit = iro.train.unit @denoiser of %unet optimizer @adamw8bit(lr 1e-05, eps 1e-08) max_grad_norm 1.0
  iro.sequence @once capture(%unet, %unit) {
    iro.train.window %unit members 2 {
      %batch = iro.call @batch() -> batch
      %loss = iro.access %unet protected_current {
      ^(%u):
        %prediction = iro.call @predict(%u, %batch) -> retained prediction
        iro.call @objective(%prediction, %batch) -> retained loss
      }
      iro.train.contribute %loss to %unit
    }
  }
}
"""


def installed() -> dict:
    def declared(name: str, inputs=(), outputs=()):
        return implementation(version=1, inputs=inputs, outputs=outputs, name=name)(lambda *args: None)

    return {
        "batch": declared("batch", outputs=("batch",)),
        "predict": declared("predict", ("protected_current lease", "batch"), ("retained prediction",)),
        "objective": declared("objective", ("retained prediction", "batch"), ("retained loss",)),
        "standard.contribute": declared("standard.contribute", ("owner_update lease", "retained loss")),
        "standard.advance": declared("standard.advance", ("owner_update lease", "owner_update lease")),
    }


def test_training_words_read_and_print_back_through_the_training_contract() -> None:
    strategy = parse(STEP, contract())
    (unit,) = [op for op in strategy.walk() if isinstance(op, UnitOp)]
    assert unit.sym_name.data == "denoiser" and unit.optimizer.root_reference.data == "adamw8bit"
    assert unit.max_grad_norm is not None and unit.max_grad_norm.value.data == 1.0
    printed = print_strategy(strategy, contract())
    assert "iro.train.unit @denoiser of %unet optimizer @adamw8bit(lr 1e-05, eps 1e-08) max_grad_norm 1.0" in printed
    assert "iro.train.window %unit members 2 {" in printed and "iro.train.contribute %loss to %unit" in printed
    assert round_trips(printed, contract())


def test_normalization_expands_them_into_core_words_with_the_member_update_leases() -> None:
    normalized = normalize(parse(STEP, contract())).strategy
    ops = list(normalized.walk())
    assert not [op for op in ops if op.name.startswith("iro.train.")], "only core words remain"
    (unit,) = [op for op in ops if isinstance(op, ResourceOp) and op.sym_name.data == "denoiser"]
    assert unit.handle.type.kind.data == "unit"
    (window,) = [op for op in ops if isinstance(op, RepeatOp)]
    assert (
        window.sym_name is not None
        and window.sym_name.data == "denoiser.window"
        and window.limit is not None
        and window.limit.value.data == 2
    )
    calls = {op.implementation.root_reference.data: op for op in ops if isinstance(op, CallOp)}
    contribute, advance = calls[CONTRIBUTE], calls[ADVANCE]
    contributing = contribute.parent_op()
    assert isinstance(contributing, AccessOp) and contributing.parent_op() is window, "each member contributes inside the window"
    advancing = advance.parent_op()
    assert isinstance(advancing, AccessOp) and advancing.parent_block() is window.parent_block()
    assert [use.mode.data for use in advancing.uses.data] == ["owner_update", "owner_update"], "the unit and its participant, together"
    for op in (window, contributing, advancing):
        ancestry = op.properties["ancestry"]
        assert isinstance(ancestry, ArrayAttr) and any(
            "iro.train" in entry.data for entry in ancestry.data if isinstance(entry, StringAttr)
        )


def test_the_loss_is_released_after_its_contribution() -> None:
    normalized = normalize(parse(STEP, contract())).text(contract())
    contribute_at = normalized.index("@standard.contribute")
    assert normalized.index("iro.release", contribute_at) > contribute_at


def test_a_strategy_with_training_words_is_accepted_with_its_pins_and_cold_loads() -> None:
    run = accepted_run(STEP, contract(), installed=installed())
    assert {name for name, _, _ in run.definitions} >= {"iro.train.unit", "iro.train.window", "iro.train.contribute", "iro.train.profile"}
    assert {name for name, _, _ in run.implementations} >= {CONTRIBUTE, ADVANCE}
    assert [p.name for p in run.seed.participants] == ["unet"], "a unit is not a participant"
    bound = [o for o in run.obligations if o.kind == ObligationKind.BOUND]
    assert {o.names for o in bound} == {("@unet",), ("@denoiser",)}
    assert cold_load(run, contract(), installed=installed()).run is run


def test_a_window_needs_its_unit_s_participants_where_it_advances() -> None:
    hidden = STEP.replace("iro.sequence @once capture(%unet, %unit) {", "iro.sequence @once capture(%unit) {").replace(
        "      %loss = iro.access %unet protected_current {\n      ^(%u):\n        %prediction = iro.call @predict(%u, %batch) -> retained prediction\n        iro.call @objective(%prediction, %batch) -> retained loss\n      }\n      iro.train.contribute %loss to %unit\n",
        "",
    )
    with pytest.raises(Rejection, match=r"iro-norm-009(.|\n)*trains @unet, which is not\s+captured where its window advances it"):
        normalize(parse(hidden, contract()))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "iro.strategy @s {\n  %u = iro.resource @u : participant\n  %x = iro.train.unit @x of %u optimizer @o profile @custom\n}\n",
            "standard profile",
        ),
        ("iro.strategy @s {\n  %p = iro.resource @p\n  %x = iro.train.unit @x of %p optimizer @o\n}\n", "members are participants"),
    ],
)
def test_a_unit_trains_participants_under_the_standard_profile(text: str, message: str) -> None:
    with pytest.raises(Rejection, match=message):
        parse(text, contract())


def test_without_the_training_contract_its_words_are_unknown() -> None:
    with pytest.raises(Rejection, match=r"unknown word iro\.train\.unit"):
        parse(STEP)


def test_verification_of_the_expanded_strategy_owes_the_unit_to_its_owner() -> None:
    verification = verify(normalize(parse(STEP, contract())))
    (unit,) = [o for o in verification.obligations if o.kind == ObligationKind.BOUND and o.names == ("@denoiser",)]
    assert unit.evaluator == "the resource's owner"


TWO_WINDOWS = """iro.strategy @s uses(@batch v1, @predict v1, @objective v1, @standard.contribute v1, @standard.advance v1) {
  %unet = iro.resource @unet : participant
  %unit = iro.train.unit @denoiser of %unet optimizer @sgd
  iro.sequence @first capture(%unet, %unit) {
    iro.train.window %unit members 1 {
      %batch = iro.call @batch() -> batch
      %loss = iro.access %unet protected_current {
      ^(%u):
        %prediction = iro.call @predict(%u, %batch) -> retained prediction
        iro.call @objective(%prediction, %batch) -> retained loss
      }
      iro.train.contribute %loss to %unit
    }
  }
  iro.sequence @second capture(%unet, %unit) {
    iro.derived @inside {
      iro.train.window %unit members 1 {
        %batch = iro.call @batch() -> batch
        %loss = iro.access %unet protected_current {
        ^(%u):
          %prediction = iro.call @predict(%u, %batch) -> retained prediction
          iro.call @objective(%prediction, %batch) -> retained loss
        }
        iro.train.contribute %loss to %unit
      }
    }
  }
}
"""


def test_windows_on_one_unit_are_numbered_in_source_order_even_inside_another_derived_word() -> None:
    normalized = normalize(parse(TWO_WINDOWS, contract())).strategy
    repeats = [op for op in normalized.walk() if isinstance(op, RepeatOp)]
    assert [op.sym_name.data for op in repeats if op.sym_name is not None] == ["denoiser.window", "denoiser.window2"]
    (inner,) = [op for op in repeats if op.sym_name is not None and op.sym_name.data == "denoiser.window2"]
    ancestry = inner.properties["ancestry"]
    assert isinstance(ancestry, ArrayAttr)
    entries = [entry.data for entry in ancestry.data if isinstance(entry, StringAttr)]
    assert any("iro.train.window" in entry for entry in entries)


def test_every_window_that_cannot_expand_is_reported() -> None:
    hidden = TWO_WINDOWS.replace("capture(%unet, %unit)", "capture(%unit)").replace(
        "      %loss = iro.access %unet protected_current {\n      ^(%u):\n        %prediction = iro.call @predict(%u, %batch) -> retained prediction\n        iro.call @objective(%prediction, %batch) -> retained loss\n      }\n      iro.train.contribute %loss to %unit\n",
        "",
    )
    hidden = hidden.replace(
        "        %loss = iro.access %unet protected_current {\n        ^(%u):\n          %prediction = iro.call @predict(%u, %batch) -> retained prediction\n          iro.call @objective(%prediction, %batch) -> retained loss\n        }\n        iro.train.contribute %loss to %unit\n",
        "",
    )
    with pytest.raises(Rejection) as raised:
        normalize(parse(hidden, contract()))
    assert str(raised.value).count("iro-norm-009") == 2


def test_a_loss_is_contributed_only_to_a_unit() -> None:
    text = """iro.strategy @s uses(@objective v1) {
  %m = iro.resource @m : participant
  iro.sequence @once capture(%m) {
    %loss = iro.access %m protected_current {
    ^(%u):
      iro.call @objective(%u) -> retained loss
    }
    iro.train.contribute %loss to %m
  }
}
"""
    with pytest.raises(Rejection, match="a loss is contributed to a unit"):
        parse(text, contract())
