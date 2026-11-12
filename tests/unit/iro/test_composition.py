import pytest
from xdsl.irdl import IRDLOperation, irdl_op_definition

from library.iro.composition import Composition, Definition, core


@irdl_op_definition
class WindowOp(IRDLOperation):
    """Contributions advanced together."""

    name = "iro.train.window"


def test_the_core_contract_holds_every_core_word_by_area() -> None:
    contract = core()
    assert contract.operation("iro.access") is not None
    assert {d.name for d in contract.words("data")} == {"iro.stream", "iro.take", "iro.borrow", "iro.dispose", "iro.filter", "iro.fork"}
    assert "iro.once" not in contract


def test_a_domain_extends_the_contract_and_changes_its_version() -> None:
    base = core()
    training = base.extend([Definition(WindowOp, "training")], name="training")
    assert "iro.train.window" in training and "iro.train.window" not in base
    assert training.version != base.version
    assert core().version == base.version
    assert training.context().get_optional_op("iro.train.window") is WindowOp


def test_a_definition_version_changes_the_contract_version() -> None:
    first = Composition([Definition(WindowOp, "training", 1)])
    second = Composition([Definition(WindowOp, "training", 2)])
    assert first.version != second.version


def test_a_word_is_defined_once() -> None:
    with pytest.raises(ValueError, match="defined twice"):
        core().extend([Definition(WindowOp, "training"), Definition(WindowOp, "training")])


def test_the_printout_lists_words_by_area_with_their_meaning() -> None:
    text = core().printout()
    assert text.startswith(f"# kuroiro contract {core().version}\n")
    assert "## access" in text
    assert "iro.fork     v1  Splits a claim into forks" in text
