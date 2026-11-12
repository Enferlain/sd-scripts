"""The written form: exact round trips, what reading builds, and source-attributed problems."""

import re
from pathlib import Path

import pytest
from xdsl.dialects.builtin import FloatAttr, IntegerAttr, StringAttr

from library.iro.diagnostics import Rejection
from library.iro.operation_contracts import ResourceSemantics, VersionSelection
from library.iro.text import parse, print_strategy, round_trips
from library.iro.words import types as t
from library.iro.words.access import AccessOp
from library.iro.words.calls import CallOp, ConstantOp
from library.iro.words.data import ForkOp
from library.iro.words.lifetimes import HandlerOp
from library.iro.words.structure import ChoiceOp, RepeatOp

ROOT = Path(__file__).resolve().parents[3]
DESIGN = (ROOT / "openspec/changes/rework-model-strategy-trainer/design.md").read_text()
WIDE = (Path(__file__).parent / "wide.iro").read_text()


def design_example(marker: str) -> str:
    block = next(b for b in re.findall(r"```text\n(.*?)```", DESIGN, re.S) if marker in b)
    return "\n".join(line for line in block.splitlines() if not line.strip().startswith("//")) + "\n"


def op_named(strategy, cls, name: str | None = None):
    return next(op for op in strategy.walk() if isinstance(op, cls) and (name is None or getattr(op, "sym_name", None) == StringAttr(name)))


def test_the_designs_data_only_example_prints_back_exactly() -> None:
    example = design_example("@score_records")
    assert print_strategy(parse(example)) == example


def test_the_designs_cache_example_is_the_canonical_form() -> None:
    snippet = design_example("@read_latent_cache")
    body = "\n".join(f"      {line}" if line else line for line in snippet.splitlines())
    text = (
        "iro.strategy @s uses(@read_latent_cache v1, @encode_latent v1, @encode_sdxl_text v1, @use v1) {\n"
        "  %vae   = iro.resource @vae : participant\n"
        "  %te1   = iro.resource @te1 : participant\n"
        "  %te2   = iro.resource @te2 : participant\n"
        "  %batch = iro.stream @batch : item\n"
        "             bounds(outstanding 2, ready 1)\n"
        "\n"
        "  iro.parallel @p capture(%batch, %vae, %te1, %te2) {\n"
        "    iro.producer @data into %batch capture(%vae, %te1, %te2) {\n"
        "    ^(%items: int):\n"
        f"{body}\n"
        "      %item = iro.call @use(%latents, %cond) -> item\n"
        "      iro.yield %item\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
    printed = print_strategy(parse(text))
    assert body in printed
    assert round_trips(printed)


def test_every_core_word_round_trips() -> None:
    assert round_trips(WIDE)


def test_a_choice_on_an_optional_hands_the_value_to_its_first_alternative() -> None:
    choice = op_named(parse(WIDE), ChoiceOp)
    assert isinstance(choice.operands[0].type, t.OptionalType)
    assert choice.regions[0].block.args[0].type == t.value("latents")
    assert [r.type for r in choice.results] == [t.value("latents")]
    # The else alternative captures the VAE and the index; the first receives the same captures.
    assert len(choice.operands) == 3
    assert len(choice.regions[0].block.args) == 3 and len(choice.regions[1].block.args) == 2


def test_one_access_takes_several_leases_and_retained_results_keep_them() -> None:
    strategy = parse(WIDE)
    access = next(op for op in strategy.walk() if isinstance(op, AccessOp) and len(op.uses.data) == 2)
    assert [u.selection for u in ResourceSemantics().describe(access).uses] == [VersionSelection.CURRENT_AT_START] * 2
    call = op_named(access, CallOp)
    assert call.retains is not None and [x.value.data for x in call.retains.data] == [0, 1]
    routed = next(op for op in strategy.walk() if isinstance(op, AccessOp) and op.uses.data[0].route.data)
    assert routed.uses.data[0].route.data == "evaluation_point"


def test_a_handler_without_a_written_entry_takes_its_requests_value_type() -> None:
    handler = op_named(parse(WIDE), HandlerOp, "evaluation")
    assert handler.request == t.int_type()
    assert handler.regions[0].block.args[-1].type == t.int_type()


def test_literals_take_pythons_type_names() -> None:
    constants = [op for op in parse(WIDE).walk() if isinstance(op, ConstantOp)]
    written = {(type(c.value).__name__, c.result.type) for c in constants}
    assert ("StringAttr", t.value("str")) in written
    assert ("FloatAttr", t.value("float")) in written
    assert ("IntegerAttr", t.value("bool")) in written
    assert ("IntegerAttr", t.value("count")) in written
    assert any(isinstance(c.value, FloatAttr) for c in constants)
    assert any(isinstance(c.value, IntegerAttr) and c.value.value.data == 3 for c in constants)


def test_carried_values_and_forks_read_back_as_written() -> None:
    strategy = parse(WIDE)
    carrying = op_named(strategy, RepeatOp, "carrying")
    assert carrying.carry_count.value.data == 1 and len(carrying.results) == 1
    assert carrying.quiescence is not None and carrying.quiescence.data[0].data == "between_settled_iterations"
    assert len(op_named(strategy, ForkOp).forks) == 2


def test_source_locations_print_only_on_request_and_round_trip() -> None:
    text = (
        "iro.strategy @s uses(@f v1) {\n"
        '  %x = iro.resource @x loc("strategies/s.py":3)\n'
        "\n"
        "  iro.sequence @once capture(%x) {\n"
        '    iro.call @f(%x) loc("strategies/s.py":5)\n'
        '  } loc("strategies/s.py":4)\n'
        '} loc("strategies/s.py":1)\n'
    )
    strategy = parse(text)
    assert print_strategy(strategy, locations=True) == text
    assert "loc(" not in print_strategy(strategy)


def rejected(text: str) -> Rejection:
    with pytest.raises(Rejection) as raised:
        parse(text, source="s.iro")
    return raised.value


@pytest.mark.parametrize(
    ("text", "code", "line", "words"),
    [
        ("iro.strategy @s {\n  iro.frobnicate\n}\n", "iro-text-005", 2, "unknown word"),
        ("iro.strategy @s {\n  iro.yield %nowhere\n}\n", "iro-text-002", 2, "unknown value"),
        ("iro.strategy @s {\n  %x = iro.resource @x\n  %x = iro.resource @y\n}\n", "iro-text-004", 3, "already defined"),
        ("iro.strategy @s {\n  %x = iro.resource @x\n", "iro-text-006", 3, "not closed"),
        ("iro.strategy @s {\n  # nope\n}\n", "iro-text-001", 2, "unexpected character"),
    ],
)
def test_reading_problems_name_their_rule_and_line(text: str, code: str, line: int, words: str) -> None:
    found = rejected(text).diagnostics[0]
    assert found.code == code and found.source == f"s.iro:{line}" and words in found.message
    assert found.fragment is not None and "^" in found.fragment


def test_an_uncaptured_value_points_at_the_capture_clause() -> None:
    text = "iro.strategy @s uses(@f v1) {\n  %x = iro.resource @x\n  iro.sequence @once {\n    iro.call @f(%x)\n  }\n}\n"
    found = rejected(text).diagnostics[0]
    assert found.code == "iro-text-003" and found.source == "s.iro:4"
    assert found.fix == "add %x to the activity's capture(...)"


def test_a_handler_entry_that_cannot_be_inferred_must_be_written() -> None:
    text = "iro.strategy @s {\n  iro.parallel @p {\n    iro.handler @h sources [@p] max 1 {\n    }\n  }\n}\n"
    found = rejected(text).diagnostics[0]
    assert found.code == "iro-text-010" and "^(%step: int):" in (found.fix or "")


def test_an_entry_of_the_wrong_type_is_rejected_where_written() -> None:
    text = "iro.strategy @s {\n  %r = iro.stream @r : record\n         bounds(outstanding 1, ready 1)\n  iro.parallel @p capture(%r) {\n    iro.producer @q into %r {\n    ^(%i: str):\n      iro.yield %i\n    }\n  }\n}\n"
    found = rejected(text).diagnostics[0]
    assert found.code == "iro-text-008" and found.source == "s.iro:6" and "int" in found.message


def test_a_shape_problem_reads_in_the_written_forms_types() -> None:
    text = (
        "iro.strategy @s uses(@keep v1) {\n"
        "  %r = iro.stream @r : record\n"
        "         bounds(outstanding 1, ready 1)\n"
        "  iro.parallel @p capture(%r) {\n"
        "    iro.sequence @q capture(%r) {\n"
        "      %c = iro.take %r\n"
        "      %k = iro.filter %c {\n"
        "      ^(%v: record):\n"
        "        iro.call @keep(%v) -> bool\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
    found = rejected(text).diagnostics[0]
    assert found.code == "iro-text-009" and "yields (), expected (bool)" in found.message


# Regressions from the first review.


def generic(strategy) -> str:
    from io import StringIO

    from xdsl.printer import Printer

    out = StringIO()
    Printer(stream=out, print_generic_format=True).print_op(strategy)
    return out.getvalue()


def test_reading_back_the_printed_form_builds_the_same_operations() -> None:
    first = parse(WIDE)
    assert generic(parse(print_strategy(first))) == generic(first)


def test_a_local_body_that_yields_nothing_after_a_result_keeps_its_yield() -> None:
    text = (
        "iro.strategy @s uses(@f v1) {\n"
        "  %x = iro.resource @x\n"
        "  iro.access %x snapshot {\n"
        "  ^(%l):\n"
        "    %y = iro.call @f(%l) -> int\n"
        "    iro.yield\n"
        "  }\n"
        "}\n"
    )
    printed = print_strategy(parse(text))
    assert "    iro.yield\n" in printed
    assert not op_named(parse(printed), AccessOp).results


@pytest.mark.parametrize("literal", ["1e-05", "1e+16", "0.00001", "2.5e-3", "1e5"])
def test_exponent_floats_round_trip(literal: str) -> None:
    text = f"iro.strategy @s uses(@f v1) {{\n  iro.call @f({literal})\n}}\n"
    printed = print_strategy(parse(text))
    assert round_trips(printed)
    constant = op_named(parse(printed), ConstantOp)
    assert isinstance(constant.value, FloatAttr) and constant.value.value.data == float(literal)


@pytest.mark.parametrize(
    "text", ["iro.strategy @s {\n  %0 = iro.resource @r\n}\n", 'iro.strategy @s uses(@f v1) {\n  iro.call @f("a\\q")\n}\n']
)
def test_malformed_names_and_strings_are_reading_problems(text: str) -> None:
    assert rejected(text).diagnostics[0].code == "iro-text-001"


def test_a_shape_problem_names_the_line_of_its_statement() -> None:
    text = "iro.strategy @s {\n  %x = iro.resource @x\n  iro.release %x\n}\n"
    found = rejected(text).diagnostics[0]
    assert found.source == "s.iro:3"
    assert "should be of base attribute iro.retained" in found.message and "!iro" not in found.message


def test_names_and_captures_are_given_once() -> None:
    twice = "iro.strategy @s {\n  %a = iro.resource @a\n  %b = iro.resource @a\n}\n"
    found = rejected(twice).diagnostics[0]
    assert found.code == "iro-text-011" and found.source == "s.iro:3"
    captured = "iro.strategy @s {\n  %a = iro.resource @a\n  %b = iro.resource @b\n  iro.sequence @q capture(%a, %b as %a) {\n  }\n}\n"
    assert rejected(captured).diagnostics[0].code == "iro-text-004"


def test_a_word_outside_the_composition_is_unknown() -> None:
    from library.iro.composition import Composition, core

    without_fork = Composition([d for d in core().definitions.values() if d.name != "iro.fork"])
    text = "iro.strategy @s {\n  %r = iro.stream @r : item\n         bounds(outstanding 1, ready 1)\n  iro.sequence @q capture(%r) {\n    %c = iro.take %r\n    %a, %b = iro.fork %c\n  }\n}\n"
    with pytest.raises(Rejection) as raised:
        parse(text, without_fork, source="s.iro")
    assert raised.value.diagnostics[0].code == "iro-text-005"


def test_a_long_header_moves_its_clauses_and_declarations_together() -> None:
    text = (
        "iro.strategy @s {\n"
        "  %r = iro.stream @r : item\n"
        "         bounds(outstanding 1, ready 1)\n"
        "  iro.sequence @q capture(%r) {\n"
        "    %first = iro.take %r\n"
        "    %last = iro.repeat @carrying_for_a_while 3 times carry(%c = %first) reproducibility(schedule_independent) {\n"
        "      iro.yield %c\n"
        "    }\n"
        "    iro.dispose %last consumed\n"
        "  }\n"
        "}\n"
    )
    printed = print_strategy(parse(text))
    assert max(len(line) for line in printed.splitlines()) <= 80
    assert "        carry(%c = %first) reproducibility(schedule_independent) {\n" in printed
    assert round_trips(printed)


def test_parallel_branches_print_with_the_clauses_names() -> None:
    strategy = parse(design_example("@score_records"))
    parallel = next(op for op in strategy.walk() if op.name == "iro.parallel")
    for argument in parallel.regions[1].block.args:
        argument.name_hint = "renamed"
    assert round_trips(print_strategy(strategy))
