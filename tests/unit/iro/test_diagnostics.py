import pytest

from library.iro.diagnostics import Diagnostic, Kind, Problems, Rejection, Stage


def test_rendering_follows_the_designed_layout() -> None:
    found = Diagnostic(
        Stage.VERIFY,
        "iro-life-004",
        "stop @data after @training does not cover @validation's pending request; @validation still takes from @batches.",
        names=("@data", "@training", "@validation"),
        source="strategies/sdxl/finetune.py:88",
        fix="stop @data after @consumers, or give @validation its own stream.",
    )
    # The layout D18 shows, line for line.
    assert found.render() == (
        "iro verify [iro-life-004] strategies/sdxl/finetune.py:88\n"
        "  stop @data after @training does not cover @validation's pending\n"
        "  request; @validation still takes from @batches.\n"
        "  fix: stop @data after @consumers, or give @validation its own stream."
    )


def test_target_limits_are_marked_and_kept_apart_from_invalid_meaning() -> None:
    limit = Diagnostic(Stage.PREPARE, "iro-target-001", "this target cannot realize protected live reads", kind=Kind.TARGET_LIMIT)
    assert limit.render().splitlines()[0] == "iro prepare [iro-target-001] (target limit)"
    assert Rejection([limit]).target_limits_only
    error = Diagnostic(Stage.VERIFY, "iro-own-001", "a claim is disposed twice")
    assert not Rejection([limit, error]).target_limits_only


def test_fragments_print_indented_under_the_message() -> None:
    found = Diagnostic(Stage.PARSE, "iro-text-001", "unknown word", fragment="iro.frobnicate %x\n  ^")
    assert found.render().splitlines()[2:] == ["    iro.frobnicate %x", "      ^"]


@pytest.mark.parametrize("code", ["life-004", "iro-Life-004", "iro-life-4", "iro--004"])
def test_rule_codes_have_one_shape(code: str) -> None:
    with pytest.raises(ValueError, match="rule code"):
        Diagnostic(Stage.VERIFY, code, "message")


def test_a_rejection_reports_every_independent_problem_once() -> None:
    problems = Problems(Stage.VERIFY)
    assert not problems
    problems.add("iro-own-001", "a claim is disposed twice", names=("%claim",))
    problems.add("iro-own-001", "a claim is disposed twice", names=("%claim",))
    problems.add("iro-life-001", "a stop covers nothing")
    assert len(problems) == 2
    with pytest.raises(Rejection) as raised:
        problems.reject_if_any()
    assert raised.value.codes == ("iro-own-001", "iro-life-001")
    assert [d.stage for d in raised.value] == [Stage.VERIFY, Stage.VERIFY]


def test_no_problems_means_no_rejection() -> None:
    Problems(Stage.NORMALIZE).reject_if_any()
    with pytest.raises(ValueError):
        Rejection([])
