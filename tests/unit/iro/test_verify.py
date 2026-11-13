"""Verification: claims, leases, accesses, calls, lifecycles and streams, with obligations and reproducibility."""

from pathlib import Path

import pytest

from library.iro.diagnostics import Rejection
from library.iro.normalize import normalize
from library.iro.operation_contracts import Reproducibility
from library.iro.text import parse
from library.iro.verify import CORE_EXECUTION, Checkpoint, Verification, verify

HERE = Path(__file__).parent
WIDE = (HERE / "wide.iro").read_text()
WIDE_NORMALIZED = (HERE / "wide.normalized.iro").read_text()


def verified(text: str, **options: bool) -> Verification:
    return verify(normalize(parse(text, source="s.iro")), **options)


def rejected(text: str, **options: bool) -> Rejection:
    with pytest.raises(Rejection) as raised:
        verified(text, **options)
    return raised.value


# The fixture.


def test_the_fixture_verifies_with_obligations_at_two_checkpoints() -> None:
    verification = verified(WIDE)
    obligations = verification.obligations
    assert obligations.checkpoints == (CORE_EXECUTION, Checkpoint("evaluation"))
    assert str(Checkpoint("evaluation")) == "request to @evaluation"
    evidence = {o.evidence for o in obligations}
    assert "participant @unet bound, materialized by @load_sdxl" in evidence
    assert "@data stops admitting once @consumers settles; its admitted items still settle" in evidence
    requested = obligations.at(Checkpoint("evaluation"))
    assert [o.evidence for o in requested] == ["protected_current lease of @unet at at_request"]
    releases = [o for o in obligations if o.evidence.startswith("the leased state stays protected")]
    assert sorted(o.names for o in releases) == [("@te1", "@te2"), ("@te1", "@te2", "@unet")], (
        "a release keeps its call's leases and retained inputs"
    )


def test_a_cold_loaded_normalized_form_derives_the_same_obligations() -> None:
    assert verified(WIDE_NORMALIZED).obligations.revision == verified(WIDE).obligations.revision


def test_the_fixture_reports_why_it_may_depend_on_scheduling() -> None:
    verification = verified(WIDE)
    assert verification.reproducibility == Reproducibility.UNKNOWN, "its calls declare nothing"
    reasons = {(c.subject, c.reproducibility) for c in verification.contributors if c.reproducibility != Reproducibility.UNKNOWN}
    assert ("@wide/@items", Reproducibility.MAY_DEPEND) in reasons, "@training and @turns take from @items concurrently"


def test_verification_reads_normalized_strategies_only() -> None:
    with pytest.raises(ValueError, match="normalize it first"):
        from library.iro.verify import Analysis

        Analysis(parse(WIDE))


# Claims.

STREAM = """iro.strategy @s uses(@work v1) {
  %items = iro.stream @items : item
             bounds(outstanding 1, ready 1)

  iro.parallel @all capture(%items) {
    iro.producer @data into %items {
    ^(%index: int):
      %item = iro.call @work(%index) -> item
      iro.yield %item
    }

    iro.repeat @consumer 2 times capture(%items) {
      BODY
    }

    iro.stop @data after @consumer
  }
}
"""


def consuming(body: str) -> str:
    return STREAM.replace("BODY", body.strip().replace("\n", "\n      "))


SETTLED = "%c = iro.take %items\niro.dispose %c consumed"


def test_a_taken_and_disposed_claim_verifies() -> None:
    verified(consuming(SETTLED))


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ("%c = iro.take %items\niro.call @work(%c)", "iro-claim-001"),
        ("%c = iro.take %items\niro.dispose %c consumed\niro.dispose %c cancelled", "iro-claim-002"),
        ("%c = iro.take %items\niro.dispose %c consumed\n%v = iro.borrow %c\niro.call @work(%v)", "iro-claim-003"),
        ("%c = iro.take %items\niro.repeat @twice 2 times capture(%c) {\n  iro.dispose %c consumed\n}", "iro-claim-004"),
        (
            "%c = iro.take %items\niro.parallel @both capture(%c) {\n  iro.dispose %c consumed\n\n  iro.dispose %c cancelled\n}",
            "iro-claim-005",
        ),
        ("%c = iro.take %items\niro.call @work(%c)\niro.dispose %c consumed", "iro-claim-006"),
    ],
)
def test_a_claim_is_settled_exactly_once_on_every_path(body: str, code: str) -> None:
    assert code in rejected(consuming(body)).codes


def test_forks_each_settled_in_their_own_branch_verify() -> None:
    body = """%c = iro.take %items
%a, %b = iro.fork %c
iro.parallel @both capture(%a, %b) {
  iro.dispose %a consumed

  iro.dispose %b filtered_out
}"""
    verified(consuming(body))


def test_a_choice_may_settle_a_claim_differently_in_each_alternative() -> None:
    body = """%c = iro.take %items
%keep = iro.call @work() -> bool
iro.choice %keep {
  iro.dispose %c consumed
} else {
  iro.dispose %c cancelled
}"""
    verified(consuming(body))
    one_sided = """%c = iro.take %items
%keep = iro.call @work() -> bool
iro.choice %keep {
  iro.dispose %c consumed
} else {
  iro.call @work()
}"""
    assert rejected(consuming(one_sided)).codes == ("iro-claim-001",)


# Leases and retained results.

MODEL = """iro.strategy @s uses(@objective v1, @step v1, @report v1) {
  %m = iro.resource @m : participant
  iro.sequence @train capture(%m) {
    BODY
  }
}
"""


def training(body: str) -> str:
    return MODEL.replace("BODY", body.strip().replace("\n", "\n    "))


LOSS = "%loss = iro.access %m protected_current {\n^(%lease):\n  iro.call @objective(%lease) -> retained loss\n}"
UPDATE = "iro.access %m owner_update {\n^(%lease):\n  iro.call @step(%lease)\n}"


def test_an_update_after_the_retained_result_is_used_up_verifies() -> None:
    verified(training(f"{LOSS}\niro.call @report(%loss)\n{UPDATE}"))


def test_an_update_while_a_retained_result_keeps_the_state_protected_is_rejected() -> None:
    assert rejected(training(f"{LOSS}\n{UPDATE}\niro.call @report(%loss)")).codes == ("iro-lease-004",)


@pytest.mark.parametrize(
    "body",
    [
        f"{LOSS}\niro.yield %loss",
        f"{LOSS}\niro.sequence @elsewhere capture(%loss) {{\n  iro.call @report(%loss)\n}}",
    ],
)
def test_a_retained_result_never_leaves_its_activity(body: str) -> None:
    assert "iro-lease-002" in rejected(training(body)).codes


def test_a_lease_never_leaves_its_access() -> None:
    body = "%l = iro.access %m protected_current {\n^(%lease):\n  iro.yield %lease\n}\niro.call @report(%l)"
    assert "iro-lease-001" in rejected(training(body)).codes


def test_a_normalized_form_with_a_missing_or_early_release_is_rejected() -> None:
    normalized = normalize(parse(training(f"{LOSS}\niro.call @report(%loss)"))).text()
    assert "iro.release %loss\n" in normalized
    missing = normalized.replace("    iro.release %loss\n", "")
    with pytest.raises(Rejection) as raised:
        normalize(parse(missing))
    assert raised.value.codes == ("iro-norm-008",)
    early = normalized.replace(
        "    iro.call @report(%loss)\n    iro.release %loss\n", "    iro.release %loss\n    iro.call @report(%loss)\n"
    )
    assert early != normalized
    assert rejected(early).codes == ("iro-lease-003",)


# Accesses.


def test_at_request_is_read_only_in_requested_work() -> None:
    body = "iro.access %m protected_current at_request {\n^(%lease):\n  iro.call @report(%lease)\n}"
    assert rejected(training(body)).codes == ("iro-access-001",)


def test_a_nested_access_that_waits_for_its_own_enclosing_access_is_rejected() -> None:
    body = "iro.access %m protected_current {\n^(%outer):\n  iro.access %m owner_update {\n  ^(%inner):\n    iro.call @step(%inner)\n  }\n}"
    assert rejected(training(body)).codes == ("iro-access-002",)


# Calls.


def test_every_implementation_and_materializer_is_pinned() -> None:
    assert rejected(training("iro.call @unpinned()")).codes == ("iro-call-001",)
    text = training("iro.call @report()").replace(
        "%m = iro.resource @m : participant", "%m = iro.resource @m : participant materialize(@load)"
    )
    assert rejected(text).codes == ("iro-call-001",)


def test_a_call_reaches_a_resource_through_a_lease() -> None:
    assert rejected(training("iro.call @report(%m)")).codes == ("iro-call-002",)


# Lifecycles.

REQUESTS = """iro.strategy @s uses(@work v1, @measure v1) {
  %items = iro.stream @items : item
             bounds(outstanding 1, ready 1)

  iro.parallel @all capture(%items) {
    iro.producer @data into %items {
    ^(%index: int):
      %item = iro.call @work(%index) -> item
      iro.yield %item
    }

    iro.repeat @training 2 times capture(%items) {
    ^(%step: int):
      %c = iro.take %items
      iro.dispose %c consumed
      iro.request @evaluation(%step) from @training at settled_success
    }

    HANDLER

    STOP
  }
}
"""
HANDLER = "iro.handler @evaluation sources [@training] max 1 {\n      iro.call @measure()\n    }"
STOP = "iro.stop @data after @training"


def lifecycle(*, handler: str = HANDLER, stop: str = STOP) -> str:
    return REQUESTS.replace("HANDLER", handler).replace("STOP", stop)


def test_requested_work_with_a_covering_stop_verifies() -> None:
    verification = verified(lifecycle())
    assert any(o.evidence.startswith("registered before @training reaches settled_success") for o in verification.obligations)


def test_a_stop_must_cover_requested_work_that_still_takes() -> None:
    handler = (
        "iro.handler @evaluation sources [@training] max 1 capture(%items) {\n"
        "      %c = iro.take %items\n"
        "      iro.dispose %c consumed\n"
        "    }"
    )
    found = rejected(lifecycle(handler=handler)).diagnostics
    assert [d.code for d in found] == ["iro-life-004"]
    assert (
        found[0].message == "stop @data after @training does not cover @evaluation's pending request; @evaluation still takes from @items."
    )
    assert found[0].fix == "stop @data after an activity that settles after @evaluation, or give @evaluation its own stream."


def test_a_producer_needs_a_stop_and_a_stop_names_a_producer() -> None:
    assert rejected(lifecycle(stop="")).codes == ("iro-life-003",)
    assert "iro-life-002" in rejected(lifecycle(stop=f"{STOP}\n\n    iro.stop @training after @data")).codes


def test_a_request_is_made_inside_its_source() -> None:
    stop = f"{STOP}\n\n    iro.sequence @elsewhere {{\n      iro.request @evaluation(1) from @training at settled_success\n    }}"
    assert "iro-life-001" in rejected(lifecycle(stop=stop)).codes


def test_points_that_wait_for_each_other_are_rejected() -> None:
    assert "iro-life-005" in rejected(lifecycle(stop="iro.stop @data after @data")).codes


# Streams.


def test_a_stream_has_exactly_one_producer() -> None:
    second = (
        "iro.producer @more into %items {\n    ^(%index: int):\n      %item = iro.call @work(%index) -> item\n      iro.yield %item\n    }\n\n    "
        + STOP
    )
    assert "iro-flow-001" in rejected(lifecycle(stop=second)).codes


def test_admission_compares_owners_production_reads() -> None:
    text = lifecycle().replace(
        "  %items = iro.stream @items : item\n             bounds(outstanding 1, ready 1)",
        "  %m = iro.resource @m : participant\n  %items = iro.stream @items : item\n             bounds(outstanding 1, ready 1)\n             admit exact_current(@m)",
    )
    assert rejected(text).codes == ("iro-flow-002",)


# Decisions.

DECLARED = """iro.strategy @s uses(@work v1) {
  iro.sequence @once {
    iro.call @work() reproducibility(schedule_independent) rank_rule(RANK)
  }
}
"""


def test_a_strategy_whose_decisions_are_all_declared_is_schedule_independent() -> None:
    assert verified(DECLARED.replace("RANK", "replicated_deterministic")).reproducibility == Reproducibility.INDEPENDENT


def test_on_several_ranks_every_decision_agrees() -> None:
    verified(DECLARED.replace("RANK", "replicated_deterministic"), replicated=True)
    distributed = verified(DECLARED.replace("RANK", "decided_once_and_distributed"), replicated=True)
    assert any(o.evidence.startswith("decided once and distributed") for o in distributed.obligations)
    assert rejected(DECLARED.replace("RANK", "local"), replicated=True).codes == ("iro-decide-001",)


def test_a_claim_in_a_nested_parallel_is_still_settled_by_one_branch() -> None:
    body = """%c = iro.take %items
iro.parallel @outer capture(%c) {
  iro.parallel @inner capture(%c) {
    iro.dispose %c consumed

    iro.dispose %c cancelled
  }
}"""
    assert "iro-claim-005" in rejected(consuming(body)).codes


def test_a_normalized_form_releasing_a_kept_result_first_is_rejected() -> None:
    text = """iro.strategy @s uses(@objective v1, @derive v1, @report v1) {
  %m = iro.resource @m : participant
  iro.sequence @train capture(%m) {
    %a = iro.access %m protected_current {
    ^(%l):
      iro.call @objective(%l) -> retained loss
    }
    %b = iro.access %m protected_current {
    ^(%l2):
      iro.call @derive(%l2, %a) -> retained pred
    }
    iro.call @report(%b)
  }
}
"""
    normalized = normalize(parse(text)).text()
    swapped = normalized.replace("    iro.release %b\n    iro.release %a\n", "    iro.release %a\n    iro.release %b\n")
    assert swapped != normalized
    assert rejected(swapped).codes == ("iro-lease-003",)


def test_a_producer_needs_a_stop_that_always_runs() -> None:
    stop = (
        "iro.sequence @maybe {\n"
        "      %flag = iro.call @measure() -> bool\n"
        "      iro.choice %flag {\n"
        "        iro.stop @data after @training\n"
        "      }\n"
        "    }"
    )
    found = rejected(lifecycle(stop=stop))
    assert found.codes == ("iro-life-003",)
    assert "may not run" in found.diagnostics[0].message


def test_admission_by_a_concurrently_updated_owner_may_depend_on_scheduling() -> None:
    text = """iro.strategy @s uses(@work v1, @step v1) {
  %m = iro.resource @m : participant
  %items = iro.stream @items : item
             bounds(outstanding 1, ready 1)
             admit exact_current(@m)

  iro.parallel @all capture(%items, %m) {
    iro.producer @data into %items capture(%m) {
    ^(%index: int):
      %item = iro.access %m protected_current {
      ^(%lease):
        iro.call @work(%lease, %index) -> item
      }
      iro.yield %item
    }

    iro.repeat @consumer 1 times capture(%items) {
      %c = iro.take %items
      iro.dispose %c consumed
    }

    iro.access %m owner_update {
    ^(%lease):
      iro.call @step(%lease)
    }

    iro.stop @data after @consumer
  }
}
"""
    reasons = [c.reason for c in verified(text).contributors if c.reproducibility == Reproducibility.MAY_DEPEND]
    assert "whether an item is admitted can depend on when @m is updated" in reasons
