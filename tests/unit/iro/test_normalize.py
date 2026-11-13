"""Normalization: derived words expanded, releases placed, relationships resolved, nothing invented."""

from pathlib import Path

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr, SymbolRefAttr
from xdsl.ir import Operation, Region, SSAValue
from xdsl.irdl import irdl_op_definition, region_def

from library.iro.diagnostics import Rejection
from library.iro.normalize import Link, normalize
from library.iro.operation_contracts import DerivedWord, LifecycleProtocol
from library.iro.text import parse, print_strategy, round_trips
from library.iro.words.access import ReleaseOp
from library.iro.words.base import Activity
from library.iro.words.calls import CallOp
from library.iro.words.structure import DerivedOp, SequenceOp, _activity

HERE = Path(__file__).parent
WIDE = (HERE / "wide.iro").read_text()
WIDE_NORMALIZED = (HERE / "wide.normalized.iro").read_text()


def ancestry(op: Operation) -> list[str] | None:
    found = op.properties.get("ancestry")
    return [a.data for a in found.data] if isinstance(found, ArrayAttr) else None


def statements(op: Operation) -> list[str]:
    return [inner.name for inner in op.walk() if inner is not op]


# The fixture.


def test_the_accepted_fixture_normalizes_to_the_normalized_fixture() -> None:
    accepted = parse(WIDE)
    normalized = normalize(accepted)
    assert normalized.text() == WIDE_NORMALIZED
    assert print_strategy(accepted) == WIDE, "the accepted strategy stays as written"
    assert round_trips(WIDE_NORMALIZED)


def test_normalizing_a_normalized_strategy_changes_nothing() -> None:
    assert normalize(parse(WIDE_NORMALIZED)).text() == WIDE_NORMALIZED


def test_relationships_are_resolved_by_field_and_kind() -> None:
    found = {(link.field, link.target, link.kind) for link in normalize(parse(WIDE)).links}
    assert found == {
        ("owners", "vae", "resource"),
        ("handler", "evaluation", "handler"),
        ("source", "training", "activity"),
        ("source", "consumers", "activity"),
        ("sources", "training", "activity"),
        ("target", "data", "activity"),
    }
    assert all(isinstance(link, Link) and link.path.startswith("@wide/") for link in normalize(parse(WIDE)).links)


# Derived words.


def test_a_derived_body_moves_unchanged_into_its_sequence() -> None:
    accepted = parse(WIDE)
    derived = next(op for op in accepted.walk() if isinstance(op, DerivedOp))
    sequence = next(op for op in normalize(accepted).strategy.walk() if isinstance(op, SequenceOp) and op.sym_name == StringAttr("weigh"))
    assert statements(sequence) == statements(derived)
    assert [v.type for v in sequence.operands] == [v.type for v in derived.operands]
    assert [r.type for r in sequence.results] == [r.type for r in derived.results]
    assert ancestry(sequence) == ["iro.derived @weigh"]
    assert all(ancestry(op) is None for op in sequence.walk() if op is not sequence), "authored statements gain no ancestry"


def _logged(op: Operation) -> tuple[list[Operation], list[SSAValue]]:
    """A domain derived word: a call to @log, then the authored body as a derived word of its own."""
    log = CallOp.create(properties={"implementation": SymbolRefAttr("log")})
    body = Region()
    op.regions[0].move_blocks(body)
    sequence = DerivedOp.create(
        operands=list(op.operands),
        result_types=[r.type for r in op.results],
        properties={"sym_name": op.properties["sym_name"]},
        regions=[body],
    )
    return [log, sequence], list(sequence.results)


def _forever(op: Operation) -> tuple[list[Operation], list[SSAValue]]:
    body = Region()
    op.regions[0].move_blocks(body)
    again = ForeverOp.create(
        operands=list(op.operands), result_types=[r.type for r in op.results], properties=dict(op.properties), regions=[body]
    )
    return [again], list(again.results)


def _dropping(op: Operation) -> tuple[list[Operation], list[SSAValue]]:
    return [], []


@irdl_op_definition
class LoggedOp(Activity):
    """A test word that logs, then runs its body."""

    name = "test.logged"
    body = region_def("single_block")
    traits = _activity(LifecycleProtocol.SEQUENCE, DerivedWord(_logged))


@irdl_op_definition
class ForeverOp(Activity):
    """A test word whose expansion is itself."""

    name = "test.forever"
    body = region_def("single_block")
    traits = _activity(LifecycleProtocol.SEQUENCE, DerivedWord(_forever))


@irdl_op_definition
class DroppingOp(Activity):
    """A test word whose expansion loses its body."""

    name = "test.dropping"
    body = region_def("single_block")
    traits = _activity(LifecycleProtocol.SEQUENCE, DerivedWord(_dropping))


NESTED = """iro.strategy @s {
  iro.sequence @outer {
    iro.derived @inner {
      iro.call @work()
      iro.derived @innermost {
        iro.call @work()
      }
    }
  }
}
"""


def as_word(strategy, name: str, cls) -> None:
    """Swaps the derived word named ``name`` for ``cls``, moving its body."""
    old = next(op for op in strategy.walk() if isinstance(op, DerivedOp) and op.sym_name == StringAttr(name))
    body = Region()
    old.body.move_blocks(body)
    new = cls.create(
        operands=list(old.operands), result_types=[r.type for r in old.results], properties=dict(old.properties), regions=[body]
    )
    old.parent_block().insert_op_before(new, old)
    old.parent_block().erase_op(old)


def test_a_domain_derived_word_records_itself_on_everything_its_expansion_creates() -> None:
    strategy = parse(NESTED)
    as_word(strategy, "inner", LoggedOp)
    normalized = normalize(strategy).strategy
    created = [op for op in normalized.walk() if ancestry(op) is not None]
    assert [(op.name, ancestry(op)) for op in created] == [
        ("iro.call", ["test.logged @inner"]),
        ("iro.sequence", ["test.logged @inner", "iro.derived @inner"]),
        ("iro.sequence", ["iro.derived @innermost"]),
    ]
    printed = print_strategy(normalized)
    assert "from(test.logged @inner, iro.derived @inner)" in printed
    assert "iro.sequence @innermost from(iro.derived @innermost)" in printed, "an authored word moved by an expansion is not its creation"
    assert round_trips(printed)


def test_an_expansion_that_never_ends_is_rejected() -> None:
    strategy = parse(NESTED)
    as_word(strategy, "innermost", ForeverOp)
    with pytest.raises(Rejection) as raised:
        normalize(strategy)
    assert raised.value.codes == ("iro-norm-006",)


def test_an_expansion_that_drops_authored_statements_is_rejected() -> None:
    strategy = parse(NESTED)
    as_word(strategy, "innermost", DroppingOp)
    with pytest.raises(Rejection) as raised:
        normalize(strategy)
    assert "iro-norm-007" in raised.value.codes


# Releases.


def released_after(strategy) -> dict[str, str]:
    """Each release, keyed by what it releases, with the statement just before it."""
    found = {}
    for op in strategy.walk():
        if isinstance(op, ReleaseOp):
            previous = op.prev_op
            found[op.product.owner.name] = previous.name if previous is not None else ""
    return found


def test_a_release_follows_the_last_use_even_inside_a_later_body() -> None:
    text = """iro.strategy @s {
  %m = iro.resource @m : participant
  iro.sequence @step capture(%m) {
    %loss = iro.access %m protected_current {
    ^(%lease):
      iro.call @objective(%lease) -> retained loss
    }
    %done = iro.call @check() -> bool
    iro.choice %done {
      iro.call @report(%loss)
    }
    iro.call @after()
  }
}
"""
    normalized = normalize(parse(text)).strategy
    release = next(op for op in normalized.walk() if isinstance(op, ReleaseOp))
    assert release.prev_op is not None and release.prev_op.name == "iro.choice"
    assert isinstance(release.parent_op(), SequenceOp), "a yielded result is released where it arrives, not inside the access"
    assert sum(isinstance(op, ReleaseOp) for op in normalized.walk()) == 1


def test_an_unused_retained_result_is_released_at_once() -> None:
    text = """iro.strategy @s {
  %m = iro.resource @m : participant
  iro.sequence @step capture(%m) {
    iro.access %m protected_current {
    ^(%lease):
      iro.call @objective(%lease) -> retained loss
    }
    iro.call @after()
  }
}
"""
    normalized = normalize(parse(text)).strategy
    release = next(op for op in normalized.walk() if isinstance(op, ReleaseOp))
    assert release.prev_op is not None and release.prev_op.name == "iro.access"


# Relationships that do not resolve.

BASE = """iro.strategy @s {
  %items = iro.stream @items : item
             bounds(outstanding 1, ready 1)

  iro.parallel @all capture(%items) {
    iro.repeat @training 2 times capture(%items) {
    ^(%step: int):
      %c = iro.take %items
      iro.dispose %c consumed
      REQUEST
    }

    HANDLER

    STOP
  }
}
"""
REQUEST = "iro.request @evaluation(%step) from @training at settled_success"
HANDLER = "iro.handler @evaluation sources [@training] max 1 {\n      iro.call @measure()\n    }"
STOP = "iro.stop @training after @training"


def strategy_with(*, request: str = REQUEST, handler: str = HANDLER, stop: str = STOP) -> str:
    return BASE.replace("REQUEST", request).replace("HANDLER", handler).replace("STOP", stop)


def test_the_relationship_base_normalizes() -> None:
    normalize(parse(strategy_with()))


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"stop": "iro.stop @nothing after @training"}, "iro-norm-002"),
        ({"stop": "iro.stop @items after @training"}, "iro-norm-003"),
        (
            {
                "handler": "iro.handler @evaluation sources [@training] max 1 {\n      ^(%r: record):\n      iro.call @measure(%r)\n    }",
            },
            "iro-norm-004",
        ),
        ({"handler": "iro.handler @evaluation sources [@all] max 1 {\n      iro.call @measure()\n    }"}, "iro-norm-005"),
    ],
)
def test_a_relationship_that_does_not_resolve_is_rejected(changes: dict[str, str], code: str) -> None:
    with pytest.raises(Rejection) as raised:
        normalize(parse(strategy_with(**changes)))
    assert raised.value.codes == (code,)


def test_a_request_names_a_handler() -> None:
    strategy = parse(strategy_with())
    request = next(op for op in strategy.walk() if op.name == "iro.request")
    request.properties["handler"] = SymbolRefAttr("training")
    with pytest.raises(Rejection) as raised:
        normalize(strategy)
    assert raised.value.codes == ("iro-norm-003",)


def test_a_stream_admits_against_resources_only() -> None:
    text = strategy_with().replace(
        "bounds(outstanding 1, ready 1)", "bounds(outstanding 1, ready 1)\n             admit exact_current(@training)"
    )
    with pytest.raises(Rejection) as raised:
        normalize(parse(text))
    assert raised.value.codes == ("iro-norm-003",)


def test_two_statements_with_one_name_are_rejected() -> None:
    strategy = parse(strategy_with())
    handler = next(op for op in strategy.walk() if op.name == "iro.handler")
    handler.properties["sym_name"] = StringAttr("training")
    with pytest.raises(Rejection) as raised:
        normalize(strategy)
    assert "iro-norm-001" in raised.value.codes


# Stages.


def test_an_accepted_strategy_has_no_releases_and_a_normalized_one_no_derived_words() -> None:
    with pytest.raises(Rejection) as raised:
        parse(WIDE_NORMALIZED.replace("@wide normalized", "@wide"))
    assert {d.code for d in raised.value.diagnostics} == {"iro-text-009", "iro-text-012"}, (
        "releases and from(...) belong to normalized forms"
    )
    assert any("normalization places releases" in d.message for d in raised.value.diagnostics)
    with pytest.raises(Rejection) as raised:
        parse(WIDE.replace("@wide", "@wide normalized", 1))
    assert all("derived words expanded" in d.message for d in raised.value.diagnostics)


def test_an_accepted_strategy_names_no_expansion() -> None:
    text = "iro.strategy @s uses(@work v1) {\n  iro.sequence @q from(iro.derived @q) {\n    iro.call @work()\n  }\n}\n"
    with pytest.raises(Rejection) as raised:
        parse(text)
    assert raised.value.codes == ("iro-text-012",)
    assert round_trips(text.replace("@s uses", "@s normalized uses"))


def test_a_retained_result_is_released_after_every_result_that_keeps_it() -> None:
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
    printed = normalize(parse(text)).text()
    assert printed.index("iro.release %b") < printed.index("iro.release %a")
    assert printed.index("iro.call @report(%b)") < printed.index("iro.release %b")
