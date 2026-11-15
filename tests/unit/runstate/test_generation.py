"""Generations and publication: one swap installs every changed section, or nothing changes (D19)."""

import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest

from library.runstate.generation import Generations, Publication
from library.runstate.owners import Mutation, Refused, Stale


@dataclass(frozen=True)
class Plain:
    owner: str
    revision: int
    held: dict = field(default_factory=dict)

    def revision_of(self, key: str) -> int | None:
        return self.held.get(key)


@dataclass
class Owner:
    refuse: str | None = None
    seen: list = field(default_factory=list)

    @contextmanager
    def admit(self, publication, current, next) -> Generator[None]:
        self.seen.append((current, next))
        if self.refuse:
            raise Refused(self.refuse)
        yield


def generations(*owners: str) -> tuple[Generations, dict[str, Owner]]:
    made = Generations("run1")
    registered = {owner: Owner() for owner in owners}
    for owner, publishes in registered.items():
        made.register(owner, publishes)
    return made, registered


def test_a_publication_installs_the_next_generation_and_keeps_the_sections_it_does_not_change() -> None:
    made, _ = generations("a", "b")
    first = made.publish(Publication(made.basis("o"), (Plain("a", 1), Plain("b", 1))))
    second = made.publish(Publication(made.basis("o"), (Plain("a", 2),)))
    assert (first.number, second.number) == (1, 2)
    assert second.section("b") is first.section("b")
    assert made.current is second
    assert first.section("a").revision == 1, "a published generation never changes"


def test_a_result_built_from_an_older_generation_never_publishes() -> None:
    made, _ = generations("a")
    made.publish(Publication(made.basis("o"), (Plain("a", 1),)))
    old = made.basis("o")
    made.publish(Publication(made.basis("o"), (Plain("a", 2),)))
    with pytest.raises(Stale):
        made.publish(Publication(old, (Plain("a", 3),)))
    assert made.current.section("a").revision == 2


@pytest.mark.parametrize(
    ("sections", "message"),
    [
        ((Plain("a", 1), Plain("a", 2)), "two sections"),
        ((Plain("c", 1),), "not registered"),
    ],
)
def test_a_malformed_publication_is_refused(sections, message: str) -> None:
    made, _ = generations("a")
    with pytest.raises(Refused, match=message):
        made.publish(Publication(made.basis("o"), sections))
    assert made.current.number == 0


def test_a_next_section_must_move_past_the_current_revision() -> None:
    made, _ = generations("a")
    made.publish(Publication(made.basis("o"), (Plain("a", 1),)))
    with pytest.raises(Refused, match="not past 1"):
        made.publish(Publication(made.basis("o"), (Plain("a", 1),)))


def test_an_owner_that_refuses_its_next_section_stops_the_whole_publication() -> None:
    made, owners = generations("a", "b")
    made.publish(Publication(made.basis("o"), (Plain("a", 1), Plain("b", 1))))
    owners["b"].refuse = "a lease is out"
    with pytest.raises(Refused, match="a lease is out"):
        made.publish(Publication(made.basis("o"), (Plain("a", 2), Plain("b", 2))))
    assert made.current.number == 1 and made.current.section("a").revision == 1, "no owner's section changed"


def test_in_place_work_waits_for_withdrawal() -> None:
    made, _ = generations("a")
    attempt = made.attempt("o", Mutation.IN_PLACE)
    with pytest.raises(Refused, match="withdraw"):
        made.publish(Publication(attempt.basis, (Plain("a", 1),), attempt))


def test_of_concurrent_publications_on_one_basis_exactly_one_installs() -> None:
    """Publications are serialized: without the GIL, two built from one generation never both swap."""
    made, _ = generations("a")
    basis = made.basis("o")
    outcomes: list[str] = []
    barrier = threading.Barrier(8)

    def publish(revision: int) -> None:
        barrier.wait()
        try:
            made.publish(Publication(basis, (Plain("a", revision),)))
            outcomes.append("installed")
        except Stale:
            outcomes.append("stale")

    threads = [threading.Thread(target=publish, args=(revision,)) for revision in range(1, 9)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["installed"] + ["stale"] * 7
    assert made.current.number == 1
