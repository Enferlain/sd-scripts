"""The owner pattern: references, versions and dependencies (D19)."""

import threading
from dataclasses import dataclass

import pytest

from library.iro.seams import Version as SeamVersion
from library.runstate.authority import ParticipantRef
from library.runstate.owners import AttemptRef, Dependencies, Mint, Reference, Version, new_run


def test_a_reference_reads_back_exactly_from_its_three_parts() -> None:
    reference = ParticipantRef("run1", 7)
    assert str(reference) == "run1/participant/7"
    assert Reference.parse(str(reference)) == reference
    assert isinstance(Reference.parse("run1/attempt/3"), AttemptRef)


@pytest.mark.parametrize("text", ["run1/participant/07", "run1/participant/x", "run1/nothing/1", "run1/participant", "a/b/participant/1"])
def test_a_string_that_is_not_exactly_a_reference_is_refused(text: str) -> None:
    with pytest.raises(ValueError):
        Reference.parse(text)


def test_references_of_different_kinds_never_compare_equal() -> None:
    assert ParticipantRef("run1", 1) != AttemptRef("run1", 1)
    with pytest.raises(TypeError):
        Reference("run1", 1)
    with pytest.raises(ValueError):
        ParticipantRef("run/1", 1)


def test_minting_never_reuses_a_serial_number_even_from_many_threads() -> None:
    mint = Mint(ParticipantRef, new_run())
    minted: list[ParticipantRef] = []
    lock = threading.Lock()

    def take() -> None:
        made = [mint() for _ in range(500)]
        with lock:
            minted.extend(made)

    threads = [threading.Thread(target=take) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(reference.serial for reference in minted) == list(range(1, 4001))
    assert Mint(ParticipantRef, "run1", after=4)().serial == 5, "a restored mint continues past the serial numbers it restored"


def test_a_version_is_what_the_owner_seam_compares() -> None:
    """The reference's serial is the incarnation, the bound revision the version lineage, the numeric version the ordinal (D19)."""
    version = Version("unet", ParticipantRef("run1", 3), 4, 12)
    assert isinstance(version, SeamVersion)
    assert (version.owner, version.incarnation, version.lineage, version.ordinal) == ("unet", 3, "4", 12)
    replaced = Version("unet", ParticipantRef("run1", 3), 5, 13)
    assert (replaced.incarnation, replaced.lineage) != (version.incarnation, version.lineage), (
        "versions from either side of a bound-state replacement are incomparable"
    )


@dataclass(frozen=True)
class Held:
    revision: int
    held: dict

    def revision_of(self, key: str) -> int | None:
        return self.held.get(key)


def test_a_dependency_on_a_whole_section_goes_stale_on_any_change() -> None:
    before = {"authority": Held(1, {"a": 1, "b": 1})}
    after = {"authority": Held(2, {"a": 1, "b": 2})}
    whole = Dependencies.on(before)
    assert whole.stale(before) == ()
    assert whole.stale(after) == ("authority",), "an unrelated change makes the conservative dependency stale"


def test_an_exact_dependency_stays_fresh_while_its_revisions_hold() -> None:
    exact = Dependencies(exact=(("authority", "a", 1),))
    assert exact.stale({"authority": Held(2, {"a": 1, "b": 2})}) == (), "an unrelated change leaves it fresh"
    assert exact.stale({"authority": Held(3, {"a": 2})}) == ("authority a",)
    assert exact.stale({}) == ("authority a",), "a section that is gone holds nothing"
