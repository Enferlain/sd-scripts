"""Projections: what each kind of consumer receives instead of the authority, a section or a generation (D19).

No projection returns a participant by its address, and none offers a
method that changes a record. Each kind of consumer gets its own type. The
persistence, capability and observability projections come with the
second pass of section 15.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from library.runstate.authority.authority import RunAuthority
from library.runstate.authority.records import Participant, ParticipantRef
from library.runstate.owners import Dependencies


class Execution:
    """What the coordinator receives for participants: the owner seam, realized by the run authority (D16).

    It offers the authority's resources, current versions and leases, taken
    at boundaries from the current generation, and none of the methods that
    change records. Python cannot hide the authority behind it; the scope is
    the interface's. Its methods are the authority's own, so a lease costs no
    extra call.
    """

    __slots__ = ("current", "lease", "resources")

    def __init__(self, authority: RunAuthority):
        self.resources = authority.resources
        self.current = authority.current
        self.lease = authority.lease


@dataclass(frozen=True)
class Preparation:
    """What a preparation attempt and the unit resolution inside it receive: the participants it requested, pinned to one generation.

    Each record carries the participant's reference, authored address,
    declaration, binding with its bound object, and current routes. The
    dependencies name the authority section the records came from. The
    attempt's result publishes only while the generation it was derived from
    is current, so any publication since, by any owner, makes it stale: the
    conservative default, stricter than these dependencies alone.
    """

    generation: int
    participants: Mapping[ParticipantRef, Participant]
    dependencies: Dependencies

    def __getitem__(self, reference: ParticipantRef) -> Participant:
        found = self.participants.get(reference)
        if found is None:
            raise KeyError(f"{reference} is not a participant this attempt requested")
        return found


__all__ = ["Execution", "Preparation"]
