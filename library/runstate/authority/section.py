"""The authority's section of a generation (D19): its records at one authority revision.

The specs call it the authority snapshot. It is immutable once built; the
authority builds the next one off to the side and publishes it whole.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from library.iro.verify import Obligations
from library.runstate.authority.records import Participant, ParticipantRef

AUTHORITY = "run authority"
"""The authority's name as an owner of a section."""


@dataclass(frozen=True)
class AuthoritySection:
    """The authority's records at one revision.

    ``addresses`` locates each declared address's current incarnation, for
    the authority's own use in transitions and at the owner seam, which
    names resources; consumers receive projections instead. A dependency
    on a participant is keyed by its reference's string, whose revision is
    its binding revision.
    """

    revision: int
    run: str
    strategy: str
    obligations: Obligations
    participants: Mapping[ParticipantRef, Participant]
    addresses: Mapping[str, ParticipantRef]
    _keys: Mapping[str, ParticipantRef] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_keys", MappingProxyType({str(reference): reference for reference in self.participants}))

    @property
    def owner(self) -> str:
        return AUTHORITY

    def revision_of(self, key: str) -> int | None:
        reference = self._keys.get(key)
        binding = self.participants[reference].binding if reference is not None else None
        return binding.revision if binding is not None else None

    def at(self, address: str) -> Participant:
        """The current incarnation declared at an address; the authority's own lookup."""
        reference = self.addresses.get(address)
        if reference is None:
            raise KeyError(f"@{address} is not a participant the accepted strategy declares")
        return self.participants[reference]


__all__ = ["AUTHORITY", "AuthoritySection"]
