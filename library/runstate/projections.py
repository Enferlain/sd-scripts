"""Projections: what each kind of consumer receives instead of the authority, a section or a generation (D19).

No projection returns a participant by its address, and none can change a
record. Each kind of consumer gets its own type.
"""

from library.runstate.authority.authority import RunAuthority


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


__all__ = ["Execution"]
