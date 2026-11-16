"""Trainer optimization: the owner of every optimization unit (D16, D19, D20).

It joins the workflow layer through the owner seam, as the run authority
does for participants. Each unit of the plan is a resource it holds, under a
:class:`UnitRef` it minted. At the seam a unit's version is its reference's
serial as incarnation, its definition revision as version lineage and its
count of updates as ordinal, and only an update lease is offered: the
standard profile's contribution and window end take one, and its handle is
the unit's :class:`LiveUnit`. A grant and a read of a unit's state, such as
its evaluation point, come with the second pass of section 16.

Its section of each generation holds the prepared units, built off to the
side by a preparation attempt and published with it (D8). It refuses a
publication that changes a prepared unit while a lease on that unit is out,
and judges the obligations on its units at a readiness checkpoint.
"""

import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

from library.iro import seams
from library.iro.operation_contracts import AccessMode
from library.iro.seams import Use
from library.iro.verify import Checkpoint, Evaluator, ObligationKind, Obligations
from library.optimization.construction import build
from library.runstate.authority import Unavailable
from library.runstate.generation import Generations, Publication, Section
from library.runstate.owners import AttemptRef, Mint, Refused, Version
from library.training.units.candidates import UnitCandidate
from library.training.units.plan import OptimizationPlan, UnitRef
from library.training.units.prepared import TRAINER, LiveUnit, PreparedUnit


@dataclass(frozen=True)
class OptimizationSection:
    """Trainer optimization's part of a generation: the prepared units by name, at one revision."""

    revision: int
    units: Mapping[str, PreparedUnit]

    @property
    def owner(self) -> str:
        return TRAINER

    def revision_of(self, key: str) -> int | None:
        """A unit's definition revision, by its name."""
        found = self.units.get(key)
        return found.definition.revision if found is not None else None


class UnitLease:
    """An update lease on one unit, granted for an access's body; its handle is the unit's live state."""

    __slots__ = ("_active", "_owner", "handle", "use", "version")

    def __init__(self, owner: TrainerOptimization, use: Use, version: Version, handle: LiveUnit):
        self._owner, self.use, self.version, self.handle = owner, use, version, handle
        self._active = True

    def release(self) -> None:
        self._owner._release(self)

    def updated(self) -> Version:
        """The runtime's notice that a call updated the unit through this lease: the unit's version advances (D16)."""
        return self._owner._advance(self)


class TrainerOptimization:
    """The owner of one run's optimization units."""

    def __init__(self, generations: Generations, plan: OptimizationPlan, obligations: Obligations):
        self.generations = generations
        self.plan = plan
        self.obligations = obligations
        mint = Mint(UnitRef, generations.run)
        self.references: Mapping[str, UnitRef] = MappingProxyType({name: mint() for name in plan.units})
        self._lock = threading.Lock()
        self._outstanding: dict[str, int] = {}
        self._writers: dict[str, UnitLease] = {}
        self._numeric: dict[str, int] = dict.fromkeys(plan.units, 0)
        self._live: dict[str, LiveUnit] = {}
        generations.register(TRAINER, self)

    @property
    def section(self) -> OptimizationSection | None:
        found = self.generations.current.sections.get(TRAINER)
        assert found is None or isinstance(found, OptimizationSection)
        return found

    def live(self, name: str) -> LiveUnit:
        """A unit's live state, for the host's reports; calls reach it only through a lease."""
        with self._lock:
            return self._live_for(self._prepared(name))

    def _live_for(self, prepared: PreparedUnit) -> LiveUnit:
        """The live state over the prepared unit now published; progress carries over to a newly prepared unit of the same reference."""
        name = prepared.definition.name
        live = self._live.get(name)
        if live is None or live.prepared is not prepared:
            carried = live.progress if live is not None and live.prepared.reference == prepared.reference else None
            live = self._live[name] = LiveUnit(prepared, carried)
        return live

    # Preparation.

    def prepare(self, candidates: Sequence[UnitCandidate], attempt: AttemptRef) -> OptimizationSection | tuple[str, ...]:
        """The next section, off to the side: each candidate's optimizer built on its final parameters, or why it cannot be.

        Nothing is published here; the preparation attempt publishes the
        section with the routes it prepared (D8).
        """
        reasons: list[str] = []
        units: dict[str, PreparedUnit] = {}
        for candidate in candidates:
            definition = candidate.definition
            policy = definition.optimizer
            try:
                optimizer = build(policy.name, candidate.parameters, dict(policy.settings))
            except Exception as failed:
                reasons.append(f"@{definition.name}: its optimizer @{policy.name} could not be built: {failed}")
                continue
            units[definition.name] = PreparedUnit(
                self.references[definition.name],
                definition,
                candidate.members,
                candidate.participants,
                candidate.modules,
                optimizer,
                candidate.dependencies,
                attempt,
            )
        missing = [name for name in self.plan.units if name not in units and not any(r.startswith(f"@{name}:") for r in reasons)]
        reasons += [f"@{name}: the attempt resolved no candidate for it" for name in missing]
        if reasons:
            return tuple(reasons)
        current = self.section
        return OptimizationSection(current.revision + 1 if current is not None else 1, MappingProxyType(units))

    # Readiness.

    def readiness(self, checkpoint: Checkpoint) -> tuple[str, ...]:
        """Why the obligations on the units at a readiness checkpoint do not hold now; empty once they all do."""
        reasons: list[str] = []
        names = {f"@{name}": name for name in self.plan.units}
        for obligation in self.obligations.at(checkpoint):
            if obligation.kind == ObligationKind.BOUND and obligation.evaluator == Evaluator.RESOURCE_OWNER:
                unit = names.get(obligation.names[0]) if len(obligation.names) == 1 else None
                if unit is not None and (reason := self._stale(unit)) is not None:
                    reasons.append(f"{reason} ({obligation.source or obligation.subject})")
            elif obligation.kind == ObligationKind.LEASE and obligation.use is not None and obligation.use.resource in self.plan.units:
                unit = obligation.use.resource
                reason = self._stale(unit)
                if reason is None and obligation.use.mode != AccessMode.OWNER_UPDATE:
                    reason = _unoffered(unit, obligation.use.mode)
                if reason is not None:
                    reasons.append(f"{reason} ({obligation.source or obligation.subject})")
        return tuple(reasons)

    def _stale(self, unit: str) -> str | None:
        section = self.section
        prepared = section.units.get(unit) if section is not None else None
        if prepared is None:
            return f"@{unit} has no prepared unit"
        changed = prepared.dependencies.stale(self.generations.current.sections)
        if changed:
            return f"@{unit} was prepared from bindings that have changed since ({', '.join(changed)})"
        return None

    # Publication checks.

    @contextmanager
    def admit(self, publication: Publication, current: Section | None, next: Section) -> Generator[None]:
        """Refuses a publication that changes a prepared unit while a lease on that unit is out."""
        del publication
        assert isinstance(next, OptimizationSection)
        assert current is None or isinstance(current, OptimizationSection)
        with self._lock:
            if current is not None:
                for name, prepared in current.units.items():
                    if next.units.get(name) is not prepared and self._outstanding.get(name):
                        raise Refused(f"@{name} has a lease out; a publication while the run executes goes through the runtime (D19)")
            yield

    # The owner seam.

    def resources(self) -> tuple[str, ...]:
        return tuple(self.plan.units)

    def current(self, resource: str) -> Version:
        prepared = self._prepared(resource)
        return Version(resource, prepared.reference, prepared.definition.revision, self._numeric[resource])

    def lease(self, uses: Sequence[Use], reserved: Mapping[str, seams.Version] | None = None) -> list[UnitLease]:
        """Update leases for the uses, taken together; if one cannot be taken, none is held."""
        del reserved
        granted: list[UnitLease] = []
        with self._lock:
            try:
                for use in uses:
                    granted.append(self._grant(use))
            except BaseException:
                for lease in granted:
                    self._release_held(lease)
                raise
        return granted

    def _prepared(self, resource: str) -> PreparedUnit:
        if resource not in self.plan.units:
            raise Unavailable(f"@{resource} is not a unit the accepted strategy declares")
        section = self.section
        prepared = section.units.get(resource) if section is not None else None
        if prepared is None:
            raise Unavailable(f"@{resource} has no prepared unit")
        return prepared

    def _grant(self, use: Use) -> UnitLease:
        prepared = self._prepared(use.resource)
        name = use.resource
        if use.mode != AccessMode.OWNER_UPDATE:
            raise Unavailable(_unoffered(name, use.mode))
        if prepared.dependencies.stale(self.generations.current.sections):
            raise Unavailable(f"@{name} was prepared from bindings that have changed since")
        if name in self._writers:
            raise Unavailable(f"@{name} already has an update lease out")
        version = Version(name, prepared.reference, prepared.definition.revision, self._numeric[name])
        lease = UnitLease(self, use, version, self._live_for(prepared))
        self._writers[name] = lease
        self._outstanding[name] = self._outstanding.get(name, 0) + 1
        return lease

    def _advance(self, lease: UnitLease) -> Version:
        with self._lock:
            name = lease.use.resource
            if not lease._active or self._writers.get(name) is not lease:
                raise Unavailable(f"@{name}: only its current update lease is told of updates")
            self._numeric[name] += 1
            previous = lease.version
            lease.version = Version(name, previous.reference, previous.bound, self._numeric[name])
            return lease.version

    def _release(self, lease: UnitLease) -> None:
        with self._lock:
            self._release_held(lease)

    def _release_held(self, lease: UnitLease) -> None:
        name = lease.use.resource
        if not lease._active:
            raise Unavailable(f"@{name}: a lease is released twice")
        lease._active = False
        remaining = self._outstanding[name] - 1
        if remaining:
            self._outstanding[name] = remaining
        else:
            del self._outstanding[name]
        if self._writers.get(name) is lease:
            del self._writers[name]


def _unoffered(unit: str, mode: AccessMode) -> str:
    if mode == AccessMode.GRANT:
        return f"@{unit} is used through a grant; grants come with the second pass of section 16"
    return f"@{unit}'s state is read through a {mode} use; reading a unit's state, such as its evaluation point, comes with the second pass of section 16"


__all__ = ["OptimizationSection", "TrainerOptimization", "UnitLease"]
