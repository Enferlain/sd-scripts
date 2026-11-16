"""Prepared units, their live state and the advancement record (D20).

A :class:`PreparedUnit` is what a preparation result carries for one unit and
a generation publishes: its reference and definition revision, resolved
members, optimizer and dependencies. It never changes once published.

A unit's live state, its open window and progress, changes every step
without changing a record, as a participant's numeric version does. Only
Trainer optimization changes it, through the update interface its unit's
update lease hands the standard profile's calls (D17, D20):

- a **contribution** runs the backward pass from a loss into the unit's own
  parameters only (autograd's ``inputs``), so units that share a forward
  pass never accumulate into each other;
- the **end of a window**, under the standard profile, clips the unit's
  gradient norm to ``max_grad_norm``, steps the optimizer, zeroes the
  gradients and records the unit's progress in an :class:`Advancement`.

A window end with nothing contributed steps nothing: its optimizer call is
not attempted and progress stays. Synchronization, unscaling and schedulers
come with the second pass of section 16, and data parallelism with section
20. With no gradient scaler nothing skips a step, so every attempted step
counts as returned and advanced; a non-finite gradient norm is stepped as
the baseline steps it. Whether a step a backend skips may report that it
changed nothing is settled with advancement outcomes in that second pass.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import torch

from library.runstate.owners import AttemptRef, Dependencies
from library.training.units.candidates import Member
from library.training.units.plan import UnitDefinition, UnitRef

TRAINER = "trainer optimization"
"""Trainer optimization's name, as an owner of a section and as the performer of the standard mechanics."""


class Outcome(StrEnum):
    """What became of one mechanic in an advancement (D20, G3.1)."""

    RETURNED = "returned"
    SKIPPED = "skipped"
    UNCERTAIN = "uncertain"
    NOT_ATTEMPTED = "not attempted"


@dataclass(frozen=True)
class Progress:
    """A unit's own progress: optimizer calls attempted, and advancements the backend confirmed were not skipped, counted apart."""

    attempted: int = 0
    advanced: int = 0


@dataclass(frozen=True)
class Advancement:
    """One window's record: what it included, each mechanic reached with its outcome and performer, and the progress after.

    ``norm`` is the gradient norm clipping measured, left on the device: the
    record never waits for the device.
    """

    unit: UnitRef
    contributions: int
    clipped: bool
    norm: object = field(repr=False, compare=False)
    optimizer: Outcome
    zeroed: bool
    progress: Progress
    performer: str = TRAINER


@dataclass(frozen=True)
class PreparedUnit:
    """A unit as one generation publishes it (D20)."""

    reference: UnitRef
    definition: UnitDefinition
    members: tuple[Member, ...]
    participants: Mapping[str, torch.nn.Module] = field(repr=False, compare=False)
    modules: Mapping[str, tuple[str, ...]]
    optimizer: torch.optim.Optimizer = field(repr=False, compare=False)
    dependencies: Dependencies
    attempt: AttemptRef

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return [member.parameter for member in self.members]


class LiveUnit:
    """A unit's live state and the update interface its update lease hands the standard profile's calls.

    It never leaves the access that holds the lease (D16).
    """

    __slots__ = ("_parameters", "last", "position", "prepared", "progress")

    def __init__(self, prepared: PreparedUnit, progress: Progress | None = None):
        self.prepared = prepared
        self._parameters = prepared.parameters
        self.position = 0  # contributions in the open window
        self.progress = progress or Progress()
        self.last: Advancement | None = None

    def contribute(self, loss: torch.Tensor) -> None:
        """Backpropagates the loss into the unit's own parameters only, accumulating into its open window."""
        torch.autograd.backward(loss, inputs=self._parameters)
        self.position += 1

    def advance(self, participants: Sequence[object]) -> Advancement:
        """Ends the window under the standard profile: clip, step, zero and count progress.

        ``participants`` are what the members' update leases reach, in the
        order the window takes them, and must be exactly the participants the
        unit trains.
        """
        prepared = self.prepared
        expected = list(prepared.participants.values())
        if len(participants) != len(expected) or any(given is not wanted for given, wanted in zip(participants, expected, strict=False)):
            raise RuntimeError(f"@{prepared.definition.name} is advanced with leases on other participants than the ones it trains")
        if self.position == 0:  # nothing was contributed, so there is nothing to step
            record = Advancement(prepared.reference, 0, False, None, Outcome.NOT_ATTEMPTED, False, self.progress)
            self.last = record
            return record
        limit = prepared.definition.max_grad_norm
        norm = torch.nn.utils.clip_grad_norm_(self._parameters, limit) if limit is not None else None
        prepared.optimizer.step()
        prepared.optimizer.zero_grad(set_to_none=True)
        self.progress = Progress(self.progress.attempted + 1, self.progress.advanced + 1)
        record = Advancement(prepared.reference, self.position, limit is not None, norm, Outcome.RETURNED, True, self.progress)
        self.position = 0
        self.last = record
        return record


__all__ = ["TRAINER", "Advancement", "LiveUnit", "Outcome", "PreparedUnit", "Progress"]
