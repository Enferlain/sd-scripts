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
counts as returned and advanced. Whether a step a backend skips may report
that it changed nothing is settled with advancement outcomes in that second
pass.

A gradient norm that is not finite fails the run (D20). The host never waits
for it: each window end copies its norm to host memory and a later window
end reads it once it has arrived, before stepping; the run's end settles the
last ones (:meth:`LiveUnit.settle`). The step of the window whose norm was
not finite has been taken by then.
"""

import math
from collections import deque
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

    ``norm`` is the unit's gradient norm, which clipping measures and the
    finiteness check needs without clipping, left on the device: the record
    never waits for the device.
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


class NonFiniteGradient(RuntimeError):
    """A unit's gradient norm was not finite; the run fails rather than train on (D20)."""


@dataclass(frozen=True)
class _Norm:
    """One window's gradient norm on its way to host memory: the host copy and, on a device, the event after the copy."""

    window: int
    host: torch.Tensor
    event: torch.Event | None

    def arrived(self) -> bool:
        return self.event is None or self.event.query()


class LiveUnit:
    """A unit's live state and the update interface its update lease hands the standard profile's calls.

    It never leaves the access that holds the lease (D16).
    """

    __slots__ = ("_norms", "_parameters", "_spare", "last", "position", "prepared", "progress")

    def __init__(self, prepared: PreparedUnit, progress: Progress | None = None):
        self.prepared = prepared
        self._parameters = prepared.parameters
        self.position = 0  # contributions in the open window
        self.progress = progress or Progress()
        self.last: Advancement | None = None
        self._norms: deque[_Norm] = deque()  # gradient norms on their way to the host, oldest first
        self._spare: list[torch.Tensor] = []  # pinned host buffers whose norm has been read

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
        reasons = self._read(wait=False)
        if reasons:
            raise NonFiniteGradient("\n".join(reasons))
        limit = prepared.definition.max_grad_norm
        if limit is not None:
            norm = torch.nn.utils.clip_grad_norm_(self._parameters, limit)
        else:
            norm = torch.nn.utils.get_total_norm([p.grad for p in self._parameters if p.grad is not None])
        self._send(self.progress.attempted + 1, norm)
        prepared.optimizer.step()
        prepared.optimizer.zero_grad(set_to_none=True)
        self.progress = Progress(self.progress.attempted + 1, self.progress.advanced + 1)
        record = Advancement(prepared.reference, self.position, limit is not None, norm, Outcome.RETURNED, True, self.progress)
        self.position = 0
        self.last = record
        return record

    def settle(self) -> tuple[str, ...]:
        """Once the run's work has ended: the windows whose gradient norm was not finite, waiting for the norms still on their way."""
        return self._read(wait=True)

    def adopt(self, previous: LiveUnit) -> None:
        """Takes over the norms the live state this one replaces still has on their way, and its host buffers."""
        self._norms.extend(previous._norms)
        previous._norms.clear()
        self._spare.extend(previous._spare)
        previous._spare.clear()

    def _send(self, window: int, norm: torch.Tensor) -> None:
        """Starts the norm's copy to host memory, after the work that computes it, without waiting for it."""
        value = norm.detach().to(torch.float32)
        if value.device.type == "cpu":
            self._norms.append(_Norm(window, value, None))
            return
        host = self._spare.pop() if self._spare else torch.empty((), dtype=torch.float32, pin_memory=True)
        host.copy_(value, non_blocking=True)  # on the current stream of the norm's device
        event = torch.Event(device=value.device)
        event.record(torch.accelerator.current_stream(value.device))
        self._norms.append(_Norm(window, host, event))

    def _read(self, *, wait: bool) -> tuple[str, ...]:
        """Reads the norms that have arrived, or all of them when waiting; why any was not finite."""
        reasons: list[str] = []
        while self._norms and (wait or self._norms[0].arrived()):
            norm = self._norms.popleft()
            if norm.event is not None:
                norm.event.synchronize()
            value = float(norm.host)
            if norm.event is not None:
                self._spare.append(norm.host)
            if not math.isfinite(value):
                reasons.append(
                    f"@{self.prepared.definition.name}: the gradient norm of window {norm.window} was {value}; "
                    "its step was taken, so the unit's parameters may no longer be finite"
                )
        return tuple(reasons)


__all__ = ["TRAINER", "Advancement", "LiveUnit", "NonFiniteGradient", "Outcome", "PreparedUnit", "Progress"]
