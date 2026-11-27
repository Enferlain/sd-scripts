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

A window's gradient is the one a single batch of everything it accumulated
would give (D20). Each contribution's loss comes with its mass, what the
loss's own reduction divides by, as an object with ``value`` and ``mass``:
the loss declares it, and a loss that declares none is refused, never taken
as one sample's worth. The window divides the sum of loss times mass once,
by the total mass. A contribution of no mass contributes nothing, and a
window of no mass has nothing to optimize, so it steps nothing, wherever its
mass is held: a window whose masses are on the device reads its total on the
host at its end, a wait that only runs whose losses report device masses
pay, and a loss whose mass can be zero on the device keeps a finite value
and derivative there, as a sum divided by its mass clamped to one does. A
contribution scales its loss by its share of the total the window expects,
the previous window's, so a window like the one before it pays one scalar
multiply per contribution; a window whose total differs divides once at its
end, inside clipping's own multiply when clipping is on.

A window end with nothing contributed steps nothing: its optimizer call is
not attempted and progress stays. Synchronization, unscaling and schedulers
come with the second pass of section 16, and data parallelism with section
20. With no gradient scaler nothing skips a step, so every attempted step
counts as returned and advanced. Whether a step a backend skips may report
that it changed nothing is settled with advancement outcomes in that second
pass.

A window whose gradient norm, or loss where nothing measures the norm, is not
finite fails the run (D20). The check reads only what a window computes
anyway: the gradient norm where clipping measures it, and otherwise the
window's loss, which misses a gradient that is not finite behind a finite
loss. Each window end folds that value into a record
kept on the device, which keeps the first window whose value was not finite,
and the host reads it only where it reads the device already
(:meth:`LiveUnit.check`): when the run's work ends (:meth:`LiveUnit.settle`),
and later before a save and with the loss the run reports. The steps up to
that read have been taken by then.
"""

from collections import defaultdict
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

    ``norm`` is the unit's gradient norm where clipping measured it, and
    ``mass`` the window's total mass, both possibly on the device: the record
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
    mass: object = field(default=None, repr=False, compare=False)


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
    default_stream: bool = False  # the optimizer steps on its device's default stream (``steps_on_default_stream``)

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return [member.parameter for member in self.members]


class LiveUnit:
    """A unit's live state and the update interface its update lease hands the standard profile's calls.

    It never leaves the access that holds the lease (D16).
    """

    __slots__ = ("_bad", "_basis", "_expected", "_mass", "_parameters", "_window_loss", "last", "position", "prepared", "progress")

    def __init__(self, prepared: PreparedUnit, progress: Progress | None = None):
        self.prepared = prepared
        self._parameters = prepared.parameters
        self.position = 0  # contributions in the open window
        self.progress = progress or Progress()
        self.last: Advancement | None = None
        self._expected: float | None = None  # the previous window's total mass, which the next window expects
        self._basis: float | None = None  # the total this window's contributions were scaled by, once one had mass
        self._mass: float | torch.Tensor = 0.0  # the open window's total mass
        self._window_loss: torch.Tensor | None = None  # the open window's losses, read for the check without clipping
        self._bad: torch.Tensor | None = None  # the first window whose value was not finite, or -1, on the device

    def contribute(self, loss: object) -> None:
        """Backpropagates the loss, by its share of the window's mass, into the unit's own parameters only."""
        value, mass = _split(loss)
        if self.position == 0:
            self._basis, self._mass, self._window_loss = self._expected, 0.0, None
        if _number(mass) and mass == 0:  # a contribution of no mass contributes nothing, its value not even read
            self.position += 1
            return
        if self._basis is None:  # the first contribution with mass: a host mass is the basis; a device one cannot be read here
            self._basis = float(mass) if _number(mass) else 1.0
        share = mass / self._basis
        torch.autograd.backward(value if _number(share) and share == 1.0 else value * share, inputs=self._parameters)
        self.position += 1
        self._mass = self._mass + mass
        if self.prepared.definition.max_grad_norm is None:
            detached = value.detach().float()
            self._window_loss = detached if self._window_loss is None else self._window_loss + detached

    def advance(self, participants: Sequence[object]) -> Advancement:
        """Ends the window under the standard profile: normalize, clip, step, zero and count progress.

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
        if isinstance(self._mass, torch.Tensor):  # the host reads a device total, so a window of no mass steps nothing wherever its mass is
            self._mass = float(self._mass)
        if self._mass == 0:  # a window of no mass has nothing to optimize
            prepared.optimizer.zero_grad(set_to_none=True)
            record = Advancement(prepared.reference, self.position, False, None, Outcome.NOT_ATTEMPTED, True, self.progress, mass=0.0)
            self.position = 0
            self.last = record
            return record
        assert self._basis is not None, "a window with mass had a contribution with mass"
        factor = self._basis / self._mass  # what turns the contributions' sum into the window's gradient
        exact = _number(factor) and factor == 1.0
        limit = prepared.definition.max_grad_norm
        grads = [p.grad for p in self._parameters if p.grad is not None]
        norm: torch.Tensor | None = None
        if limit is not None and exact:
            norm = torch.nn.utils.clip_grad_norm_(self._parameters, limit)
        elif limit is not None:
            norm = torch.nn.utils.get_total_norm(grads) * factor
            _multiply(grads, torch.clamp(limit / (norm + 1e-6), max=1.0) * factor)  # clipping's own multiply carries the division
        elif not exact:
            _multiply(grads, factor)
        checked = norm if norm is not None else self._window_loss
        if checked is not None:
            self._fold(self.progress.attempted + 1, checked)
        _step(prepared)
        prepared.optimizer.zero_grad(set_to_none=True)
        self.progress = Progress(self.progress.attempted + 1, self.progress.advanced + 1)
        record = Advancement(
            prepared.reference, self.position, limit is not None, norm, Outcome.RETURNED, True, self.progress, mass=self._mass
        )
        self._expected = float(self._mass)
        self.position = 0
        self.last = record
        return record

    def check(self) -> tuple[str, ...]:
        """Reads the record, waiting for the device: why the unit's windows were not finite, if any were.

        Called only where the host reads the device already.
        """
        if self._bad is None:
            return ()
        window = int(self._bad)
        if window < 0:
            return ()
        what = "gradient norm" if self.prepared.definition.max_grad_norm is not None else "loss"
        return (
            f"@{self.prepared.definition.name}: the {what} of window {window} was not finite; "
            "its step was taken, so the unit's parameters may no longer be finite",
        )

    def settle(self) -> tuple[str, ...]:
        """Once the run's work has ended: the windows whose value was not finite."""
        return self.check()

    def adopt(self, previous: LiveUnit) -> None:
        """Takes over the record of the live state this one replaces, and what its next window expects."""
        if previous._bad is not None:
            self._bad = previous._bad if self._bad is None else torch.where(previous._bad >= 0, previous._bad, self._bad)
        previous._bad = None
        self._expected = previous._expected

    def _fold(self, window: int, value: torch.Tensor) -> None:
        """Folds a window's value into the record on the device: a few small operations, no copy and no wait."""
        if self._bad is None:
            self._bad = torch.full((), -1, dtype=torch.int64, device=value.device)
        self._bad = torch.where(torch.isfinite(value) | (self._bad >= 0), self._bad, window)


def _step(prepared: PreparedUnit) -> None:
    """Steps the unit's optimizer, on the device's default stream where its optimizer needs it, ordered after and before the current stream's work."""
    if not prepared.default_stream or not torch.cuda.is_available():
        prepared.optimizer.step()
        return
    device = next((p.device for p in prepared.parameters if p.device.type == "cuda"), None)
    current = torch.cuda.current_stream(device) if device is not None else None
    default = torch.cuda.default_stream(device) if device is not None else None
    if current is None or default is None or current == default:
        prepared.optimizer.step()
        return
    default.wait_stream(current)
    try:
        with torch.cuda.stream(default):
            prepared.optimizer.step()
    finally:  # a step that raised may have queued work, which what follows on the current stream, freeing included, waits for
        current.wait_stream(default)


def _split(loss: object) -> tuple[torch.Tensor, float | torch.Tensor]:
    """A contribution's loss and its mass, from an object with ``value`` and ``mass`` (D20)."""
    mass = getattr(loss, "mass", None)
    if mass is None:
        raise TypeError(
            "a contribution's loss declares its mass, what its reduction divides by, as an object with value and mass; "
            "a bare loss is not taken as one sample's worth (D20)"
        )
    if isinstance(mass, torch.Tensor):
        if mass.numel() != 1:
            raise ValueError(f"a contribution's mass is one number, and this one has {mass.numel()} elements")
        mass = mass.detach()  # a mass divides; no derivative goes into it
    value = getattr(loss, "value", None)  # duck-typed: a loss reports its value and its mass
    assert isinstance(value, torch.Tensor), "a loss with a mass has a value"
    return value, mass


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _multiply(grads: list[torch.Tensor], factor: float | torch.Tensor) -> None:
    """Multiplies the gradients in place, one fused multiply per device."""
    by_device: dict[torch.device, list[torch.Tensor]] = defaultdict(list)
    for grad in grads:
        by_device[grad.device].append(grad)
    for device, group in by_device.items():
        torch._foreach_mul_(group, factor.to(device) if isinstance(factor, torch.Tensor) else factor)


__all__ = ["TRAINER", "Advancement", "LiveUnit", "Outcome", "PreparedUnit", "Progress"]
