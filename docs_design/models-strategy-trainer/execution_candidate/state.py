"""Candidate state owners and a CPU-only current-use/publication protocol.

The installed image is a coherent aggregate, not a second binding authority.
Only its binding section owns participant state. Units, policies and input
providers keep their own mutable state. No distributed guarantees are implied.
"""

from __future__ import annotations

import asyncio
import copy
from collections import deque
from collections.abc import AsyncGenerator, Hashable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from types import MappingProxyType

import torch
from torch import nn

from .access import participant_access, remaining_uses, retained_uses
from .graph import Accepted, NotReady, OutOfBounds, Retain, Unit, Use, View
from .optimizer import EagerSGD
from .progress import Progress


@dataclass(frozen=True)
class Stamp:
    participant_ref: str
    binding: int
    numerical_epoch: int


@dataclass(frozen=True)
class Binding:
    ref: str
    revision: int
    model: nn.Module


@dataclass(frozen=True)
class Product:
    key: Hashable
    value: object
    provenance: tuple[Stamp, ...] = ()


@dataclass(frozen=True)
class Contribution:
    invocation: int
    units: tuple[str, ...]
    prepared_keys: tuple[object, ...]
    region: object | None = field(default=None, repr=False)
    origins: tuple[int, ...] = ()
    destination_stamps: tuple[tuple[str, Stamp], ...] = ()


@dataclass(frozen=True)
class Replacement:
    participant: str
    expected_binding: int
    model: nn.Module


@dataclass
class UnitOutcome:
    unit_ref: str
    invocation: int
    contribution: str = "not_attempted"
    optimizer: str = "not_attempted"
    reset: str = "not_attempted"
    advancement_invocation: int | None = None
    region: object | None = field(default=None, repr=False)
    source_stamps: tuple[Stamp, ...] = ()


@dataclass
class _PendingWork:
    """Live owner evidence, independent of bounded diagnostic history.

    Entry exists before numerical or granted work starts. A verified handback
    makes it ready; advancement consumes that readiness before any fallible
    mechanics. Partial joint completion retains every unit's evidence until
    the complete selected advancement has returned.
    """

    units: tuple[str, ...]
    prepared_keys: tuple[object, ...]
    region: object | None
    destination_stamps: tuple[tuple[str, Stamp], ...]
    records: dict[int, UnitOutcome] = field(default_factory=dict)
    ready: Contribution | None = None
    advancing: bool = False


@dataclass
class UnitRuntime:
    definition: Unit
    ref: str
    parameters: tuple[nn.Parameter, ...]
    optimizer: EagerSGD
    steps: int = 0
    outcomes: deque[UnitOutcome] = field(default_factory=lambda: deque(maxlen=32))
    prepared_key: object = field(default_factory=object, repr=False)
    _work: _PendingWork | None = field(default=None, repr=False)

    @property
    def unfinished(self) -> bool:
        return self._work is not None

    @property
    def window(self) -> Contribution | None:
        """Read-only diagnostic projection, not independent readiness state."""
        return self._work.ready if self._work is not None else None

    @property
    def pending(self) -> int | None:
        return self.window.invocation if self.window is not None else None

    @property
    def active_outcomes(self) -> Mapping[int, UnitOutcome]:
        """Diagnostic snapshot; editing a report cannot change live evidence."""
        return MappingProxyType({key: copy.copy(record) for key, record in self._work.records.items()} if self._work is not None else {})

    def check_entry(self, previous: Contribution | None) -> None:
        if self._work is None:
            if previous is None:
                return
        elif previous is not None and self.accepts(previous):
            return
        raise NotReady("An existing contribution window needs an explicit verified continuation")

    def begin(
        self,
        invocation: int,
        units: tuple[str, ...],
        prepared_keys: tuple[object, ...],
        region: object | None,
        destination_stamps: tuple[tuple[str, Stamp], ...],
        source_stamps: tuple[Stamp, ...],
        *,
        previous: Contribution | None = None,
        granted: bool = False,
    ) -> None:
        self.check_entry(previous)
        if self._work is None:
            self._work = _PendingWork(units, prepared_keys, region, destination_stamps)
        if invocation in self._work.records:
            raise NotReady("Contribution invocation is already active")
        record = UnitOutcome(
            self.ref,
            invocation,
            contribution="unverified_grant" if granted else "uncertain",
            region=region,
            source_stamps=source_stamps,
        )
        self._work.records[invocation] = record
        self.outcomes.append(record)

    def accepts(self, contribution: Contribution) -> bool:
        return (
            self._work is not None
            and not self._work.advancing
            and self._work.ready is contribution
            and tuple(self._work.records) == contribution.origins
            and all(parameter.grad is not None for parameter in self.parameters)
        )

    def handback(self, contribution: Contribution, *, granted: bool) -> None:
        assert self._work is not None
        self._work.records[contribution.invocation].contribution = "granted_ready" if granted else "ready"
        self._work.ready = contribution

    def claim_advancement(self) -> None:
        assert self._work is not None and self._work.ready is not None and not self._work.advancing
        self._work.advancing = True

    def start_advancement(self, invocation: int) -> tuple[UnitOutcome, ...]:
        assert self._work is not None and self._work.advancing
        records = tuple(self._work.records.values())
        for record in records:
            record.advancement_invocation = invocation
        return records

    def advancement_returned(self) -> None:
        assert self._work is not None and self._work.advancing
        self._work.ready = None

    def finish(self) -> None:
        assert self._work is not None and self._work.advancing and self._work.ready is None
        assert all(record.optimizer == record.reset == "returned" for record in self._work.records.values())
        self._work = None

    def evidence(self, invocation: int, region: object | None) -> tuple[UnitOutcome, ...]:
        active = self._work.records.values() if self._work is not None else ()
        records = {record.invocation: record for record in (*self.outcomes, *active)}
        return tuple(
            copy.copy(record)
            for record in records.values()
            if record.invocation == invocation or (region is not None and record.region is region)
        )


@dataclass(frozen=True)
class Image:
    generation: int
    bindings: Mapping[str, Binding]
    units: Mapping[str, UnitRuntime]
    views: Mapping[str, PreparedView]


@dataclass(frozen=True)
class Projection:
    models: Mapping[str, nn.Module]
    stamps: Mapping[str, Stamp]


def _cut_view_output(value):
    """Cut tensor paths in supported result shapes, never mutate the inputs.

    Opaque results and tensor-valued keys cannot certify a cut in this target.
    Detaching is NOT a storage-isolation or arbitrary-Python sandbox guarantee.
    """
    if isinstance(value, torch.Tensor):
        return value.detach()
    scalars = (bool, int, float, complex, str, bytes)
    if type(value) is tuple:
        return tuple(_cut_view_output(item) for item in value)
    if type(value) is list:
        return [_cut_view_output(item) for item in value]
    if type(value) is dict:
        if any(key is not None and type(key) not in scalars for key in value):
            raise NotReady("No-gradient view has unsupported output keys")
        return {key: _cut_view_output(item) for key, item in value.items()}
    if value is None or type(value) in scalars:
        return value
    raise NotReady(f"No-gradient view has unsupported output {type(value).__name__}")


@dataclass(frozen=True)
class PreparedView:
    """Synchronous CPU view attachment; mode changes cannot span an await.

    Both roles retain the same underlying participant. This is NOT evidence
    that live mutable module modes are safe under threads or async backends.
    """

    definition: View
    model: nn.Module

    def __call__(self, *args, **kwargs):
        modes = tuple((module, module.training) for module in self.model.modules()) if self.definition.evaluation else ()
        result, primary = None, None
        try:
            if self.definition.evaluation:
                self.model.eval()
            with torch.set_grad_enabled(self.definition.gradients):
                result = self.model(*args, **kwargs)
                if not self.definition.gradients:
                    result = _cut_view_output(result)
        except BaseException as error:
            primary = error
        cleanup = []
        for module, training in modes:
            try:
                module.training = training
            except BaseException as error:
                cleanup.append(error)
        if cleanup:
            raise BaseExceptionGroup("View execution/mode restoration failed", ((primary,) if primary is not None else ()) + tuple(cleanup))
        if primary is not None:
            raise primary
        return result


@dataclass
class RetainedState:
    """A source version and its lifetime/correspondence evidence, not a new model identity."""

    projection: Projection
    views: Mapping[str, PreparedView]
    parameters: Mapping[str, tuple[nn.Parameter, ...]]
    prepared_keys: Mapping[str, object]
    isolated: bool
    region: object
    lease: AbstractAsyncContextManager[Projection] | None
    uses: tuple[Use, ...]
    token: object = field(default_factory=object, repr=False)
    released: bool = False
    closing: bool = False


def _footprint(model: nn.Module) -> set[tuple[str, int]]:
    """Cold-path evidence for this target's disjoint-participant restriction.

    Shared storage across distinct participants needs relationship-aware
    protection/invalidation, which this target has not implemented. Views of
    ONE participant use its binding and protection. Do not mistake separate authored
    participant addresses or Python Parameter objects for independent storage.
    Evidence covers registered modules, parameters and buffers, using shared
    objects or matching storage base addresses. It does not inspect private
    Python attributes, arbitrary external memory overlap or the whole heap.
    These process-local keys are evidence, never semantic identities.
    """
    keys = {("module", id(module)) for module in model.modules()}
    parameter_storage: dict[int, int] = {}
    for tensor in (*model.parameters(), *model.buffers()):
        keys.add(("tensor", id(tensor)))
        try:
            pointer = tensor.untyped_storage().data_ptr() if tensor.numel() else None
        except (NotImplementedError, RuntimeError, ValueError) as error:
            raise NotReady("This target cannot establish a concrete storage footprint") from error
        if pointer is not None:
            keys.add(("storage", pointer))
            if isinstance(tensor, nn.Parameter) and parameter_storage.setdefault(pointer, id(tensor)) != id(tensor):
                raise NotReady("Distinct Parameter views sharing storage require an unimplemented alias mapping")
    return keys


def _registered_tensors(model: nn.Module) -> dict[tuple[str, str], torch.Tensor]:
    return {
        (kind, path): tensor
        for kind, members in (
            ("parameter", model.named_parameters(remove_duplicate=False)),
            ("buffer", model.named_buffers(remove_duplicate=False)),
        )
        for path, tensor in members
    }


def _verify_registered_copy(source: nn.Module, copied: nn.Module) -> None:
    """Cold-path registered topology/modes, tensor values and path-to-alias evidence."""
    source_aliases: dict[int, tuple[str, str]] = {}
    copied_aliases: dict[int, tuple[str, str]] = {}
    source_modules = dict(source.named_modules(remove_duplicate=False))
    copied_modules = dict(copied.named_modules(remove_duplicate=False))
    if source_modules.keys() != copied_modules.keys():
        raise NotReady("An isolated version must preserve its registered module state")
    for name, original in source_modules.items():
        version = copied_modules[name]
        path = ("module", name)
        if source_aliases.setdefault(id(original), path) != copied_aliases.setdefault(id(version), path):
            raise NotReady("Retained version lost accepted alias/member correspondence")
        if type(original) is not type(version) or original.training != version.training:
            raise NotReady(f"An isolated version must preserve its registered module state: {name}")
    before, after = _registered_tensors(source), _registered_tensors(copied)
    if before.keys() != after.keys():
        raise NotReady("An isolated version must preserve its registered source state")
    for path, original in before.items():
        version = after[path]
        if source_aliases.setdefault(id(original), path) != copied_aliases.setdefault(id(version), path):
            raise NotReady("Retained version lost accepted alias/member correspondence")
        if (
            original.shape != version.shape
            or original.dtype != version.dtype
            or original.device != version.device
            or original.layout != version.layout
            or original.requires_grad != version.requires_grad
            or not torch.equal(original, version)
        ):
            raise NotReady(f"An isolated version must preserve its registered source state: {path}")


@dataclass(frozen=True)
class Capture:
    """An in-memory, finished-run CPU cut, NOT durable exact restoration."""

    authority: str
    generation: int
    participant_refs: Mapping[str, str]
    unit_refs: Mapping[str, str]
    source_stamps: Mapping[str, Stamp]
    weights: Mapping[str, object]
    owner_states: Mapping[str, object]
    optimization: Mapping[str, object]
    coordinates: Mapping[str, object]


class Feed:
    """Addressed, bounded handoff owned by input coordination.

    Work identity/provenance is a selected payload contract. No image, caption,
    epoch or global-step fields are built into the mechanism.
    """

    def __init__(self, capacity: int, changed: asyncio.Condition, *, progress: Progress | None = None, name: str = ""):
        self.capacity = capacity
        self.changed = changed
        self.progress, self.name = progress, name
        self.ready: dict[Hashable, Product] = {}
        self.closed = False
        self.failure: BaseException | None = None
        self.handed_off: deque[tuple[Hashable, int, tuple[Stamp, ...]]] = deque(maxlen=32)
        self.discarded: deque[tuple[Hashable, tuple[Stamp, ...]]] = deque(maxlen=32)
        self.high_water = 0

    async def publish(self, product: Product) -> None:
        async with self.changed:
            if product.key in self.ready:
                raise OutOfBounds(f"Duplicate ready work {product.key!r}")

            def available():
                return len(self.ready) < self.capacity or self.closed

            progress = self.progress
            if not available() and progress is not None:
                with progress.waiting(
                    "publication:" + self.name, "any", lambda: None if available() else progress.claimants(self.name, self.ready)
                ):
                    await self.changed.wait_for(available)
            else:
                await self.changed.wait_for(available)
            if self.closed:
                raise NotReady("Publication after producer shutdown")
            # Recheck after waiting: another worker may publish the same key.
            if product.key in self.ready:
                raise OutOfBounds(f"Duplicate ready work {product.key!r}")
            self.ready[product.key] = product
            self.high_water = max(self.high_water, len(self.ready))
            self.changed.notify_all()
            if self.progress is not None:
                # Publication can make an earlier unknown demand concrete.
                # Ready contents remain actual evidence if this check fails.
                self.progress.check()

    async def close(self, failure: BaseException | None = None) -> None:
        async with self.changed:
            if failure is not None:
                self.failure = failure
            self.closed = True
            self.changed.notify_all()

    async def end_demand(self) -> None:
        """This target's explicitly accepted ready-work discard policy."""
        async with self.changed:
            self.closed = True
            self.discarded.extend((key, product.provenance) for key, product in self.ready.items())
            self.ready.clear()
            self.changed.notify_all()


def _condition(feeds: tuple[Feed, ...]) -> asyncio.Condition:
    if not feeds or any(feed.changed is not feeds[0].changed for feed in feeds):
        raise NotReady("Correlated feeds must share one handoff condition")
    return feeds[0].changed


def _capacity_conflict(feeds: tuple[Feed, ...], key: Hashable) -> bool:
    return any(key not in feed.ready and len(feed.ready) >= feed.capacity and not feed.closed for feed in feeds)


def _available(feeds: tuple[Feed, ...], key: Hashable) -> bool:
    return (
        all(key in feed.ready for feed in feeds)
        or any(feed.failure is not None or (feed.closed and key not in feed.ready) for feed in feeds)
        or _capacity_conflict(feeds, key)
    )


def _check_available(feeds: tuple[Feed, ...], key: Hashable) -> None:
    if any(feed.failure is not None for feed in feeds):
        raise NotReady("An input producer failed")
    if _capacity_conflict(feeds, key):
        # A concrete fail-closed reordering policy, NOT a proof of global
        # deadlock. Another policy might service other demands or reserve slots.
        # This target neither evicts products nor guesses that schedule.
        raise NotReady("Bounded reorder capacity cannot serve the current input demand")
    if any(key not in feed.ready for feed in feeds):
        raise NotReady(f"Missing requested work {key!r}")


async def wait_for_products(feeds: tuple[Feed, ...], key: Hashable) -> None:
    """Readiness wait with no participant/state lease and no input claim."""
    condition = _condition(feeds)
    async with condition:
        await condition.wait_for(lambda: _available(feeds, key))
        _check_available(feeds, key)


async def join(
    feeds: tuple[Feed, ...],
    key: Hashable,
    invocation: int,
    admit,
    projection: Projection,
    *,
    wait: bool = True,
) -> tuple[object, ...]:
    """Check EVERY member, then claim all together without an intervening await.

    A failed second member does not silently consume the first. This atomic
    handoff claim is local input coordination, not an optimizer transaction.
    """
    condition = _condition(feeds)
    async with condition:
        if wait:
            await condition.wait_for(lambda: _available(feeds, key))
        elif not any(feed.failure is not None for feed in feeds) and any(key not in feed.ready for feed in feeds):
            # Readiness was observed before leasing. If another consumer claimed
            # it meanwhile, fail without waiting under a conflicting lease.
            raise NotReady("Input claim lost readiness before protected admission")
        _check_available(feeds, key)
        products = tuple(feed.ready[key] for feed in feeds)
        for product in products:
            if product.key != key or not admit(product, projection):
                raise NotReady(f"Inadmissible work {key!r}; actual provenance retained")
        for feed, product in zip(feeds, products, strict=True):
            del feed.ready[key]
            feed.handed_off.append((key, invocation, product.provenance))
        condition.notify_all()
        return tuple(product.value for product in products)


class RunState:
    """One installed image and separate owner-held mutable sections."""

    def __init__(self, accepted: Accepted):
        self.accepted = accepted
        self.image: Image | None = None
        self.owners = {name: spec.initialize() for name, spec in accepted.states.items()}
        self.numerical_epochs = dict.fromkeys(accepted.participants, 0)
        self.coordinates: dict[str, dict[str, int]] = {}
        self.unusable: set[str] = set()
        self.unusable_states: set[str] = set()
        self.condition = asyncio.Condition()
        self.progress = Progress()
        self.active_producers: dict[str, tuple[str, ...]] = {}
        self.active_retentions: dict[object, RetainedState] = {}
        self.running = False
        self.completed = False
        self.invocation = 0
        self.trace: deque[tuple[str, object]] = deque(maxlen=256)
        self.handoffs = asyncio.Condition()
        self.feeds = {
            name: Feed(spec.capacity, self.handoffs, progress=self.progress, name=name) for name, spec in accepted.channels.items()
        }

    @property
    def readers(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for lease in self.progress.leases.values():
            for (kind, name), mode in lease.access.items():
                if kind == "participant" and mode == "read":
                    result[name] = result.get(name, 0) + 1
        return result

    @property
    def writers(self) -> set[str]:
        return {
            name
            for lease in self.progress.leases.values()
            for (kind, name), mode in lease.access.items()
            if kind == "participant" and mode == "write"
        }

    @property
    def state_locks(self) -> set[str]:
        return {name for lease in self.progress.leases.values() for kind, name in lease.access if kind == "state"}

    @property
    def generation(self) -> int:
        return self.image.generation if self.image is not None else 0

    def candidate(self, models: Mapping[str, nn.Module], *, replacing: str | None = None) -> Image:
        """Fallible CPU preparation, including final member correspondence.

        Known aliases within one unit are consolidated. Competing units reject.
        Reuse of unaffected units has explicit member-only dependency evidence
        for this plain CPU backend; it is not assumed for backend composites.
        """
        if set(models) != set(self.accepted.participants):
            raise NotReady("Incomplete participant preparation")
        bindings: dict[str, Binding] = {}
        for name, declaration in self.accepted.participants.items():
            model = models[name]
            declaration.verify(model)
            if any(tensor.device.type != "cpu" for tensor in (*model.parameters(), *model.buffers())):
                raise NotReady("Candidate backend supports synchronous CPU tensors only")
            old = self.image.bindings[name] if self.image is not None else None
            revision = old.revision + int(name == replacing) if old else 1
            bindings[name] = Binding(self.accepted.participant_refs[name], revision, model)

        physical_owners: dict[tuple[str, int], str] = {}
        current_keys = (
            set().union(*(_footprint(binding.model) for binding in self.image.bindings.values())) if self.image is not None else set()
        )
        for name, binding in bindings.items():
            keys = _footprint(binding.model)
            if name == replacing and keys & current_keys:
                raise NotReady("Replacement-only preparation needs a fresh realization, not borrowed current objects/storage")
            for key in keys:
                if physical_owners.setdefault(key, name) != name:
                    raise NotReady("Shared participant objects/storage need unimplemented relationship-aware protection")

        units: dict[str, UnitRuntime] = {}
        ownership: dict[int, str] = {}
        selected: set[int] = set()
        for name, definition in self.accepted.units.items():
            parameters: dict[int, nn.Parameter] = {}
            for participant, path in definition.members:
                try:
                    parameter = bindings[participant].model.get_parameter(path)
                except AttributeError as error:
                    raise NotReady(f"{name}: unresolved accepted member {participant}/{path}") from error
                owner = ownership.setdefault(id(parameter), name)
                if owner != name:
                    raise NotReady(f"Shared physical member claimed by {owner} and {name}")
                parameters[id(parameter)] = parameter
                selected.add(id(parameter))
            old_unit = self.image.units[name] if self.image is not None else None
            affected = old_unit is None or any(participant == replacing for participant, _ in definition.members)
            if affected:
                optimizer = EagerSGD(parameters.values(), lr=definition.learning_rate)
                # Resetting a physical optimizer is not resetting the stable
                # unit's progress or erasing previously observed outcomes.
                units[name] = UnitRuntime(
                    definition,
                    self.accepted.unit_refs[name],
                    tuple(parameters.values()),
                    optimizer,
                    steps=old_unit.steps if old_unit is not None else 0,
                    outcomes=deque((copy.copy(record) for record in old_unit.outcomes), maxlen=32)
                    if old_unit is not None
                    else deque(maxlen=32),
                )
            else:
                if len(old_unit.parameters) != len(parameters) or any(
                    a is not b for a, b in zip(old_unit.parameters, parameters.values(), strict=True)
                ):
                    raise NotReady("Unaffected unit lost final member correspondence")
                units[name] = old_unit

        # Only new/unpublished realizations may be configured here.
        for name, binding in bindings.items():
            if self.image is None or name == replacing:
                for parameter in binding.model.parameters():
                    parameter.requires_grad_(id(parameter) in selected)
        views = {name: PreparedView(definition, bindings[definition.participant].model) for name, definition in self.accepted.views.items()}
        return Image(self.generation + 1, MappingProxyType(bindings), MappingProxyType(units), MappingProxyType(views))

    def install(self, candidate: Image, expected: int) -> None:
        if expected != self.generation or candidate.generation != expected + 1:
            raise NotReady("Stale preparation attempt")
        if self.readers or self.writers or self.state_locks:
            raise NotReady("Publication conflicts with protected current use")
        if self.active_retentions:
            raise NotReady("Publication needs a quiescent retained/contribution boundary: retained state is still live")
        self.require_optimization_idle("Publication needs a quiescent retained/contribution boundary: optimization work is unfinished")
        # Validation is over. No external callbacks or owner initialization here.
        self.image = candidate

    def require_optimization_idle(self, message: str) -> None:
        if self.image is not None and any(unit.unfinished for unit in self.image.units.values()):
            raise NotReady(message)

    @asynccontextmanager
    async def acquire(self, uses: tuple[Use, ...], states: tuple[str, ...] = ()) -> AsyncGenerator[Projection]:
        access = participant_access(uses)
        requested = {("participant", name): mode for name, mode in access.items()} | {("state", name): "write" for name in states}
        async with self.condition:

            def unavailable():
                return self.image is None or bool(set(access) & self.unusable or set(states) & self.unusable_states)

            def free():
                return not self.progress.blockers(requested)

            # A failed close may still hold physical protection. Such state is
            # not something a waiter can make usable by waiting for that lease.
            if not unavailable() and not free():
                description = "access:" + ",".join(f"{kind}/{name}:{mode}" for (kind, name), mode in requested.items())
                with self.progress.waiting(description, "all", lambda: None if unavailable() else self.progress.blockers(requested)):
                    await self.condition.wait_for(lambda: unavailable() or free())
            if unavailable():
                raise NotReady("Required current state is unavailable")
            assert self.image is not None
            token = self.progress.claim(requested)
            projection = Projection(
                MappingProxyType({name: self.image.bindings[name].model for name in access}),
                MappingProxyType({name: self.stamp(name) for name in access}),
            )
        try:
            yield projection
        finally:
            async with self.condition:
                del self.progress.leases[token]
                self.condition.notify_all()

    def _region_sources(self, region: object | None) -> tuple[RetainedState, ...]:
        if region is None:
            return ()
        return tuple(retained for retained in self.active_retentions.values() if retained.region is region)

    @asynccontextmanager
    async def use(
        self,
        uses: tuple[Use, ...],
        states: tuple[str, ...] = (),
        *,
        region: object | None = None,
        publication: bool = False,
    ) -> AsyncGenerator[Projection]:
        """Supply current state, borrowing only this region's protected reads.

        The caller keeps this scope open through selected execution and physical
        completion. Borrowing never releases or extends the retained lifetime;
        only its owner can end that lifetime. Isolated copies cover no current use.
        """
        sources = self._region_sources(region)
        if publication and sources:
            raise NotReady("A sequence cannot wait on its own protected old-state last use")
        live = tuple(retained for retained in sources if not retained.isolated)
        held = participant_access(tuple(use for retained in live for use in retained.uses))
        remaining = remaining_uses(uses, held, reuse=True)
        names = {use.participant for use in uses}
        covered = {}
        for retained in live:
            selected = names & retained.projection.models.keys()
            if selected:
                self.check_retained(retained, region)
                covered.update({name: (retained.projection.models[name], retained.projection.stamps[name]) for name in selected})
        async with self.acquire(remaining, states) as projection:
            # The acquisition may have waited. Recheck borrowed state before
            # supplying it, just as acquire checks its newly reserved members.
            for retained in live:
                if names & retained.projection.models.keys():
                    self.check_retained(retained, region)
            yield (
                Projection(
                    MappingProxyType({**{name: value[0] for name, value in covered.items()}, **projection.models}),
                    MappingProxyType({**{name: value[1] for name, value in covered.items()}, **projection.stamps}),
                )
                if covered
                else projection
            )

    @asynccontextmanager
    async def use_source(self, allowed: tuple[Use, ...], uses: tuple[Use, ...] | None = None) -> AsyncGenerator[Projection]:
        """A provider's bounded authority and returned effects share its lease."""
        uses = allowed if uses is None else uses
        authority = participant_access(allowed)
        if any(
            use.participant not in authority
            or use.view is not None
            or use.access not in ("read", "write")
            or (use.access == "write" and authority[use.participant] != "write")
            for use in uses
        ):
            raise OutOfBounds("Provider source access exceeds accepted authority")
        async with self.use(uses) as projection:
            try:
                yield projection
            except BaseException:
                self.unusable.update(use.participant for use in uses if use.access == "write")
                raise
            else:
                self.returned_writes(uses)

    def stamp(self, participant: str) -> Stamp:
        if self.image is None:
            raise NotReady("No current preparation")
        binding = self.image.bindings[participant]
        return Stamp(binding.ref, binding.revision, self.numerical_epochs[participant])

    def returned_writes(self, uses: tuple[Use, ...]) -> None:
        # Returned declared effects, not proof of a numerical tensor change.
        for name in {use.participant for use in uses if use.access == "write"}:
            self.numerical_epochs[name] += 1

    async def retain(self, declaration: Retain, region: object, *, exclusive: tuple[str, ...] = ()) -> RetainedState:
        """Explicit copy policy within one awaited, sequential region lifetime.

        Concurrent direct retain/use/release calls for the same region are not
        supported; lowering never overlaps those ownership transitions.
        """
        uses = retained_uses(declaration, exclusive)
        held = participant_access(tuple(use for retained in self._region_sources(region) if not retained.isolated for use in retained.uses))
        remaining_uses(uses, held, reuse=False)
        lease = self.acquire(uses)
        projection = await lease.__aenter__()
        keep_lease = False
        try:
            assert self.image is not None
            runtimes = {name: self.image.units[name] for name in declaration.units}
            for runtime in runtimes.values():
                if runtime.unfinished:
                    if runtime.window is None:
                        raise NotReady("Retention needs a verified contribution window")
                    # A later microbatch may retain current numerical state
                    # while its region owns an unchanged accumulated window.
                    # A diagnostic token can also survive claimed advancement;
                    # the owner check, not token presence, decides readiness.
                    self.check_contribution(runtime.window, runtime.window.units, region)
            if declaration.isolated:
                models = {name: copy.deepcopy(model) for name, model in projection.models.items()}
                current_keys = set().union(*(_footprint(binding.model) for binding in self.image.bindings.values()))
                for name, model in models.items():
                    self.accepted.participants[name].verify(model)
                    keys = _footprint(model)
                    if keys & current_keys:
                        raise NotReady("An isolated version must not borrow current objects/storage")
                    current_keys.update(keys)
                    before, after = projection.models[name].state_dict(), model.state_dict()
                    if before.keys() != after.keys() or any(not torch.equal(before[key], after[key]) for key in before):
                        raise NotReady("An isolated version must preserve its declared source state")
                    _verify_registered_copy(projection.models[name], model)
                projection = Projection(MappingProxyType(models), projection.stamps)
            parameters = {}
            for name, runtime in runtimes.items():
                selected = tuple(
                    {
                        id(parameter): parameter
                        for participant, path in runtime.definition.members
                        for parameter in (projection.models[participant].get_parameter(path),)
                    }.values()
                )
                if len(selected) != len(runtime.parameters):
                    raise NotReady("Retained version lost accepted alias/member correspondence")
                parameters[name] = selected
            retained = RetainedState(
                projection,
                MappingProxyType(
                    {
                        name: PreparedView(view, projection.models[view.participant]) if declaration.isolated else self.image.views[name]
                        for name, view in self.accepted.views.items()
                        if view.participant in projection.models
                    }
                ),
                MappingProxyType(parameters),
                MappingProxyType({name: runtime.prepared_key for name, runtime in runtimes.items()}),
                declaration.isolated,
                region,
                None if declaration.isolated else lease,
                uses,
            )
            self.active_retentions[retained.token] = retained
            keep_lease = not declaration.isolated
            self.trace.append(("retained", (declaration.name, declaration.isolated, tuple(projection.stamps.values()))))
            return retained
        finally:
            if not keep_lease:
                await lease.__aexit__(None, None, None)

    def check_retained(self, retained: RetainedState, region: object | None) -> None:
        if (
            retained.released
            or retained.closing
            or self.active_retentions.get(retained.token) is not retained
            or retained.region is not region
        ):
            raise NotReady("Retained state is closing, released, foreign or outside its accepted lifetime")
        assert self.image is not None
        if any(self.image.units[name].prepared_key is not key for name, key in retained.prepared_keys.items()):
            raise NotReady("Retained derivative member correspondence is stale")
        if not retained.isolated and self.unusable & retained.projection.models.keys():
            raise NotReady("Required current state is unavailable")
        if not retained.isolated and any(self.stamp(name) != stamp for name, stamp in retained.projection.stamps.items()):
            raise NotReady("Protected old state changed before last use")

    async def release(self, retained: RetainedState, region: object, *, aborted: bool = False) -> None:
        if retained.region is not region or self.active_retentions.get(retained.token) is not retained:
            raise NotReady("Cannot release a foreign retained lifetime")
        if retained.closing:
            raise NotReady("Retained lifetime is already closing; an uncertain release cannot be retried")
        if not aborted:
            self.check_retained(retained, region)
        # End admission before any fallible/awaited physical release. Failure
        # cannot leave a registered token lending protection that may be gone.
        retained.closing = True
        try:
            if retained.lease is not None:
                await retained.lease.__aexit__(None, None, None)
        except BaseException:
            if not retained.isolated:
                self.unusable.update(retained.projection.models)
            async with self.condition:
                self.condition.notify_all()
            raise
        retained.released = True
        retained.closing = False
        del self.active_retentions[retained.token]
        self.trace.append(("released", (retained.isolated, tuple(retained.projection.stamps.values()))))

    async def release_region(self, region: object) -> None:
        """Drain this region's lifetimes without erasing failed-work evidence."""
        failures = []
        for retained in self._region_sources(region):
            try:
                await self.release(retained, region, aborted=True)
            except BaseException as error:
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("Retained lifetime cleanup failed", failures)

    @asynccontextmanager
    async def producer_owner(self, name: str, states: tuple[str, ...]) -> AsyncGenerator[Mapping[str, object]]:
        """Pin continuation identity for the actual activity AND its cleanup.

        These are activity-lifetime registrations, not participant leases held
        while waiting for work. Reset/migration needs quiescence in this target;
        merely looking up a fresh object would not fix private workers holding
        the former state. Preserve means the accepted owner can continue.
        """
        async with self.condition:
            if name in self.active_producers or set(states) & self.unusable_states or self.image is None:
                raise NotReady("Provider continuation state is unavailable or already active")
            self.active_producers[name] = states
            owners = MappingProxyType({key: self.owners[key] for key in states})
        try:
            yield owners
        finally:
            async with self.condition:
                del self.active_producers[name]
                self.condition.notify_all()

    def contribute(
        self,
        loss: torch.Tensor,
        names: tuple[str, ...],
        invocation: int,
        *,
        retained: RetainedState | None = None,
        region: object | None = None,
        keep_graph: bool = False,
        previous: Contribution | None = None,
    ) -> Contribution:
        assert self.image is not None
        runtimes = tuple(self.image.units[name] for name in names)
        if retained is not None:
            self.check_retained(retained, region)
            if set(names) - retained.parameters.keys():
                raise NotReady("No retained derivative destination correspondence")
        parameters = tuple(
            parameter
            for name, runtime in zip(names, runtimes, strict=True)
            for parameter in (retained.parameters[name] if retained is not None else runtime.parameters)
        )
        self.begin_contribution(
            names,
            invocation,
            region=region,
            previous=previous,
            source_stamps=tuple(retained.projection.stamps.values()) if retained is not None else (),
        )
        gradients = torch.autograd.grad(loss, parameters, retain_graph=keep_graph)
        accumulated = []
        for parameter, gradient in zip((parameter for runtime in runtimes for parameter in runtime.parameters), gradients, strict=True):
            value = gradient.detach().clone()
            if previous is not None:
                if parameter.grad is None:
                    raise NotReady("An accumulated gradient is missing")
                value = parameter.grad.detach() + value
            accumulated.append(value)
        offset = 0
        for runtime in runtimes:
            for parameter in runtime.parameters:
                # A custom autograd backward can return an alias of source
                # state. A cross-block gradient needs its own storage.
                parameter.grad = accumulated[offset]
                offset += 1
        return self.complete_contribution(names, invocation, region=region)

    def begin_contribution(
        self,
        names: tuple[str, ...],
        invocation: int,
        *,
        region: object | None = None,
        previous: Contribution | None = None,
        source_stamps: tuple[Stamp, ...] = (),
        granted: bool = False,
    ) -> None:
        """Register ownership before either ordinary or granted numerical work."""
        assert self.image is not None
        runtimes = tuple(self.image.units[name] for name in names)
        if previous is not None:
            if granted:
                raise NotReady("This target's granted work cannot extend an existing contribution window")
            self.check_contribution(previous, names, region)
            if region is None or invocation in previous.origins:
                raise NotReady("Accumulation requires a new block in the same execution region")
            stamps = previous.destination_stamps
        else:
            stamps = tuple(
                (name, self.stamp(name)) for name in dict.fromkeys(member[0] for unit in runtimes for member in unit.definition.members)
            )
        # Reject conflicts for the entire selected group before registering any
        # of it. Once entry succeeds, uncertain evidence survives failed work.
        # All entry paths keep the group's origins/regions in lockstep; handback
        # checks that correspondence against the exact entered destination group.
        for runtime in runtimes:
            runtime.check_entry(previous)
        prepared_keys = tuple(runtime.prepared_key for runtime in runtimes)
        for runtime in runtimes:
            runtime.begin(invocation, names, prepared_keys, region, stamps, source_stamps, previous=previous, granted=granted)

    def complete_contribution(
        self, names: tuple[str, ...], invocation: int, *, region: object | None = None, granted: bool = False
    ) -> Contribution:
        """Verify one handback and publish its token to every destination owner."""
        assert self.image is not None
        runtimes = tuple(self.image.units[name] for name in names)
        work = runtimes[0]._work
        if work is None or not work.records:
            raise NotReady("No entered contribution to hand back")
        origins = tuple(work.records)
        status = "unverified_grant" if granted else "uncertain"
        if origins[-1] != invocation or any(
            runtime._work is None
            or runtime._work.units != names
            or runtime._work.prepared_keys != tuple(unit.prepared_key for unit in runtimes)
            or runtime._work.region is not region
            or runtime._work.advancing
            or tuple(runtime._work.records) != origins
            or runtime._work.records[invocation].region is not region
            or runtime._work.records[invocation].contribution != status
            or any(parameter.grad is None for parameter in runtime.parameters)
            for runtime in runtimes
        ):
            raise NotReady("Incomplete or duplicate contribution handback")
        # A granted executor must have restored its temporary edits before this
        # call. Its accepted returned writes have already advanced conservative
        # epochs; ordinary backward retains the pre-execution dependency stamps.
        stamps = tuple((name, self.stamp(name)) for name, _ in work.destination_stamps) if granted else work.destination_stamps
        contribution = Contribution(invocation, names, work.prepared_keys, region, origins, stamps)
        for runtime in runtimes:
            runtime.handback(contribution, granted=granted)
        return contribution

    def check_contribution(self, contribution: Contribution, names: tuple[str, ...], region: object | None) -> None:
        assert self.image is not None
        runtimes = tuple(self.image.units[name] for name in names)
        if (
            not isinstance(contribution, Contribution)
            or not contribution.origins
            or contribution.region is not region
            or contribution.units != names
            or len(contribution.prepared_keys) != len(runtimes)
            or any(key is not unit.prepared_key for key, unit in zip(contribution.prepared_keys, runtimes, strict=True))
            or any(not unit.accepts(contribution) for unit in runtimes)
            or any(self.stamp(name) != stamp for name, stamp in contribution.destination_stamps)
        ):
            raise NotReady("Stale, misaddressed or unverified gradient handback")

    def advance(self, contribution: Contribution, names: tuple[str, ...], invocation: int, *, region: object | None = None) -> None:
        assert self.image is not None
        runtimes = tuple(self.image.units[name] for name in names)
        self.check_contribution(contribution, names, region)
        if region is None and contribution.invocation != invocation:
            raise NotReady("Stale, misaddressed or unverified gradient handback")
        subjects = {member[0] for runtime in runtimes for member in runtime.definition.members}
        if any(not retained.isolated and subjects & retained.projection.models.keys() for retained in self.active_retentions.values()):
            raise NotReady("Ready gradients do not permit an update before protected old-state last use")
        # Claim the entire addressed advancement before callbacks/backend work.
        # Neither a reentrant call nor an exception can reuse the old readiness.
        for runtime in runtimes:
            runtime.claim_advancement()
        for runtime in runtimes:
            records = runtime.start_advancement(invocation)
            if runtime.definition.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(runtime.parameters, runtime.definition.clip_norm)
            for record in records:
                record.optimizer = "uncertain"  # entering backend, not proof of a change
            runtime.optimizer.step()
            for record in records:
                record.optimizer = "returned"
            runtime.steps += 1
            for participant in {member[0] for member in runtime.definition.members}:
                self.numerical_epochs[participant] += 1  # conservative source-state epoch
            for record in records:
                record.reset = "uncertain"
            runtime.optimizer.zero_grad(set_to_none=True)
            for record in records:
                record.reset = "returned"
            runtime.advancement_returned()
        # Keep the full pending-window evidence through partial joint failure;
        # the bounded completed-history deque must not evict live origins.
        for runtime in runtimes:
            runtime.finish()

    async def replace(self, proposal: Replacement, participant: str) -> None:
        """Replacement-only path; other structural changes are not implemented."""
        async with self.condition:

            def closing():
                return any(retained.closing for retained in self.active_retentions.values())

            await self.condition.wait_for(lambda: closing() or (not self.readers and not self.writers and not self.state_locks))
            if closing():
                raise NotReady("Replacement with closing retained work is unsupported")
            assert self.image is not None
            if proposal.participant != participant or proposal.expected_binding != self.image.bindings[participant].revision:
                raise NotReady("Stale or misaddressed replacement proposal")
            if proposal.model is self.image.bindings[participant].model:
                raise NotReady("In-place preparation requires an unimplemented destructive protocol")
            self.require_optimization_idle("Replacement with unfinished gradients is unsupported")
            if self.active_retentions:
                raise NotReady("Replacement with unfinished retained derivative work is unsupported")
            affected_states = {name: spec for name, spec in self.accepted.states.items() if participant in spec.dependencies}
            if any(spec.on_replace == "reject" for spec in affected_states.values()):
                raise NotReady("No accepted owner-state continuity for replacement")
            resetting = {name for name, spec in affected_states.items() if spec.on_replace == "reset"}
            if any(resetting & set(names) for names in self.active_producers.values()):
                raise NotReady("Owner reset requires the active producer to finish cleanup first")
            resets = {name: spec.initialize() for name, spec in affected_states.items() if spec.on_replace == "reset"}
            models = {name: binding.model for name, binding in self.image.bindings.items()}
            models[participant] = proposal.model
            expected = self.generation
            candidate = self.candidate(models, replacing=participant)
            new_owners = dict(self.owners)
            new_owners.update(resets)
            self.install(candidate, expected)
            self.owners = new_owners
            self.unusable.discard(participant)
            self.unusable_states.difference_update(resets)
            self.trace.append(("replacement_published", self.stamp(participant)))

    def capture(self) -> Capture:
        """A narrow executable cut check, not a production snapshot protocol."""
        if (
            not self.completed
            or self.running
            or self.readers
            or self.writers
            or self.state_locks
            or self.active_producers
            or self.active_retentions
            or self.unusable
            or self.unusable_states
        ):
            raise NotReady("Capture needs a finished, usable, quiescent run")
        if self.image is None:
            raise NotReady("Capture cannot omit an unfinished contribution")
        self.require_optimization_idle("Capture cannot omit an unfinished contribution")
        if any(not feed.closed or feed.ready or feed.failure for feed in self.feeds.values()):
            raise NotReady("Capture cannot reconstruct unfinished provider/handoff work")
        return Capture(
            self.accepted.authority,
            self.generation,
            self.accepted.participant_refs,
            self.accepted.unit_refs,
            {name: self.stamp(name) for name in self.image.bindings},
            {name: copy.deepcopy(binding.model.state_dict()) for name, binding in self.image.bindings.items()},
            copy.deepcopy(self.owners),
            {name: {"state": copy.deepcopy(unit.optimizer.state_dict()), "steps": unit.steps} for name, unit in self.image.units.items()},
            copy.deepcopy(self.coordinates),
        )
