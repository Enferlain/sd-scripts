"""Candidate state owners and a CPU-only current-use/publication protocol.

The installed image is a coherent aggregate, not a second binding authority.
Only its binding section owns participant state. Units, policies and input
providers keep their own mutable state. No distributed guarantees are implied.
"""

from __future__ import annotations

import asyncio
import copy
from collections import deque
from collections.abc import AsyncIterator, Hashable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from types import MappingProxyType

import torch
from torch import nn

from .graph import Accepted, NotReady, OutOfBounds, Retain, Unit, Use, View


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
class UnitRuntime:
    definition: Unit
    ref: str
    parameters: tuple[nn.Parameter, ...]
    optimizer: torch.optim.Optimizer
    pending: int | None = None
    steps: int = 0
    outcomes: deque[UnitOutcome] = field(default_factory=lambda: deque(maxlen=32))
    prepared_key: object = field(default_factory=object, repr=False)

    def outcome(self, invocation: int) -> UnitOutcome:
        for record in reversed(self.outcomes):
            if record.invocation == invocation:
                return record
        record = UnitOutcome(self.ref, invocation)
        self.outcomes.append(record)
        return record


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
    token: object = field(default_factory=object, repr=False)
    released: bool = False


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

    def __init__(self, capacity: int, changed: asyncio.Condition):
        self.capacity = capacity
        self.changed = changed
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
            await self.changed.wait_for(lambda: len(self.ready) < self.capacity or self.closed)
            if self.closed:
                raise NotReady("Publication after producer shutdown")
            # Recheck after waiting: another worker may publish the same key.
            if product.key in self.ready:
                raise OutOfBounds(f"Duplicate ready work {product.key!r}")
            self.ready[product.key] = product
            self.high_water = max(self.high_water, len(self.ready))
            self.changed.notify_all()

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
        for feed, product in zip(feeds, products):
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
        self.readers: dict[str, int] = {}
        self.writers: set[str] = set()
        self.state_locks: set[str] = set()
        self.active_producers: dict[str, tuple[str, ...]] = {}
        self.active_retentions: dict[object, RetainedState] = {}
        self.running = False
        self.completed = False
        self.invocation = 0
        self.trace: deque[tuple[str, object]] = deque(maxlen=256)
        self.handoffs = asyncio.Condition()
        self.feeds = {name: Feed(spec.capacity, self.handoffs) for name, spec in accepted.channels.items()}

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
                optimizer = torch.optim.SGD(parameters.values(), lr=definition.learning_rate)
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
                if len(old_unit.parameters) != len(parameters) or any(a is not b for a, b in zip(old_unit.parameters, parameters.values())):
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
        if self.active_retentions or (self.image is not None and any(unit.pending is not None for unit in self.image.units.values())):
            raise NotReady("Publication needs a quiescent retained/contribution boundary in this target")
        # Validation is over. No external callbacks or owner initialization here.
        self.image = candidate

    @asynccontextmanager
    async def acquire(self, uses: tuple[Use, ...], states: tuple[str, ...] = ()) -> AsyncIterator[Projection]:
        access: dict[str, str] = {}
        for use in uses:
            if access.get(use.participant) != "write":
                access[use.participant] = use.access
        async with self.condition:

            def free():
                return not (set(states) & self.state_locks) and all(
                    name not in self.writers and (mode == "read" or not self.readers.get(name)) for name, mode in access.items()
                )

            await self.condition.wait_for(free)
            if self.image is None or set(access) & self.unusable or set(states) & self.unusable_states:
                raise NotReady("Required current state is unavailable")
            self.state_locks.update(states)
            for name, mode in access.items():
                if mode == "write":
                    self.writers.add(name)
                else:
                    self.readers[name] = self.readers.get(name, 0) + 1
            projection = Projection(
                MappingProxyType({name: self.image.bindings[name].model for name in access}),
                MappingProxyType({name: self.stamp(name) for name in access}),
            )
        try:
            yield projection
        finally:
            async with self.condition:
                for name, mode in access.items():
                    if mode == "write":
                        self.writers.remove(name)
                    else:
                        self.readers[name] -= 1
                        if not self.readers[name]:
                            del self.readers[name]
                self.state_locks.difference_update(states)
                self.condition.notify_all()

    def stamp(self, participant: str) -> Stamp:
        if self.image is None:
            raise NotReady("No current preparation")
        binding = self.image.bindings[participant]
        return Stamp(binding.ref, binding.revision, self.numerical_epochs[participant])

    def returned_writes(self, uses: tuple[Use, ...]) -> None:
        # Returned declared effects, not proof of a numerical tensor change.
        for name in {use.participant for use in uses if use.access == "write"}:
            self.numerical_epochs[name] += 1

    async def retain(self, declaration: Retain, region: object) -> RetainedState:
        """The copy policy is explicit; revision labels alone never isolate storage."""
        lease = self.acquire(tuple(Use(name) for name in declaration.participants))
        projection = await lease.__aenter__()
        keep_lease = False
        try:
            assert self.image is not None
            runtimes = {name: self.image.units[name] for name in declaration.units}
            if any(runtime.pending is not None for runtime in runtimes.values()):
                raise NotReady("Retention starts from quiescent selected contribution windows")
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
            )
            self.active_retentions[retained.token] = retained
            keep_lease = not declaration.isolated
            self.trace.append(("retained", (declaration.name, declaration.isolated, tuple(projection.stamps.values()))))
            return retained
        finally:
            if not keep_lease:
                await lease.__aexit__(None, None, None)

    def check_retained(self, retained: RetainedState, region: object | None) -> None:
        if retained.released or self.active_retentions.get(retained.token) is not retained or retained.region is not region:
            raise NotReady("Retained state is released, foreign or outside its accepted lifetime")
        assert self.image is not None
        if any(self.image.units[name].prepared_key is not key for name, key in retained.prepared_keys.items()):
            raise NotReady("Retained derivative member correspondence is stale")
        if not retained.isolated and any(self.stamp(name) != stamp for name, stamp in retained.projection.stamps.items()):
            raise NotReady("Protected old state changed before last use")

    async def release(self, retained: RetainedState, region: object, *, aborted: bool = False) -> None:
        if retained.region is not region or self.active_retentions.get(retained.token) is not retained:
            raise NotReady("Cannot release a foreign retained lifetime")
        if not aborted:
            self.check_retained(retained, region)
        if retained.lease is not None:
            await retained.lease.__aexit__(None, None, None)
        retained.released = True
        del self.active_retentions[retained.token]
        self.trace.append(("released", (retained.isolated, tuple(retained.projection.stamps.values()))))

    @asynccontextmanager
    async def producer_owner(self, name: str, states: tuple[str, ...]) -> AsyncIterator[Mapping[str, object]]:
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
    ) -> Contribution:
        assert self.image is not None
        runtimes = tuple(self.image.units[name] for name in names)
        if any(runtime.pending is not None for runtime in runtimes):
            raise NotReady("This probe profile requires a quiescent contribution window")
        if retained is not None:
            self.check_retained(retained, region)
            if set(names) - retained.parameters.keys():
                raise NotReady("No retained derivative destination correspondence")
        parameters = tuple(
            parameter
            for name, runtime in zip(names, runtimes)
            for parameter in (retained.parameters[name] if retained is not None else runtime.parameters)
        )
        gradients = torch.autograd.grad(loss, parameters, retain_graph=keep_graph)
        offset = 0
        for runtime in runtimes:
            record = runtime.outcome(invocation)
            record.region = region
            record.source_stamps = tuple(retained.projection.stamps.values()) if retained is not None else ()
            for parameter in runtime.parameters:
                # A custom autograd backward can return an alias of source
                # state. A cross-block gradient needs its own storage.
                parameter.grad = gradients[offset].detach().clone()
                offset += 1
            runtime.pending = invocation
            record.contribution = "ready"
        return Contribution(invocation, names, tuple(unit.prepared_key for unit in runtimes), region)

    def advance(self, contribution: Contribution, names: tuple[str, ...], invocation: int, *, region: object | None = None) -> None:
        assert self.image is not None
        runtimes = tuple(self.image.units[name] for name in names)
        if (
            (contribution.region is not region or (region is None and contribution.invocation != invocation))
            or contribution.units != names
            or len(contribution.prepared_keys) != len(runtimes)
            or any(key is not runtime.prepared_key for key, runtime in zip(contribution.prepared_keys, runtimes))
            or any(unit.pending != contribution.invocation for unit in runtimes)
        ):
            raise NotReady("Stale, misaddressed or unverified gradient handback")
        subjects = {member[0] for runtime in runtimes for member in runtime.definition.members}
        if any(not retained.isolated and subjects & retained.projection.models.keys() for retained in self.active_retentions.values()):
            raise NotReady("Ready gradients do not permit an update before protected old-state last use")
        for runtime in runtimes:
            record = runtime.outcome(contribution.invocation)
            record.advancement_invocation = invocation
            if runtime.definition.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(runtime.parameters, runtime.definition.clip_norm)
            record.optimizer = "uncertain"  # entering backend, not proof of a change
            runtime.optimizer.step()
            record.optimizer = "returned"
            runtime.steps += 1
            for participant in {member[0] for member in runtime.definition.members}:
                self.numerical_epochs[participant] += 1  # conservative source-state epoch
            record.reset = "uncertain"
            runtime.optimizer.zero_grad(set_to_none=True)
            record.reset = "returned"
            runtime.pending = None

    async def replace(self, proposal: Replacement, participant: str) -> None:
        """Replacement-only path; other structural changes are not implemented."""
        async with self.condition:
            await self.condition.wait_for(lambda: not self.readers and not self.writers and not self.state_locks)
            assert self.image is not None
            if proposal.participant != participant or proposal.expected_binding != self.image.bindings[participant].revision:
                raise NotReady("Stale or misaddressed replacement proposal")
            if proposal.model is self.image.bindings[participant].model:
                raise NotReady("In-place preparation requires an unimplemented destructive protocol")
            if any(unit.pending is not None for unit in self.image.units.values()):
                raise NotReady("Replacement with unfinished gradients is unsupported")
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
        if self.image is None or any(unit.pending is not None for unit in self.image.units.values()):
            raise NotReady("Capture cannot omit an unfinished contribution")
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
