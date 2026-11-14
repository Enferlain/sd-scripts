"""The coordinator: one per process, executing a prepared run (D15, D16).

It realizes every lifecycle protocol of the accepted strategy as the
reference semantics defines it, and reports the same owner-visible facts, so
each realization is checked against the reference. It runs on one event
loop and never blocks on selected work: calls run on worker threads through
the implementation and device seams, a fused group as one submission, and
the coordinator waits for their completion without blocking.

It executes the run's lowered form (:mod:`library.iro.prepare.lowered`): a
body's values sit in numbered slots and each step is bound to its fixed data,
so a round dispatches on step kinds and never reads the strategy again. The
coordinator keeps runtime control state only; owners hold the state leases
reach. That state is split among components, each owning its part and
reaching only the components it uses:

- :mod:`~library.iro.runtime.kernel`: facts, failure and closing, waits and
  wakes, tasks and the handoff to workers;
- :mod:`~library.iro.runtime.turns`: device turns and admissions;
- :mod:`~library.iro.runtime.storage`: device storage work, snapshot copies
  and leaving the device;
- :mod:`~library.iro.runtime.protection`: leases, protection, retained
  results, owner updates and request reservations;
- :mod:`~library.iro.runtime.requests`: requests, handlers and stops;
- :mod:`~library.iro.runtime.flow`: streams, claims, producers and filters;
- :mod:`~library.iro.runtime.calls`: the call path.

The coordinator itself runs the strategy: invocations and their lifecycle,
bodies and their end checks, and the structural protocols.

Bookkeeping is bounded: an invocation's records live while it or its scope
is live, productions' observed versions while they produce, a participant's
pending device completions until later device work on it finds them done,
and facts go to the observer as they happen.

In checked mode, for conformance tests, every check that lowering narrows
also runs in full, and the run fails if the full check finds what lowering
excluded.
"""

import random
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from library.iro.operation_contracts import LifecycleProtocol
from library.iro.prepare import HOST, Prepared
from library.iro.prepare.lowered import (
    BORROW,
    DISPOSE,
    FORK,
    INVOKE,
    RELATIONSHIP,
    RELEASE,
    REQUEST,
    RESOURCE,
    STOP,
    STREAM,
    TAKE,
    UNREALIZED,
    Body,
    Calls,
    Invoke,
    Named,
    Step,
    lower,
)
from library.iro.runtime.calls import CallRunner
from library.iro.runtime.flow import Flow
from library.iro.runtime.kernel import Kernel, Observer
from library.iro.runtime.protection import Protections
from library.iro.runtime.requests import Requests
from library.iro.runtime.scopes import Records, scopes
from library.iro.runtime.state import TRANSITIONS, Cancelled, Identity, Outcome, Product, Protection, RunFailure, State, Ticket
from library.iro.runtime.storage import DeviceStorage
from library.iro.runtime.turns import Turns
from library.iro.seams import Device, Implementation, Owner


class Coordinator:
    """Executes one prepared run against its owners, implementations and devices."""

    def __init__(
        self,
        prepared: Prepared,
        owners: Mapping[str, Owner],
        installed: Mapping[str, Implementation],
        devices: Mapping[str, Device],
        *,
        observe: Observer | None = None,
        explore: random.Random | None = None,
        checked: bool = False,
    ):
        """``explore`` perturbs the schedule for conformance testing: at every
        wait and submission the coordinator yields to other activities a
        random number of times. It changes when things happen, never what may happen.

        ``checked`` runs every check lowering narrows in full beside the
        lowered one, and fails the run if the full check finds anything the
        lowered one excluded.
        """
        self.prepared = prepared
        self.explore = explore
        self.checked = checked
        self.lowered = lower(prepared, installed)
        self.owners, self.installed, self.devices = dict(owners), dict(installed), dict(devices)
        if HOST.name not in self.devices:
            raise ValueError("the coordinator needs the host device")
        self.kernel = kernel = Kernel(observe, explore, checked)
        protection: dict[str, Protection] = defaultdict(Protection)
        self.storage = DeviceStorage(kernel, prepared, self.owners, self.devices, protection)
        unknown = set(self.devices) - {facts.name for facts in prepared.target.devices}
        if unknown:
            raise ValueError(f"adapters for devices the run was not prepared for: {', '.join(sorted(unknown))}")
        for facts in prepared.target.devices:
            adapter = self.devices.get(facts.name)
            if adapter is None:
                raise ValueError(f"the run was prepared for {facts.name}, and no device adapter realizes it")
            if adapter.name != facts.name:
                raise ValueError(f"the adapter given for {facts.name} is {adapter.name}")
            if adapter.kind != facts.kind or not facts.offers <= adapter.offers:
                raise ValueError(
                    f"{facts.name}: the adapter is a {adapter.kind} device offering {sorted(adapter.offers)}, "
                    f"and the run was prepared for a {facts.kind} device offering {sorted(facts.offers)}"
                )
        # The invocations running, and the scope records the waits for a source to settle read.
        self.outcomes: dict[Identity, Outcome] = {}
        self._ordinals: dict[Identity, dict[str, int]] = defaultdict(dict)
        self._waited = self.lowered.waited
        self._records = Records()
        self._all_records = Records()  # checked mode: every invocation's records, to compare with the lowered ones
        self.turns = Turns(kernel)
        self.protections = Protections(kernel, self.owners, self.turns, self.storage, protection, frozenset(prepared.updated))
        self.requests = Requests(kernel, self.protections, self.outcomes, self._records, self._all_records, self._enter)
        self.flow = Flow(kernel, self.owners, self.protections, self.requests.stopped, self._enter, self._run_entry)
        self.calls = CallRunner(kernel, self.devices, self.protections, self.storage, self.turns, self.requests.request_of)
        self._statements: dict[str, Callable[[Any, tuple[object, ...], Identity], Awaitable[tuple[object, ...]]]] = {
            INVOKE: self._invoke,
            RESOURCE: self._resource,
            RELATIONSHIP: self._relationship,
            STREAM: self.flow.stream,
            RELEASE: self._release,
            TAKE: self.flow.take_step,
            BORROW: self.flow.borrow,
            DISPOSE: self.flow.dispose,
            FORK: self.flow.fork,
            REQUEST: self.requests.request,
            STOP: self.requests.stop,
            UNREALIZED: self._unrealized,
        }
        self._protocols: dict[LifecycleProtocol, Callable[[Invoke, tuple[object, ...], Identity], Awaitable[tuple[object, ...]]]] = {
            LifecycleProtocol.SEQUENCE: self._sequence,
            LifecycleProtocol.CHOICE: self._choice,
            LifecycleProtocol.PARALLEL: self._parallel,
            LifecycleProtocol.REPEAT: self._repeat,
            LifecycleProtocol.ACCESS: self._access,
            LifecycleProtocol.FILTER: self.flow.filter,
            LifecycleProtocol.PRODUCER: self.flow.producer,
            LifecycleProtocol.HANDLER: self.requests.handler,
        }

    @property
    def observe(self) -> Observer:
        return self.kernel.observe

    @observe.setter
    def observe(self, observe: Observer) -> None:
        self.kernel.observe = observe

    @property
    def errors(self) -> list[str]:
        return self.kernel.errors

    @property
    def cancelled(self) -> bool:
        return self.kernel.cancelled

    # Running.

    async def run(self) -> Outcome:
        """Runs the strategy to settlement and returns the root's outcome."""
        kernel = self.kernel
        with ThreadPoolExecutor(max_workers=self.prepared.target.workers, thread_name_prefix="iro-work") as executor:
            kernel.executor = executor
            name = self.lowered.path
            root = self._identity((), name)
            self._begin(root, name)
            outcome = self.outcomes[root]
            for state in (State.PENDING, State.ADMITTED, State.RUNNING):
                self._state(root, state)
            try:
                try:
                    await self._body(self.lowered.root, (), root)
                except RunFailure:
                    raise
                except Exception as error:
                    raise RunFailure(f"{name}: {error!r}") from error
                outcome.status = "succeeded"
            except RunFailure as failed:
                kernel.note_failure(failed)
                outcome.status = "cancelled" if kernel.cancelled or isinstance(failed, Cancelled) else "failed"
                outcome.error = str(failed)
            finally:
                self._state(root, State.CLOSING)
                self._state(root, State.SETTLING)
                await kernel.until(lambda: kernel.closing or not self.requests.pending_under(root), ("requests",))
                self.requests.settle(root)
                await self.storage.drain()
                if self.checked and kernel.failed and not kernel.cancelled and outcome.status != "failed":
                    # A miss closes admission without raising where it was found, so the root saw only that.
                    outcome.status, outcome.error = "failed", "; ".join(kernel.errors)
                self._state(root, State.TERMINAL)
        return outcome

    def cancel(self) -> None:
        """Run cancellation: admission closes while entered work settles."""
        self.kernel.cancel()

    # Invocations' lifecycle.

    def _identity(self, parent: Identity, owner: str) -> Identity:
        ordinals = self._ordinals[parent]
        ordinal = ordinals.get(owner, 0)
        ordinals[owner] = ordinal + 1
        return (*parent, (owner, ordinal))

    def _begin(self, identity: Identity, owner: str) -> None:
        self.outcomes[identity] = Outcome(owner, identity)
        commons = self._waited.get(owner)
        if commons is not None:
            self._records.begin(owner, scopes(identity, commons))
        if self.checked:
            self._all_records.begin(owner, scopes(identity))

    def _state(self, identity: Identity, state: State) -> None:
        outcome = self.outcomes[identity]
        if state not in TRANSITIONS[outcome.state]:
            raise RunFailure(f"{outcome.owner}: {outcome.state} cannot become {state}")
        outcome.state = state
        self.kernel.fact(outcome.owner, identity, "state", str(state))
        if state == State.TERMINAL:
            self.kernel.fact(outcome.owner, identity, "outcome", outcome.status, outcome.error)
            self._retire(identity, outcome.owner)

    def _retire(self, identity: Identity, owner: str) -> None:
        """Bookkeeping for a finished invocation, and for the scope it closed, is dropped (bounded bookkeeping)."""
        del self.outcomes[identity]
        commons = self._waited.get(owner)
        settled = commons is not None and self._records.retire(owner, identity, scopes(identity, commons))
        if self._ordinals:
            self._ordinals.pop(identity, None)
        self._records.close(identity)
        if self.checked:
            self._all_records.retire(owner, identity, scopes(identity))
            self._all_records.close(identity)
        self.requests.scope_closed(identity)
        if settled:
            self.kernel.changed(("settled", owner))

    # Bodies.

    async def _body(self, body: Body, args: tuple[object, ...] | list[object], identity: Identity) -> tuple[object, ...]:
        if len(args) != body.entries:
            raise RunFailure(f"{body.path}: a body receives {body.entries} entries, given {len(args)}")
        frame = [*args, *body.template]
        returned: tuple[object, ...] = ()
        failed: BaseException | None = None
        try:
            for step in body.steps:
                if isinstance(step, Calls):
                    await self.calls.run(step, frame, identity)
                    continue
                result = await self._statements[step.kind](step, tuple(map(frame.__getitem__, step.operands)), identity)
                if len(result) != len(step.results):
                    raise RunFailure(f"{step.path}: gave {len(result)} results for {len(step.results)}")
                for slot, value in zip(step.results, result, strict=True):
                    frame[slot] = value
            returned = tuple(map(frame.__getitem__, body.returns))
            return returned
        except BaseException as error:
            failed = error
            raise
        finally:
            if body.retained or body.claims or self.checked:
                self._end_checks(body, frame, returned, identity, failed)

    def _end_checks(
        self, body: Body, frame: list[object], returned: tuple[object, ...], identity: Identity, failed: BaseException | None
    ) -> None:
        """What a body leaves when it ends: retained results are released and claims settled, and leaving them fails it (D16)."""
        kernel = self.kernel
        given = {id(v) for v in returned}
        leaked: RunFailure | None = None
        for slot in reversed(body.retained):
            value = frame[slot]
            if isinstance(value, Product) and value.active and id(value) not in given:
                if failed is None and leaked is None:
                    leaked = RunFailure(f"{value.key[0]}: a retained result was never released")
                self.protections.release_product(value, "cancelled" if kernel.cancelled else "failed")
        for slot in body.claims:
            value = frame[slot]
            if (
                isinstance(value, Ticket)
                and value.owner == identity
                and value.disposition is None
                and not value.forks
                and id(value) not in given
            ):
                if failed is None and leaked is None:
                    leaked = RunFailure(f"{value.stream}/{value.key}: a claim was never settled")
                self.flow.settle(value, "cancelled" if kernel.cancelled or isinstance(failed, Cancelled) else "failed", identity)
        if self.checked:
            for slot, value in enumerate(frame):
                if isinstance(value, Product) and slot in body.results and slot not in body.retained:
                    kernel.missed(f"{body.path}: a retained result in a slot the end check skips")
                if isinstance(value, Ticket) and slot not in body.claims:
                    kernel.missed(f"{body.path}: a claim in a slot the end check skips")
        if leaked is not None:
            raise leaked

    # Statements the coordinator realizes itself; the rest belong to their components.

    async def _resource(self, step: Named, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        if step.word not in self.owners:
            raise RunFailure(f"{step.path}: no owner holds @{step.word}")
        return (step.word,)

    async def _relationship(self, step: Step, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        return ((args[0], args[1]),)

    async def _release(self, step: Step, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self.kernel.admission_point()
        product = args[0]
        if not isinstance(product, Product):
            raise RunFailure(f"{step.path}: releases something that is not a retained result")
        self.protections.release_product(product, "released")
        return ()

    async def _unrealized(self, step: Step, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self.kernel.admission_point()
        raise RunFailure(step.problem or f"{step.path}: the runtime does not realize it")

    # Invocations.

    async def _invoke(
        self, step: Invoke, args: tuple[object, ...], parent: Identity, *, body: list[object] | None = None
    ) -> tuple[object, ...]:
        kernel, requests = self.kernel, self.requests
        identity = self._identity(parent, step.path)
        self._begin(identity, step.path)
        outcome = self.outcomes[identity]
        self._state(identity, State.PENDING)
        request_of = requests.request_of
        if request_of and parent in request_of:
            request_of[identity] = request_of[parent]
        try:
            kernel.admission_point()
            self._state(identity, State.ADMITTED)
            self._state(identity, State.RUNNING)
            if step.captures_claims or self.checked:
                for ticket in (v for v in (body if body is not None else args) if isinstance(v, Ticket)):
                    if not step.captures_claims:
                        kernel.missed(f"{step.path}: a captured claim")
                    if ticket.owner != parent or ticket.disposition is not None:
                        raise RunFailure(f"{step.path}: a captured claim is not held by the capturing body")
                    ticket.owner = identity
            try:
                if body is not None:
                    result = await self._body(step.bodies[0], body, identity)
                else:
                    result = await self._protocol(step, args, identity)
            except RunFailure:
                raise
            except Exception as error:
                raise RunFailure(f"{step.path}: {error!r}") from error
            if step.owns_claims or self.checked:
                for value in result:
                    if isinstance(value, Ticket):
                        if not step.owns_claims:
                            kernel.missed(f"{step.path}: a claim among its results")
                        value.owner = parent
            outcome.status = "succeeded"
            return result
        except RunFailure as failed:
            outcome.status = "cancelled" if isinstance(failed, Cancelled) else "failed"
            outcome.error = str(failed)
            kernel.note_failure(failed)
            raise
        finally:
            self._state(identity, State.CLOSING)
            self._state(identity, State.SETTLING)
            if step.owns_claims or self.checked:
                self.flow.settle_owned(step, identity)
            if requests.emissions:
                for request_step, source, request_args in requests.emissions.pop(identity, ()):
                    await requests.emit(request_step, source, request_args, outcome.status)
            if request_of:
                request_of.pop(identity, None)
            self._state(identity, State.TERMINAL)

    def _entry(self, step: Invoke, args: tuple[object, ...], supplied: tuple[object, ...] = (), *, role: int = 0) -> list[object]:
        entry = step.entries[role]
        values: list[object] = [None] * entry.size
        for argument, operand in entry.captures:
            values[argument] = args[operand]
        if len(entry.supplied) != len(supplied):
            raise RunFailure(f"{step.path}: the protocol supplies {len(supplied)} entries, the body takes {len(entry.supplied)}")
        for position, value in zip(entry.supplied, supplied, strict=True):
            values[position] = value
        return values

    async def _enter(self, step: Invoke, args: tuple[object, ...], identity: Identity, supplied: tuple[object, ...]) -> tuple[object, ...]:
        """The step's body as a new invocation, with the protocol's supplied entries: a production, a request's handling."""
        return await self._invoke(step, (), identity, body=self._entry(step, args, supplied))

    async def _run_entry(
        self, step: Invoke, args: tuple[object, ...], identity: Identity, supplied: tuple[object, ...]
    ) -> tuple[object, ...]:
        """The step's body run in place, with the protocol's supplied entries: a filter's."""
        return await self._body(step.bodies[0], self._entry(step, args, supplied), identity)

    async def _protocol(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        if step.problem is not None:
            raise RunFailure(step.problem)
        return await self._protocols[step.protocol](step, args, identity)

    async def _sequence(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        return await self._body(step.bodies[0], self._entry(step, args), identity)

    async def _choice(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        chosen = args[0]
        if step.optional:
            if chosen is not None:
                return await self._body(step.bodies[0], self._entry(step, args, (chosen,)), identity)
            return await self._body(step.bodies[1], self._entry(step, args, role=1), identity)
        if type(chosen) is not bool:
            raise RunFailure(f"{step.path}: a choice chooses on a bool")
        role = 0 if chosen else 1
        return await self._body(step.bodies[role], self._entry(step, args, role=role), identity)

    async def _parallel(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        kernel, request_of = self.kernel, self.requests.request_of
        branches = []
        for index, body in enumerate(step.bodies):
            branch = (*identity, (step.branches[index], 0))
            if identity in request_of:
                request_of[branch] = request_of[identity]
            used = step.branch_claims[index]
            for position, value in enumerate(args):
                if isinstance(value, Ticket) and position in used:
                    value.owner = branch
            branches.append(kernel.spawn(self._branch(body, self._entry(step, args, role=index), branch)))
        results = await kernel.join(branches)
        errors = [r for r in results if isinstance(r, BaseException)]
        if kernel.cancelled:
            raise Cancelled("the run was cancelled")
        if errors:
            raise next((e for e in errors if not isinstance(e, Cancelled)), errors[0])
        return ()

    async def _branch(self, body: Body, args: list[object], branch: Identity) -> tuple[object, ...]:
        try:
            try:
                return await self._body(body, args, branch)
            except RunFailure:
                raise
            except Exception as error:
                raise RunFailure(f"{branch[-1][0]}: {error!r}") from error
        except RunFailure as failed:
            self.kernel.note_failure(failed)
            raise
        finally:
            self.requests.request_of.pop(branch, None)
            self._ordinals.pop(branch, None)

    async def _repeat(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        assert step.limit is not None  # lowering gave a repeat by policy its problem
        split = len(args) - step.carried
        captures, carries = args[:split], args[split:]
        for iteration in range(step.limit):
            carries = await self._invoke(step, (), identity, body=self._entry(step, (*captures, *carries), (iteration, *carries)))
        return carries

    async def _access(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        uses = step.uses
        device = step.device
        protections, turns = self.protections, self.turns
        # Leases and the device turn are taken together: nothing waits for one while holding the other (D14).
        await turns.queue(device, identity, lambda: all(protections.can_lease(name, use) for name, use in uses), step.keys)
        request_of = self.requests.request_of
        leases = protections.acquire(step, identity, request_of.get(identity) if request_of else None)
        if device is not None:
            turns.grant(device, identity, identity[:-1])
        try:
            storage = self.storage
            for lease in leases:
                if lease.capture is not None or lease.resource in storage.leaving:
                    await storage.ready(lease)
            return await self._body(step.bodies[0], self._entry(step, args, leases), identity)
        finally:
            protections.ended(leases)
            if device is not None:
                turns.work_ended(identity)


__all__ = ["Coordinator", "Observer"]
