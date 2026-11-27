"""The call path: a fused group is one submission to a worker thread on its device (D15, D16).

A group resolves its leases and retained results, waits for what it must
follow, and runs its calls in order inside the device's context on a worker
thread. Its results become slot values, owner updates and retained results.
Calls not declared safe for concurrent use run one at a time per
participant. A group waits until the snapshot copies already asked for of a
participant's device storage it reaches are queued (D15).

On the worker, inside the device's context, the group's call setups are
entered with its annotations and the handles its calls receive, and left
when the calls return (D16). A lease its owner attached a held setup to, as
a route's realization does, has the setup applied there too, under the
group's annotations, and the group uses that resource alone while it
applies it. The setup's hold covers the leases its calls use and binds while
one of them is in use by a running group or kept by a retained result:
groups needing the same state share it, and a group that needs another
waits only while it binds. A hold that no longer binds leaves its change in
place; one whose setup undoes its change has it undone first in the next
group that needs another state, or on its own once its last lease is
released, and at the run's end however the run ended.

Fusing calls keeps each call's admission (design: a lowering must not erase
a required opportunity for admission): before each call, the first one
included, the group's worker checks whether admission has closed, because a
sibling failed or the run was cancelled. A call that has not entered by then
does not run and gets no facts, the calls before it keep their outcomes, and
the group is cancelled, as the reference cancels a call at its entry.
"""

import asyncio
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

from library.iro.evidence import setup_digest
from library.iro.prepare.lowered import Call, Calls
from library.iro.runtime.kernel import Kernel
from library.iro.runtime.protection import Applied, Hold, Protections
from library.iro.runtime.state import Cancelled, Held, Identity, Product, Request, RunFailure
from library.iro.runtime.storage import DeviceStorage
from library.iro.runtime.turns import Turns
from library.iro.seams import Annotations, CallContext, CallSetup, Completion, Device, HeldSetup, Implementation


class SetupFailed(RuntimeError):
    """A held setup failed to set its state before the group's calls ran (D16)."""


@dataclass
class NotEntered:
    """A call's report when admission closed before it entered: it did not run, and it is neither entered nor failed."""

    failure: BaseException | None = None  # leaving the group's context failed afterwards, with no call left to fail


class CallRunner:
    """Runs the run's call groups, counting each call's ordinals."""

    def __init__(
        self,
        kernel: Kernel,
        devices: Mapping[str, Device],
        protections: Protections,
        storage: DeviceStorage,
        turns: Turns,
        request_of: Mapping[Identity, Request],
    ):
        self.kernel, self.devices = kernel, devices
        self.protections, self.storage, self.turns = protections, storage, turns
        self.request_of = request_of  # the request each invocation handles, which its calls' facts carry
        self.busy: set[str] = set()  # resources a group not declared safe for concurrent use is running on
        self.alone: set[str] = set()  # resources a group applying its lease's held setup is running on, alone
        self.running_on: Counter[str] = Counter()  # how many running groups use each resource
        self.call_counts: dict[str, int] = defaultdict(int)
        self._undos: list[asyncio.Task] = []
        protections.undo_hold = self.undo_at_end

    async def run(self, step: Calls, frame: list[object], identity: Identity) -> None:
        kernel = self.kernel
        kernel.admission_point()
        if step.problem is not None:
            raise RunFailure(step.problem)
        device = self.devices.get(step.group.device)
        if device is None:
            raise RunFailure(f"{step.path}: the coordinator has no device {step.group.device}")
        turns = self.turns
        own_admission = not step.host and turns.admission_for(identity, step.group.device) is None
        if own_admission:
            await turns.queue(step.group.device, identity)
            turns.grant(step.group.device, identity, identity)
        borrowed: list[Product] = []
        leased: list[Held] = []
        try:
            arguments: list[list[object]] = []
            program = []
            for call in step.calls:
                covering: list[tuple[CallSetup, object]] = []
                values = [frame[s] for s in call.slots] if not call.locals else [frame[s] if s >= 0 else None for s in call.slots]
                given = values
                if call.leases or call.retained:
                    given = list(values)
                    for position in call.leases:
                        held = values[position]
                        assert isinstance(held, Held)
                        if not held.active:
                            raise RunFailure(f"{call.path}: a lease is used after its release")
                        given[position] = held.handle
                        leased.append(held)
                        for (
                            setup
                        ) in held.call_setups:  # a lease's call setups cover this call alone, entered once per setup and handle (D16)
                            if not any(s is setup and h is held.handle for s, h in covering):
                                covering.append((setup, held.handle))
                    for position in call.retained:
                        product = values[position]
                        assert isinstance(product, Product)
                        if not product.active:
                            raise RunFailure(f"{call.path}: a retained result is used after its release")
                        product.borrowers += 1
                        borrowed.append(product)
                        given[position] = product.value
                if kernel.checked:
                    self._check_arguments(call, values)
                arguments.append(values)
                program.append((call.implementation, given, call.locals, len(call.results), covering))
            if len(leased) > 1:
                leased = list(dict.fromkeys(leased))
            if kernel.explore is not None:
                await kernel.jitter()
            storage = self.storage
            touching = storage.touching(leased, borrowed)
            resources = {lease.resource for lease in leased}
            busy = resources if not step.safe else set()
            annotations = step.group.annotations
            holding = [lease for lease in leased if lease.setup is not None]
            declared = {need.resource: need.key for need in step.group.holds}
            for lease in holding:
                assert lease.setup is not None
                needed = lease.setup.key(annotations)
                if lease.resource in declared and declared[lease.resource] != needed:
                    raise RunFailure(
                        f"{step.path}: @{lease.setup.name} needs ({', '.join(needed)}) on @{lease.resource}, and preparation was told "
                        f"({', '.join(declared[lease.resource])}); the domain that attaches the setup declares what it needs (D16)"
                    )
            # What realization state each resource needs, as its setups say; a group's key on a resource covers all its leases there.
            keys = {
                r: frozenset(
                    (lease.setup.name, setup_digest(lease.setup), lease.setup.key(annotations))
                    for lease in holding
                    if lease.resource == r and lease.setup
                )
                for r in {h.resource for h in holding}
            }
            alone = set(keys)
            held_off = any(self._held_off(r, keys.get(r)) for r in resources)
            if busy or alone or held_off or (resources & self.alone) or (touching and storage.capturing):
                capturing = storage.capturing
                await kernel.until(
                    lambda: (
                        kernel.closing
                        or not (
                            busy & self.busy
                            or alone & self.running_on.keys()
                            or resources & self.alone
                            or any(r in capturing for r in touching)
                            or any(self._held_off(r, keys.get(r)) for r in resources)
                        )
                    ),
                    *(("resource", r) for r in resources.union(touching)),
                )
                kernel.admission_point()
            handles = [lease.handle for lease in leased]
            protection = self.protections.by_resource
            joined: list[tuple[str, Hold]] = []
            self.busy |= busy
            self.alone |= alone
            self.running_on.update(resources)
            storage.calls_started(touching)
            try:
                undo, apply = self._take_holds(step.group.device, resources, keys, holding, joined)
                if undo:  # a displaced hold's change, undone on its own device before anything of this group runs (D16)
                    await self._undo_now(undo)
                after = [c for lease in leased for c in protection[lease.resource].pending]
                after += [p.completion for p in borrowed if p.completion is not None]
                held = [(a.setup, a.handle) for a in apply]
                reports, completions = await kernel.submit(
                    run_group,
                    device,
                    step.group.context,
                    after,
                    step.group.setups,
                    annotations,
                    handles,
                    held,
                    program,
                    step.wait,
                    kernel.closed.is_set,
                )
                if apply and reports and isinstance(reports[0], SetupFailed):  # what failed to be set is not taken as in place
                    for _, hold in joined:
                        hold.applied[:] = [a for a in hold.applied if a not in apply]
            finally:
                for resource, hold in joined:
                    hold.running -= 1
                    self.protections.end_if_done(resource, hold)
                self.busy -= busy
                self.alone -= alone
                self.running_on.subtract(resources)
                for resource in resources:
                    if not self.running_on[resource]:
                        del self.running_on[resource]
                storage.calls_ended(touching)
            if not step.wait:
                kernel.queued(*completions)  # the run's settlement waits for it (D15)
                # Later work on these participants follows each call's completion (D15).
                queued = [c for c in completions if c is not None]
                for lease in leased:
                    pending = protection[lease.resource].pending
                    pending[:] = [c for c in pending if not c.done()]
                    pending.extend(queued)
            self._results(step, frame, identity, arguments, reports, completions)
        finally:
            for product in borrowed:
                product.borrowers -= 1
            if own_admission:
                turns.work_ended(identity)
            if leased:
                kernel.changed(*(("resource", lease.resource) for lease in leased))

    # Held setups (D16).

    def _held_off(self, resource: str, need: frozenset | None) -> bool:
        """Whether work on the resource needing this state, or none, must wait for the hold in force: it binds, or its undo runs."""
        hold = self.protections.holds.get(resource)
        if hold is None:
            return False
        if hold.undoing:
            return True
        if need is not None and hold.key == need:
            return False
        if need is None and not hold.changes_storage:
            return False
        return self.protections.binding(hold)

    def _take_holds(
        self, device: str, resources: set[str], keys: dict[str, frozenset], holding: list[Held], joined: list[tuple[str, Hold]]
    ) -> tuple[list[Applied], list[Applied]]:
        """After the wait: the changes to undo first and the setups to apply; each hold the group joins goes into ``joined``.

        A hold the group needs another state of, or no state while it holds a
        storage change, no longer binds: it ends, its change undone first
        where its setup undoes it. A setup is applied only to a handle no
        setup of its name and code has been applied to under the hold.
        """
        holds = self.protections.holds
        undo: list[Applied] = []
        apply: list[Applied] = []
        for lease in holding:
            if lease.setup is not None and lease.setup.changes_storage and not lease.setup.undoes:
                raise RunFailure(
                    f"@{lease.setup.name} changes @{lease.resource}'s storage and does not undo it; a storage change is undone (D16)"
                )
        for resource in sorted(resources):
            hold, need = holds.get(resource), keys.get(resource)
            if hold is not None and ((need is None and hold.changes_storage) or (need is not None and hold.key != need)):
                undo.extend(a for a in hold.applied if a.setup.undoes)
                del holds[resource]
                hold = None
            if need is None:
                continue
            if hold is None:
                hold = holds[resource] = Hold(need)
            hold.running += 1
            joined.append((resource, hold))
            for lease in holding:
                if lease.resource != resource or lease.setup is None:
                    continue
                hold.leases.add(lease.token)
                hold.routes.add(lease.lease.use.route)
                setup = lease.setup
                code = (setup.name, setup_digest(setup))
                if not any((a.setup.name, setup_digest(a.setup)) == code and a.handle is lease.handle for a in hold.applied):
                    applied = Applied(resource, setup, lease.handle, device, lease.token)
                    hold.applied.append(applied)
                    apply.append(applied)
        return undo, apply

    async def _undo_now(self, undo: list[Applied]) -> None:
        """Undoes changes on the devices they were made on, each after its resource's queued work; a failure fails the run (D16)."""
        protection = self.protections.by_resource
        by_place: dict[tuple[str, str], list[Applied]] = defaultdict(list)
        for applied in undo:
            by_place[applied.resource, applied.device].append(applied)
        for (resource, device), items in by_place.items():
            try:
                completion = await self.kernel.submit(
                    run_undo, self.devices[device], list(protection[resource].pending), [(a.setup, a.handle) for a in items]
                )
            except Exception as failed:
                raise RunFailure(f"@{resource}: undoing a held setup failed, so its state is uncertain: {failed}") from failed
            if completion is not None:
                protection[resource].pending.append(completion)
                self.kernel.queued(completion)

    def undo_at_end(self, resource: str, hold: Hold) -> None:
        """Starts a hold's undo once its last lease is released, or a lease it excludes is wanted, off the coordinator (D16)."""
        hold.undoing = True
        self._undos.append(self.kernel.spawn(self._undo(resource, hold)))

    async def _undo(self, resource: str, hold: Hold) -> None:
        try:
            await self._undo_now([a for a in hold.applied if a.setup.undoes])
        except RunFailure as failed:
            self.kernel.note_failure(failed)
        finally:
            if self.protections.holds.get(resource) is hold:
                del self.protections.holds[resource]
            self.kernel.changed(("resource", resource))

    async def settle_holds(self) -> list[str]:
        """At the run's end, however it ended: undo every change still in place, and wait for the undos under way (D16)."""
        failures: list[str] = []
        for task in list(self._undos):
            await asyncio.gather(task, return_exceptions=True)
        for resource, hold in list(self.protections.holds.items()):
            if hold.undoes and not hold.undoing:
                try:
                    await self._undo_now([a for a in hold.applied if a.setup.undoes])
                except RunFailure as failed:
                    failures.append(f"at the run's end, {failed}")
            del self.protections.holds[resource]
        return failures

    def _results(
        self,
        step: Calls,
        frame: list[object],
        identity: Identity,
        arguments: list[list[object]],
        reports: list[Any],
        completions: list[Completion | None],
    ) -> None:
        """Each call's report becomes its facts, owner updates, retained results and slot values, in order."""
        kernel = self.kernel
        request_of = self.request_of
        made: list[tuple[object, ...]] = []
        for index, call in enumerate(step.calls):
            if index >= len(reports):
                break
            report = reports[index]
            if isinstance(report, NotEntered):  # admission closed before it entered: no facts, and the calls before it keep theirs
                if report.failure is not None:
                    raise RunFailure(f"{step.path}: leaving the group's context failed: {report.failure}") from report.failure
                kernel.admission_point()
                raise Cancelled("admission closed before the call entered")  # unreached: the guard saw admission closed
            ordinal = self.call_counts[call.path]
            self.call_counts[call.path] = ordinal + 1
            kernel.fact(call.path, identity, "call_entered", ordinal)
            if isinstance(report, BaseException):
                kernel.fact(call.path, identity, "call_failed", ordinal, str(report))
                raise RunFailure(f"{call.path}: {report}") from report
            results: tuple[object, ...] = report
            if len(results) != len(call.results):
                kernel.fact(call.path, identity, "call_invalid", ordinal)
                raise RunFailure(f"{call.path}: returned {len(results)} results for {len(call.results)}")
            resolved = arguments[index]
            if call.updates or call.retains:
                resolved = list(resolved)
                for position, earlier, result in call.locals:
                    resolved[position] = made[earlier][result]
                for position in call.updates:
                    self.protections.updated(call.path, resolved[position], identity)
            request = request_of.get(identity) if request_of else None
            shown = ("retained",) if call.retains else results
            kernel.fact(call.path, identity, "call_completed", ordinal, shown, request.payload if request else None)
            if call.retains:
                product = self.protections.retain(call, ordinal, results[0], resolved, identity, step.group.device)
                product.completion = None if step.wait else completions[index]
                results = (product,)
            made.append(results)
            for slot, value in zip(call.results, results, strict=True):
                frame[slot] = value

    def _check_arguments(self, call: Call, values: list[object]) -> None:
        for position, value in enumerate(values):
            if isinstance(value, Held) and position not in call.leases:
                self.kernel.missed(f"{call.path}: a lease passed where its type says none can be")
            if isinstance(value, Product) and position not in call.retained:
                self.kernel.missed(f"{call.path}: a retained result passed where its type says none can be")


def run_group(
    device: Device,
    context: CallContext,
    after: list[Completion],
    setups: tuple[CallSetup, ...],
    annotations: Annotations,
    handles: list[object],
    held: list[tuple[HeldSetup, object]],
    program: list[tuple[Implementation, list[object], tuple[tuple[int, int, int], ...], int, list[tuple[CallSetup, object]]]],
    wait: bool,
    closed: Callable[[], bool],
) -> tuple[list[Any], list[Completion | None]]:
    """Runs a group's calls in order inside the device's context for them, on a worker thread.

    Inside the device's context, the group's call setups are entered with its
    annotations and the handles its calls receive, and left when the calls
    return; then each held setup its leases carry is applied under the same
    annotations to the handle the lease gives (D16). The call setups a
    lease carries are entered around each call the lease is passed to, and
    only those. A setup that fails to enter or apply fails the group's first
    call, and one entered around a call fails that call.

    Each call's report is its results or what it raised, beside its
    completion; the group stops at the first call that fails or gives the
    wrong number of results, so no later call reads a result that is not
    there. Before each call, the first one included, once the group's context
    and setups are in place, ``closed`` says whether the run's admission has
    closed; if it has, the call does not enter, its report is
    :class:`NotEntered`, and the group stops there. Entering the context, which follows earlier work, or leaving it
    failing fails the call it was entered for. With ``wait`` the worker waits
    for the work's completion, as for host work or results that leave device
    work; otherwise later device work follows each completion in the device's
    own order.
    """
    reports: list[Any] = []
    completions: list[Completion | None] = []
    try:
        with device.context(context, after), ExitStack() as entered:
            for setup in setups:
                entered.enter_context(setup.entered(annotations, handles))
            for held_setup, handle in held:
                try:
                    held_setup.hold(annotations, handle)
                except Exception as failed:
                    raise SetupFailed(f"@{held_setup.name} failed to set its state, so that state is uncertain: {failed}") from failed
            for implementation, given, passed, count, covering in program:
                if closed():  # this call's own admission: a sibling failed or the run was cancelled since the group was admitted
                    reports.append(NotEntered())
                    completions.append(None)
                    break
                if passed:
                    given = list(given)
                    for position, earlier, result in passed:
                        given[position] = reports[earlier][result]
                try:
                    with ExitStack() as around:
                        for call_setup, handle in covering:
                            around.enter_context(call_setup.entered(annotations, (handle,)))
                        results, completion = implementation(device, *given)
                except Exception as failed:
                    reports.append(failed)
                    completions.append(None)
                    break
                results = tuple(results)
                reports.append(results)
                completions.append(completion)
                if len(results) != count:
                    break
    except Exception as failed:
        if not reports:
            reports.append(failed)
            completions.append(None)
        elif isinstance(reports[-1], NotEntered):
            reports[-1].failure = failed
        elif not isinstance(reports[-1], BaseException):
            reports[-1] = failed
    if wait:
        for index, completion in enumerate(completions):
            if completion is None:
                continue
            try:
                completion.wait()
            except Exception as failed:
                reports[index] = failed  # physical completion failed: the call failed
    return reports, completions


def run_undo(device: Device, after: list[Completion], applied: Sequence[tuple[HeldSetup, object]]) -> Completion | None:
    """Undoes a hold's changes on a worker, inside the device's context and after the resource's work: their completion, if the device gives one."""
    with device.context(CallContext(), after):
        for setup, handle in applied:
            if setup.undoes:
                setup.undo(handle)
        return device.completion()


__all__ = ["CallRunner", "run_group", "run_undo"]
