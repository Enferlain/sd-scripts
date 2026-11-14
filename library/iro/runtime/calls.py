"""The call path: a fused group is one submission to a worker thread on its device (D15, D16).

A group resolves its leases and retained results, waits for what it must
follow, and runs its calls in order inside the device's context on a worker
thread. Its results become slot values, owner updates and retained results.
Calls not declared safe for concurrent use run one at a time per
participant, and a group that reaches a participant's device storage waits
until the snapshot copies already asked for of it are queued (D15).
"""

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from library.iro.prepare.lowered import Call, Calls
from library.iro.runtime.kernel import Kernel
from library.iro.runtime.protection import Protections
from library.iro.runtime.state import Held, Identity, Product, Request, RunFailure
from library.iro.runtime.storage import DeviceStorage
from library.iro.runtime.turns import Turns
from library.iro.seams import CallContext, Completion, Device, Implementation


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
        self.busy: set[str] = set()
        self.call_counts: dict[str, int] = defaultdict(int)

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
                program.append((call.implementation, given, call.locals, len(call.results)))
            if len(leased) > 1:
                leased = list(dict.fromkeys(leased))
            if kernel.explore is not None:
                await kernel.jitter()
            storage = self.storage
            busy: set[str] = set()
            touching = storage.touching(leased, borrowed)
            if not step.safe and leased:
                busy = {lease.resource for lease in leased}
            if busy or (touching and storage.capturing):
                capturing = storage.capturing
                await kernel.until(
                    lambda: kernel.closing or not (busy & self.busy or any(r in capturing for r in touching)),
                    *(("resource", r) for r in busy.union(touching)),
                )
                kernel.admission_point()
            protection = self.protections.by_resource
            after = [c for lease in leased for c in protection[lease.resource].pending]
            after += [p.completion for p in borrowed if p.completion is not None]
            reaches = tuple(lease.handle for lease in leased)
            self.busy |= busy
            storage.calls_started(touching)
            try:
                reports, completions = await kernel.submit(run_group, device, step.group.context, after, reaches, program, step.wait)
            finally:
                self.busy -= busy
                storage.calls_ended(touching)
            if not step.wait:
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
    reaches: tuple[object, ...],
    program: list[tuple[Implementation, list[object], tuple[tuple[int, int, int], ...], int]],
    wait: bool,
) -> tuple[list[Any], list[Completion | None]]:
    """Runs a group's calls in order inside the device's context for them, on a worker thread.

    Each call's report is its results or what it raised, beside its
    completion; the group stops at the first call that fails or gives the
    wrong number of results, so no later call reads a result that is not
    there. Entering the context, which follows earlier work, or leaving it
    failing fails the call it was entered for. With ``wait`` the worker waits
    for the work's completion, as for host work or results that leave device
    work; otherwise later device work follows each completion in the device's
    own order.
    """
    reports: list[Any] = []
    completions: list[Completion | None] = []
    try:
        with device.context(context, after, reaches):
            for implementation, given, passed, count in program:
                if passed:
                    given = list(given)
                    for position, earlier, result in passed:
                        given[position] = reports[earlier][result]
                try:
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


__all__ = ["CallRunner", "run_group"]
