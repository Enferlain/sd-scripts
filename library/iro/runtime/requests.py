"""Requests, handlers and stops (D14).

A request is armed where it is written and made when its source reaches
the boundary it names; it waits for request capacity and for updates of
what it reads at the request, then holds those versions. A handler takes
requests one at a time up to its bound and ends when nothing is pending and
its sources have settled. A stop ends a producer once its source has settled
in the scope they share.
"""

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy

from library.iro.operation_contracts import Boundary
from library.iro.prepare.lowered import Invoke, RequestStep, Settling, StopStep
from library.iro.runtime.kernel import Kernel
from library.iro.runtime.protection import Protections
from library.iro.runtime.scopes import Records, scope
from library.iro.runtime.state import Cancelled, Enter, HandlerState, Identity, Outcome, Request, RunFailure, State


class Requests:
    """The run's handlers, the requests made and in flight, and the stops requested."""

    def __init__(
        self,
        kernel: Kernel,
        protections: Protections,
        outcomes: Mapping[Identity, Outcome],
        records: Records,
        all_records: Records,
        enter: Enter,
    ):
        self.kernel, self.protections = kernel, protections
        self.outcomes = outcomes
        self.records, self.all_records = records, all_records
        self.enter = enter
        self.handlers: dict[str, HandlerState] = defaultdict(HandlerState)
        self.request_of: dict[Identity, Request] = {}
        self.emissions: dict[Identity, list[tuple[RequestStep, Identity, tuple[object, ...]]]] = defaultdict(list)
        self.stops: set[tuple[Identity, str]] = set()

    # Requests.

    async def request(self, step: RequestStep, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        """The request statement: armed now, made when its source reaches its boundary."""
        self.kernel.admission_point()
        running = [
            identity[:length]
            for length in range(1, len(identity) + 1)
            if identity[length - 1][0] == step.source
            and (outcome := self.outcomes.get(identity[:length])) is not None
            and outcome.state == State.RUNNING
        ]
        if not running:
            raise RunFailure(f"{step.path}: a request is made inside its source")
        self.emissions[running[-1]].append((step, running[-1], args))
        return ()

    async def emit(self, step: RequestStep, source: Identity, args: tuple[object, ...], status: str) -> None:
        kernel = self.kernel
        if step.boundary == Boundary.COMPLETED:
            raise RunFailure(f"{step.path}: requests at completion are not realized yet")
        if (step.boundary == Boundary.SETTLED_SUCCESS and status != "succeeded") or kernel.closing:
            kernel.fact(step.handler_name, source, "request_disposed", "unfulfilled")
            return
        target = step.handler
        handler = self.handlers[target.path]
        reads = [name for name, _ in target.at_request]
        protection = self.protections.by_resource
        await kernel.until(
            lambda: (
                kernel.closing
                or (not any(protection[name].writer is not None for name in reads) and len(handler.pending) + handler.active < target.bound)
            ),
            ("handler", target.path),
            *(("resource", name) for name in reads),
        )
        if kernel.closing:
            kernel.fact(step.handler_name, source, "request_disposed", "cancelled")
            return
        request = Request(target.path, handler.next_ordinal, source, deepcopy(args[0]))
        handler.next_ordinal += 1
        self.protections.reserve(request, target)
        handler.pending.append(request)
        kernel.fact(target.path, source, "request_pending", request.ordinal, request.payload)
        kernel.changed(("handler", target.path), ("requests",))

    def pending_under(self, identity: Identity) -> list[Request]:
        return [r for h in self.handlers.values() for r in h.pending if r.cause[: len(identity)] == identity]

    def settle(self, identity: Identity) -> None:
        """Requests still pending under an ending invocation are disposed of, cancelled."""
        for request in self.pending_under(identity):
            self.handlers[request.handler].pending.remove(request)
            self.protections.end_reservations(request)
            self.kernel.fact(request.handler, identity, "request_disposed", request.ordinal, "cancelled")
            self.kernel.changed(("handler", request.handler), ("requests",))

    # Handlers.

    async def handler(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        kernel = self.kernel
        handler = self.handlers[step.path]
        sources = step.sources
        key = ("handler", step.path)
        while True:
            await kernel.until(
                lambda: kernel.closing or bool(handler.pending) or all(self.source_settled(s, identity) for s in sources), *step.keys
            )
            if kernel.closing:
                while handler.pending:
                    request = handler.pending.popleft()
                    self.protections.end_reservations(request)
                    kernel.fact(step.path, identity, "request_disposed", request.ordinal, "cancelled")
                kernel.changed(key, ("requests",))
                return ()
            if not handler.pending:
                return ()
            handler.active += 1
            request = handler.pending.popleft()
            self.request_of[identity] = request
            disposition = "failed"
            try:
                await self.enter(step, args, identity, (deepcopy(request.payload),))
                disposition = "completed"
            except RunFailure as failed:
                disposition = "cancelled" if isinstance(failed, Cancelled) else "failed"
                raise
            finally:
                handler.active -= 1
                self.protections.end_reservations(request)
                kernel.fact(step.path, identity, "request_settled", request.ordinal, disposition)
                del self.request_of[identity]
                kernel.changed(key, ("requests",))

    # Stops and settling.

    async def stop(self, step: StopStep, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        """The stop statement: once its source has settled in the scope they share, the target stops there."""
        kernel = self.kernel
        kernel.admission_point()
        settling = step.settling
        await kernel.until(lambda: kernel.closing or self.source_settled(settling, identity), *settling.keys)
        kernel.admission_point()
        self.stops.add((scope(step.scope, identity), step.target))
        kernel.fact(step.target, identity, "stop_requested", step.source)
        kernel.changed(("stop", step.target))
        return ()

    def stopped(self, target: str, identity: Identity) -> bool:
        return bool(self.stops) and any(t == target and identity[: len(s)] == s for s, t in self.stops)

    def scope_closed(self, identity: Identity) -> None:
        if self.stops:
            self.stops = {(s, target) for s, target in self.stops if s != identity}

    def source_settled(self, settling: Settling, identity: Identity) -> bool:
        """Whether the source, and every statement between it and the shared scope, has run and ended there."""
        shared = scope(settling.common, identity)
        settled = self.records.settled(settling.chain, shared)
        if self.kernel.checked and self.all_records.settled(settling.chain, shared) != settled:
            self.kernel.missed(f"{settling.source}: the scope records lowering kept disagree with the full records")
        return settled


__all__ = ["Requests"]
