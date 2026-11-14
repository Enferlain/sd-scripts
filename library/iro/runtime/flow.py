"""Streams, claims, producers and filters (D14).

A producer reserves a claim within the stream's outstanding bound, produces
the item, and publishes it within the ready bound; a take admits the item
against the stream's version policy and the consumer owns its claim until it
settles. A claim settles exactly once; a forked claim settles when all its
forks have. A filter settles the claim it is given, forwarding the item or
filtering it out.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy

from library.iro.operation_contracts import Admission
from library.iro.prepare.lowered import Invoke, Named, Step, StreamStep
from library.iro.runtime.kernel import Kernel
from library.iro.runtime.protection import Protections
from library.iro.runtime.state import Cancelled, Enter, Identity, RunFailure, Stream, Ticket
from library.iro.seams import Owner, Version

Stopped = Callable[[str, Identity], bool]


class Flow:
    """The run's streams and the claims on their items."""

    def __init__(self, kernel: Kernel, owners: Mapping[str, Owner], protections: Protections, stopped: Stopped, enter: Enter, body: Enter):
        self.kernel, self.owners, self.protections = kernel, owners, protections
        self.stopped = stopped
        self.enter, self.body = enter, body  # an invocation's body as a new invocation, and run in place
        self.streams: dict[str, Stream] = {}

    # Statements.

    async def stream(self, step: StreamStep, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self.streams[step.name] = Stream(step.name, step.flow)
        return (step.name,)

    async def take_step(self, step: Step, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        kernel = self.kernel
        kernel.admission_point()
        stream = self.streams[str(args[0])]
        while True:
            await kernel.until(lambda: kernel.closing or bool(stream.ready) or stream.closed, ("stream", stream.name))
            kernel.admission_point()
            ticket = self._take(step, stream, identity)
            if ticket is not None:
                return (ticket,)

    async def borrow(self, step: Step, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self.kernel.admission_point()
        return (deepcopy(self.owned(step, args[0], identity).payload),)

    async def dispose(self, step: Named, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self.kernel.admission_point()
        self.settle(self.owned(step, args[0], identity), step.word, identity)
        return ()

    async def fork(self, step: Step, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        self.kernel.admission_point()
        ticket = self.owned(step, args[0], identity)
        ticket.forks = [Ticket(ticket.stream, ticket.key, identity, ticket.payload, ticket.versions, ticket) for _ in step.results]
        ticket.owner = None
        self.kernel.fact(ticket.stream, identity, "forked", ticket.key, len(ticket.forks))
        return tuple(ticket.forks)

    # Claims.

    def owned(self, step: Step, value: object, identity: Identity) -> Ticket:
        if not isinstance(value, Ticket) or value.owner != identity or value.disposition is not None or value.forks:
            raise RunFailure(f"{step.path}: the claim is not held here")
        return value

    def settle(self, ticket: Ticket, disposition: str, identity: Identity) -> None:
        kernel = self.kernel
        if ticket.disposition is not None:
            raise RunFailure(f"{ticket.stream}/{ticket.key}: a claim is settled twice")
        ticket.disposition = (disposition,)
        kernel.fact(ticket.stream, identity, "disposition", ticket.key, disposition)
        while ticket.parent is not None:
            parent = ticket.parent
            if any(fork.disposition is None for fork in parent.forks):
                kernel.changed(("stream", ticket.stream))
                return
            parent.disposition = tuple(d for fork in parent.forks for d in fork.disposition or ())
            ticket = parent
        stream = self.streams[ticket.stream]
        if stream.outstanding.get(ticket.key) is not ticket:
            raise RunFailure(f"{ticket.stream}/{ticket.key}: settled without its reservation")
        stream.outstanding.pop(ticket.key)
        if ticket in stream.ready:
            stream.ready.remove(ticket)
        kernel.fact(ticket.stream, (), "settled", ticket.key, ticket.disposition)
        kernel.changed(("stream", ticket.stream))

    def settle_owned(self, step: Invoke, identity: Identity) -> None:
        """Claims an ending invocation still owns are settled, cancelled or failed."""
        for stream in self.streams.values():
            for ticket in tuple(stream.outstanding.values()):
                for held in leaves(ticket):
                    if held.owner == identity and held.disposition is None:
                        if not step.owns_claims:
                            self.kernel.missed(f"{step.path}: a claim it still owned when it settled")
                        self.settle(held, "cancelled" if self.kernel.cancelled else "failed", identity)

    def _take(self, step: Step, stream: Stream, identity: Identity) -> Ticket | None:
        if not stream.ready:
            raise RunFailure(f"{step.path}: @{stream.name} closed before this take")
        ticket = stream.ready.popleft()
        admitted = self._admit(stream, ticket)
        self.kernel.fact(step.path, identity, "admission", ticket.key, ticket.versions, admitted)
        ticket.owner = identity
        self.kernel.changed(("stream", ticket.stream))
        if not admitted:
            self.settle(ticket, "rejected", identity)
            return None
        return ticket

    def _admit(self, stream: Stream, ticket: Ticket) -> bool:
        policy = stream.flow.versions
        observed: dict[str, Version] = {}
        for version in ticket.versions:
            if observed.setdefault(version.owner, version) != version:
                return False
        if policy.admission == Admission.ANY_WITH_PROVENANCE:
            return True
        allowed = policy.max_lag if policy.admission == Admission.BOUNDED_LAG else 0
        for owner in policy.owners:
            seen, now = observed.get(owner), self.owners[owner].current(owner)
            if seen is None or (seen.incarnation, seen.lineage) != (now.incarnation, now.lineage) or seen.ordinal > now.ordinal:
                return False
            if now.ordinal - seen.ordinal > allowed:
                return False
        return True

    # Protocols.

    async def filter(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        ticket = self.owned(step, args[0], identity)
        disposition = "failed"
        try:
            kept = await self.body(step, args, identity, (deepcopy(ticket.payload),))
            if len(kept) != 1 or type(kept[0]) is not bool:
                raise RunFailure(f"{step.path}: a filter's body yields one bool")
            disposition = "forwarded" if kept[0] else "filtered_out"
            return (deepcopy(ticket.payload) if kept[0] else None,)
        except Cancelled:
            disposition = "cancelled"
            raise
        finally:
            self.settle(ticket, disposition, identity)

    async def producer(self, step: Invoke, args: tuple[object, ...], identity: Identity) -> tuple[object, ...]:
        kernel = self.kernel
        name = str(args[0])
        stream = self.streams[name]
        keys = (("stream", name), ("stop", step.path))
        outstanding, ready = stream.flow.max_outstanding, stream.flow.ready_bound
        try:
            index = 0
            while not self.stopped(step.path, identity) and not kernel.closing:
                await kernel.until(
                    lambda: kernel.closing or self.stopped(step.path, identity) or len(stream.outstanding) < outstanding,
                    *keys,
                )
                kernel.admission_point()
                if self.stopped(step.path, identity):
                    break
                ticket = Ticket(name, stream.next_key, identity)
                stream.next_key += 1
                stream.outstanding[ticket.key] = ticket
                production = (*identity, (step.path, index))
                self.protections.begin_production(identity)
                try:
                    (value,) = await self.enter(step, args, identity, (index,))
                except RunFailure as failed:
                    if ticket.disposition is None:
                        self.settle(ticket, "cancelled" if isinstance(failed, Cancelled) else "failed", production)
                    raise
                finally:
                    observed = self.protections.end_production(identity)
                index += 1
                ticket.versions = tuple(dict.fromkeys(observed))
                if step.optional and value is None:
                    stream.outstanding.pop(ticket.key)
                    kernel.fact(name, identity, "nothing_produced", ticket.key)
                    kernel.changed(("stream", name))
                    continue
                try:
                    await kernel.until(
                        lambda: kernel.closing or self.stopped(step.path, identity) or len(stream.ready) < ready,
                        *keys,
                    )
                    kernel.admission_point()
                except RunFailure as failed:
                    if ticket.disposition is None:
                        self.settle(ticket, "cancelled" if isinstance(failed, Cancelled) else "failed", identity)
                    raise
                ticket.payload, ticket.owner = deepcopy(value), None
                if self.stopped(step.path, identity):
                    self.settle(ticket, "cancelled", identity)
                else:
                    stream.ready.append(ticket)
                    kernel.fact(name, identity, "published", ticket.key, ticket.versions)
                kernel.changed(("stream", name))
        finally:
            stream.closed = True
            kernel.changed(("stream", name))
            # Wait for consumers to settle what they took; what the producer still holds settles at the drain.
            await kernel.until(
                lambda: all(
                    leaf.owner is None or leaf.owner == identity
                    for t in stream.outstanding.values()
                    if t not in stream.ready
                    for leaf in leaves(t)
                ),
                ("stream", name),
            )
            for ticket in tuple(stream.outstanding.values()):
                if ticket in stream.ready or ticket.owner == identity:
                    self.settle(ticket, "cancelled", identity)
        return ()


def leaves(ticket: Ticket) -> list[Ticket]:
    if not ticket.forks:
        return [ticket]
    return [leaf for fork in ticket.forks for leaf in leaves(fork)]


__all__ = ["Flow", "leaves"]
