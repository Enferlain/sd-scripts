"""Device storage work the runtime queues on a device's stream: snapshot copies, leaving and returning (D15).

An owner never reads, copies or changes a participant's device storage
itself; it names the storage, and the work here runs through the device's
adapter, in order with the participant's other work.

**Snapshot copies.** An owner's snapshot lease names the storage its version
lives in and how its handle is built over copies (``seams.Capture``). The
copy is queued between the participant's calls: after the calls on it
already being queued, and before later ones, which wait until it is queued.
So it holds what the work before it left, including storage a route or an
implementation moves within a use. The access's calls receive the handle the
owner builds over the copies, and follow the copy's completion.

**Leaving the device.** A permitted participant's storage leaves when its
last lease is released, unless it is leased again before that move begins,
and comes back as the first work of the next access that leases it. Its
moves run one after another; leaving and returning change no version and
report no fact.

Bookkeeping is by participant on a device, so bounded by those participants.
"""

import asyncio
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

from library.iro.operation_contracts import AccessMode
from library.iro.prepare import Prepared
from library.iro.runtime.kernel import Kernel
from library.iro.runtime.state import Held, Product, Protection, RunFailure
from library.iro.seams import Capture, Completion, CopiesStorage, Device, HoldsDeviceStorage, Owner, Version


class DeviceStorage:
    """The device storage work of one run: which participants may leave, where each lives, and the work in flight."""

    def __init__(
        self,
        kernel: Kernel,
        prepared: Prepared,
        owners: Mapping[str, Owner],
        devices: Mapping[str, Device],
        protection: Mapping[str, Protection],
    ):
        self.kernel = kernel
        self.owners, self.devices, self.protection = owners, devices, protection
        # Participants that may leave the device between uses, by the device their storage leaves from.
        self.leaving = dict(prepared.leaving)
        for name, device in self.leaving.items():
            if not isinstance(owners.get(name), HoldsDeviceStorage):
                raise ValueError(f"@{name} may leave the device, and its owner does not say what its device storage is")
            if not isinstance(devices.get(device), CopiesStorage):
                raise ValueError(f"@{name} may leave {device}, and its adapter does not copy storage")
        self.away: set[str] = set()
        self.holding: dict[str, int] = defaultdict(int)
        self.moves: dict[str, asyncio.Task] = {}
        # Where each participant's storage lives, the copies still being queued, and the calls reaching
        # its storage still being queued, which a copy follows.
        self.placed = dict(prepared.placed)
        self.capturing: dict[str, set[asyncio.Task]] = {}
        self.using: dict[str, int] = defaultdict(int)

    # Leases.

    def capturer(self, name: str, mode: AccessMode, version: Version) -> CopiesStorage:
        """The adapter that copies a snapshot of this participant's storage, or why none can."""
        if mode != AccessMode.SNAPSHOT:
            raise RunFailure(f"@{name}: only a snapshot lease is captured, and the owner asked to capture a {mode} lease")
        device = self.placed.get(name)
        if device is None:
            raise RunFailure(
                f"@{name}: a snapshot of device storage is copied on the one device whose work uses it, "
                "and device work on no device or on several uses it"
            )
        adapter = self.devices[device]
        if not isinstance(adapter, CopiesStorage) or "copies" not in adapter.offers:
            raise RunFailure(f"@{name}: a snapshot of device storage needs storage copies, and {device}'s adapter does not offer them")
        if version != self.owners[name].current(name):
            raise RunFailure(f"@{name}: a capture copies live storage, and the snapshot's version is not the current one")
        return adapter

    def held(self, lease: Held, capture: Capture | None, adapter: CopiesStorage | None) -> None:
        """A lease was taken: a participant that may leave stays while it is held, and a snapshot's copy is asked for."""
        name = lease.resource
        if name in self.leaving:
            self.holding[name] += 1
        if capture is not None:
            assert adapter is not None
            # Only frozen storage may leave, and frozen storage is never copied (Protections.hold).
            assert name not in self.leaving, f"@{name} may leave the device, and a copy of it was asked for"
            lease.capture = task = self.kernel.spawn(self.capture(lease, capture, adapter))
            self.capturing.setdefault(name, set()).add(task)
            task.add_done_callback(lambda ended: self._captured(name, ended))

    def released(self, name: str) -> None:
        if name in self.leaving:
            self._unhold(name)

    def _unhold(self, name: str) -> None:
        self.holding[name] -= 1
        if not self.holding[name]:
            self.moves[name] = self.kernel.spawn(self._leave(name, self.moves.get(name)))

    async def ready(self, lease: Held) -> None:
        """Before an access's body: its snapshot's copy is queued, and storage it reaches is on the device."""
        if lease.capture is not None:
            await asyncio.shield(lease.capture)  # the body's calls receive the copies
        elif lease.resource in self.leaving:
            await self.present(lease.resource)  # the first work of the admission that uses it

    # Calls reaching device storage.

    def touching(self, leased: Sequence[Held], borrowed: Sequence[Product]) -> list[str]:
        """The participants on a device whose storage a call group reaches, through a lease or a retained result."""
        placed = self.placed
        if not placed:
            return []
        reached = [*leased, *(lease for product in borrowed for lease in product.leases)]
        return [lease.resource for lease in reached if lease.capture is None and lease.resource in placed]

    def calls_started(self, touching: Sequence[str]) -> None:
        for name in touching:
            self.using[name] += 1

    def calls_ended(self, touching: Sequence[str]) -> None:
        for name in touching:
            self.using[name] -= 1
            if not self.using[name] and name in self.capturing:
                self.kernel.changed(("resource", name))

    def follows(self, name: str, completion: Completion) -> None:
        """Later work on the participant follows the completion; those already done are dropped, so the list stays short."""
        pending = self.protection[name].pending
        pending[:] = [c for c in pending if not c.done()]
        pending.append(completion)
        self.kernel.queued(completion)  # the run's settlement waits for it (D15)

    # Snapshot copies.

    async def capture(self, lease: Held, capture: Capture, adapter: CopiesStorage) -> None:
        kernel = self.kernel
        name = lease.resource
        try:
            if self.using[name]:
                await kernel.until(lambda: kernel.closing or not self.using[name], ("resource", name))
                kernel.admission_point()
            if not lease.active:
                return  # released before its copy was queued, as when its request was cancelled: nothing reads it
            # A guard: no writer was leased when the snapshot was, and updates leased since wait for this copy.
            if lease.version != self.owners[name].current(name):
                raise RunFailure(f"@{name}: an update moved it past its snapshot's version before the capture was queued")
            after = list(self.protection[name].pending)
            copies, completion = await kernel.submit(adapter.capture, capture.storage, after)
            lease.built, lease.over_storage = capture.over(copies), True
            # The access's calls, and any other later work on it off this stream, follow the copy.
            self.follows(name, completion)
        except RunFailure as failed:
            kernel.note_failure(failed)
            raise
        except Exception as error:
            failed = RunFailure(f"@{name}: capturing its snapshot failed: {error!r}")
            kernel.note_failure(failed)
            raise failed from error

    def _captured(self, name: str, task: asyncio.Task) -> None:
        """A copy ended, however it ended, cancelled before it began included: later calls on it may go."""
        if not task.cancelled():
            task.exception()  # a failure is already noted; this marks it seen
        tasks = self.capturing[name]
        tasks.discard(task)
        if not tasks:
            del self.capturing[name]
        self.kernel.changed(("resource", name))

    # Leaving the device.

    async def present(self, name: str) -> None:
        """Waits until the participant's storage is on its device, bringing it back if it left."""
        while True:
            move = self.moves.get(name)
            if move is not None and not move.done():
                await asyncio.shield(move)
                continue
            if name not in self.away:
                return
            self.moves[name] = self.kernel.spawn(self._return(name))

    async def _return(self, name: str) -> None:
        adapter = cast(CopiesStorage, self.devices[self.leaving[name]])
        storage = cast(HoldsDeviceStorage, self.owners[name]).storage(name)
        completion = await self.kernel.submit(adapter.fill, storage)
        self.away.discard(name)
        # Host work on it waits for the copy on the CPU; device work on its stream follows it.
        self.follows(name, completion)

    async def _leave(self, name: str, before: asyncio.Task | None) -> None:
        if before is not None and not before.done():
            await asyncio.shield(before)
        if self.holding[name] or name in self.away:
            return  # leased again meanwhile, or already gone
        device = self.leaving[name]
        adapter = cast(CopiesStorage, self.devices[device])
        storage = cast(HoldsDeviceStorage, self.owners[name]).storage(name)
        try:
            await self.kernel.submit(adapter.empty, storage)
            self.away.add(name)
        except Exception as error:
            self.kernel.note_failure(RunFailure(f"@{name}: leaving {device} failed: {error!r}"))

    async def drain(self) -> None:
        """At the run's end: the copies still being queued, then the moves still running."""
        await asyncio.gather(*(task for tasks in list(self.capturing.values()) for task in tasks), return_exceptions=True)
        await asyncio.gather(*self.moves.values(), return_exceptions=True)


__all__ = ["DeviceStorage"]
