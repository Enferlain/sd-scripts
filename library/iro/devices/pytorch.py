"""The base of every PyTorch backend's adapter: what does not depend on the vendor (D15).

There is one adapter per PyTorch backend, named by PyTorch's device type, and
each builds on this. Work that takes turns on one device shares one stream
that the adapter owns: PyTorch's caching allocator reuses freed memory only
for the stream that freed it, so separate streams for work in turn would keep
a cache per stream or synchronize the device at every turn. A call's
completion is an event recorded on that stream after its work. Work the call
must follow is ordered on the device: an event from this stream needs
nothing, one from another stream of the same backend makes this stream wait
on it, and any other completion is waited for on the CPU, in the worker. The
coordinator itself never waits for the device.

Everything here goes through PyTorch's vendor-neutral types and calls:
``torch.Stream``, ``torch.Event``, ``torch.accelerator``, and the backend's
device module for random state and memory facts. A backend's adapter names
its device type and adds what only its vendor offers. Only a run whose
target selects such a device imports this module.
"""

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import ClassVar

import torch

from library.iro.devices.torch_context import RandomStates, entered
from library.iro.seams import CallContext, Completion


class Recorded:
    """The completion of work queued on a stream: an event recorded after it."""

    def __init__(self, event: torch.Event, stream: torch.Stream):
        self.event, self.stream = event, stream

    def done(self) -> bool:
        return self.event.query()

    def wait(self) -> None:
        self.event.synchronize()


class PyTorchDevice:
    """One device of a PyTorch backend: its stream, its callers' random states and its capacity facts.

    ``offers`` says what the adapter realizes (D15): ``ordering``
    (completion ordered on the device) and, where the backend reports it,
    ``memory`` (free and total device memory). The target records it and the
    coordinator refuses an adapter offering less than the run was prepared
    for; checking a run's needs against it comes with the first need.
    """

    kind: ClassVar[str]

    def __init__(self, index: int = 0, *, seed: int = 0):
        self.device = torch.device(self.kind, index)
        self.name = f"{self.kind}:{index}"
        self.stream = torch.Stream(device=self.device)
        self.random = RandomStates(self.device, seed)
        self._memory = getattr(torch.get_device_module(self.kind), "mem_get_info", None)
        self.offers = frozenset({"ordering", *(("memory",) if self._memory is not None else ())})

    @contextmanager
    def context(self, call: CallContext, after: Sequence[Completion], reaches: Sequence[object]) -> Generator[None]:
        # The stream is made current per call, never by entering the shared torch.Stream object: that
        # keeps its restore state on the object, and two threads entering it at once abort the process.
        index = self.device.index
        with torch.accelerator.device_index(index):
            previous = torch.accelerator.current_stream(index)
            torch.accelerator.set_stream(self.stream)
            try:
                for completion in after:
                    if isinstance(completion, Recorded) and completion.stream.device_type == self.stream.device_type:
                        if completion.stream != self.stream:
                            self.stream.wait_event(completion.event)
                    else:
                        completion.wait()
                with entered(call, reaches, self.device, self.random):
                    yield
            finally:
                torch.accelerator.set_stream(previous)

    def completion(self) -> Recorded:
        return Recorded(self.stream.record_event(), self.stream)

    def capacity(self) -> Mapping[str, int]:
        if self._memory is None:
            return {}
        free, total = self._memory(self.device)
        return {"free_bytes": free, "total_bytes": total}


__all__ = ["PyTorchDevice", "Recorded"]
