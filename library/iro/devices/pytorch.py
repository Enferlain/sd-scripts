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
    (completion ordered on the device), ``copies`` (an owner's device
    storage leaving the device and coming back) and, where the backend
    reports it, ``memory`` (free and total device memory). The target records
    it, preparation checks a run's needs against it, and the coordinator
    refuses an adapter offering less than the run was prepared for.
    """

    kind: ClassVar[str]

    def __init__(self, index: int = 0, *, seed: int = 0):
        self.device = torch.device(self.kind, index)
        self.name = f"{self.kind}:{index}"
        self.stream = torch.Stream(device=self.device)
        self.random = RandomStates(self.device, seed)
        self._memory = getattr(torch.get_device_module(self.kind), "mem_get_info", None)
        self.offers = frozenset({"ordering", *(("copies",) if _pins() else ()), *(("memory",) if self._memory is not None else ())})
        self._host: dict[int, _HostCopy] = {}  # by the id of each tensor an owner names

    @contextmanager
    def _current(self) -> Generator[None]:
        """This device and its stream, current for the calling thread.

        The stream is made current per use, never by entering the shared
        torch.Stream object: that keeps its restore state on the object, and
        two threads entering it at once abort the process.
        """
        index = self.device.index
        with torch.accelerator.device_index(index):
            previous = torch.accelerator.current_stream(index)
            torch.accelerator.set_stream(self.stream)
            try:
                yield
            finally:
                torch.accelerator.set_stream(previous)

    @contextmanager
    def context(self, call: CallContext, after: Sequence[Completion], reaches: Sequence[object]) -> Generator[None]:
        with self._current():
            for completion in after:
                if isinstance(completion, Recorded) and completion.stream.device_type == self.stream.device_type:
                    if completion.stream != self.stream:
                        self.stream.wait_event(completion.event)
                else:
                    completion.wait()
            with entered(call, reaches, self.device, self.random):
                yield

    def empty(self, storage: Sequence[object]) -> None:
        """Frees each tensor's device memory, keeping an exact copy of its whole storage in pinned host memory.

        The copy is of the storage's bytes, so views, offsets and tensors
        sharing one storage come back whole. Only frozen storage leaves, so
        the copy made the first time stays exact. It is queued on this stream
        after the work already there, and after what the calling thread's
        stream had queued, such as loading the weights; the freed memory is
        marked as used by this stream, so the caching allocator reuses it only
        after that work and the copy finish.
        """
        outer = torch.accelerator.current_stream(self.device.index)
        with self._current():
            self.stream.wait_stream(outer)
            # Addresses are read before anything is freed: an emptied storage's address reads as 0.
            pieces = []
            for tensor in storage:
                assert isinstance(tensor, torch.Tensor) and tensor.device == self.device
                memory = tensor.untyped_storage()
                pieces.append((tensor, memory, memory.data_ptr()))
            emptied: dict[int, _HostCopy] = {}  # by storage address, for tensors sharing one storage
            for tensor, memory, address in pieces:
                if address in emptied:
                    self._host[id(tensor)] = emptied[address]
                    continue
                copy = self._host.get(id(tensor))
                if copy is None:
                    if memory.nbytes() == 0:
                        raise RuntimeError("the storage is already empty: it left the device in an earlier run")
                    host = torch.empty(memory.nbytes(), dtype=torch.uint8, pin_memory=True)
                    host.copy_(_bytes(memory, self.device), non_blocking=True)
                    copy = self._host[id(tensor)] = _HostCopy(tensor, host)
                emptied[address] = copy
                tensor.record_stream(self.stream)
                memory.resize_(0)

    def fill(self, storage: Sequence[object]) -> Recorded:
        """Brings each tensor's storage back from its host copy, on this stream, before any later work on it.

        The storage comes back at a new address; the tensors keep their
        identity, but anything holding the old address, such as a captured
        graph, would be stale, which is why only storage nothing else refers
        to may leave (D15).
        """
        with self._current():
            filled: set[int] = set()
            for tensor in storage:
                assert isinstance(tensor, torch.Tensor)
                copy = self._host[id(tensor)]
                if id(copy) in filled:
                    continue
                filled.add(id(copy))
                memory = tensor.untyped_storage()
                memory.resize_(copy.host.numel())
                _bytes(memory, self.device).copy_(copy.host, non_blocking=True)
            return self.completion()

    def completion(self) -> Recorded:
        return Recorded(self.stream.record_event(), self.stream)

    def capacity(self) -> Mapping[str, int]:
        if self._memory is None:
            return {}
        free, total = self._memory(self.device)
        return {"free_bytes": free, "total_bytes": total}


class _HostCopy:
    """A storage's exact bytes in pinned host memory, with the tensor it was taken from kept alive."""

    def __init__(self, tensor: torch.Tensor, host: torch.Tensor):
        self.tensor, self.host = tensor, host


def _bytes(memory: torch.UntypedStorage, device: torch.device) -> torch.Tensor:
    """A tensor of a storage's bytes, whatever views of it exist."""
    return torch.empty(0, dtype=torch.uint8, device=device).set_(memory)


def _pins() -> bool:
    """Whether this PyTorch build can pin host memory, which storage copies need."""
    try:
        torch.empty(1, pin_memory=True)
    except RuntimeError:
        return False
    return True


__all__ = ["PyTorchDevice", "Recorded"]
