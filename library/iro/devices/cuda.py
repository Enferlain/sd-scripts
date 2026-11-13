"""The PyTorch CUDA device adapter (D15).

Work that takes turns on one device shares one stream that the adapter
owns: PyTorch's caching allocator reuses freed memory only for the stream
that freed it, so separate streams for work in turn would keep a cache per
stream or synchronize the device at every turn. A call's completion is an
event recorded on that stream after its work. Work the call must follow is
ordered on the device: a completion from this stream needs nothing, one from
another stream makes this stream wait on its event, and any other completion
is waited for on the CPU, in the worker. The coordinator itself never waits
for the device.

Only a run whose prepared target selects a CUDA device imports this module.
"""

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager

import torch

from library.iro.devices.torch_context import RandomStates, entered
from library.iro.seams import CallContext, Completion


class Recorded:
    """The completion of work queued on a CUDA stream: an event recorded after it."""

    def __init__(self, event: torch.cuda.Event, stream: torch.cuda.Stream):
        self.event, self.stream = event, stream

    def done(self) -> bool:
        return self.event.query()

    def wait(self) -> None:
        self.event.synchronize()


class Cuda:
    """One CUDA device: its stream, its callers' random states and its capacity facts."""

    def __init__(self, index: int = 0, *, seed: int = 0):
        self.device = torch.device("cuda", index)
        self.name = f"cuda:{index}"
        self.stream = torch.cuda.Stream(self.device)
        self.random = RandomStates(self.device, seed)

    @contextmanager
    def context(self, call: CallContext, after: Sequence[Completion], reaches: Sequence[object]) -> Generator[None]:
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            for completion in after:
                if isinstance(completion, Recorded):
                    if completion.stream != self.stream:
                        self.stream.wait_event(completion.event)
                else:
                    completion.wait()
            with entered(call, reaches, self.device, self.random):
                yield

    def completion(self) -> Recorded:
        event = torch.cuda.Event()
        event.record(self.stream)
        return Recorded(event, self.stream)

    def capacity(self) -> Mapping[str, int]:
        free, total = torch.cuda.mem_get_info(self.device)
        return {"free_bytes": free, "total_bytes": total}


__all__ = ["Cuda", "Recorded"]
