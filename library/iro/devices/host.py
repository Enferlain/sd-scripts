"""The host device adapter: work on the CPU, where a call's return is its completion (D15)."""

import sys
import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager

from library.iro.seams import CallContext, Completion


class Returned:
    """The completion of host work: it is done when the call returns."""

    def done(self) -> bool:
        return True

    def wait(self) -> None:
        return None


class Host:
    """The host as a device: work runs on the worker thread and nothing is known of its capacity."""

    name = "host"
    kind = "host"
    offers: frozenset[str] = frozenset()

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._random = None
        self._creating = threading.Lock()

    @contextmanager
    def context(self, call: CallContext, after: Sequence[Completion], reaches: Sequence[object]) -> Generator[None]:
        """Waits on the CPU for the work the call follows, and establishes the call's Torch settings.

        A run that has not loaded Torch has no Torch settings to establish.
        A call that loads Torch itself has run under the thread's defaults,
        so it fails, saying to import Torch before the run (D15: a call never
        inherits a thread's ambient state).
        """
        for completion in after:
            completion.wait()
        if "torch" not in sys.modules:
            yield
            if "torch" in sys.modules:
                raise RuntimeError("the call loaded Torch itself, so its Torch settings were not established; import Torch before the run")
            return
        import torch

        from library.iro.devices.torch_context import RandomStates, entered

        if self._random is None:
            with self._creating:
                if self._random is None:
                    self._random = RandomStates(torch.device("cpu"), self.seed)
        with entered(call, reaches, torch.device("cpu"), self._random):
            yield

    def completion(self) -> Returned:
        return Returned()

    def capacity(self) -> Mapping[str, int]:
        return {}


__all__ = ["Host", "Returned"]
