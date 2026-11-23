"""The Torch side of a call's context: a neutral baseline and the random state (D15, D16).

Device adapters use it for calls that run Torch code. It is imported only by
an adapter that a prepared target selects, so the runtime core never loads
Torch. A call never inherits a thread's ambient state: on entry grad mode is
off and any autocast the thread has on is turned off, and the call's setups,
entered inside this context, turn on what a domain or an owner decided for
it, such as grad mode for the calls training differentiates or autocast for
a route's mixed precision (D16). The previous settings are restored on exit,
so a worker thread never carries one call's state into the next.

Random state is kept per owner: each owner that draws randomness has its own
generator state per device, swapped in for its call and saved back after.
Calls that declare randomness swap one device's state one at a time, under a
lock. Code that draws randomness without declaring it is not covered: on the
host, where calls run side by side, it may draw from another owner's state
(D18 qualifies declarations).

Module mode is not set here: it is the state of a participant's modules, not
of the thread, and the held setup of the route a call's lease goes through
sets it (D16, D20).
"""

import threading
from collections.abc import Generator
from contextlib import ExitStack, contextmanager

import torch

from library.iro.seams import CallContext


class RandomStates:
    """Each owner's random state on one device, seeded from the run's seed and the owner's name."""

    def __init__(self, device: torch.device, seed: int = 0):
        self.device = device
        self.seed = seed
        self.states: dict[str, torch.Tensor] = {}
        self.lock = threading.Lock()

    def _get(self) -> torch.Tensor:
        if self.device.type == "cpu":
            return torch.get_rng_state()
        return torch.get_device_module(self.device.type).get_rng_state(self.device)

    def _set(self, state: torch.Tensor) -> None:
        if self.device.type == "cpu":
            torch.set_rng_state(state)
        else:
            torch.get_device_module(self.device.type).set_rng_state(state, self.device)

    @contextmanager
    def drawing(self, owner: str) -> Generator[None]:
        with self.lock:
            previous = self._get()
            state = self.states.get(owner)
            if state is None:
                generator = torch.Generator(self.device).manual_seed(self.seed + _stable_hash(owner))
                state = generator.get_state()
            self._set(state)
            try:
                yield
            finally:
                self.states[owner] = self._get()
                self._set(previous)


def _stable_hash(name: str) -> int:
    value = 0
    for character in name.encode():
        value = (value * 131 + character) % (2**31)
    return value


@contextmanager
def entered(call: CallContext, device: torch.device, random: RandomStates) -> Generator[None]:
    """Establishes a neutral baseline and the call's random state for Torch code, and restores the previous settings."""
    with ExitStack() as stack:
        stack.enter_context(torch.set_grad_enabled(False))
        if torch.is_autocast_enabled(device.type):
            stack.enter_context(torch.autocast(device.type, enabled=False))
        if call.random is not None:
            stack.enter_context(random.drawing(call.random))
        yield


__all__ = ["RandomStates", "entered"]
