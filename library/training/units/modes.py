"""Module mode follows trainability (D20).

Module runtime mode is not unit state: validation runs alongside training,
so a run-wide switch into evaluation mode has no right moment. Instead each
route that training's preparation prepares establishes its participant's
mode for each call through it (D15): train mode for the modules of the units
the call's derivatives go to, evaluation mode for the rest of the
participant. A frozen network on the gradient path between a trained module
and the loss therefore keeps evaluation mode, and its dropout and
normalization statistics stay as they are.

Modules are named by path within the participant, the empty path being the
whole participant, so the same setup holds for whatever handle a call
receives, the live module or a snapshot's copy, and survives sharding, which
keeps module structure. A mode is changed only where the previous call left
a different one: setting a mode walks the whole module tree, which a large
model would pay for on every call otherwise. It is not restored afterwards;
a call whose lease establishes a mode uses its participant alone, so no other
activity sees it.
"""

import threading
import weakref
from collections.abc import Mapping

import torch

from library.iro.seams import CallContext

_applied: weakref.WeakKeyDictionary[torch.nn.Module, frozenset[str]] = weakref.WeakKeyDictionary()
_lock = threading.Lock()  # the record of what was applied is shared by every route on a participant


class ModuleModes:
    """What a route establishes for each call: its participant's module mode, from the units that train it.

    ``trains`` holds, for each unit that trains part of the participant, the
    paths of the modules it trains there.
    """

    def __init__(self, trains: Mapping[str, tuple[str, ...]]):
        self.trains = dict(trains)

    def establish(self, call: CallContext, handle: object) -> None:
        if not isinstance(handle, torch.nn.Module):
            return
        paths = frozenset(path for unit in call.derivatives_to for path in self.trains.get(unit, ())) if call.grad else frozenset()
        whole = "" in paths
        with _lock:
            if handle.training == whole and _applied.get(handle) == paths:
                return
            _applied.pop(handle, None)  # until it is applied, the next call sets it again
        handle.train(whole)
        for path in sorted(paths - {""}):
            handle.get_submodule(path).train(True)
        with _lock:
            _applied[handle] = paths


__all__ = ["ModuleModes"]
