"""Module mode follows trainability (D20).

Module runtime mode is not unit state: validation runs alongside training,
so a run-wide switch into evaluation mode has no right moment. Instead each
route that training's preparation prepares carries a held setup that sets its
participant's mode for each call through it (D16): train mode for the
modules of the units the call is annotated with, the units whose losses it
reaches from their trained state (:mod:`library.training.units.derivation`),
and evaluation mode for the rest of the participant. A frozen network on the
gradient path between a trained module and the loss therefore keeps
evaluation mode, and its dropout and normalization statistics stay as they
are.

The setup is held: checkpointing recomputes the forward inside the later
backward, so the mode holds while the call runs and while its retained
results keep its leases, and a call that needs another mode waits only then
(D16, D20). Its key is the units a call serves that train this participant,
which training also declares for each call at preparation, so a wait inside
one activity is refused before the run.

Modules are named by path within the participant, the empty path being the
whole participant, so the same setup holds for whatever handle a call
receives, the live module or a snapshot's copy, and survives sharding, which
keeps module structure. A mode is changed only where the previous call left
a different one: setting a mode walks the whole module tree, which a large
model would pay for on every call otherwise. It is not restored afterwards:
the next setup sets the mode it needs.
"""

import threading
import weakref
from collections.abc import Mapping

import torch

from library.iro.seams import Annotations
from library.training.units.derivation import UNITS

_applied: weakref.WeakKeyDictionary[torch.nn.Module, frozenset[str]] = weakref.WeakKeyDictionary()
_lock = threading.Lock()  # the record of what was applied is shared by every route on a participant


class ModuleModes:
    """A route's held setup: its participant's module mode for each call, from the units that train it.

    ``trains`` holds, for each unit that trains part of the participant, the
    paths of the modules it trains there.
    """

    name = "training.module_mode"
    version = "1"
    undoes = False  # it leaves its mode for the next setup to set
    changes_storage = False

    def __init__(self, trains: Mapping[str, tuple[str, ...]]):
        self.trains = dict(trains)

    def key(self, annotations: Annotations) -> tuple[str, ...]:
        """The units a call serves that train this participant, which fix the modules it runs in train mode, since units never share modules."""
        return tuple(sorted(unit for unit in dict(annotations).get(UNITS, ()) if unit in self.trains))

    def hold(self, annotations: Annotations, handle: object) -> None:
        if not isinstance(handle, torch.nn.Module):
            return
        paths = frozenset(path for unit in self.key(annotations) for path in self.trains[unit])
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

    def undo(self, handle: object) -> None:
        """Nothing: the next setup sets the mode it needs."""


__all__ = ["ModuleModes"]
