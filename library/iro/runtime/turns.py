"""Device turns and admissions: one admission at a time per device, granted in request order (D15).

An admission covers the work it was taken for and, until the retained
results it produced are released, the rest of the body that holds them: a
window member's forward is in an access, its backward after it.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from library.iro.prepare import HOST
from library.iro.runtime.kernel import Kernel
from library.iro.runtime.state import Identity, Product


@dataclass
class Turn:
    """One device's turns: the admission holding it and those waiting, in request order (D15)."""

    holder: Identity | None = None
    depth: int = 0
    waiting: list[Identity] = field(default_factory=list)


@dataclass(eq=False)
class Admission:
    """A device admission in progress: it ends when its work has ended and its retained results are released."""

    device: str
    identity: Identity
    body: Identity
    products: set[int] = field(default_factory=set)
    work_ended: bool = False


class Turns:
    """Every device's turns and the admissions holding them."""

    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.by_device: dict[str, Turn] = defaultdict(Turn)
        self.admissions: dict[int, Admission] = {}
        self.admission_of: dict[Identity, Admission] = {}

    def admission_for(self, identity: Identity, device: str | None = None) -> Admission | None:
        """The innermost admission on the device covering work of this invocation."""
        found: Admission | None = None
        for admission in self.admission_of.values():
            if device is not None and admission.device != device:
                continue
            inside = identity[: len(admission.identity)] == admission.identity and not admission.work_ended
            after = bool(admission.products) and identity[: len(admission.body)] == admission.body
            if (inside or after) and (found is None or len(admission.identity) > len(found.identity)):
                found = admission
        return found

    def _ready(self, device: str, identity: Identity) -> bool:
        turn = self.by_device[device]
        if turn.holder is not None:
            return self.admission_for(identity, device) is not None
        return bool(turn.waiting) and turn.waiting[0] == identity

    async def queue(
        self, device: str | None, identity: Identity, also: Callable[[], bool] = lambda: True, keys: tuple[tuple, ...] = ()
    ) -> None:
        """Waits, in request order, until the device turn and ``also`` are both available; ``keys`` wake ``also``."""
        kernel = self.kernel
        if device is None:
            await kernel.until(lambda: kernel.closing or also(), *keys)
            kernel.admission_point()
            return
        turn = self.by_device[device]
        turn.waiting.append(identity)
        try:
            await kernel.until(lambda: kernel.closing or (self._ready(device, identity) and also()), ("turn", device), *keys)
        finally:
            turn.waiting.remove(identity)
            kernel.changed(("turn", device))
        kernel.admission_point()

    def grant(self, device: str, identity: Identity, body: Identity) -> None:
        turn = self.by_device[device]
        if turn.holder is None:
            turn.holder, turn.depth = identity, 1
        else:
            turn.depth += 1
        admission = Admission(device, identity, body)
        self.admission_of[identity] = admission
        self.admissions[id(admission)] = admission

    def work_ended(self, identity: Identity) -> None:
        admission = self.admission_of.get(identity)
        if admission is None:
            return
        admission.work_ended = True
        self._end_if_done(admission)

    def retained(self, product: Product, identity: Identity, device: str) -> None:
        """A retained result produced under an admission keeps it until its release."""
        admission = self.admission_for(identity, device) if device != HOST.name else self.admission_for(identity)
        if admission is not None:
            admission.products.add(id(product))

    def released(self, product: Product) -> None:
        for admission in list(self.admissions.values()):
            if id(product) in admission.products:
                admission.products.discard(id(product))
                self._end_if_done(admission)

    def _end_if_done(self, admission: Admission) -> None:
        if not admission.work_ended or admission.products or id(admission) not in self.admissions:
            return
        del self.admissions[id(admission)]
        if self.admission_of.get(admission.identity) is admission:
            del self.admission_of[admission.identity]
        turn = self.by_device[admission.device]
        turn.depth -= 1
        if turn.depth == 0:
            turn.holder = None
        self.kernel.changed(("turn", admission.device))


__all__ = ["Admission", "Turn", "Turns"]
