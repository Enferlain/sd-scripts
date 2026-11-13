"""The seams: plain host-side stand-ins satisfy each interface structurally."""

from contextlib import nullcontext
from dataclasses import dataclass, field

from library.iro.operation_contracts import AccessMode, VersionSelection
from library.iro.seams import Completion, Declaration, Device, Implementation, Lease, Owner, Use, Version


@dataclass(frozen=True)
class Seen:
    owner: str
    incarnation: int
    lineage: str
    ordinal: int


@dataclass
class Granted:
    use: Use
    version: Seen
    handle: object
    released: bool = False

    def release(self) -> None:
        self.released = True


@dataclass
class Counter:
    """An owner of one integer, updated in place."""

    value: int = 0
    ordinal: int = 0

    def resources(self):
        return ("counter",)

    def current(self, resource):
        return Seen("counter", 0, "main", self.ordinal)

    def lease(self, uses, reserved=None):
        return [Granted(use, (reserved or {}).get(use.resource, self.current(use.resource)), self) for use in uses]


class Done:
    def done(self) -> bool:
        return True

    def wait(self) -> None:
        return None


@dataclass
class Host:
    name: str = "host"
    work: list = field(default_factory=list)

    def context(self):
        return nullcontext()

    def completion(self):
        return Done()

    def capacity(self):
        return {}


DOUBLE = Declaration("double", "1", ("int",), ("int",), concurrency_safe=True)


@dataclass(frozen=True)
class Double:
    declaration: Declaration = DOUBLE

    def __call__(self, device, *arguments):
        with device.context():
            return (arguments[0] * 2,), device.completion()


def test_host_stand_ins_satisfy_the_seams() -> None:
    owner, device, implementation = Counter(), Host(), Double()
    assert isinstance(owner, Owner) and isinstance(device, Device) and isinstance(implementation, Implementation)
    assert isinstance(owner.current("counter"), Version)
    (lease,) = owner.lease([Use("counter", AccessMode.PROTECTED_CURRENT, VersionSelection.CURRENT_AT_START)])
    assert isinstance(lease, Lease) and lease.version.ordinal == 0
    results, completion = implementation(device, 21)
    assert isinstance(completion, Completion) and completion.done() and results == (42,)


def test_a_reserved_version_is_the_one_an_at_request_lease_observes() -> None:
    owner = Counter(ordinal=5)
    reserved = owner.current("counter")
    owner.ordinal = 7
    (lease,) = owner.lease([Use("counter", AccessMode.SNAPSHOT, VersionSelection.AT_REQUEST)], {"counter": reserved})
    assert lease.version.ordinal == 5
