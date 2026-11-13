"""The host device adapter: work on the CPU, where a call's return is its completion (D15)."""

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext


class Returned:
    """The completion of host work: it is done when the call returns."""

    def done(self) -> bool:
        return True

    def wait(self) -> None:
        return None


class Host:
    """The host as a device: no context of its own and nothing known of its capacity."""

    name = "host"

    def context(self) -> AbstractContextManager[object]:
        return nullcontext()

    def completion(self) -> Returned:
        return Returned()

    def capacity(self) -> Mapping[str, int]:
        return {}


__all__ = ["Host", "Returned"]
