"""The selected target, as preparation sees it (D15, D16).

A target names the devices a run may use, where device work goes and how
many worker threads run selected calls. It is a description: the device
adapters that realize it live in ``library/iro/devices`` and are chosen when
the prepared run is lowered. Preparation reads only these facts, so it stays
Torch-free.
"""

import re
from dataclasses import dataclass

from library.iro.seams import Device


@dataclass(frozen=True)
class DeviceFacts:
    """One device: its name, its kind and what its adapter offers (D15).

    The kind is ``host`` or a PyTorch device type, such as ``cuda`` or
    ``xpu``; a device of a PyTorch backend is named ``<kind>:<index>``.
    """

    name: str
    kind: str
    offers: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if (self.kind == "host") != (self.name == "host"):
            raise ValueError(f"only the host is named host and of kind host, not {self.name!r} of kind {self.kind!r}")
        if self.kind != "host" and not re.fullmatch(rf"{re.escape(self.kind)}:(0|[1-9]\d*)", self.name):
            raise ValueError(f"a {self.kind} device is named {self.kind}:<index>, not {self.name!r}")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.kind):
            raise ValueError(f"a device kind is host or a PyTorch device type, not {self.kind!r}")
        object.__setattr__(self, "offers", frozenset(self.offers))

    @classmethod
    def of(cls, device: Device) -> DeviceFacts:
        """The facts of a device adapter, as a target records them."""
        return cls(device.name, device.kind, device.offers)


HOST = DeviceFacts("host", "host")


@dataclass(frozen=True)
class Target:
    """What a run is prepared for.

    ``device`` is where device work goes unless ``placement`` places an
    activity, by name, on another device; without it, device work goes to the
    first device that is not the host. ``fuse`` permits fusing local chains
    of calls. ``workers`` bounds the threads that run selected calls; the
    runtime reads it when the prepared run is lowered.
    """

    devices: tuple[DeviceFacts, ...] = (HOST,)
    device: str | None = None
    placement: tuple[tuple[str, str], ...] = ()
    workers: int = 2
    fuse: bool = True

    def __post_init__(self) -> None:
        names = [d.name for d in self.devices]
        if HOST not in self.devices:
            raise ValueError("a target always has the host")
        if len(set(names)) != len(names):
            raise ValueError("device names are unique")
        for device in (self.device, *(device for _, device in self.placement)):
            if device is not None and (device not in names or device == HOST.name):
                raise ValueError(f"device work goes to a device of the target, not {device!r}")
        placed = [activity for activity, _ in self.placement]
        if len(set(placed)) != len(placed):
            raise ValueError("an activity is placed once")
        if type(self.workers) is not int or self.workers < 1:
            raise ValueError("a target has at least one worker")

    @property
    def device_work(self) -> str | None:
        """Where device work goes when nothing places it, or None on a host-only target."""
        if self.device is not None:
            return self.device
        return next((d.name for d in self.devices if d.kind != "host"), None)

    def describe(self) -> str:
        devices = ", ".join(f"{d.name} ({'; '.join([d.kind, *sorted(d.offers)])})" for d in self.devices)
        placed = "".join(f"; @{activity} on {device}" for activity, device in self.placement)
        return f"devices {devices}; device work on {self.device_work or 'none'}{placed}; workers {self.workers}; fuse {self.fuse}"


__all__ = ["HOST", "DeviceFacts", "Target"]
