"""Precision realizations of a participant's routes (D15).

A participant declares how the calls through its routes compute, such as
``realize(@mixed_precision, "bfloat16")``, and preparation installs the
realization as each route's call setup: the runtime enters it around every
call through the route, on the call's worker, and leaves it when the call
returns. Only those calls run in it; a loss computed outside the route keeps
its own precision. PyTorch keeps autocast per thread, so entering it changes
no other worker's calls, and checkpointing carries it into its recompute.

The realization owns only how the calls compute. Loss scaling is a separate
choice with its own owner, the unit's unscaling mechanic (D20), as PyTorch
keeps autocast and gradient scaling separate.
"""

import warnings
from collections.abc import Sequence

import torch

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}
"""The dtypes mixed precision computes in, by the name a strategy writes."""


class MixedPrecision:
    """Mixed precision for the calls through a route: PyTorch's autocast in one dtype on the device the calls run on."""

    name = "mixed_precision"

    def __init__(self, device_type: str, dtype: torch.dtype):
        self.device_type = device_type
        self.dtype = dtype

    def entered(self, annotations: object, handles: Sequence[object]) -> torch.autocast:
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)


def mixed_precision(device: object, dtype: str) -> MixedPrecision:
    """The realization a strategy names as ``realize(@mixed_precision, "<dtype>")``, for the device adapter the calls run on.

    Refuses a dtype the device's autocast would not compute in, rather than
    let the calls run in another precision than the one declared.
    """
    found = DTYPES.get(dtype)
    if found is None:
        raise ValueError(f"mixed precision computes in {', '.join(sorted(DTYPES))}, not {dtype!r}")
    torch_device = getattr(device, "device", None)  # a PyTorch adapter's device; the host runs Torch on the CPU
    kind = torch_device.type if isinstance(torch_device, torch.device) else "cpu"
    if not torch.amp.is_autocast_available(kind):
        raise ValueError(f"PyTorch has no mixed precision on {kind}")
    if kind == "cuda" and found == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError(f"{torch_device} does not compute in bfloat16")
    with warnings.catch_warnings(record=True) as caught:  # autocast warns and turns itself off for a dtype the device does not support
        warnings.simplefilter("always")
        with torch.autocast(device_type=kind, dtype=found):
            enabled = torch.is_autocast_enabled(kind)
    if not enabled:
        reasons = "; ".join(str(w.message) for w in caught)
        raise ValueError(f"PyTorch's mixed precision on {kind} does not compute in {dtype}: {reasons}")
    return MixedPrecision(kind, found)


__all__ = ["DTYPES", "MixedPrecision", "mixed_precision"]
