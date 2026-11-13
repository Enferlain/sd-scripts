"""The adapter for PyTorch's ``cuda`` backend (D15).

It is the shared PyTorch base (:mod:`library.iro.devices.pytorch`) on the
``cuda`` device type: NVIDIA GPUs, and AMD GPUs on PyTorch's ROCm builds,
which present them as ``cuda`` (not run here, so untested). Features only
this backend has, such as CUDA graphs, stream priorities and memory pools,
are added here when a run first needs them, and reported in ``offers``.

Only a run whose prepared target selects a CUDA device imports this module.
"""

from library.iro.devices.pytorch import PyTorchDevice, Recorded


class Cuda(PyTorchDevice):
    """One CUDA device."""

    kind = "cuda"


__all__ = ["Cuda", "Recorded"]
