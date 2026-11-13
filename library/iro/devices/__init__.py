"""Device adapters: execution context, physical completion and capacity facts for one backend (D15, D21).

The host adapter is plain Python. Adapters that need Torch, such as PyTorch
CUDA, live in their own modules and are imported only by a run that uses them.
"""

from library.iro.devices.host import Host, Returned

__all__ = ["Host", "Returned"]
