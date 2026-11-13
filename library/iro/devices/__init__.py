"""Device adapters: execution context, physical completion and capacity facts for one backend (D15, D21).

The host adapter is plain Python. Each PyTorch backend has its own adapter,
built on the shared base in ``pytorch.py`` and named by PyTorch's device
type; so far ``cuda.py``, the only one run here. Other backends need an
adapter naming their device type and are untested until someone runs them.
Those modules need Torch and are imported only by a run that uses them.
"""

from library.iro.devices.host import Host, Returned

__all__ = ["Host", "Returned"]
