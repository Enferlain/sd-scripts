"""A call's settings on the host, through the shared Torch context (14.1): no GPU needed.

The CUDA adapter's own test (``test_cuda.py``) covers the same settings on a
device; this one keeps them covered where no GPU is present.
"""

import torch

from library.iro.devices import Host
from library.iro.seams import CallContext


def test_a_host_call_establishes_its_settings_and_restores_them() -> None:
    host = Host(seed=5)
    module = torch.nn.Dropout(0.5)
    module.train(False)
    seen = {}
    with torch.enable_grad():
        with host.context(CallContext(grad=False, module_mode="train", precision="bfloat16", random="a"), (), (module,)):
            seen["grad"] = torch.is_grad_enabled()
            seen["training"] = module.training
            seen["dtype"] = (torch.ones(4, 4) @ torch.ones(4, 4)).dtype
            first = torch.rand(4)
        assert torch.is_grad_enabled() and not module.training, "the previous settings are restored"
    outside = torch.get_rng_state()
    with host.context(CallContext(random="b"), (), ()):
        torch.rand(4)
    assert torch.equal(torch.get_rng_state(), outside), "a call's draws leave the thread's own random state alone"
    with host.context(CallContext(random="a"), (), ()):
        second = torch.rand(4)
    assert seen == {"grad": False, "training": True, "dtype": torch.bfloat16}
    with Host(seed=5).context(CallContext(random="a"), (), ()):
        again = torch.rand(4)
    assert torch.equal(first, again), "a caller's random state starts from the seed and its name"
    assert not torch.equal(first, second), "and continues from where that caller's last draw left it"
