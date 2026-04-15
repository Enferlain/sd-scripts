import torch
from torch import nn

from library.models.sd3.loader import _materialize_meta_module


class DummyMetaTextModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3, device="meta")
        self.register_buffer("position_ids", torch.zeros((1, 4), dtype=torch.long, device="meta"), persistent=False)


def test_materialize_meta_module_handles_missing_nonpersistent_buffers():
    module = DummyMetaTextModule()
    state_dict = {
        "proj.weight": torch.randn(3, 4),
        "proj.bias": torch.randn(3),
    }

    missing_keys, unexpected_keys = _materialize_meta_module(module, state_dict, device="cpu")

    assert missing_keys == []
    assert unexpected_keys == []
    assert module.proj.weight.device.type == "cpu"
    assert module.position_ids.device.type == "cpu"
    assert not module.position_ids.is_meta
