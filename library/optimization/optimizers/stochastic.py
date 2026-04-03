import torch


def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """Copy float32 values into a lower-precision tensor via stochastic rounding."""
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)
    target.copy_(result.view(dtype=torch.float32))
    del result
