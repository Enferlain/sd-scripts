import torch


def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """Copy values into a target tensor, using stochastic rounding for float32-like sources."""
    if source.dtype == torch.float64:
        source_fp32 = source.to(dtype=torch.float32)
    elif source.dtype == torch.float32:
        source_fp32 = source
    else:
        target.copy_(source.to(dtype=target.dtype))
        return

    result = torch.randint_like(source_fp32, dtype=torch.int32, low=0, high=(1 << 16))
    result.add_(source_fp32.view(dtype=torch.int32))
    result.bitwise_and_(-65536)
    target.copy_(result.view(dtype=torch.float32))
    del result
