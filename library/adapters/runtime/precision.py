from __future__ import annotations

import torch
from torch import Tensor


def is_autocast_active() -> bool:
    if torch.is_autocast_enabled():
        return True
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:
        cpu_autocast_enabled = getattr(torch, "is_autocast_cpu_enabled", None)
        return bool(cpu_autocast_enabled()) if callable(cpu_autocast_enabled) else False


def cast_input_for_compute(x: Tensor, dtype: torch.dtype) -> Tensor:
    if is_autocast_active() or x.dtype == dtype:
        return x
    return x.to(dtype)


def cast_bias_for_compute(bias: Tensor | None, dtype: torch.dtype) -> Tensor | None:
    if bias is None or is_autocast_active() or bias.dtype == dtype:
        return bias
    return bias.to(dtype, non_blocking=True)


def restore_output_dtype(result: Tensor, original_dtype: torch.dtype) -> Tensor:
    if is_autocast_active() or result.dtype == original_dtype:
        return result
    return result.to(original_dtype)


def needs_explicit_dtype_fallback(x: Tensor, *branch_dtypes: torch.dtype) -> bool:
    if is_autocast_active():
        return False
    return any(dtype != x.dtype for dtype in branch_dtypes) or len(set(branch_dtypes)) > 1
