import torch

from library.adapters.runtime.precision import (
    cast_bias_for_compute,
    cast_input_for_compute,
    needs_explicit_dtype_fallback,
    restore_output_dtype,
)


def test_cast_input_for_compute_preserves_input_under_autocast():
    inputs = torch.randn(2, 4, dtype=torch.bfloat16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        casted = cast_input_for_compute(inputs, torch.float32)

    assert casted.dtype == inputs.dtype
    assert casted.data_ptr() == inputs.data_ptr()


def test_restore_output_dtype_preserves_autocast_result_dtype():
    result = torch.randn(2, 4, dtype=torch.bfloat16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        restored = restore_output_dtype(result, torch.float64)

    assert restored.dtype == result.dtype
    assert restored.data_ptr() == result.data_ptr()


def test_cast_bias_for_compute_preserves_bias_under_autocast():
    bias = torch.randn(4, dtype=torch.bfloat16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        casted = cast_bias_for_compute(bias, torch.float32)

    assert casted is bias


def test_needs_explicit_dtype_fallback_ignores_mismatch_under_autocast():
    inputs = torch.randn(2, 4, dtype=torch.bfloat16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert needs_explicit_dtype_fallback(inputs, torch.float32, torch.float16) is False


def test_needs_explicit_dtype_fallback_requires_no_autocast_mismatch():
    inputs = torch.randn(2, 4, dtype=torch.float64)

    assert needs_explicit_dtype_fallback(inputs, torch.float32, torch.float32) is True
    assert needs_explicit_dtype_fallback(inputs, torch.float64, torch.float64) is False
