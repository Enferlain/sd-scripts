"""
Unit tests for library/performance/fp8_optimization_utils.py

Tests FP8 quantization math and weight quantization logic.
"""

import pytest
import torch

from library.performance.fp8_optimization_utils import (
    calculate_fp8_maxval,
    quantize_fp8,
    quantize_weight,
)


# =============================================================================
# calculate_fp8_maxval Tests
# =============================================================================

@pytest.mark.unit
class TestCalculateFP8Maxval:
    """Test FP8 maximum value calculation."""
    
    def test_e4m3_default(self):
        """E4M3 (4-bit exponent, 3-bit mantissa) format."""
        result = calculate_fp8_maxval(exp_bits=4, mantissa_bits=3, sign_bits=1)
        
        expected = torch.finfo(torch.float8_e4m3fn).max
        assert result == expected
        assert result == 448.0  # Known E4M3 max
    
    def test_e5m2_format(self):
        """E5M2 (5-bit exponent, 2-bit mantissa) format."""
        result = calculate_fp8_maxval(exp_bits=5, mantissa_bits=2, sign_bits=1)
        
        expected = torch.finfo(torch.float8_e5m2).max
        assert result == expected
        assert result == 57344.0  # Known E5M2 max
    
    def test_invalid_format_raises(self):
        """Unsupported FP8 format should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported FP8 format"):
            calculate_fp8_maxval(exp_bits=3, mantissa_bits=4, sign_bits=1)
    
    def test_bits_must_sum_to_8(self):
        """Total bits must equal 8."""
        with pytest.raises(AssertionError):
            calculate_fp8_maxval(exp_bits=4, mantissa_bits=4, sign_bits=1)  # 9 bits


# =============================================================================
# quantize_fp8 Tests
# =============================================================================

@pytest.mark.unit
class TestQuantizeFP8:
    """Test FP8 quantization function."""
    
    def test_basic_quantization_e4m3(self):
        """Basic tensor quantization to E4M3."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scale = 1.0
        max_val = 448.0
        min_val = -448.0
        
        result = quantize_fp8(tensor, scale, torch.float8_e4m3fn, max_val, min_val)
        
        assert result.dtype == torch.float8_e4m3fn
        assert result.shape == tensor.shape
    
    def test_scaling_applied(self):
        """Scale factor should be applied before quantization."""
        tensor = torch.tensor([100.0, 200.0])
        scale = 100.0  # Divide by 100
        max_val = 448.0
        min_val = -448.0
        
        result = quantize_fp8(tensor, scale, torch.float8_e4m3fn, max_val, min_val)
        
        # After scaling: [1.0, 2.0]
        # Convert back to check approximate values
        result_float = result.to(torch.float32)
        assert torch.allclose(result_float, torch.tensor([1.0, 2.0]), atol=0.1)
    
    def test_clamping_to_max(self):
        """Values exceeding max should be clamped."""
        tensor = torch.tensor([1000.0, 500.0])  # Exceeds E4M3 max of 448
        scale = 1.0
        max_val = 448.0
        min_val = -448.0
        
        result = quantize_fp8(tensor, scale, torch.float8_e4m3fn, max_val, min_val)
        result_float = result.to(torch.float32)
        
        # Both should be clamped to max
        assert result_float[0] == 448.0
        assert result_float[1] == 448.0
    
    def test_handles_nan(self):
        """NaN values should be replaced with 0."""
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        scale = 1.0
        max_val = 448.0
        min_val = -448.0
        
        result = quantize_fp8(tensor, scale, torch.float8_e4m3fn, max_val, min_val)
        result_float = result.to(torch.float32)
        
        # NaN should become 0
        assert result_float[1] == 0.0
    
    def test_negative_values(self):
        """Negative values should be quantized correctly."""
        tensor = torch.tensor([-1.0, -10.0, -100.0])
        scale = 1.0
        max_val = 448.0
        min_val = -448.0
        
        result = quantize_fp8(tensor, scale, torch.float8_e4m3fn, max_val, min_val)
        result_float = result.to(torch.float32)
        
        assert (result_float < 0).all()


# =============================================================================
# quantize_weight Tests
# =============================================================================

@pytest.mark.unit
class TestQuantizeWeight:
    """Test weight quantization with different modes."""
    
    def test_per_tensor_quantization(self):
        """Per-tensor quantization mode."""
        tensor = torch.randn(64, 128)
        max_val = 448.0
        min_val = -448.0
        
        quantized, scale = quantize_weight(
            "test.weight", tensor, torch.float8_e4m3fn, max_val, min_val,
            quantization_mode="tensor"
        )
        
        assert quantized.dtype == torch.float8_e4m3fn
        assert quantized.shape == tensor.shape
        assert isinstance(scale, torch.Tensor)
    
    def test_per_channel_quantization(self):
        """Per-channel (row-wise) quantization mode."""
        tensor = torch.randn(64, 128)
        max_val = 448.0
        min_val = -448.0
        
        quantized, scale = quantize_weight(
            "test.weight", tensor, torch.float8_e4m3fn, max_val, min_val,
            quantization_mode="channel"
        )
        
        assert quantized.dtype == torch.float8_e4m3fn
        assert quantized.shape == tensor.shape
        # Scale should be per output channel: [64, 1]
        assert scale.shape == (64, 1)
    
    def test_block_quantization(self):
        """Block-wise quantization mode."""
        # in_features must be divisible by block_size
        tensor = torch.randn(64, 128)  # 128 % 64 == 0
        max_val = 448.0
        min_val = -448.0
        
        quantized, scale = quantize_weight(
            "test.weight", tensor, torch.float8_e4m3fn, max_val, min_val,
            quantization_mode="block", block_size=64
        )
        
        assert quantized.dtype == torch.float8_e4m3fn
        assert quantized.shape == tensor.shape
        # Scale shape: [out_features, num_blocks, 1] = [64, 2, 1]
        assert scale.shape == (64, 2, 1)
    
    def test_block_fallback_to_channel(self, caplog):
        """Block mode should fallback to channel if not divisible."""
        # 100 is not divisible by 64
        tensor = torch.randn(64, 100)
        max_val = 448.0
        min_val = -448.0
        
        import logging
        with caplog.at_level(logging.WARNING):
            quantized, scale = quantize_weight(
                "test.weight", tensor, torch.float8_e4m3fn, max_val, min_val,
                quantization_mode="block", block_size=64
            )
        
        assert quantized.shape == tensor.shape
        # Should fallback to per-channel
        assert scale.shape == (64, 1)
        assert "fallback to per-channel" in caplog.text
    
    def test_non_2d_fallback_to_tensor(self):
        """Non-2D tensors should fallback to per-tensor mode."""
        tensor = torch.randn(64, 3, 3)  # 3D tensor (like Conv kernel)
        max_val = 448.0
        min_val = -448.0
        
        quantized, scale = quantize_weight(
            "test.weight", tensor, torch.float8_e4m3fn, max_val, min_val,
            quantization_mode="block"
        )
        
        assert quantized.shape == tensor.shape
        # Should fallback to per-tensor, scale is scalar tensor
        assert scale.dim() == 0 or scale.numel() == 1
    
    def test_scale_is_float32(self):
        """Scale should always be float32."""
        tensor = torch.randn(64, 128)
        max_val = 448.0
        min_val = -448.0
        
        _, scale = quantize_weight(
            "test.weight", tensor, torch.float8_e4m3fn, max_val, min_val,
            quantization_mode="tensor"
        )
        
        assert scale.dtype == torch.float32


# =============================================================================
# apply_fp8_monkey_patch Tests
# =============================================================================

@pytest.mark.unit
class TestApplyFP8MonkeyPatch:
    """Test FP8 monkey patching logic."""
    
    def test_patches_linear_with_scale_weight(self):
        """Linear layers with scale_weight in state dict should be patched."""
        import torch.nn as nn
        from library.performance.fp8_optimization_utils import apply_fp8_monkey_patch
        
        # Create a simple model with a Linear layer
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        
        # Create an optimized state dict with scale_weight keys
        optimized_state_dict = {
            "0.weight": torch.randn(32, 64).to(torch.float8_e4m3fn),
            "0.bias": torch.randn(32),
            "0.scale_weight": torch.tensor([[1.0]] * 32),  # [32, 1] shape
            "2.weight": torch.randn(16, 32).to(torch.float8_e4m3fn),
            "2.bias": torch.randn(16),
            "2.scale_weight": torch.tensor([[1.0]] * 16),  # [16, 1] shape
        }
        
        # Apply monkey patch
        patched_model = apply_fp8_monkey_patch(model, optimized_state_dict)
        
        # Verify scale_weight buffers were registered
        assert hasattr(model[0], "scale_weight")
        assert hasattr(model[2], "scale_weight")
        
        # Verify the model is returned (same instance)
        assert patched_model is model
    
    def test_does_not_patch_layers_without_scale(self):
        """Layers without scale_weight should not be patched."""
        import torch.nn as nn
        from library.performance.fp8_optimization_utils import apply_fp8_monkey_patch
        
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.Linear(32, 16),
        )
        
        # Only first layer has scale_weight
        optimized_state_dict = {
            "0.weight": torch.randn(32, 64).to(torch.float8_e4m3fn),
            "0.scale_weight": torch.tensor([[1.0]] * 32),
            "1.weight": torch.randn(16, 32),  # No scale, not FP8
        }
        
        apply_fp8_monkey_patch(model, optimized_state_dict)
        
        # First layer patched
        assert hasattr(model[0], "scale_weight")
        # Second layer NOT patched
        assert not hasattr(model[1], "scale_weight")
    
    def test_patched_forward_runs(self):
        """Patched forward method should execute correctly."""
        import torch.nn as nn
        from library.performance.fp8_optimization_utils import apply_fp8_monkey_patch
        
        # Wrap in Sequential so named_modules() gives proper paths
        model = nn.Sequential(nn.Linear(64, 32))
        
        optimized_state_dict = {
            "0.weight": torch.randn(32, 64).to(torch.float8_e4m3fn),
            "0.bias": torch.randn(32),
            "0.scale_weight": torch.ones(32, 1),  # Per-channel scale
        }
        
        apply_fp8_monkey_patch(model, optimized_state_dict)
        model.load_state_dict(optimized_state_dict)
        
        # Run forward pass
        x = torch.randn(4, 64)
        output = model(x)
        
        assert output.shape == (4, 32)
        assert not torch.isnan(output).any()
    
    def test_scale_shape_determines_quantization_mode(self):
        """Scale shape determines dequantization behavior."""
        import torch.nn as nn
        from library.performance.fp8_optimization_utils import apply_fp8_monkey_patch
        
        # Wrap in Sequential so named_modules() gives proper paths
        model = nn.Sequential(nn.Linear(128, 64))
        
        # Block-wise scale: [out, num_blocks, 1]
        block_scale = torch.ones(64, 2, 1)  # 128 / 64 = 2 blocks
        
        optimized_state_dict = {
            "0.weight": torch.randn(64, 128).to(torch.float8_e4m3fn),
            "0.bias": torch.randn(64),
            "0.scale_weight": block_scale,
        }
        
        apply_fp8_monkey_patch(model, optimized_state_dict)
        
        # Verify scale buffer has correct shape
        assert model[0].scale_weight.shape == (64, 2, 1)


# =============================================================================
# fp8_linear_forward_patch Tests
# =============================================================================

@pytest.mark.unit
class TestFP8LinearForwardPatch:
    """Test the patched Linear forward method."""
    
    def test_dequantization_per_tensor(self):
        """Per-tensor dequantization should broadcast scale."""
        import torch.nn as nn
        from library.performance.fp8_optimization_utils import fp8_linear_forward_patch
        
        # Create a Linear layer with FP8 weights
        layer = nn.Linear(64, 32, bias=False)
        layer.weight.data = torch.randn(32, 64).to(torch.float8_e4m3fn)
        layer.scale_weight = torch.tensor(2.0)  # Scalar scale
        
        x = torch.randn(4, 64)
        output = fp8_linear_forward_patch(layer, x, use_scaled_mm=False)
        
        assert output.shape == (4, 32)
    
    def test_dequantization_per_channel(self):
        """Per-channel dequantization should use row-wise scales."""
        import torch.nn as nn
        from library.performance.fp8_optimization_utils import fp8_linear_forward_patch
        
        layer = nn.Linear(64, 32, bias=True)
        layer.weight.data = torch.randn(32, 64).to(torch.float8_e4m3fn)
        layer.scale_weight = torch.ones(32, 1)  # Per-channel
        
        x = torch.randn(4, 64)
        output = fp8_linear_forward_patch(layer, x, use_scaled_mm=False)
        
        assert output.shape == (4, 32)
    
    def test_dequantization_block_wise(self):
        """Block-wise dequantization should reshape weights."""
        import torch.nn as nn
        from library.performance.fp8_optimization_utils import fp8_linear_forward_patch
        
        layer = nn.Linear(128, 64, bias=False)
        layer.weight.data = torch.randn(64, 128).to(torch.float8_e4m3fn)
        layer.scale_weight = torch.ones(64, 2, 1)  # Block-wise: 2 blocks
        
        x = torch.randn(4, 128)
        output = fp8_linear_forward_patch(layer, x, use_scaled_mm=False)
        
        assert output.shape == (4, 64)

