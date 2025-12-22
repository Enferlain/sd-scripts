"""
Unit tests for library/models/original_unet.py and sdxl_original_unet.py

Tests pure utility functions that don't require full model instantiation.
"""

import pytest
import torch
import torch.nn as nn

from library.models.original_unet import (
    get_timestep_embedding,
    get_parameter_dtype,
    get_parameter_device,
    resize_like,
)


# =============================================================================
# get_timestep_embedding Tests (SD 1.x/2.x version)
# =============================================================================

@pytest.mark.unit
class TestGetTimestepEmbedding:
    """Test sinusoidal timestep embedding generation."""
    
    def test_output_shape_even_dim(self):
        """Even embedding_dim should produce correct output shape."""
        timesteps = torch.tensor([0, 100, 500, 999])
        embedding_dim = 256
        
        result = get_timestep_embedding(timesteps, embedding_dim)
        
        assert result.shape == (4, 256)
    
    def test_output_shape_odd_dim(self):
        """Odd embedding_dim should zero-pad to correct shape."""
        timesteps = torch.tensor([0, 100])
        embedding_dim = 255  # Odd
        
        result = get_timestep_embedding(timesteps, embedding_dim)
        
        assert result.shape == (2, 255)
    
    def test_different_timesteps_produce_different_embeddings(self):
        """Different timesteps should produce different embeddings."""
        t0 = torch.tensor([0])
        t500 = torch.tensor([500])
        t999 = torch.tensor([999])
        
        emb0 = get_timestep_embedding(t0, 128)
        emb500 = get_timestep_embedding(t500, 128)
        emb999 = get_timestep_embedding(t999, 128)
        
        assert not torch.allclose(emb0, emb500)
        assert not torch.allclose(emb500, emb999)
        assert not torch.allclose(emb0, emb999)
    
    def test_flip_sin_to_cos(self):
        """flip_sin_to_cos should swap sin and cos portions."""
        timesteps = torch.tensor([100])
        
        emb_normal = get_timestep_embedding(timesteps, 128, flip_sin_to_cos=False)
        emb_flipped = get_timestep_embedding(timesteps, 128, flip_sin_to_cos=True)
        
        # First half of flipped should equal second half of normal
        # and second half of flipped should equal first half of normal
        assert torch.allclose(emb_flipped[:, :64], emb_normal[:, 64:])
        assert torch.allclose(emb_flipped[:, 64:], emb_normal[:, :64])
    
    def test_scale_parameter(self):
        """Scale parameter should multiply the embedding values."""
        timesteps = torch.tensor([100])
        
        emb_scale1 = get_timestep_embedding(timesteps, 128, scale=1.0)
        emb_scale2 = get_timestep_embedding(timesteps, 128, scale=2.0)
        
        # After scaling and before sin/cos, the relationship is complex
        # but the embeddings should be different
        assert not torch.allclose(emb_scale1, emb_scale2)
    
    def test_values_bounded(self):
        """Sinusoidal embeddings should be bounded by [-1, 1]."""
        timesteps = torch.tensor([0, 250, 500, 750, 999], dtype=torch.float32)
        
        result = get_timestep_embedding(timesteps, 256)
        
        assert result.min() >= -1.0
        assert result.max() <= 1.0
    
    def test_requires_1d_input(self):
        """Should assert on non-1D input."""
        timesteps = torch.tensor([[0, 100], [200, 300]])  # 2D
        
        with pytest.raises(AssertionError):
            get_timestep_embedding(timesteps, 128)


# =============================================================================
# get_parameter_dtype / get_parameter_device Tests
# =============================================================================

@pytest.mark.unit
class TestGetParameterDtypeDevice:
    """Test parameter introspection utilities."""
    
    def test_get_parameter_dtype_float32(self):
        """Should return float32 dtype for default Linear."""
        layer = nn.Linear(10, 10)
        
        result = get_parameter_dtype(layer)
        
        assert result == torch.float32
    
    def test_get_parameter_dtype_float16(self):
        """Should return float16 dtype for half-precision module."""
        layer = nn.Linear(10, 10).half()
        
        result = get_parameter_dtype(layer)
        
        assert result == torch.float16
    
    def test_get_parameter_device_cpu(self):
        """Should return CPU device for default module."""
        layer = nn.Linear(10, 10)
        
        result = get_parameter_device(layer)
        
        assert result.type == "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_parameter_device_cuda(self):
        """Should return CUDA device for GPU module."""
        layer = nn.Linear(10, 10).cuda()
        
        result = get_parameter_device(layer)
        
        assert result.type == "cuda"


# =============================================================================
# resize_like Tests
# =============================================================================

@pytest.mark.unit
class TestResizeLike:
    """Test interpolation utility."""
    
    def test_no_resize_when_same_size(self):
        """Should return unchanged tensor when sizes match."""
        x = torch.randn(1, 3, 64, 64)
        target = torch.randn(1, 3, 64, 64)
        
        result = resize_like(x, target)
        
        assert torch.allclose(result, x)
    
    def test_upsample(self):
        """Should upsample to match larger target."""
        x = torch.randn(1, 3, 32, 32)
        target = torch.randn(1, 3, 64, 64)
        
        result = resize_like(x, target)
        
        assert result.shape == (1, 3, 64, 64)
    
    def test_downsample(self):
        """Should downsample to match smaller target."""
        x = torch.randn(1, 3, 64, 64)
        target = torch.randn(1, 3, 32, 32)
        
        result = resize_like(x, target)
        
        assert result.shape == (1, 3, 32, 32)
    
    def test_nearest_mode(self):
        """Should work with nearest interpolation mode."""
        x = torch.randn(1, 3, 32, 32)
        target = torch.randn(1, 3, 64, 64)
        
        result = resize_like(x, target, mode="nearest")
        
        assert result.shape == (1, 3, 64, 64)
    
    def test_bfloat16_conversion(self):
        """Should handle bfloat16 by converting to float32 and back."""
        x = torch.randn(1, 3, 32, 32).to(torch.bfloat16)
        target = torch.randn(1, 3, 64, 64)
        
        result = resize_like(x, target)
        
        assert result.dtype == torch.bfloat16
        assert result.shape == (1, 3, 64, 64)
    
    def test_preserves_float32_dtype(self):
        """Should preserve float32 dtype."""
        x = torch.randn(1, 3, 32, 32)
        target = torch.randn(1, 3, 64, 64)
        
        result = resize_like(x, target)
        
        assert result.dtype == torch.float32
    
    def test_non_square_resize(self):
        """Should handle non-square aspect ratio changes."""
        x = torch.randn(1, 3, 32, 64)
        target = torch.randn(1, 3, 64, 128)
        
        result = resize_like(x, target)
        
        assert result.shape == (1, 3, 64, 128)
