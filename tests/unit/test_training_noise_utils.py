"""
Unit tests for library/training/noise_utils.py

Tests noise-related utility functions for diffusion training.
"""

import pytest
import torch

from library.training.noise_utils import (
    prepare_scheduler_for_custom_training,
    fix_noise_scheduler_betas_for_zero_terminal_snr,
    pyramid_noise_like,
    apply_noise_offset,
)


# =============================================================================
# Mock Scheduler for Testing
# =============================================================================

class MockNoiseScheduler:
    """Mock noise scheduler with realistic attributes for testing."""
    
    def __init__(self, num_timesteps=1000):
        # Create realistic beta schedule (linear)
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)


# =============================================================================
# prepare_scheduler_for_custom_training Tests
# =============================================================================

@pytest.mark.training
@pytest.mark.unit
class TestPrepareSchedulerForCustomTraining:
    """Test prepare_scheduler_for_custom_training function."""
    
    def test_adds_all_snr_attribute(self):
        """Test that all_snr attribute is added to scheduler."""
        scheduler = MockNoiseScheduler()
        assert not hasattr(scheduler, "all_snr")
        
        prepare_scheduler_for_custom_training(scheduler, torch.device("cpu"))
        
        assert hasattr(scheduler, "all_snr")
        assert isinstance(scheduler.all_snr, torch.Tensor)
        
    def test_all_snr_shape_matches_timesteps(self):
        """Test that all_snr has correct shape."""
        num_timesteps = 1000
        scheduler = MockNoiseScheduler(num_timesteps=num_timesteps)
        
        prepare_scheduler_for_custom_training(scheduler, torch.device("cpu"))
        
        assert scheduler.all_snr.shape == (num_timesteps,)
        
    def test_all_snr_values_positive(self):
        """Test that SNR values are all positive."""
        scheduler = MockNoiseScheduler()
        
        prepare_scheduler_for_custom_training(scheduler, torch.device("cpu"))
        
        assert torch.all(scheduler.all_snr > 0)
        
    def test_all_snr_decreases_over_time(self):
        """Test that SNR generally decreases over timesteps (more noise)."""
        scheduler = MockNoiseScheduler()
        
        prepare_scheduler_for_custom_training(scheduler, torch.device("cpu"))
        
        # SNR should be higher at beginning (less noise) than end (more noise)
        assert scheduler.all_snr[0] > scheduler.all_snr[-1]
        
    def test_skips_if_already_prepared(self):
        """Test that function skips if all_snr already exists."""
        scheduler = MockNoiseScheduler()
        scheduler.all_snr = torch.tensor([1.0, 2.0, 3.0])  # Pre-set value
        
        prepare_scheduler_for_custom_training(scheduler, torch.device("cpu"))
        
        # Should not be overwritten
        assert torch.equal(scheduler.all_snr, torch.tensor([1.0, 2.0, 3.0]))


# =============================================================================
# fix_noise_scheduler_betas_for_zero_terminal_snr Tests
# =============================================================================

@pytest.mark.training
@pytest.mark.unit
class TestFixNoiseSchedulerBetasForZeroTerminalSNR:
    """Test fix_noise_scheduler_betas_for_zero_terminal_snr function."""
    
    def test_modifies_betas(self):
        """Test that betas are modified."""
        scheduler = MockNoiseScheduler()
        original_betas = scheduler.betas.clone()
        
        fix_noise_scheduler_betas_for_zero_terminal_snr(scheduler)
        
        # Betas should be different after fixing
        assert not torch.allclose(scheduler.betas, original_betas)
        
    def test_modifies_alphas(self):
        """Test that alphas are recalculated."""
        scheduler = MockNoiseScheduler()
        original_alphas = scheduler.alphas.clone()
        
        fix_noise_scheduler_betas_for_zero_terminal_snr(scheduler)
        
        assert not torch.allclose(scheduler.alphas, original_alphas)
        
    def test_modifies_alphas_cumprod(self):
        """Test that alphas_cumprod is recalculated."""
        scheduler = MockNoiseScheduler()
        original_alphas_cumprod = scheduler.alphas_cumprod.clone()
        
        fix_noise_scheduler_betas_for_zero_terminal_snr(scheduler)
        
        assert not torch.allclose(scheduler.alphas_cumprod, original_alphas_cumprod)
        
    def test_terminal_snr_approaches_zero(self):
        """Test that the terminal (last) alphas_cumprod approaches zero."""
        scheduler = MockNoiseScheduler()
        
        fix_noise_scheduler_betas_for_zero_terminal_snr(scheduler)
        
        # The last alphas_cumprod should be very close to zero
        assert scheduler.alphas_cumprod[-1] < 1e-6
        
    def test_initial_alphas_cumprod_preserved(self):
        """Test that initial alphas_cumprod is approximately preserved."""
        scheduler = MockNoiseScheduler()
        original_first = scheduler.alphas_cumprod[0].item()
        
        fix_noise_scheduler_betas_for_zero_terminal_snr(scheduler)
        
        # First value should be close to original
        assert abs(scheduler.alphas_cumprod[0].item() - original_first) < 0.01


# =============================================================================
# pyramid_noise_like Tests
# =============================================================================

@pytest.mark.training
@pytest.mark.unit
class TestPyramidNoiseLike:
    """Test pyramid_noise_like function."""
    
    def test_output_shape_matches_input(self):
        """Test that output shape matches input."""
        noise = torch.randn(2, 4, 64, 64)
        
        result = pyramid_noise_like(noise, torch.device("cpu"), iterations=4)
        
        assert result.shape == noise.shape
        
    def test_output_roughly_unit_variance(self):
        """Test that output has roughly unit variance (normalized)."""
        noise = torch.randn(4, 4, 64, 64)
        
        result = pyramid_noise_like(noise, torch.device("cpu"), iterations=4)
        
        # Standard deviation should be close to 1 (normalized)
        assert 0.8 < result.std().item() < 1.2
        
    def test_different_iteration_counts(self):
        """Test with different iteration counts."""
        noise = torch.randn(2, 4, 32, 32)
        
        for iterations in [1, 3, 6]:
            result = pyramid_noise_like(noise.clone(), torch.device("cpu"), iterations=iterations)
            assert result.shape == noise.shape
            
    def test_different_discount_values(self):
        """Test with different discount values."""
        noise = torch.randn(2, 4, 32, 32)
        
        for discount in [0.2, 0.4, 0.6]:
            result = pyramid_noise_like(noise.clone(), torch.device("cpu"), iterations=4, discount=discount)
            assert result.shape == noise.shape
            
    def test_adds_pyramid_noise(self):
        """Test that pyramid noise is added (result differs from original)."""
        noise = torch.randn(2, 4, 32, 32)
        original_noise = noise.clone()
        
        result = pyramid_noise_like(noise, torch.device("cpu"), iterations=4)
        
        # Result should be different from original (multires noise added, then normalized)
        assert not torch.allclose(result, original_noise)


# =============================================================================
# apply_noise_offset Tests
# =============================================================================

@pytest.mark.training
@pytest.mark.unit
class TestApplyNoiseOffset:
    """Test apply_noise_offset function."""
    
    def test_returns_unchanged_when_offset_none(self):
        """Test that noise is returned unchanged when offset is None."""
        latents = torch.randn(2, 4, 64, 64)
        noise = torch.randn(2, 4, 64, 64)
        original_noise = noise.clone()
        
        result = apply_noise_offset(latents, noise, noise_offset=None, adaptive_noise_scale=None)
        
        assert torch.equal(result, original_noise)
        
    def test_modifies_noise_when_offset_applied(self):
        """Test that noise is modified when offset is applied."""
        latents = torch.randn(2, 4, 64, 64)
        noise = torch.randn(2, 4, 64, 64)
        original_noise = noise.clone()
        
        result = apply_noise_offset(latents, noise, noise_offset=0.1, adaptive_noise_scale=None)
        
        # Noise should be modified
        assert not torch.equal(result, original_noise)
        
    def test_output_shape_preserved(self):
        """Test that output shape matches input."""
        latents = torch.randn(2, 4, 64, 64)
        noise = torch.randn(2, 4, 64, 64)
        
        result = apply_noise_offset(latents, noise, noise_offset=0.1, adaptive_noise_scale=None)
        
        assert result.shape == noise.shape
        
    def test_adaptive_noise_scale_modifies_offset(self):
        """Test that adaptive noise scale affects the offset."""
        latents = torch.randn(2, 4, 64, 64)
        noise = torch.randn(2, 4, 64, 64)
        
        # With adaptive scale, the offset should depend on latent mean
        result = apply_noise_offset(latents, noise, noise_offset=0.1, adaptive_noise_scale=0.5)
        
        assert result.shape == noise.shape
        
    def test_different_offset_magnitudes(self):
        """Test with different offset magnitudes."""
        latents = torch.randn(2, 4, 64, 64)
        noise = torch.randn(2, 4, 64, 64)
        
        for offset in [0.01, 0.1, 0.5]:
            result = apply_noise_offset(latents, noise.clone(), noise_offset=offset, adaptive_noise_scale=None)
            assert result.shape == noise.shape
            
    def test_tensor_offset_value(self):
        """Test with tensor offset value (as used in training)."""
        latents = torch.randn(2, 4, 64, 64)
        noise = torch.randn(2, 4, 64, 64)
        offset_tensor = torch.tensor(0.1)
        
        result = apply_noise_offset(latents, noise, noise_offset=offset_tensor, adaptive_noise_scale=None)
        
        assert result.shape == noise.shape
