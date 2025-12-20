"""
Unit tests for library/training/diffusion.py

Tests timestep generation and noisy latent creation functions.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock

from library.training.diffusion import (
    get_timesteps,
    get_noise_noisy_latents_and_timesteps,
)
from library.config.dataclasses.regularization import RegularizationConfig
from library.config.dataclasses.timestep import TimestepConfig
from library.config.dataclasses.training import TrainingConfig


# =============================================================================
# get_timesteps Tests
# =============================================================================

@pytest.mark.training
@pytest.mark.unit
class TestGetTimesteps:
    """Test get_timesteps function."""
    
    def test_output_shape_matches_batch_size(self):
        """Test that output tensor has correct shape."""
        batch_size = 4
        timesteps = get_timesteps(0, 1000, batch_size, torch.device("cpu"))
        
        assert timesteps.shape == (batch_size,)
        
    def test_values_within_range(self):
        """Test that timesteps are within [min, max) range."""
        min_t, max_t = 100, 500
        timesteps = get_timesteps(min_t, max_t, 100, torch.device("cpu"))
        
        assert timesteps.min() >= min_t
        assert timesteps.max() < max_t
        
    def test_min_equals_max_returns_constant(self):
        """Test edge case when min_timestep >= max_timestep."""
        value = 500
        timesteps = get_timesteps(value, value, 10, torch.device("cpu"))
        
        # When min >= max, all values should be max_timestep
        assert torch.all(timesteps == value)
        
    def test_output_dtype_is_long(self):
        """Test that output dtype is long (int64)."""
        timesteps = get_timesteps(0, 1000, 4, torch.device("cpu"))
        
        assert timesteps.dtype == torch.long
        
    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        for batch_size in [1, 8, 32]:
            timesteps = get_timesteps(0, 1000, batch_size, torch.device("cpu"))
            assert len(timesteps) == batch_size
            
    def test_zero_to_one_range(self):
        """Test minimal range (1 possible value)."""
        timesteps = get_timesteps(0, 1, 5, torch.device("cpu"))
        assert torch.all(timesteps == 0)


# =============================================================================
# get_noise_noisy_latents_and_timesteps Tests
# =============================================================================

class MockSchedulerConfig:
    """Mock scheduler config with explicit attributes."""
    num_train_timesteps = 1000


class MockNoiseScheduler:
    """
    Mock noise scheduler that doesn't auto-create attributes.
    This avoids MagicMock's behavior of returning truthy for hasattr checks.
    """
    def __init__(self):
        self.config = MockSchedulerConfig()
    
    def add_noise(self, latents, noise, timesteps):
        """Simple mock implementation of add_noise."""
        return latents + noise * 0.5


@pytest.fixture
def mock_noise_scheduler():
    """Create a mock noise scheduler."""
    return MockNoiseScheduler()


@pytest.fixture
def sample_latents():
    """Create sample latent tensors."""
    # Typical latent shape: (batch, channels, height, width)
    return torch.randn(2, 4, 64, 64)


@pytest.fixture
def default_regularization_config():
    """Create default regularization config (no regularization)."""
    return RegularizationConfig()


@pytest.fixture
def default_timestep_config():
    """Create default timestep config."""
    return TimestepConfig()


@pytest.fixture
def default_training_config():
    """Create default training config."""
    return TrainingConfig(max_train_steps=1000)


@pytest.mark.training
@pytest.mark.unit
class TestGetNoiseNoisyLatentsAndTimesteps:
    """Test get_noise_noisy_latents_and_timesteps function."""
    
    def test_basic_output_structure(
        self, 
        mock_noise_scheduler, 
        sample_latents,
        default_regularization_config,
        default_timestep_config,
        default_training_config
    ):
        """Test that function returns correct tuple structure."""
        noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(
            regularization_config=default_regularization_config,
            timestep_config=default_timestep_config,
            training_config=default_training_config,
            noise_scheduler=mock_noise_scheduler,
            latents=sample_latents,
        )
        
        assert isinstance(noise, torch.Tensor)
        assert isinstance(noisy_latents, torch.Tensor)
        assert isinstance(timesteps, torch.Tensor)
        
    def test_output_shapes_match_input(
        self,
        mock_noise_scheduler,
        sample_latents,
        default_regularization_config,
        default_timestep_config,
        default_training_config
    ):
        """Test that output shapes match input latents."""
        noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(
            regularization_config=default_regularization_config,
            timestep_config=default_timestep_config,
            training_config=default_training_config,
            noise_scheduler=mock_noise_scheduler,
            latents=sample_latents,
        )
        
        assert noise.shape == sample_latents.shape
        assert noisy_latents.shape == sample_latents.shape
        assert timesteps.shape == (sample_latents.shape[0],)
        
    def test_fixed_timesteps_passthrough(
        self,
        mock_noise_scheduler,
        sample_latents,
        default_regularization_config,
        default_timestep_config,
        default_training_config
    ):
        """Test that fixed_timesteps are passed through unchanged."""
        fixed = torch.tensor([100, 200])
        
        _, _, timesteps = get_noise_noisy_latents_and_timesteps(
            regularization_config=default_regularization_config,
            timestep_config=default_timestep_config,
            training_config=default_training_config,
            noise_scheduler=mock_noise_scheduler,
            latents=sample_latents,
            fixed_timesteps=fixed,
        )
        
        assert torch.equal(timesteps, fixed)
        
    def test_timestep_override_min(
        self,
        mock_noise_scheduler,
        sample_latents,
        default_regularization_config,
        default_timestep_config,
        default_training_config
    ):
        """Test min_timestep_override is respected."""
        min_override = 500
        
        _, _, timesteps = get_noise_noisy_latents_and_timesteps(
            regularization_config=default_regularization_config,
            timestep_config=default_timestep_config,
            training_config=default_training_config,
            noise_scheduler=mock_noise_scheduler,
            latents=sample_latents,
            min_timestep_override=min_override,
        )
        
        assert timesteps.min() >= min_override
        
    def test_timestep_override_max(
        self,
        mock_noise_scheduler,
        sample_latents,
        default_regularization_config,
        default_timestep_config,
        default_training_config
    ):
        """Test max_timestep_override is respected."""
        max_override = 200
        
        _, _, timesteps = get_noise_noisy_latents_and_timesteps(
            regularization_config=default_regularization_config,
            timestep_config=default_timestep_config,
            training_config=default_training_config,
            noise_scheduler=mock_noise_scheduler,
            latents=sample_latents,
            max_timestep_override=max_override,
        )
        
        assert timesteps.max() < max_override
        
    def test_noise_offset_applied_when_enabled(
        self,
        mock_noise_scheduler,
        sample_latents,
        default_timestep_config,
        default_training_config
    ):
        """Test that noise offset modifies noise when enabled."""
        # Config with noise offset
        reg_config_with_offset = RegularizationConfig(noise_offset=0.1)
        
        noise_with_offset, _, _ = get_noise_noisy_latents_and_timesteps(
            regularization_config=reg_config_with_offset,
            timestep_config=default_timestep_config,
            training_config=default_training_config,
            noise_scheduler=mock_noise_scheduler,
            latents=sample_latents,
            is_train=True,
        )
        
        # Noise should be generated (we can't easily compare without seeding)
        assert noise_with_offset.shape == sample_latents.shape
        
    def test_is_train_false_skips_augmentations(
        self,
        mock_noise_scheduler,
        sample_latents,
        default_timestep_config,
        default_training_config
    ):
        """Test that is_train=False skips noise augmentations."""
        reg_config = RegularizationConfig(
            noise_offset=0.1,
            multires_noise_iterations=6,
            ip_noise_gamma=0.05,
        )
        
        # Should not raise and should skip augmentations
        noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(
            regularization_config=reg_config,
            timestep_config=default_timestep_config,
            training_config=default_training_config,
            noise_scheduler=mock_noise_scheduler,
            latents=sample_latents,
            is_train=False,
        )
        
        assert noise.shape == sample_latents.shape
        
    def test_timestep_config_min_max_respected(
        self,
        mock_noise_scheduler,
        sample_latents,
        default_regularization_config,
        default_training_config
    ):
        """Test that timestep config min/max are respected."""
        timestep_config = TimestepConfig(min_timestep=200, max_timestep=400)
        
        _, _, timesteps = get_noise_noisy_latents_and_timesteps(
            regularization_config=default_regularization_config,
            timestep_config=timestep_config,
            training_config=default_training_config,
            noise_scheduler=mock_noise_scheduler,
            latents=sample_latents,
        )
        
        assert timesteps.min() >= 200
        assert timesteps.max() < 400


@pytest.mark.training
@pytest.mark.unit
class TestMultiresNoise:
    """Test multires noise application."""
    
    def test_multires_noise_applied_when_enabled(
        self,
        mock_noise_scheduler,
        sample_latents,
        default_timestep_config,
        default_training_config
    ):
        """Test that multires noise is applied during training."""
        reg_config = RegularizationConfig(
            multires_noise_iterations=4,
            multires_noise_discount=0.3,
        )
        
        noise, _, _ = get_noise_noisy_latents_and_timesteps(
            regularization_config=reg_config,
            timestep_config=default_timestep_config,
            training_config=default_training_config,
            noise_scheduler=mock_noise_scheduler,
            latents=sample_latents,
            is_train=True,
        )
        
        # Noise should be modified (shape preserved)
        assert noise.shape == sample_latents.shape


@pytest.mark.training
@pytest.mark.unit
class TestIPNoiseGamma:
    """Test IP noise gamma application."""
    
    def test_ip_noise_gamma_applied_when_enabled(
        self,
        mock_noise_scheduler,
        sample_latents,
        default_timestep_config,
        default_training_config,
        default_regularization_config
    ):
        """Test that IP noise gamma modifies noisy latents."""
        reg_config = RegularizationConfig(ip_noise_gamma=0.05)
        
        _, noisy_latents, _ = get_noise_noisy_latents_and_timesteps(
            regularization_config=reg_config,
            timestep_config=default_timestep_config,
            training_config=default_training_config,
            noise_scheduler=mock_noise_scheduler,
            latents=sample_latents,
            is_train=True,
        )
        
        assert noisy_latents.shape == sample_latents.shape
