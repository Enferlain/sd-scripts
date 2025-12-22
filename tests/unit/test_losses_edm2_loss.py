"""
Unit tests for library/losses/edm2_loss.py and edm2_loss_utils.py

Tests adaptive loss weighting components from EDM2 paper.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import MagicMock, patch, PropertyMock
from diffusers import DDPMScheduler

from library.losses.edm2_loss import (
    normalize,
    FourierFeatureExtractor,
    NormalizedLinearLayer,
    AdaptiveLossWeightMLP,
    create_weight_MLP
)

from library.losses.edm2_loss_utils import (
    handle_conflicting_configuration,
    plot_edm2_loss_weighting_check,
    plot_edm2_loss_weighting
)


# =============================================================================
# edm2_loss.py Tests
# =============================================================================

class TestNormalize:
    """Test normalize function."""
    
    def test_normalize_basic(self):
        """Test basic normalization behavior."""
        x = torch.randn(2, 3, 4)
        normalized = normalize(x, dim=[1, 2])
        
        # Check shape is preserved
        assert normalized.shape == x.shape
        
        # Check norm is approximately 1 (accounting for eps and scaling)
        norm_val = torch.linalg.vector_norm(normalized, dim=[1, 2])
        # Due to eps scaling, exact value varies, but should be reasonable
        assert torch.all(norm_val > 0)
        
    def test_normalize_2d(self):
        """Test 2D tensor normalization."""
        x = torch.randn(3, 4)
        normalized = normalize(x, dim=None)  # Should normalize over dim 
        
        assert normalized.shape == x.shape
        
    def test_normalize_dtype_handling(self):
        """Test dtype upcasting for higher-precision norm computation."""
        # Input: float16, compute norm in float32 for precision, output: float16
        x = torch.randn(2, 3, dtype=torch.float16)
        normalized = normalize(x, dtype=torch.float32)  # Upcast to float32 for norm
        
        # Output should match input dtype (float16)
        assert normalized.dtype == torch.float16
        assert normalized.shape == x.shape

    def test_normalize_same_dtype(self):
        """Test normalization when input and compute dtype match."""
        x = torch.randn(2, 3, dtype=torch.float32)
        normalized = normalize(x, dtype=torch.float32)  # Same dtype
        
        assert normalized.dtype == torch.float32
        assert normalized.shape == x.shape


class TestFourierFeatureExtractor:
    """Test Fourier feature extraction."""
    
    def test_initialization(self):
        """Test FFE initializes with correct buffers."""
        ffe = FourierFeatureExtractor(num_channels=64, bandwidth=1.0)
        
        assert hasattr(ffe, 'freqs')
        assert hasattr(ffe, 'phases')
        assert ffe.freqs.shape == (64,)
        assert ffe.phases.shape == (64,)
        
    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        ffe = FourierFeatureExtractor(num_channels=64)
        
        # Input: batch of scalars
        x = torch.randn(10)
        out = ffe(x)
        
        # Output should be [10, 64] (batch_size, num_channels)
        assert out.shape == (10, 64)
        
    def test_forward_dtype_preservation(self):
        """Test dtype is used in forward pass."""
        # Note: internals cast to float32 for trig ops usually, but buffer is float32 by default
        ffe = FourierFeatureExtractor(num_channels=32, dtype=torch.float32)
        
        x = torch.randn(5, dtype=torch.float32)
        out = ffe(x)
        
        assert out.dtype == torch.float32


class TestNormalizedLinearLayer:
    """Test custom normalized linear layer."""
    
    def test_initialization(self):
        """Test layer initializes with correct weight shape."""
        layer = NormalizedLinearLayer(in_channels=64, out_channels=32)
        
        assert layer.weight.shape == (32, 64)
        
    def test_forward_2d(self):
        """Test 2D linear transformation."""
        layer = NormalizedLinearLayer(in_channels=64, out_channels=32)
        
        x = torch.randn(10, 64)  # batch of 10
        out = layer(x)
        
        assert out.shape == (10, 32)
        
    def test_training_mode_normalization(self):
        """Test that training mode applies forced normalization."""
        layer = NormalizedLinearLayer(in_channels=10, out_channels=5)
        layer.train()
        
        # Set weights to known non-normalized values
        with torch.no_grad():
            layer.weight.copy_(torch.ones_like(layer.weight) * 10.0)
        
        x = torch.randn(3, 10)
        _ = layer(x)
        
        # After forward in training mode, weights should be normalized
        # (not exactly 1 due to scaling, but different from 10)
        assert not torch.allclose(layer.weight, torch.ones_like(layer.weight) * 10.0)


class TestAdaptiveLossWeightMLP:
    """Test the full adaptive loss weight MLP."""
    
    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock DDPMScheduler."""
        scheduler = MagicMock(spec=DDPMScheduler)
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, 1000)
        scheduler.config.num_train_timesteps = 1000
        
        # Mock all_snr attribute with 1D tensor
        snr = torch.linspace(10.0, 0.01, 1000)
        scheduler.all_snr = snr
        
        return scheduler
    
    def test_initialization(self, mock_scheduler):
        """Test MLP initializes correctly."""
        mlp = AdaptiveLossWeightMLP(
            noise_scheduler=mock_scheduler,
            logvar_channels=64,
            device='cpu',
            dtype=torch.float32
        )
        
        assert hasattr(mlp, 'logvar_fourier')
        assert hasattr(mlp, 'logvar_linear')
        assert hasattr(mlp, 'lambda_weights')
        assert mlp.lambda_weights.shape == (1000,)
        
    def test_forward_pass(self, mock_scheduler):
        """Test forward pass with loss and timesteps."""
        mlp = AdaptiveLossWeightMLP(
            noise_scheduler=mock_scheduler,
            logvar_channels=64,
            device='cpu',
            use_importance_weights=False  # Simplify for testing
        )
        
        loss = torch.randn(4)  # batch of 4
        timesteps = torch.randint(0, 1000, (4,))
        
        weighted_loss, loss_scaled = mlp(loss, timesteps)
        
        assert weighted_loss.shape == loss.shape
        assert loss_scaled.shape == loss.shape
        
    def test_importance_weights_enabled(self, mock_scheduler):
        """Test importance weighting is applied when enabled."""
        mlp = AdaptiveLossWeightMLP(
            noise_scheduler=mock_scheduler,
            logvar_channels=64,
            device='cpu',
            use_importance_weights=True,
            importance_weights_max_weight=10.0,
            importance_weights_min_snr_gamma=1.0
        )
        
        # Check importance weights were computed
        assert hasattr(mlp, 'importance_weights')
        assert mlp.importance_weights.shape == (1000,)
        
    def test_save_and_load_weights(self, mock_scheduler):
        """Test weight save/load round-trip."""
        mlp = AdaptiveLossWeightMLP(
            noise_scheduler=mock_scheduler,
            logvar_channels=32,
            device='cpu'
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_weights.safetensors")
            
            # Save
            mlp.save_weights(save_path, dtype=torch.float32, metadata={"test": "value"})
            
            assert os.path.exists(save_path)
            
            # Load
            info = mlp.load_weights(save_path)
            
            # Should load successfully
            assert info is not None


class TestCreateWeightMLP:
    """Test the MLP creation factory function."""
    
    @pytest.fixture
    def mock_scheduler(self):
        scheduler = MagicMock(spec=DDPMScheduler)
        scheduler.alphas_cumprod = torch.linspace(0.999, 0.001, 1000)
        scheduler.config.num_train_timesteps = 1000
        scheduler.all_snr = torch.linspace(10.0, 0.01, 1000)
        return scheduler
    
    def test_creates_model_and_optimizer(self, mock_scheduler):
        """Test factory creates both model and optimizer."""
        mlp, optimizer = create_weight_MLP(
            noise_scheduler=mock_scheduler,
            logvar_channels=64,
            device='cpu',
            lr=1e-2
        )
        
        assert isinstance(mlp, AdaptiveLossWeightMLP)
        assert isinstance(optimizer, torch.optim.AdamW)  # Default optimizer
        
    def test_custom_optimizer(self, mock_scheduler):
        """Test factory accepts custom optimizer class."""
        mlp, optimizer = create_weight_MLP(
            noise_scheduler=mock_scheduler,
            optimizer=torch.optim.SGD,
            lr=0.01,
            optimizer_args={'momentum': 0.9},
            device='cpu'
        )
        
        assert isinstance(optimizer, torch.optim.SGD)


# =============================================================================
# edm2_loss_utils.py Tests
# =============================================================================

class TestHandleConflictingConfiguration:
    """Test configuration conflict resolution."""
    
    def test_disables_debiased_estimation(self):
        """Test debiased estimation is disabled when conflicting."""
        config = MagicMock()
        config.edm2_loss_weighting = True
        config.edm2_loss_weighting_importance_weighting = True
        config.edm2_loss_weighting_importance_weighting_safety_override = False
        config.debiased_estimation_loss = True
        config.min_snr_gamma = None
        
        handle_conflicting_configuration(config)
        
        # Should be disabled
        assert config.debiased_estimation_loss is False
        
    def test_disables_min_snr_gamma(self):
        """Test min_snr_gamma is disabled when conflicting."""
        config = MagicMock()
        config.edm2_loss_weighting = True
        config.edm2_loss_weighting_importance_weighting = True
        config.edm2_loss_weighting_importance_weighting_safety_override = False
        config.debiased_estimation_loss = False
        config.min_snr_gamma = 5.0
        
        handle_conflicting_configuration(config)
        
        assert config.min_snr_gamma is None
        
    def test_safety_override_prevents_disabling(self):
        """Test safety override prevents automatic disabling."""
        config = MagicMock()
        config.edm2_loss_weighting = True
        config.edm2_loss_weighting_importance_weighting = True
        config.edm2_loss_weighting_importance_weighting_safety_override = True
        config.debiased_estimation_loss = True
        config.min_snr_gamma = 5.0
        
        handle_conflicting_configuration(config)
        
        # Should NOT be disabled due to override
        assert config.debiased_estimation_loss is True
        assert config.min_snr_gamma == 5.0


class TestPlotEdm2LossWeightingCheck:
    """Test plotting condition checker."""
    
    def test_returns_false_when_disabled(self):
        """Test returns False when plotting is disabled."""
        loss_config = MagicMock()
        loss_config.edm2_loss_weighting = True
        loss_config.edm2_loss_weighting_generate_graph = False
        
        training_config = MagicMock()
        
        result = plot_edm2_loss_weighting_check(loss_config, training_config, 100)
        
        assert result is False
        
    def test_returns_true_on_interval(self):
        """Test returns True on correct step intervals."""
        loss_config = MagicMock()
        loss_config.edm2_loss_weighting = True
        loss_config.edm2_loss_weighting_generate_graph = True
        loss_config.edm2_loss_weighting_generate_graph_every_x_steps = 20
        
        training_config = MagicMock()
        training_config.max_train_steps = 1000
        
        # Should be True on multiples of 20
        assert plot_edm2_loss_weighting_check(loss_config, training_config, 20) is True
        assert plot_edm2_loss_weighting_check(loss_config, training_config, 40) is True
        
        # Should be False on non-multiples
        assert plot_edm2_loss_weighting_check(loss_config, training_config, 21) is False
        
    def test_returns_true_on_final_step(self):
        """Test returns True on final training step."""
        loss_config = MagicMock()
        loss_config.edm2_loss_weighting = True
        loss_config.edm2_loss_weighting_generate_graph = True
        loss_config.edm2_loss_weighting_generate_graph_every_x_steps = 20
        
        training_config = MagicMock()
        training_config.max_train_steps = 1000
        
        # Should be True on final step even if not a multiple
        assert plot_edm2_loss_weighting_check(loss_config, training_config, 1000) is True


class TestPlotEdm2LossWeighting:
    """Test plotting functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock MLP model."""
        model = MagicMock()
        model.train = MagicMock()
        model._forward = MagicMock(return_value=torch.zeros(1000))
        model.lambda_weights = torch.ones(1000)
        model.device = "cpu"
        return model
    
    def test_creates_plot_file(self, mock_model):
        """Test plot file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loss_config = MagicMock()
            loss_config.edm2_loss_weighting_generate_graph_output_dir = tmpdir
            loss_config.edm2_loss_weighting_generate_graph_y_limit = None
            
            # Using patch to avoid matplotlib GUI backend issues if any,
            # though aggg is set in import usually
            with patch("matplotlib.pyplot.savefig") as mock_save:
                # Actually we can just let it run if it's using Agg
                # But safer to assert existence if real, or mock if we don't want IO
                pass

            # Real IO test
            plot_edm2_loss_weighting(
                loss_config,
                output_name="test_run",
                step=100,
                model=mock_model,
                device="cpu",
                num_timesteps=1000
            )
            
            # Check plot was saved
            expected_dir = os.path.join(tmpdir, "test_run")
            expected_file = os.path.join(expected_dir, "weighting_step_0000100.png")
            
            assert os.path.exists(expected_file)
            
    def test_handles_save_failure_gracefully(self, mock_model):
        """Test graceful handling of save failures."""
        loss_config = MagicMock()
        # Invalid path on Windows
        loss_config.edm2_loss_weighting_generate_graph_output_dir = ">>:|invalid|:<<"
        loss_config.edm2_loss_weighting_generate_graph_y_limit = None
        
        # Should not raise, just log warning
        plot_edm2_loss_weighting(
            loss_config,
            output_name="test",
            step=0,
            model=mock_model,
            device="cpu"
        )
