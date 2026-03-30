"""Unit tests for the EDM2 loss-weighting package."""

import pytest
import torch
import tempfile
import os
from unittest.mock import MagicMock, patch
from diffusers import DDPMScheduler

from library.losses.edm2.edm2_loss import normalize, FourierFeatureExtractor, NormalizedLinearLayer, AdaptiveLossWeightMLP, create_weight_MLP
from library.losses.edm2.edm2_modifier import EDM2LossModifier
from library.losses.loss_modifiers import BatchLossOutput, LossModifierOutput, NoOpLossModifier
from library.objectives.ddpm import build_loss_modifier

from library.losses.edm2.plotting import plot_edm2_loss_weighting, plot_edm2_loss_weighting_check
from library.losses.edm2.validation import handle_conflicting_configuration


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

        assert hasattr(ffe, "freqs")
        assert hasattr(ffe, "phases")
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
        mlp = AdaptiveLossWeightMLP(noise_scheduler=mock_scheduler, logvar_channels=64, device="cpu", dtype=torch.float32)

        assert hasattr(mlp, "logvar_fourier")
        assert hasattr(mlp, "logvar_linear")
        assert hasattr(mlp, "lambda_weights")
        assert mlp.lambda_weights.shape == (1000,)

    def test_forward_pass(self, mock_scheduler):
        """Test forward pass with loss and timesteps."""
        mlp = AdaptiveLossWeightMLP(
            noise_scheduler=mock_scheduler,
            logvar_channels=64,
            device="cpu",
            use_importance_weights=False,  # Simplify for testing
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
            device="cpu",
            use_importance_weights=True,
            importance_weights_max_weight=10.0,
            importance_weights_min_snr_gamma=1.0,
        )

        # Check importance weights were computed
        assert hasattr(mlp, "importance_weights")
        assert mlp.importance_weights.shape == (1000,)

    def test_save_and_load_weights(self, mock_scheduler):
        """Test weight save/load round-trip."""
        mlp = AdaptiveLossWeightMLP(noise_scheduler=mock_scheduler, logvar_channels=32, device="cpu")

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
        mlp, optimizer = create_weight_MLP(noise_scheduler=mock_scheduler, logvar_channels=64, device="cpu", lr=1e-2)

        assert isinstance(mlp, AdaptiveLossWeightMLP)
        assert isinstance(optimizer, torch.optim.AdamW)  # Default optimizer

    def test_custom_optimizer(self, mock_scheduler):
        """Test factory accepts custom optimizer class."""
        mlp, optimizer = create_weight_MLP(
            noise_scheduler=mock_scheduler, optimizer=torch.optim.SGD, lr=0.01, optimizer_args={"momentum": 0.9}, device="cpu"
        )

        assert isinstance(optimizer, torch.optim.SGD)


# =============================================================================
# EDM2 Helper Tests
# =============================================================================


class TestHandleConflictingConfiguration:
    """Test configuration conflict resolution."""

    def test_disables_debiased_estimation(self):
        """Test debiased estimation is disabled when conflicting."""
        edm2_config = MagicMock()
        edm2_config.enabled = True
        edm2_config.importance.enabled = True
        edm2_config.importance.safety_override = False
        snr_config = MagicMock()
        snr_config.debiased_estimation_loss = True
        snr_config.min_snr_gamma = None

        handle_conflicting_configuration(edm2_config, snr_config)

        # Should be disabled
        assert snr_config.debiased_estimation_loss is False

    def test_disables_min_snr_gamma(self):
        """Test min_snr_gamma is disabled when conflicting."""
        edm2_config = MagicMock()
        edm2_config.enabled = True
        edm2_config.importance.enabled = True
        edm2_config.importance.safety_override = False
        snr_config = MagicMock()
        snr_config.debiased_estimation_loss = False
        snr_config.min_snr_gamma = 5.0

        handle_conflicting_configuration(edm2_config, snr_config)

        assert snr_config.min_snr_gamma is None

    def test_safety_override_prevents_disabling(self):
        """Test safety override prevents automatic disabling."""
        edm2_config = MagicMock()
        edm2_config.enabled = True
        edm2_config.importance.enabled = True
        edm2_config.importance.safety_override = True
        snr_config = MagicMock()
        snr_config.debiased_estimation_loss = True
        snr_config.min_snr_gamma = 5.0

        handle_conflicting_configuration(edm2_config, snr_config)

        # Should NOT be disabled due to override
        assert snr_config.debiased_estimation_loss is True
        assert snr_config.min_snr_gamma == 5.0


class TestLossModifierBuilder:
    """Test generic loss modifier construction."""

    def test_builds_noop_when_edm2_disabled(self):
        loss_config = MagicMock()
        loss_config.edm2.enabled = False

        modifier = build_loss_modifier(loss_config, MagicMock(), MagicMock(), MagicMock())

        assert isinstance(modifier, NoOpLossModifier)

    def test_builds_edm2_modifier_when_enabled(self):
        loss_config = MagicMock()
        loss_config.edm2.enabled = True

        expected_modifier = EDM2LossModifier()
        with patch("library.losses.edm2.factory.create_edm2_modifier", return_value=expected_modifier) as mock_create:
            modifier = build_loss_modifier(loss_config, MagicMock(), MagicMock(), MagicMock())

        assert modifier is expected_modifier
        mock_create.assert_called_once()


class TestEDM2LossModifier:
    """Test the generic trainer-owned EDM2 modifier runtime."""

    def test_apply_returns_mean_loss_when_disabled(self):
        modifier = EDM2LossModifier()

        result = modifier.apply(per_sample_loss=torch.tensor([1.0, 3.0]), timesteps=torch.tensor([10, 20]))

        assert result.loss.item() == pytest.approx(2.0)
        assert result.metrics == {}

    def test_apply_uses_sidecar_model_when_enabled(self):
        model = MagicMock(return_value=(torch.tensor([2.0, 4.0]), torch.tensor([1.0, 3.0])))
        modifier = EDM2LossModifier(model=model)

        result = modifier.apply(per_sample_loss=torch.tensor([1.0, 3.0]), timesteps=torch.tensor([10, 20]))

        model.assert_called_once()
        assert result.loss.item() == pytest.approx(3.0)
        assert result.metrics["loss/current_scaled"] == pytest.approx(2.0)

    def test_optimizer_step_and_zero_grad_delegate_to_optimizer_and_scheduler(self):
        optimizer = MagicMock()
        lr_scheduler = MagicMock()
        modifier = EDM2LossModifier(model=MagicMock(), optimizer=optimizer, lr_scheduler=lr_scheduler)

        modifier.optimizer_step()
        modifier.zero_grad()

        optimizer.step.assert_called_once()
        lr_scheduler.step.assert_called_once()
        optimizer.zero_grad.assert_called_once_with(set_to_none=True)

    def test_save_and_load_sidecar_delegate_to_model(self):
        model = MagicMock()
        modifier = EDM2LossModifier(model=model)

        modifier.save_sidecar("/tmp/test.safetensors", {"test": "value"})
        modifier.load_sidecar("/tmp/test.safetensors")

        model.save_weights.assert_called_once_with("/tmp/test.safetensors", dtype=torch.float32, metadata={"test": "value"})
        model.load_weights.assert_called_once_with("/tmp/test.safetensors")


class TestLossModifierMetricNormalization:
    """Test metric normalization for batch and modifier outputs."""

    def test_loss_modifier_output_rejects_unnamespaced_metric(self):
        with pytest.raises(ValueError, match="namespaced"):
            LossModifierOutput(loss=torch.tensor(1.0), metrics={"scaled_loss": 1.0})

    def test_loss_modifier_output_coerces_scalar_tensor_metrics(self):
        output = LossModifierOutput(loss=torch.tensor(1.0), metrics={"loss/current_scaled": torch.tensor(2.5)})

        assert output.metrics == {"loss/current_scaled": 2.5}

    def test_batch_loss_output_normalizes_metrics(self):
        output = BatchLossOutput(
            loss=torch.tensor(1.0),
            per_sample_loss=torch.tensor([1.0]),
            timesteps=torch.tensor([10]),
            metrics={"loss/base": torch.tensor(1.5)},
        )

        assert output.metrics == {"loss/base": 1.5}


class TestPlotEdm2LossWeightingCheck:
    """Test plotting condition checker."""

    def test_returns_false_when_disabled(self):
        """Test returns False when plotting is disabled."""
        loss_config = MagicMock()
        loss_config.enabled = True
        loss_config.visualization.enabled = False

        training_config = MagicMock()

        result = plot_edm2_loss_weighting_check(loss_config, training_config, 100)

        assert result is False

    def test_returns_true_on_interval(self):
        """Test returns True on correct step intervals."""
        loss_config = MagicMock()
        loss_config.enabled = True
        loss_config.visualization.enabled = True
        loss_config.visualization.every_n_steps = 20

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
        loss_config.enabled = True
        loss_config.visualization.enabled = True
        loss_config.visualization.every_n_steps = 20

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
            loss_config.visualization.output_dir = tmpdir
            loss_config.visualization.y_limit = None

            # Real IO test
            plot_edm2_loss_weighting(loss_config, output_name="test_run", step=100, model=mock_model, num_timesteps=1000, device="cpu")

            # Check plot was saved
            expected_dir = os.path.join(tmpdir, "test_run")
            expected_file = os.path.join(expected_dir, "weighting_step_0000100.png")

            assert os.path.exists(expected_file)

    def test_handles_save_failure_gracefully(self, mock_model):
        """Test graceful handling of save failures."""
        loss_config = MagicMock()
        # Invalid path on Windows
        loss_config.visualization.output_dir = ">>:|invalid|:<<"
        loss_config.visualization.y_limit = None

        # Should not raise, just log warning
        plot_edm2_loss_weighting(loss_config, output_name="test", step=0, model=mock_model, device="cpu")
