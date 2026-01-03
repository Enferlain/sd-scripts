"""
Unit tests for library/performance/deepspeed_utils.py

Tests the DeepSpeed configuration preparation and plugin creation functions.
"""

import pytest
from unittest.mock import patch, MagicMock

from library.config.dataclasses.data import LoaderConfig
from library.config.dataclasses.performance import DeepSpeedConfig, PrecisionConfig
from library.config.dataclasses.performance import PerformanceConfig
from library.config.dataclasses.training import TrainingConfig
from library.performance.deepspeed_utils import (
    prepare_deepspeed_config,
    prepare_deepspeed_plugin,
    prepare_deepspeed_model,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_deepspeed_config():
    """DeepSpeedConfig with deepspeed disabled."""
    return DeepSpeedConfig(deepspeed=False)


@pytest.fixture
def enabled_deepspeed_config():
    """DeepSpeedConfig with deepspeed enabled."""
    return DeepSpeedConfig(
        deepspeed=True,
        zero_stage=2,
        offload_optimizer_device=None,
        offload_param_device=None,
    )


@pytest.fixture
def performance_config_disabled():
    """PerformanceConfig with deepspeed disabled."""
    return PerformanceConfig(
        deepspeed=DeepSpeedConfig(deepspeed=False),
    )


@pytest.fixture
def deepspeed_config_disabled():
    """DeepSpeedConfig with deepspeed disabled."""
    return DeepSpeedConfig(deepspeed=False)


@pytest.fixture
def precision_config_default():
    """PrecisionConfig with default settings."""
    return PrecisionConfig(mixed_precision="no", full_fp16=False)


@pytest.fixture
def performance_config_enabled():
    """PerformanceConfig with deepspeed enabled."""
    return PerformanceConfig(
        deepspeed=DeepSpeedConfig(
            deepspeed=True,
            zero_stage=2,
            offload_optimizer_device=None,
            offload_param_device=None,
        ),
    )


@pytest.fixture
def deepspeed_config_enabled():
    """DeepSpeedConfig with deepspeed enabled."""
    return DeepSpeedConfig(
        deepspeed=True,
        zero_stage=2,
        offload_optimizer_device=None,
        offload_param_device=None,
    )


@pytest.fixture
def precision_config_fp16():
    """PrecisionConfig with fp16 mixed precision."""
    return PrecisionConfig(mixed_precision="fp16", full_fp16=False)


@pytest.fixture
def training_config():
    """Basic TrainingConfig for testing."""
    return TrainingConfig(
        gradient_accumulation_steps=4,
        train_batch_size=2,
    )


@pytest.fixture
def loader_config():
    """Basic LoaderConfig for testing."""
    return LoaderConfig(
        max_workers=8,
        persistent_workers=False,
    )


# =============================================================================
# Tests: prepare_deepspeed_config
# =============================================================================


class TestPrepareDeepspeedConfig:
    """Tests for prepare_deepspeed_config function."""

    def test_disabled_deepspeed_does_nothing(self, deepspeed_config_disabled, loader_config):
        """When deepspeed is disabled, loader_config should not be modified."""
        original_workers = loader_config.max_workers

        prepare_deepspeed_config(deepspeed_config_disabled, loader_config)

        # Should remain unchanged
        assert loader_config.max_workers == original_workers

    def test_enabled_deepspeed_sets_workers_to_one(self, deepspeed_config_enabled, loader_config):
        """When deepspeed is enabled, max_workers should be set to 1."""
        assert loader_config.max_workers == 8  # Initial value

        prepare_deepspeed_config(deepspeed_config_enabled, loader_config)

        assert loader_config.max_workers == 1

    def test_enabled_deepspeed_no_loader_config(self, deepspeed_config_enabled):
        """When loader_config is None, should not raise error."""
        # Should not raise
        prepare_deepspeed_config(deepspeed_config_enabled, None)

    def test_disabled_deepspeed_no_loader_config(self, deepspeed_config_disabled):
        """When both disabled and no loader_config, should not raise error."""
        # Should not raise
        prepare_deepspeed_config(deepspeed_config_disabled, None)


# =============================================================================
# Tests: prepare_deepspeed_plugin
# =============================================================================


class TestPrepareDeepspeedPlugin:
    """Tests for prepare_deepspeed_plugin function."""

    def test_disabled_returns_none(self, deepspeed_config_disabled, precision_config_default):
        """When deepspeed is disabled, should return None."""
        result = prepare_deepspeed_plugin(deepspeed_config_disabled, precision_config_default)
        assert result is None

    def test_disabled_with_training_config_returns_none(self, deepspeed_config_disabled, precision_config_default, training_config):
        """When deepspeed is disabled with training config, should return None."""
        result = prepare_deepspeed_plugin(deepspeed_config_disabled, precision_config_default, training_config)
        assert result is None

    @patch("library.performance.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_enabled_creates_plugin(self, mock_plugin_class, deepspeed_config_enabled, precision_config_fp16, training_config):
        """When deepspeed is enabled, should create and return a DeepSpeedPlugin."""
        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {"fp16": {}}
        mock_plugin_class.return_value = mock_plugin

        result = prepare_deepspeed_plugin(deepspeed_config_enabled, precision_config_fp16, training_config)

        assert result is mock_plugin
        mock_plugin_class.assert_called_once()

    @patch("library.performance.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_plugin_receives_correct_zero_stage(self, mock_plugin_class, precision_config_default, training_config):
        """Plugin should receive zero_stage from config."""
        ds_config = DeepSpeedConfig(deepspeed=True, zero_stage=3)

        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {}
        mock_plugin_class.return_value = mock_plugin

        prepare_deepspeed_plugin(ds_config, precision_config_default, training_config)

        call_kwargs = mock_plugin_class.call_args.kwargs
        assert call_kwargs["zero_stage"] == 3

    @patch("library.performance.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_plugin_receives_gradient_accumulation_from_training_config(
        self, mock_plugin_class, deepspeed_config_enabled, precision_config_fp16, training_config
    ):
        """Plugin should receive gradient_accumulation_steps from training_config."""
        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {"fp16": {}}
        mock_plugin_class.return_value = mock_plugin

        prepare_deepspeed_plugin(deepspeed_config_enabled, precision_config_fp16, training_config)

        call_kwargs = mock_plugin_class.call_args.kwargs
        assert call_kwargs["gradient_accumulation_steps"] == 4  # From fixture

    @patch("library.performance.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_plugin_defaults_gradient_accumulation_without_training_config(
        self, mock_plugin_class, deepspeed_config_enabled, precision_config_fp16
    ):
        """Without training_config, gradient_accumulation_steps should default to 1."""
        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {"fp16": {}}
        mock_plugin_class.return_value = mock_plugin

        prepare_deepspeed_plugin(deepspeed_config_enabled, precision_config_fp16, None)

        call_kwargs = mock_plugin_class.call_args.kwargs
        assert call_kwargs["gradient_accumulation_steps"] == 1

    @patch("library.performance.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_batch_size_set_in_config(self, mock_plugin_class, deepspeed_config_enabled, precision_config_fp16, training_config):
        """train_micro_batch_size_per_gpu should be set in deepspeed_config."""
        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {"fp16": {}}
        mock_plugin_class.return_value = mock_plugin

        prepare_deepspeed_plugin(deepspeed_config_enabled, precision_config_fp16, training_config)

        assert mock_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == 2  # From fixture

    @patch("library.performance.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_mixed_precision_applied(self, mock_plugin_class, deepspeed_config_enabled, precision_config_fp16, training_config):
        """set_mixed_precision should be called with the config value."""
        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {"fp16": {}}
        mock_plugin_class.return_value = mock_plugin

        prepare_deepspeed_plugin(deepspeed_config_enabled, precision_config_fp16, training_config)

        mock_plugin.set_mixed_precision.assert_called_once_with("fp16")

    @patch("library.performance.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_fp16_initial_scale_power_set(self, mock_plugin_class, training_config):
        """When mixed_precision is fp16, initial_scale_power should be set to 0."""
        ds_config = DeepSpeedConfig(deepspeed=True)
        prec_config = PrecisionConfig(mixed_precision="fp16")

        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {"fp16": {}}
        mock_plugin_class.return_value = mock_plugin

        prepare_deepspeed_plugin(ds_config, prec_config, training_config)

        assert mock_plugin.deepspeed_config["fp16"]["initial_scale_power"] == 0

    @patch("library.performance.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_full_fp16_with_cpu_offload_zero2(self, mock_plugin_class, training_config):
        """With full_fp16, cpu offload, and zero_stage=2, fp16_master_weights_and_grads should be True."""
        ds_config = DeepSpeedConfig(
            deepspeed=True,
            zero_stage=2,
            offload_optimizer_device="cpu",
        )
        prec_config = PrecisionConfig(mixed_precision="fp16", full_fp16=True)

        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {"fp16": {}}
        mock_plugin_class.return_value = mock_plugin

        prepare_deepspeed_plugin(ds_config, prec_config, training_config)

        assert mock_plugin.deepspeed_config["fp16"]["fp16_master_weights_and_grads"] is True

    @patch("library.performance.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_offload_params_passed_correctly(self, mock_plugin_class, precision_config_default, training_config):
        """Offload parameters should be passed to the plugin."""
        ds_config = DeepSpeedConfig(
            deepspeed=True,
            zero_stage=2,
            offload_optimizer_device="cpu",
            offload_optimizer_nvme_path="/nvme",
            offload_param_device="cpu",
            offload_param_nvme_path="/local_nvme",
            zero3_init_flag=True,
            zero3_save_16bit_model=True,
        )

        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {}
        mock_plugin_class.return_value = mock_plugin

        prepare_deepspeed_plugin(ds_config, precision_config_default, training_config)

        call_kwargs = mock_plugin_class.call_args.kwargs
        assert call_kwargs["offload_optimizer_device"] == "cpu"
        assert call_kwargs["offload_optimizer_nvme_path"] == "/nvme"
        assert call_kwargs["offload_param_device"] == "cpu"
        assert call_kwargs["offload_param_nvme_path"] == "/local_nvme"
        assert call_kwargs["zero3_init_flag"] is True
        assert call_kwargs["zero3_save_16bit_model"] is True


# =============================================================================
# Tests: prepare_deepspeed_model
# =============================================================================


class TestPrepareDeepspeedModel:
    """Tests for prepare_deepspeed_model function."""

    def test_filters_none_models(self):
        """None models should be filtered out."""
        import torch.nn as nn

        cfg = MagicMock()
        cfg.mixed_precision = "no"
        model1 = nn.Linear(10, 10)

        result = prepare_deepspeed_model(cfg, unet=model1, vae=None, text_encoder=None)

        # Should only have unet
        assert "unet" in result.models
        assert "vae" not in result.models
        assert "text_encoder" not in result.models

    def test_wraps_single_model(self):
        """Single model should be wrapped in ModuleDict."""
        import torch.nn as nn

        cfg = MagicMock()
        cfg.mixed_precision = "no"
        model = nn.Linear(10, 10)

        result = prepare_deepspeed_model(cfg, unet=model)

        assert isinstance(result.models, nn.ModuleDict)
        assert "unet" in result.models

    def test_wraps_multiple_models(self):
        """Multiple models should all be in ModuleDict."""
        import torch.nn as nn

        cfg = MagicMock()
        cfg.mixed_precision = "no"
        unet = nn.Linear(10, 10)
        vae = nn.Conv2d(3, 3, 3)

        result = prepare_deepspeed_model(cfg, unet=unet, vae=vae)

        assert "unet" in result.models
        assert "vae" in result.models

    def test_wraps_model_list(self):
        """List of models should be converted to ModuleList."""
        import torch.nn as nn

        cfg = MagicMock()
        cfg.mixed_precision = "no"
        encoders = [nn.Linear(10, 10), nn.Linear(10, 10)]

        result = prepare_deepspeed_model(cfg, text_encoders=encoders)

        assert "text_encoders" in result.models
        assert isinstance(result.models["text_encoders"], nn.ModuleList)
        assert len(result.models["text_encoders"]) == 2

    def test_get_models_returns_module_dict(self):
        """get_models() should return the ModuleDict."""
        import torch.nn as nn

        cfg = MagicMock()
        cfg.mixed_precision = "no"
        model = nn.Linear(10, 10)

        result = prepare_deepspeed_model(cfg, unet=model)

        assert result.get_models() is result.models

    def test_mixed_precision_wraps_forward(self):
        """With mixed precision, forward should be wrapped with autocast."""
        import torch
        import torch.nn as nn

        cfg = MagicMock()
        cfg.mixed_precision = "fp16"

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
                self._device = torch.device("cpu")

            @property
            def device(self):
                return self._device

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        result = prepare_deepspeed_model(cfg, unet=model)

        # The model's forward should now be wrapped
        # We can verify by calling forward (it shouldn't error)
        wrapped_model = result.models["unet"]
        x = torch.randn(2, 10)
        output = wrapped_model(x)
        assert output.shape == (2, 10)

    def test_no_mixed_precision_doesnt_wrap_forward(self):
        """Without mixed precision, forward should not be wrapped."""
        import torch.nn as nn

        cfg = MagicMock()
        cfg.mixed_precision = "no"
        model = nn.Linear(10, 10)
        original_forward = model.forward

        result = prepare_deepspeed_model(cfg, unet=model)

        # With no mixed precision, forward should be unchanged
        # (though it's still in the wrapper, the forward method itself is not modified)
        assert result.models["unet"] is model  # Same reference
