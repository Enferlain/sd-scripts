"""
Unit tests for library/optimizations/deepspeed_utils.py

Tests the DeepSpeed configuration preparation and plugin creation functions.
"""

import pytest
from unittest.mock import patch, MagicMock

from library.config.dataclasses.deepspeed import DeepSpeedConfig
from library.config.dataclasses.performance import PerformanceConfig
from library.config.dataclasses.training import TrainingConfig
from library.optimizations.deepspeed_utils import (
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
        mixed_precision="no",
        full_fp16=False,
        deepspeed=DeepSpeedConfig(deepspeed=False),
    )


@pytest.fixture
def performance_config_enabled():
    """PerformanceConfig with deepspeed enabled."""
    return PerformanceConfig(
        mixed_precision="fp16",
        full_fp16=False,
        deepspeed=DeepSpeedConfig(
            deepspeed=True,
            zero_stage=2,
            offload_optimizer_device=None,
            offload_param_device=None,
        ),
    )


@pytest.fixture
def training_config():
    """Basic TrainingConfig for testing."""
    return TrainingConfig(
        gradient_accumulation_steps=4,
        train_batch_size=2,
        max_data_loader_n_workers=8,
    )


# =============================================================================
# Tests: prepare_deepspeed_config
# =============================================================================

class TestPrepareDeepspeedConfig:
    """Tests for prepare_deepspeed_config function."""
    
    def test_disabled_deepspeed_does_nothing(self, performance_config_disabled, training_config):
        """When deepspeed is disabled, training_config should not be modified."""
        original_workers = training_config.max_data_loader_n_workers
        
        prepare_deepspeed_config(performance_config_disabled, training_config)
        
        # Should remain unchanged
        assert training_config.max_data_loader_n_workers == original_workers
    
    def test_enabled_deepspeed_sets_workers_to_one(self, performance_config_enabled, training_config):
        """When deepspeed is enabled, max_data_loader_n_workers should be set to 1."""
        assert training_config.max_data_loader_n_workers == 8  # Initial value
        
        prepare_deepspeed_config(performance_config_enabled, training_config)
        
        assert training_config.max_data_loader_n_workers == 1
    
    def test_enabled_deepspeed_no_training_config(self, performance_config_enabled):
        """When training_config is None, should not raise error."""
        # Should not raise
        prepare_deepspeed_config(performance_config_enabled, None)
    
    def test_disabled_deepspeed_no_training_config(self, performance_config_disabled):
        """When both disabled and no training_config, should not raise error."""
        # Should not raise
        prepare_deepspeed_config(performance_config_disabled, None)


# =============================================================================
# Tests: prepare_deepspeed_plugin
# =============================================================================

class TestPrepareDeepspeedPlugin:
    """Tests for prepare_deepspeed_plugin function."""
    
    def test_disabled_returns_none(self, performance_config_disabled):
        """When deepspeed is disabled, should return None."""
        result = prepare_deepspeed_plugin(performance_config_disabled)
        assert result is None
    
    def test_disabled_with_training_config_returns_none(self, performance_config_disabled, training_config):
        """When deepspeed is disabled with training config, should return None."""
        result = prepare_deepspeed_plugin(performance_config_disabled, training_config)
        assert result is None
    
    @patch("library.optimizations.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_enabled_creates_plugin(self, mock_plugin_class, performance_config_enabled, training_config):
        """When deepspeed is enabled, should create and return a DeepSpeedPlugin."""
        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {"fp16": {}}
        mock_plugin_class.return_value = mock_plugin
        
        result = prepare_deepspeed_plugin(performance_config_enabled, training_config)
        
        assert result is mock_plugin
        mock_plugin_class.assert_called_once()
    
    @patch("library.optimizations.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_plugin_receives_correct_zero_stage(self, mock_plugin_class, training_config):
        """Plugin should receive zero_stage from config."""
        perf_config = PerformanceConfig(
            mixed_precision="no",
            deepspeed=DeepSpeedConfig(
                deepspeed=True,
                zero_stage=3,
            ),
        )
        
        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {}
        mock_plugin_class.return_value = mock_plugin
        
        prepare_deepspeed_plugin(perf_config, training_config)
        
        call_kwargs = mock_plugin_class.call_args.kwargs
        assert call_kwargs["zero_stage"] == 3
    
    @patch("library.optimizations.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_plugin_receives_gradient_accumulation_from_training_config(
        self, mock_plugin_class, performance_config_enabled, training_config
    ):
        """Plugin should receive gradient_accumulation_steps from training_config."""
        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {"fp16": {}}
        mock_plugin_class.return_value = mock_plugin
        
        prepare_deepspeed_plugin(performance_config_enabled, training_config)
        
        call_kwargs = mock_plugin_class.call_args.kwargs
        assert call_kwargs["gradient_accumulation_steps"] == 4  # From fixture
    
    @patch("library.optimizations.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_plugin_defaults_gradient_accumulation_without_training_config(
        self, mock_plugin_class, performance_config_enabled
    ):
        """Without training_config, gradient_accumulation_steps should default to 1."""
        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {"fp16": {}}
        mock_plugin_class.return_value = mock_plugin
        
        prepare_deepspeed_plugin(performance_config_enabled, None)
        
        call_kwargs = mock_plugin_class.call_args.kwargs
        assert call_kwargs["gradient_accumulation_steps"] == 1
    
    @patch("library.optimizations.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_batch_size_set_in_config(self, mock_plugin_class, performance_config_enabled, training_config):
        """train_micro_batch_size_per_gpu should be set in deepspeed_config."""
        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {"fp16": {}}
        mock_plugin_class.return_value = mock_plugin
        
        prepare_deepspeed_plugin(performance_config_enabled, training_config)
        
        assert mock_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == 2  # From fixture
    
    @patch("library.optimizations.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_mixed_precision_applied(self, mock_plugin_class, performance_config_enabled, training_config):
        """set_mixed_precision should be called with the config value."""
        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {"fp16": {}}
        mock_plugin_class.return_value = mock_plugin
        
        prepare_deepspeed_plugin(performance_config_enabled, training_config)
        
        mock_plugin.set_mixed_precision.assert_called_once_with("fp16")
    
    @patch("library.optimizations.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_fp16_initial_scale_power_set(self, mock_plugin_class, training_config):
        """When mixed_precision is fp16, initial_scale_power should be set to 0."""
        perf_config = PerformanceConfig(
            mixed_precision="fp16",
            deepspeed=DeepSpeedConfig(deepspeed=True),
        )
        
        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {"fp16": {}}
        mock_plugin_class.return_value = mock_plugin
        
        prepare_deepspeed_plugin(perf_config, training_config)
        
        assert mock_plugin.deepspeed_config["fp16"]["initial_scale_power"] == 0
    
    @patch("library.optimizations.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_full_fp16_with_cpu_offload_zero2(self, mock_plugin_class, training_config):
        """With full_fp16, cpu offload, and zero_stage=2, fp16_master_weights_and_grads should be True."""
        perf_config = PerformanceConfig(
            mixed_precision="fp16",
            full_fp16=True,
            deepspeed=DeepSpeedConfig(
                deepspeed=True,
                zero_stage=2,
                offload_optimizer_device="cpu",
            ),
        )
        
        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {"fp16": {}}
        mock_plugin_class.return_value = mock_plugin
        
        prepare_deepspeed_plugin(perf_config, training_config)
        
        assert mock_plugin.deepspeed_config["fp16"]["fp16_master_weights_and_grads"] is True
    
    @patch("library.optimizations.deepspeed_utils.DeepSpeedPlugin")
    @patch.dict("sys.modules", {"deepspeed": MagicMock()})
    def test_offload_params_passed_correctly(self, mock_plugin_class, training_config):
        """Offload parameters should be passed to the plugin."""
        perf_config = PerformanceConfig(
            mixed_precision="no",
            deepspeed=DeepSpeedConfig(
                deepspeed=True,
                zero_stage=2,
                offload_optimizer_device="cpu",
                offload_optimizer_nvme_path="/nvme",
                offload_param_device="cpu",
                offload_param_nvme_path="/local_nvme",
                zero3_init_flag=True,
                zero3_save_16bit_model=True,
            ),
        )
        
        mock_plugin = MagicMock()
        mock_plugin.deepspeed_config = {}
        mock_plugin_class.return_value = mock_plugin
        
        prepare_deepspeed_plugin(perf_config, training_config)
        
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
        import torch
        import torch.nn as nn
        
        cfg = MagicMock()
        cfg.mixed_precision = "no"
        model = nn.Linear(10, 10)
        original_forward = model.forward
        
        result = prepare_deepspeed_model(cfg, unet=model)
        
        # With no mixed precision, forward should be unchanged
        # (though it's still in the wrapper, the forward method itself is not modified)
        assert result.models["unet"] is model  # Same reference
