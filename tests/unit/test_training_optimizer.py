"""
Unit tests for library/training/optimizer.py

Tests optimizer creation, scheduler setup, and config-based initialization.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import torch

from library.training.optimizer import (
    get_optimizer,
    is_schedulefree_optimizer,
    is_wrapper_optimizer,
    get_dummy_scheduler,
    parse_string_to_type,
)
from library.config.dataclasses.optimizer import OptimizerConfig, SchedulerConfig, LearningRatesConfig
from library.config.dataclasses.peft import PeftConfig
from library.config.dataclasses.dataset import DatasetConfig


# =============================================================================
# Basic Optimizer Creation Tests
# =============================================================================

@pytest.mark.training
@pytest.mark.unit
class TestGetOptimizer:
    """Test get_optimizer function with different configurations."""
    
    def test_default_adamw_optimizer(self, mock_model_parameters):
        """Test creating default AdamW optimizer."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            learning_rates=LearningRatesConfig(base=1e-4)
        )
        
        optimizer_name, optimizer_class_name, optimizer = get_optimizer(
            config, mock_model_parameters
        )
        
        assert optimizer is not None
        assert "adamw" in optimizer_name.lower()
        assert isinstance(optimizer, torch.optim.Optimizer)
        
    def test_adamw8bit_optimizer(self, mock_model_parameters):
        """Test creating AdamW8bit optimizer."""
        pytest.importorskip("bitsandbytes")
        
        config = OptimizerConfig(
            optimizer_type="AdamW8bit",
            learning_rates=LearningRatesConfig(base=1e-4)
        )
        
        optimizer_name, optimizer_class_name, optimizer = get_optimizer(
            config, mock_model_parameters
        )
        
        assert optimizer is not None
        assert "8bit" in optimizer_name.lower()
        
    def test_optimizer_with_custom_lr(self, mock_model_parameters):
        """Test optimizer with custom learning rate."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            learning_rates=LearningRatesConfig(base=5e-5)
        )
        
        _, _, optimizer = get_optimizer(config, mock_model_parameters)
        
        # Check that the learning rate is set correctly
        param_groups = optimizer.param_groups
        assert len(param_groups) > 0
        assert param_groups[0]['lr'] == 5e-5
        
    def test_optimizer_with_args(self, mock_model_parameters):
        """Test optimizer with additional arguments."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=["weight_decay=0.01", "betas=(0.9,0.999)"]
        )
        
        _, _, optimizer = get_optimizer(config, mock_model_parameters)
        
        assert optimizer is not None
        # Arguments should be parsed and applied
        param_groups = optimizer.param_groups
        assert param_groups[0]['weight_decay'] == 0.01
        
    def test_empty_optimizer_type_defaults_to_adamw(self, mock_model_parameters):
        """Test that empty optimizer_type defaults to AdamW."""
        config = OptimizerConfig(
            optimizer_type="",
            learning_rates=LearningRatesConfig(base=1e-4)
        )
        
        optimizer_name, _, optimizer = get_optimizer(config, mock_model_parameters)
        
        assert optimizer is not None
        assert "adamw" in optimizer_name.lower()
        
    def test_sgd_optimizer(self, mock_model_parameters):
        """Test creating SGD optimizer."""
        config = OptimizerConfig(
            optimizer_type="SGD",
            learning_rates=LearningRatesConfig(base=0.01)
        )
        
        optimizer_name, _, optimizer = get_optimizer(config, mock_model_parameters)
        
        assert optimizer is not None
        assert "sgd" in optimizer_name.lower()


# =============================================================================
# Optimizer Detection Tests
# =============================================================================

@pytest.mark.training
@pytest.mark.unit
class TestOptimizerDetection:
    """Test functions that detect optimizer types."""
    
    def test_is_schedulefree_optimizer_false(self, mock_model_parameters):
        """Test schedulefree detection for regular optimizer."""
        config = OptimizerConfig(optimizer_type="AdamW")
        _, _, optimizer = get_optimizer(config, mock_model_parameters)
        
        assert not is_schedulefree_optimizer(optimizer, config)
        
    def test_is_wrapper_optimizer_false(self):
        """Test wrapper detection for regular optimizer config."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            optimizer_schedulefree_wrapper=False
        )
        
        assert not is_wrapper_optimizer(config)
        
    def test_optimizer_schedulefree_wrapper_config(self):
        """Test that scheduler-free wrapper config is preserved."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            optimizer_schedulefree_wrapper=True
        )
        
        # Just verify the config preserves the value
        assert config.optimizer_schedulefree_wrapper == True


# =============================================================================
# Scheduler Tests
# =============================================================================

@pytest.mark.training
@pytest.mark.unit
class TestScheduler:
    """Test scheduler-related functions."""
    
    def test_get_dummy_scheduler(self, mock_model_parameters):
        """Test dummy scheduler creation."""
        config = OptimizerConfig(optimizer_type="AdamW", learning_rates=LearningRatesConfig(base=1e-4))
        _, _, optimizer = get_optimizer(config, mock_model_parameters)
        
        scheduler = get_dummy_scheduler(optimizer)
        
        assert scheduler is not None
        assert hasattr(scheduler, 'step')
        assert hasattr(scheduler, 'get_last_lr')
        
        # Test that dummy scheduler works
        initial_lr = scheduler.get_last_lr()
        scheduler.step()
        after_step_lr = scheduler.get_last_lr()
        
        # Dummy scheduler should not change LR
        assert initial_lr == after_step_lr
        
    def test_dummy_scheduler_preserves_lr(self, mock_model_parameters):
        """Test that dummy scheduler doesn't modify learning rate."""
        config = OptimizerConfig(optimizer_type="AdamW", learning_rates=LearningRatesConfig(base=3e-5))
        _, _, optimizer = get_optimizer(config, mock_model_parameters)
        
        initial_lr = optimizer.param_groups[0]['lr']
        scheduler = get_dummy_scheduler(optimizer)
        
        # Step multiple times
        for _ in range(5):
            scheduler.step()
            
        # LR should remain unchanged
        assert optimizer.param_groups[0]['lr'] == initial_lr


# =============================================================================
# Utility Function Tests
# =============================================================================

@pytest.mark.training
@pytest.mark.unit
class TestOptimizerUtils:
    """Test utility functions."""
    
    def test_parse_string_to_type_int(self):
        """Test parsing integer strings."""
        assert parse_string_to_type("42") == 42
        assert parse_string_to_type("0") == 0
        assert parse_string_to_type("-10") == -10
        
    def test_parse_string_to_type_float(self):
        """Test parsing float strings."""
        assert parse_string_to_type("3.14") == 3.14
        assert parse_string_to_type("1e-4") == 1e-4
        assert parse_string_to_type("-0.5") == -0.5
        assert parse_string_to_type("2.0e-6") == 2.0e-6
        
    def test_parse_string_to_type_string(self):
        """Test parsing non-numeric strings."""
        assert parse_string_to_type("hello") == "hello"
        assert parse_string_to_type("True") == "True"  # Returns as string if not matched
        
    def test_parse_string_to_type_escaped(self):
        """Test parsing returns string for complex expressions."""
        # parse_string_to_type returns strings for non-numeric values
        result = parse_string_to_type("(0.9, 0.999)")
        assert isinstance(result, str)
        
    def test_parse_string_to_type_bool_str(self):
        """Test parsing boolean-like strings."""
        result = parse_string_to_type("True")
        assert isinstance(result, str)


# =============================================================================
# Integration Tests with OptimizerConfig
# =============================================================================

@pytest.mark.training
@pytest.mark.integration
class TestOptimizerConfigIntegration:
    """Test optimizer creation with various OptimizerConfig settings."""
    
    def test_use_8bit_adam_flag(self):
        """Test that use_8bit_adam flag sets optimizer_type."""
        pytest.importorskip("bitsandbytes")
        
        config = OptimizerConfig(
            use_8bit_adam=True,
            learning_rates=LearningRatesConfig(base=1e-4)
        )
        
        # The __post_init__ should set optimizer_type to AdamW8bit
        assert config.optimizer_type == "AdamW8bit"
        
    def test_use_lion_optimizer_flag(self):
        """Test that use_lion_optimizer flag sets optimizer_type."""
        pytest.importorskip("lion_pytorch")
        
        config = OptimizerConfig(
            use_lion_optimizer=True,
            learning_rates=LearningRatesConfig(base=1e-4)
        )
        
        # The __post_init__ should set optimizer_type to Lion
        assert config.optimizer_type == "Lion"
        
    def test_lr_scheduler_constant(self):
        """Test that lr_scheduler config value is preserved."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            scheduler=SchedulerConfig(lr_scheduler="constant"),
            learning_rates=LearningRatesConfig(base=1e-4)
        )
        
        assert config.scheduler.lr_scheduler == "constant"
        
    def test_lr_warmup_steps(self):
        """Test that lr_warmup_steps config value is preserved."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            scheduler=SchedulerConfig(lr_scheduler="cosine", lr_warmup_steps=100),
            learning_rates=LearningRatesConfig(base=1e-4)
        )
        
        assert config.scheduler.lr_warmup_steps == 100
        assert config.scheduler.lr_scheduler == "cosine"
        
    def test_multiple_optimizer_args(self, mock_model_parameters):
        """Test multiple optimizer arguments parsing."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=[
                "weight_decay=0.01",
                "eps=1e-8",
                "betas=(0.9,0.999)"
            ]
        )
        
        _, _, optimizer = get_optimizer(config, mock_model_parameters)
        
        assert optimizer is not None
        param_groups = optimizer.param_groups[0]
        assert param_groups['weight_decay'] == 0.01
        assert param_groups['eps'] == 1e-8
        
    def test_max_grad_norm_preserved(self):
        """Test that max_grad_norm config is preserved."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            learning_rates=LearningRatesConfig(base=1e-4),
            max_grad_norm=0.5
        )
        
        assert config.max_grad_norm == 0.5
        
    def test_fused_backward_pass_flag(self):
        """Test fused_backward_pass flag is preserved."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            learning_rates=LearningRatesConfig(base=1e-4),
            fused_backward_pass=True
        )
        
        # Note: fused_backward_pass only works with Adafactor
        # This test just verifies the config preserves the value
        assert config.fused_backward_pass == True


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

@pytest.mark.training
@pytest.mark.unit
class TestOptimizerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_small_learning_rate(self, mock_model_parameters):
        """Test optimizer with very small learning rate."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            learning_rates=LearningRatesConfig(base=1e-10)
        )
        
        _, _, optimizer = get_optimizer(config, mock_model_parameters)
        
        assert optimizer.param_groups[0]['lr'] == 1e-10
        
    def test_large_learning_rate(self, mock_model_parameters):
        """Test optimizer with large learning rate."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            learning_rates=LearningRatesConfig(base=0.1)
        )
        
        _, _, optimizer = get_optimizer(config, mock_model_parameters)
        
        assert optimizer.param_groups[0]['lr'] == 0.1
        
    def test_zero_warmup_steps(self):
        """Test config with zero warmup steps."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            scheduler=SchedulerConfig(lr_scheduler="cosine", lr_warmup_steps=0)
        )
        
        assert config.scheduler.lr_warmup_steps == 0
