"""
Unit tests for library/training/optimizer_utils.py

Tests optimizer creation, scheduler setup, and config-based initialization.
"""

import pytest
import torch
from diffusers.optimization import SchedulerType as DiffusersSchedulerType

from library.optimization.arguments import parse_key_value_args
from library.optimization.optimizer_utils import (
    is_schedulefree_optimizer,
    is_wrapper_optimizer,
    parse_string_to_type,
)
from library.optimization.scheduler import get_dummy_scheduler, get_scheduler_fix
from library.optimization.optimizer_factory import get_optimizer
from library.optimization.types import ParameterGroup, materialize_parameter_groups
from library.config.dataclasses.optimizer import OptimizerConfig, SchedulerConfig, LearningRatesConfig
from library.config.dataclasses.training import TrainingConfig


# =============================================================================
# Basic Optimizer Creation Tests
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestGetOptimizer:
    """Test get_optimizer function with different configurations."""

    def test_default_adamw_optimizer(self, mock_model_parameters):
        """Test creating default AdamW optimizer."""
        config = OptimizerConfig(optimizer_type="AdamW", learning_rates=LearningRatesConfig(base=1e-4))

        optimizer_name, optimizer_class_name, optimizer = get_optimizer(
            config, config.learning_rates, config.scheduler, mock_model_parameters
        )

        assert optimizer is not None
        assert "adamw" in optimizer_name.lower()
        assert isinstance(optimizer, torch.optim.Optimizer)

    def test_adamw8bit_optimizer(self, mock_model_parameters):
        """Test creating AdamW8bit optimizer."""
        pytest.importorskip("bitsandbytes")

        config = OptimizerConfig(optimizer_type="AdamW8bit", learning_rates=LearningRatesConfig(base=1e-4))

        optimizer_name, optimizer_class_name, optimizer = get_optimizer(
            config, config.learning_rates, config.scheduler, mock_model_parameters
        )

        assert optimizer is not None
        assert "8bit" in optimizer_name.lower()

    def test_optimizer_with_custom_lr(self, mock_model_parameters):
        """Test optimizer with custom learning rate."""
        config = OptimizerConfig(optimizer_type="AdamW", learning_rates=LearningRatesConfig(base=5e-5))

        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        # Check that the learning rate is set correctly
        param_groups = optimizer.param_groups
        assert len(param_groups) > 0
        assert param_groups[0]["lr"] == 5e-5

    def test_optimizer_with_args(self, mock_model_parameters):
        """Test optimizer with additional arguments."""
        config = OptimizerConfig(
            optimizer_type="AdamW", learning_rates=LearningRatesConfig(base=1e-4), optimizer_args=["weight_decay=0.01", "betas=(0.9,0.999)"]
        )

        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert optimizer is not None
        # Arguments should be parsed and applied
        param_groups = optimizer.param_groups
        assert param_groups[0]["weight_decay"] == 0.01

    def test_empty_optimizer_type_defaults_to_adamw(self, mock_model_parameters):
        """Test that empty optimizer_type defaults to AdamW."""
        config = OptimizerConfig(optimizer_type="", learning_rates=LearningRatesConfig(base=1e-4))

        optimizer_name, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert optimizer is not None
        assert "adamw" in optimizer_name.lower()

    def test_sgd_optimizer(self, mock_model_parameters):
        """Test creating SGD optimizer."""
        config = OptimizerConfig(optimizer_type="SGD", learning_rates=LearningRatesConfig(base=0.01))

        optimizer_name, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert optimizer is not None
        assert "sgd" in optimizer_name.lower()

    def test_sgdnesterov_defaults_momentum(self, mock_model_parameters):
        """Registry-backed SGDNesterov construction should still supply default momentum."""
        config = OptimizerConfig(optimizer_type="SGDNesterov", learning_rates=LearningRatesConfig(base=0.01))

        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_adafactor_registry_path_preserves_relative_step_behavior(self, mock_model_parameters):
        """Registry-backed Adafactor construction should keep relative-step preprocessing behavior."""
        config = OptimizerConfig(
            optimizer_type="Adafactor",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=["relative_step=True"],
        )

        optimizer_name, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert "adafactor" in optimizer_name.lower()
        assert optimizer.__class__.__name__ == "Adafactor"
        assert config.learning_rates.base == 0.0

    def test_fully_qualified_optimizer_keeps_fallback_path(self, mock_model_parameters):
        """Unregistered fully-qualified optimizers should still build through the shared fallback path."""
        config = OptimizerConfig(
            optimizer_type="torch.optim.AdamW",
            learning_rates=LearningRatesConfig(base=2e-4),
        )

        optimizer_name, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert optimizer.__class__.__name__ == "AdamW"
        assert optimizer.param_groups[0]["lr"] == 2e-4
        assert optimizer_name == "torch.optim.adamw.AdamW"


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
        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert not is_schedulefree_optimizer(optimizer, config)

    def test_is_wrapper_optimizer_false(self):
        """Test wrapper detection for regular optimizer config."""
        config = OptimizerConfig(optimizer_type="AdamW", optimizer_schedulefree_wrapper=False)

        assert not is_wrapper_optimizer(config)

    def test_optimizer_schedulefree_wrapper_config(self):
        """Test that scheduler-free wrapper config is preserved."""
        config = OptimizerConfig(optimizer_type="AdamW", optimizer_schedulefree_wrapper=True)

        # Just verify the config preserves the value
        assert config.optimizer_schedulefree_wrapper

    def test_is_wrapper_optimizer_true_for_schedulefree_wrapper_flag(self):
        """The legacy wrapper config should now participate in wrapper detection."""
        config = OptimizerConfig(optimizer_type="AdamW", optimizer_schedulefree_wrapper=True)

        assert is_wrapper_optimizer(config)

    def test_is_schedulefree_optimizer_true_for_schedulefree_wrapper_flag(self, mock_model_parameters):
        """The legacy wrapper config should count as schedule-free for train/eval handling."""
        pytest.importorskip("schedulefree")
        config = OptimizerConfig(optimizer_type="AdamW", optimizer_schedulefree_wrapper=True)
        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert is_schedulefree_optimizer(optimizer, config)

    def test_optimizer_schedulefree_wrapper_wraps_base_optimizer(self, mock_model_parameters):
        """The legacy wrapper config should create a real schedule-free wrapper around the base optimizer."""
        pytest.importorskip("schedulefree")
        config = OptimizerConfig(
            optimizer_type="AdamW",
            optimizer_schedulefree_wrapper=True,
            schedulefree_wrapper_args=["momentum=0.95"],
        )

        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert hasattr(optimizer, "base_optimizer")
        assert optimizer.base_optimizer.__class__.__name__ == "AdamW"

    def test_scheduler_uses_base_optimizer_for_schedulefree_wrapper_flag(self, mock_model_parameters):
        """Wrapped optimizers should still schedule their base optimizer."""
        pytest.importorskip("schedulefree")
        optimizer_config = OptimizerConfig(
            optimizer_type="AdamW",
            optimizer_schedulefree_wrapper=True,
            schedulefree_wrapper_args=["momentum=0.95"],
            scheduler=SchedulerConfig(lr_scheduler="constant_with_warmup", lr_warmup_steps=5),
        )
        training_config = TrainingConfig(max_train_steps=25)
        _, _, optimizer = get_optimizer(
            optimizer_config,
            optimizer_config.learning_rates,
            optimizer_config.scheduler,
            mock_model_parameters,
        )

        scheduler = get_scheduler_fix(
            optimizer_config.scheduler,
            optimizer_config,
            training_config,
            optimizer,
            num_processes=1,
        )

        assert scheduler.optimizer is optimizer.base_optimizer


# =============================================================================
# Scheduler Tests
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestScheduler:
    """Test scheduler-related functions."""

    @staticmethod
    def _build_optimizer_and_training_config(
        mock_model_parameters,
        *,
        optimizer_type: str = "AdamW",
        learning_rate: float = 1e-4,
        scheduler_config: SchedulerConfig | None = None,
        max_train_steps: int = 25,
    ):
        optimizer_config = OptimizerConfig(
            optimizer_type=optimizer_type,
            learning_rates=LearningRatesConfig(base=learning_rate),
            scheduler=scheduler_config or SchedulerConfig(),
        )
        _, _, optimizer = get_optimizer(
            optimizer_config,
            optimizer_config.learning_rates,
            optimizer_config.scheduler,
            mock_model_parameters,
        )
        training_config = TrainingConfig(max_train_steps=max_train_steps)
        return optimizer_config, training_config, optimizer

    def test_get_dummy_scheduler(self, mock_model_parameters):
        """Test dummy scheduler creation."""
        config = OptimizerConfig(optimizer_type="AdamW", learning_rates=LearningRatesConfig(base=1e-4))
        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        scheduler = get_dummy_scheduler(optimizer)

        assert scheduler is not None
        assert hasattr(scheduler, "step")
        assert hasattr(scheduler, "get_last_lr")

        # Test that dummy scheduler works
        initial_lr = scheduler.get_last_lr()
        scheduler.step()
        after_step_lr = scheduler.get_last_lr()

        # Dummy scheduler should not change LR
        assert initial_lr == after_step_lr

    def test_dummy_scheduler_preserves_lr(self, mock_model_parameters):
        """Test that dummy scheduler doesn't modify learning rate."""
        config = OptimizerConfig(optimizer_type="AdamW", learning_rates=LearningRatesConfig(base=3e-5))
        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        initial_lr = optimizer.param_groups[0]["lr"]
        scheduler = get_dummy_scheduler(optimizer)

        # Step multiple times
        for _ in range(5):
            scheduler.step()

        # LR should remain unchanged
        assert optimizer.param_groups[0]["lr"] == initial_lr

    def test_constant_with_warmup_constructs_through_registry(self, mock_model_parameters):
        """Registered transformers schedulers should build through the shared dispatch path."""
        optimizer_config, training_config, optimizer = self._build_optimizer_and_training_config(
            mock_model_parameters,
            scheduler_config=SchedulerConfig(lr_scheduler="constant_with_warmup", lr_warmup_steps=5),
        )

        scheduler = get_scheduler_fix(
            optimizer_config.scheduler,
            optimizer_config,
            training_config,
            optimizer,
            num_processes=1,
        )

        assert scheduler.optimizer is optimizer
        assert scheduler.__class__.__name__ == "LambdaLR"

    def test_piecewise_constant_constructs_through_registry(self, mock_model_parameters):
        """Registered diffusers schedulers should build through the shared dispatch path."""
        optimizer_config, training_config, optimizer = self._build_optimizer_and_training_config(
            mock_model_parameters,
            scheduler_config=SchedulerConfig(
                lr_scheduler=DiffusersSchedulerType.PIECEWISE_CONSTANT.value,
                lr_scheduler_args=["step_rules='1:10,0.1:20,0.01'"],
            ),
        )

        scheduler = get_scheduler_fix(
            optimizer_config.scheduler,
            optimizer_config,
            training_config,
            optimizer,
            num_processes=1,
        )

        assert scheduler.optimizer is optimizer
        assert scheduler.__class__.__name__ == "LambdaLR"

    def test_custom_scheduler_type_keeps_one_entrypoint(self, mock_model_parameters):
        """Custom scheduler classes should still route through the shared scheduler entrypoint."""
        optimizer_config, training_config, optimizer = self._build_optimizer_and_training_config(
            mock_model_parameters,
            scheduler_config=SchedulerConfig(
                lr_scheduler="constant",
                lr_scheduler_type="torch.optim.lr_scheduler.StepLR",
                lr_scheduler_args=["step_size=5", "gamma=0.1"],
            ),
        )

        scheduler = get_scheduler_fix(
            optimizer_config.scheduler,
            optimizer_config,
            training_config,
            optimizer,
            num_processes=1,
        )

        assert scheduler.__class__.__name__ == "StepLR"


# =============================================================================
# Utility Function Tests
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestOptimizerUtils:
    """Test utility functions."""

    def test_parse_key_value_args(self):
        """Shared key=value parsing preserves literal types."""
        parsed = parse_key_value_args(["weight_decay=0.01", "betas=(0.9, 0.999)", "name='adamw'"])

        assert parsed["weight_decay"] == 0.01
        assert parsed["betas"] == (0.9, 0.999)
        assert parsed["name"] == "adamw"

    def test_materialize_parameter_groups(self):
        """Typed parameter groups convert to legacy optimizer dicts."""
        param = torch.nn.Parameter(torch.randn(2, 2))
        groups = [ParameterGroup(params=[param], lr=1e-4, label="denoiser")]

        materialized = materialize_parameter_groups(groups)

        assert isinstance(materialized, list)
        assert materialized[0]["params"] == [param]
        assert materialized[0]["lr"] == 1e-4

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

        config = OptimizerConfig(use_8bit_adam=True, learning_rates=LearningRatesConfig(base=1e-4))

        # The __post_init__ should set optimizer_type to AdamW8bit
        assert config.optimizer_type == "AdamW8bit"

    def test_use_lion_optimizer_flag(self):
        """Test that use_lion_optimizer flag sets optimizer_type."""
        pytest.importorskip("lion_pytorch")

        config = OptimizerConfig(use_lion_optimizer=True, learning_rates=LearningRatesConfig(base=1e-4))

        # The __post_init__ should set optimizer_type to Lion
        assert config.optimizer_type == "Lion"

    def test_lr_scheduler_constant(self):
        """Test that lr_scheduler config value is preserved."""
        config = OptimizerConfig(
            optimizer_type="AdamW", scheduler=SchedulerConfig(lr_scheduler="constant"), learning_rates=LearningRatesConfig(base=1e-4)
        )

        assert config.scheduler.lr_scheduler == "constant"

    def test_lr_warmup_steps(self):
        """Test that lr_warmup_steps config value is preserved."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            scheduler=SchedulerConfig(lr_scheduler="cosine", lr_warmup_steps=100),
            learning_rates=LearningRatesConfig(base=1e-4),
        )

        assert config.scheduler.lr_warmup_steps == 100
        assert config.scheduler.lr_scheduler == "cosine"

    def test_multiple_optimizer_args(self, mock_model_parameters):
        """Test multiple optimizer arguments parsing."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            learning_rates=LearningRatesConfig(base=1e-4),
            optimizer_args=["weight_decay=0.01", "eps=1e-8", "betas=(0.9,0.999)"],
        )

        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert optimizer is not None
        param_groups = optimizer.param_groups[0]
        assert param_groups["weight_decay"] == 0.01
        assert param_groups["eps"] == 1e-8

    def test_max_grad_norm_preserved(self):
        """Test that max_grad_norm config is preserved."""
        config = OptimizerConfig(optimizer_type="AdamW", learning_rates=LearningRatesConfig(base=1e-4), max_grad_norm=0.5)

        assert config.max_grad_norm == 0.5

    def test_fused_backward_pass_flag(self):
        """Test fused_backward_pass flag is preserved."""
        config = OptimizerConfig(optimizer_type="AdamW", learning_rates=LearningRatesConfig(base=1e-4), fused_backward_pass=True)

        # Note: fused_backward_pass only works with Adafactor
        # This test just verifies the config preserves the value
        assert config.fused_backward_pass


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestOptimizerEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_learning_rate(self, mock_model_parameters):
        """Test optimizer with very small learning rate."""
        config = OptimizerConfig(optimizer_type="AdamW", learning_rates=LearningRatesConfig(base=1e-10))

        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert optimizer.param_groups[0]["lr"] == 1e-10

    def test_large_learning_rate(self, mock_model_parameters):
        """Test optimizer with large learning rate."""
        config = OptimizerConfig(optimizer_type="AdamW", learning_rates=LearningRatesConfig(base=0.1))

        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert optimizer.param_groups[0]["lr"] == 0.1

    def test_zero_warmup_steps(self):
        """Test config with zero warmup steps."""
        config = OptimizerConfig(optimizer_type="AdamW", scheduler=SchedulerConfig(lr_scheduler="cosine", lr_warmup_steps=0))

        assert config.scheduler.lr_warmup_steps == 0
