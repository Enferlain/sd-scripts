"""
Unit tests for library/training/optimizer_utils.py

Tests optimizer creation, scheduler setup, and config-based initialization.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.optimization import SchedulerType as DiffusersSchedulerType

from library.optimization.arguments import parse_key_value_args
from library.optimization.grouping import (
    build_adapter_grouping,
    build_finetune_grouping,
    resolve_adapter_target_selection,
    resolve_finetune_trainability,
    resolve_learning_rate_groups,
)
from library.optimization.optimizer_utils import (
    apply_optimizer_runtime_mode,
    _load_optimizer_class_for_signature,
    is_schedulefree_optimizer,
    is_wrapper_optimizer,
    parse_string_to_type,
    resolve_optimizer_runtime_metadata,
)
from library.optimization.scheduler import get_dummy_scheduler, get_scheduler_fix
from library.optimization.optimizer_factory import get_optimizer
from library.optimization.types import (
    LogicalParameterGroup,
    OptimizationPlan,
    OptimizerRuntimeMetadata,
    ParameterGroup,
    SchedulerRuntimeMetadata,
    build_parameter_group,
    build_logical_parameter_group,
    materialize_parameter_groups,
)
from library.adapters.shared import AdapterTrainableParameterRef
from library.optimization.wrappers.schedulefree import ScheduleFreeWrapper
from library.config.dataclasses.optimizer import LearningRateGroupConfig, OptimizerConfig, SchedulerConfig, LearningRatesConfig
from library.config.dataclasses.training import TrainingConfig
from library.models.parameter_dump import NamedParameterComponentNames


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

    def test_adamw8bit_grouped_optimizer_uses_explicit_group_lr_when_base_is_null(self, mock_model_parameters):
        """bitsandbytes should initialize cleanly when grouped params already define their LRs."""
        pytest.importorskip("bitsandbytes")

        config = OptimizerConfig(optimizer_type="AdamW8bit", learning_rates=LearningRatesConfig(base=None))
        trainable_params = [{"params": mock_model_parameters, "lr": 5e-5}]

        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, trainable_params)

        assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)

    def test_optimizer_with_custom_lr(self, mock_model_parameters):
        """Test optimizer with custom learning rate."""
        config = OptimizerConfig(optimizer_type="AdamW", learning_rates=LearningRatesConfig(base=5e-5))

        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        # Check that the learning rate is set correctly
        param_groups = optimizer.param_groups
        assert len(param_groups) > 0
        assert param_groups[0]["lr"] == 5e-5

    @patch("library.optimization.optimizer_factory.load_target")
    def test_grouped_optimizer_omits_constructor_lr_when_base_is_null(self, mock_load_target, mock_model_parameters):
        """Explicit group LRs should be enough when the shared base fallback is null."""

        class DummyGroupedOptimizer(torch.optim.Optimizer):
            def __init__(self, params, **kwargs):
                self.received_kwargs = dict(kwargs)
                super().__init__(params, {"lr": kwargs.get("lr", 1e-3)})

        mock_load_target.return_value = DummyGroupedOptimizer

        config = OptimizerConfig(
            optimizer_type="custom.DummyGroupedOptimizer",
            learning_rates=LearningRatesConfig(base=None),
        )
        trainable_params = [{"params": mock_model_parameters, "lr": 5e-5}]

        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, trainable_params)

        assert "lr" not in optimizer.received_kwargs
        assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)

    @patch("library.optimization.optimizer_factory.materialize_parameter_groups")
    @patch("library.optimization.optimizer_factory.load_target")
    def test_fully_qualified_adammini_keeps_param_name_metadata_opt_in(
        self,
        mock_load_target,
        mock_materialize,
        mock_model_parameters,
    ):
        """Fully qualified AdamMini targets should still opt into safe param-name metadata."""

        class DummyOptimizer(torch.optim.Optimizer):
            def __init__(self, params, **kwargs):
                super().__init__(params, {"lr": kwargs.get("lr", 1e-3)})

        mock_load_target.return_value = DummyOptimizer
        mock_materialize.return_value = [{"params": mock_model_parameters, "lr": 1e-3, "param_names": ["toy.weight"]}]

        config = OptimizerConfig(
            optimizer_type="library.optimization.optimizers.adammini.AdamMini",
            learning_rates=LearningRatesConfig(base=1e-3),
        )

        get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert mock_materialize.call_args.kwargs["include_metadata_keys"] == {"param_names"}

    def test_grouped_optimizer_requires_explicit_group_lrs_when_base_is_null(self, mock_model_parameters):
        """Null base LR should fail fast if grouped optimizer params omit explicit LRs."""
        config = OptimizerConfig(optimizer_type="AdamW", learning_rates=LearningRatesConfig(base=None))
        trainable_params = [{"params": mock_model_parameters}]

        with pytest.raises(ValueError, match="base=null.*group 0 does not define an explicit lr"):
            get_optimizer(config, config.learning_rates, config.scheduler, trainable_params)

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

    @patch("library.optimization.optimizer_factory.prepare_windows_compiler_env_for_torch_compile")
    def test_torchao_optimizer_bootstraps_windows_compile_env(self, mock_prepare_compile_env, mock_model_parameters):
        """TorchAO-backed optimizers should prepare the Windows compiler env before upstream compile paths run."""
        pytest.importorskip("torchao.optim.adam")

        config = OptimizerConfig(optimizer_type="AdamW8bitAO", learning_rates=LearningRatesConfig(base=1e-4))

        optimizer_name, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert optimizer is not None
        assert "adamw8bitao" in optimizer_name.lower()
        mock_prepare_compile_env.assert_called_once()

    @pytest.mark.parametrize("optimizer_type", ["ADOPTAOScheduleFree", "CompassAO"])
    def test_torchao_repo_optimizers_tensorize_group_learning_rates(self, optimizer_type, mock_model_parameters):
        """TorchAO repo-owned optimizers should normalize explicit group LRs into tensors."""
        pytest.importorskip("torchao.optim.adam")

        config = OptimizerConfig(optimizer_type=optimizer_type, learning_rates=LearningRatesConfig(base=1e-4))
        trainable_params = [{"params": mock_model_parameters, "lr": 5e-5}]

        _, _, optimizer = get_optimizer(
            config,
            config.learning_rates,
            config.scheduler,
            trainable_params,
        )

        assert isinstance(optimizer.param_groups[0]["lr"], torch.Tensor)
        assert optimizer.param_groups[0]["lr"].item() == pytest.approx(5e-5)


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
        config = OptimizerConfig(optimizer_type="AdamW", optimizer_schedulefree_wrapper=True)
        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        assert is_schedulefree_optimizer(optimizer, config)

    def test_optimizer_schedulefree_wrapper_wraps_base_optimizer(self, mock_model_parameters):
        """The legacy wrapper config should create a real schedule-free wrapper around the base optimizer."""
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

    def test_resolve_optimizer_runtime_metadata_detects_train_eval_toggle(self, mock_model_parameters):
        """Schedule-free optimizer runtimes should advertise train/eval participation explicitly."""
        config = OptimizerConfig(
            optimizer_type="AdamW",
            optimizer_schedulefree_wrapper=True,
            schedulefree_wrapper_args=["momentum=0.95"],
        )
        _, _, optimizer = get_optimizer(config, config.learning_rates, config.scheduler, mock_model_parameters)

        runtime = resolve_optimizer_runtime_metadata(optimizer, config)

        assert runtime == OptimizerRuntimeMetadata(supports_train_eval_toggle=True)

    def test_apply_optimizer_runtime_mode_noops_when_plan_disables_toggle(self):
        """Runtime helper should safely no-op when the plan says no optimizer mode switch is needed."""
        optimizer = MagicMock()
        plan = OptimizationPlan(optimizer_runtime=OptimizerRuntimeMetadata(supports_train_eval_toggle=False))

        apply_optimizer_runtime_mode(optimizer, plan, training=False)
        apply_optimizer_runtime_mode(optimizer, plan, training=True)

        optimizer.eval.assert_not_called()
        optimizer.train.assert_not_called()


@pytest.mark.training
@pytest.mark.unit
class TestOptimizerSignatureLookup:
    """Test optimizer signature probing for default orthograd resolution."""

    def test_registered_repo_owned_torchao_optimizer_uses_registry_target(self):
        """Registered repo-owned optimizers should not be probed through torch.optim."""
        pytest.importorskip("torchao.optim.adam")

        config = OptimizerConfig(optimizer_type="AdamW8bitAO")

        optimizer_class = _load_optimizer_class_for_signature(config, {})

        assert optimizer_class is not None
        assert optimizer_class.__name__ == "AdamW8bitAO"
        assert optimizer_class.__module__ == "library.optimization.optimizers.adamw.adamw_low_bit"

    def test_wrapper_optimizer_uses_registered_base_target_for_signature(self):
        """Wrapper optimizers should resolve named base optimizers through the registry."""
        pytest.importorskip("bitsandbytes")

        config = OptimizerConfig(optimizer_type="CPUOffloadOptimizer")

        optimizer_class = _load_optimizer_class_for_signature(config, {"base_optimizer_type": "AdamW8bit"})

        assert optimizer_class is not None
        assert optimizer_class.__name__ == "AdamW8bit"


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
        optimizer_args: list[str] | None = None,
        max_train_steps: int = 25,
    ):
        optimizer_config = OptimizerConfig(
            optimizer_type=optimizer_type,
            learning_rates=LearningRatesConfig(base=learning_rate),
            optimizer_args=optimizer_args or [],
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

    def test_ranger21_rejects_external_scheduler_when_internal_scheduler_is_enabled(self, mock_model_parameters):
        """Ranger21 should fail fast instead of silently stacking internal and external LR schedules."""
        optimizer_config, training_config, optimizer = self._build_optimizer_and_training_config(
            mock_model_parameters,
            optimizer_type="Ranger21",
            optimizer_args=["num_iterations=25"],
            scheduler_config=SchedulerConfig(lr_scheduler="cosine", lr_warmup_steps=5),
        )

        with pytest.raises(ValueError, match="Ranger21 manages learning-rate scheduling internally"):
            get_scheduler_fix(
                optimizer_config.scheduler,
                optimizer_config,
                training_config,
                optimizer,
                num_processes=1,
            )

    def test_ranger21_allows_external_scheduler_when_internal_scheduler_is_disabled(self, mock_model_parameters):
        """Ranger21 should still support normal external schedulers when its internal LR schedule is disabled."""
        optimizer_config, training_config, optimizer = self._build_optimizer_and_training_config(
            mock_model_parameters,
            optimizer_type="Ranger21",
            optimizer_args=["num_iterations=25", "disable_lr_scheduler=True"],
            scheduler_config=SchedulerConfig(lr_scheduler="cosine", lr_warmup_steps=5),
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

    def test_plan_scheduler_runtime_none_uses_dummy_scheduler(self, mock_model_parameters):
        """Plan metadata should be able to declare that no external scheduler is owned."""
        optimizer_config, training_config, optimizer = self._build_optimizer_and_training_config(
            mock_model_parameters,
            scheduler_config=SchedulerConfig(lr_scheduler="constant_with_warmup", lr_warmup_steps=5),
        )
        optimization_plan = OptimizationPlan(
            scheduler_runtime=SchedulerRuntimeMetadata(mode="none", target="optimizer"),
        )

        scheduler = get_scheduler_fix(
            optimizer_config.scheduler,
            optimizer_config,
            training_config,
            optimizer,
            num_processes=1,
            optimization_plan=optimization_plan,
        )

        assert scheduler.optimizer is optimizer
        assert scheduler.__class__.__name__ == "DummyScheduler"

    def test_plan_scheduler_runtime_can_target_base_optimizer(self, mock_model_parameters):
        """Plan metadata should drive scheduler attachment onto the wrapped base optimizer."""
        base_optimizer = torch.optim.AdamW(mock_model_parameters, lr=1e-4)
        wrapped_optimizer = ScheduleFreeWrapper(base_optimizer, momentum=0.95)
        optimizer_config = OptimizerConfig(
            optimizer_type="AdamW",
            learning_rates=LearningRatesConfig(base=1e-4),
            scheduler=SchedulerConfig(lr_scheduler="constant_with_warmup", lr_warmup_steps=5),
        )
        training_config = TrainingConfig(max_train_steps=25)
        optimization_plan = OptimizationPlan(
            scheduler_runtime=SchedulerRuntimeMetadata(mode="external", target="base_optimizer"),
        )

        scheduler = get_scheduler_fix(
            optimizer_config.scheduler,
            optimizer_config,
            training_config,
            wrapped_optimizer,
            num_processes=1,
            optimization_plan=optimization_plan,
        )

        assert scheduler.optimizer is wrapped_optimizer.base_optimizer


# =============================================================================
# Utility Function Tests
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestOptimizerUtils:
    """Test utility functions."""

    def test_parse_key_value_args(self):
        """Shared key=value parsing preserves literal types."""
        parsed = parse_key_value_args(
            [
                "weight_decay=0.01",
                "betas=(0.9, 0.999)",
                "name='adamw'",
                "enabled=true",
                "offload=false",
                "missing=null",
            ]
        )

        assert parsed["weight_decay"] == 0.01
        assert parsed["betas"] == (0.9, 0.999)
        assert parsed["name"] == "adamw"
        assert parsed["enabled"] is True
        assert parsed["offload"] is False
        assert parsed["missing"] is None

    def test_materialize_parameter_groups(self):
        """Typed parameter groups convert to legacy optimizer dicts."""
        param = torch.nn.Parameter(torch.randn(2, 2))
        groups = [ParameterGroup(params=[param], lr=1e-4, label="denoiser")]

        materialized = materialize_parameter_groups(groups)

        assert isinstance(materialized, list)
        assert materialized[0]["params"] == [param]
        assert materialized[0]["lr"] == 1e-4

    def test_build_logical_parameter_group(self):
        """Logical groups preserve trainer-facing metadata independently of execution groups."""
        param = torch.nn.Parameter(torch.randn(2, 2))

        logical_group = build_logical_parameter_group(
            "denoiser",
            [param],
            lr=1e-4,
            label="denoiser",
            execution_group_indices=(0,),
        )

        assert isinstance(logical_group, LogicalParameterGroup)
        assert logical_group.metric_name == "denoiser"
        assert logical_group.parameter_count == 4
        assert logical_group.execution_group_indices == (0,)

    def test_optimization_plan_lr_descriptions(self):
        """Optimization plans expose stable LR-reporting names from logical groups."""
        plan = OptimizationPlan(
            logical_groups=[
                LogicalParameterGroup(key="denoiser", label="denoiser", params=[], execution_group_indices=(0,)),
                LogicalParameterGroup(key="text_encoder1", label="text_encoder1", params=[], execution_group_indices=(1,)),
            ]
        )

        assert plan.lr_descriptions == ["denoiser", "text_encoder1"]

    def test_optimization_plan_execution_groups_preserve_parameter_group_compatibility(self):
        """Execution-group naming should coexist with the older parameter-group accessor."""
        param = torch.nn.Parameter(torch.randn(2, 2))
        execution_group = ParameterGroup(params=[param], lr=1e-4, label="denoiser")
        plan = OptimizationPlan(execution_groups=[execution_group])

        assert plan.execution_groups == [execution_group]
        assert plan.parameter_groups == [execution_group]
        assert plan.materialize_execution_groups()[0]["lr"] == 1e-4

    def test_materialize_parameter_groups_keeps_metadata_out_of_generic_runtime_payload(self):
        """Execution-group metadata should stay out of generic optimizer payloads by default."""
        param = torch.nn.Parameter(torch.randn(2, 2))
        group = build_parameter_group(
            [param],
            lr=1e-4,
            label="denoiser",
            metadata={"param_names": ["unet.linear.weight"]},
            weight_decay=0.01,
        )

        materialized = materialize_parameter_groups([group])[0]

        assert materialized["params"] == [param]
        assert materialized["lr"] == pytest.approx(1e-4)
        assert materialized["weight_decay"] == pytest.approx(0.01)
        assert "param_names" not in materialized

    def test_materialize_parameter_groups_can_include_selected_safe_metadata(self):
        """Optimizer-specific adapters can opt into safe string metadata when needed."""
        param = torch.nn.Parameter(torch.randn(2, 2))
        group = build_parameter_group(
            [param],
            lr=1e-4,
            label="denoiser",
            metadata={"param_names": ["unet.linear.weight"], "debug_refs": [param]},
        )

        materialized = materialize_parameter_groups([group], include_metadata_keys={"param_names"})[0]

        assert materialized["param_names"] == ["unet.linear.weight"]
        assert "debug_refs" not in materialized

    def test_build_finetune_grouping_preserves_order_and_indices(self):
        """Shared grouping keeps trainer-facing order stable for the base fine-tune path."""
        denoiser = torch.nn.Linear(4, 4)
        text_encoder_1 = torch.nn.Linear(3, 3)
        text_encoder_2 = torch.nn.Linear(2, 2)

        grouping = build_finetune_grouping(
            denoiser=denoiser,
            train_denoiser=True,
            text_encoders=[text_encoder_1, text_encoder_2],
            te_train_flags=[True, False],
            learning_rates=LearningRatesConfig(base=1e-5, denoiser=2e-5, text_encoders=[3e-5, 4e-5]),
        )

        assert [group.metric_name for group in grouping.logical_groups] == ["denoiser", "text_encoder1"]
        assert [group.execution_group_indices for group in grouping.logical_groups] == [(0,), (1,)]
        assert [group.label for group in grouping.execution_groups] == ["denoiser", "text_encoder1"]
        assert grouping.parameter_groups == grouping.execution_groups

    def test_build_finetune_grouping_uses_configured_learning_rates(self):
        """Shared grouping preserves denoiser overrides and text-encoder LR fallback behavior."""
        grouping = build_finetune_grouping(
            denoiser=torch.nn.Linear(4, 4),
            train_denoiser=True,
            text_encoders=[torch.nn.Linear(3, 3), torch.nn.Linear(2, 2)],
            te_train_flags=[True, True],
            learning_rates=LearningRatesConfig(base=1e-5, denoiser=2e-5, text_encoders=[3e-5]),
        )

        assert [group.lr for group in grouping.logical_groups] == [2e-5, 3e-5, 1e-5]

    def test_zero_lr_keeps_baseline_component_frozen_and_excludes_optimizer_groups(self):
        """Explicit zero LR should freeze the baseline component instead of creating a zero-LR optimizer group."""
        denoiser = torch.nn.Linear(4, 4)
        train_denoiser, te_flags = resolve_finetune_trainability(
            denoiser=denoiser,
            text_encoders=[],
            learning_rates=LearningRatesConfig(base=None, denoiser=0.0),
        )

        grouping = build_finetune_grouping(
            denoiser=denoiser,
            train_denoiser=train_denoiser,
            text_encoders=[],
            te_train_flags=te_flags,
            learning_rates=LearningRatesConfig(base=None, denoiser=0.0),
        )

        assert train_denoiser is False
        assert te_flags == []
        assert grouping.execution_groups == []
        assert grouping.logical_groups == []

    def test_resolve_finetune_trainability_uses_groups_when_base_is_missing(self):
        """Explicit groups can make a component trainable even without a base fallback LR."""
        train_denoiser, te_flags = resolve_finetune_trainability(
            denoiser=torch.nn.Linear(4, 4),
            text_encoders=[torch.nn.Linear(3, 3), torch.nn.Linear(2, 2)],
            learning_rates=LearningRatesConfig(base=None, denoiser=None, text_encoders=[0.0, 0.0]),
            groups=[LearningRateGroupConfig(name="attention", lr=5e-5, match=["denoiser.*weight"])],
        )

        assert train_denoiser is True
        assert te_flags == [False, False]

    def test_resolve_finetune_trainability_uses_component_qualified_match_names(self):
        """Component-qualified selectors should activate the expected model-facing component."""
        train_denoiser, te_flags = resolve_finetune_trainability(
            denoiser=torch.nn.Linear(4, 4),
            text_encoders=[torch.nn.Linear(3, 3), torch.nn.Linear(2, 2)],
            learning_rates=LearningRatesConfig(base=None, denoiser=None, text_encoders=[0.0, 0.0]),
            groups=[LearningRateGroupConfig(name="attention", lr=5e-5, match=["unet.*weight"])],
            component_names=NamedParameterComponentNames(
                text_encoder_names=("clip_l", "clip_g"),
                vae_name="vae",
                denoiser_name="unet",
            ),
        )

        assert train_denoiser is True
        assert te_flags == [False, False]

    def test_resolve_adapter_target_selection_is_optimization_owned_for_peft(self):
        """Adapter target selection should be resolved by optimization policy, not PeftMode."""
        clip_l = torch.nn.Linear(3, 3)
        clip_g = torch.nn.Linear(2, 2)
        denoiser = torch.nn.Linear(4, 4)

        selection = resolve_adapter_target_selection(
            model_type="sdxl",
            denoiser=denoiser,
            text_encoders=[clip_l, clip_g],
            learning_rates=LearningRatesConfig(base=None, denoiser=1e-4, text_encoders=[0.0, 5e-5]),
        )

        assert selection.train_denoiser is True
        assert selection.te_train_flags == [False, True]
        assert [target.component for target in selection.resolved_targets.targets] == ["clip_g", "unet"]

    def test_build_adapter_grouping_uses_repo_owned_trainable_refs(self):
        """Adapter grouping should consume repo-owned refs instead of legacy optimizer hooks."""
        clip_l_param = torch.nn.Parameter(torch.randn(2, 2))
        unet_param = torch.nn.Parameter(torch.randn(2, 2))

        class FakeAdapter:
            def describe_trainable_parameter_refs(self):
                return [
                    AdapterTrainableParameterRef(
                        param=clip_l_param,
                        name="lora_te1_block.lora_down.weight",
                        algorithm="lora",
                        component="clip_l",
                        component_key="text_encoder1",
                        target_path="lora_te1_block",
                    ),
                    AdapterTrainableParameterRef(
                        param=unet_param,
                        name="lora_unet_block.lora_down.weight",
                        algorithm="lora",
                        component="unet",
                        component_key="denoiser",
                        target_path="lora_unet_block",
                    ),
                ]

        grouping = build_adapter_grouping(
            adapter=FakeAdapter(),
            learning_rates=LearningRatesConfig(base=1e-5, denoiser=2e-5, text_encoders=[3e-5]),
        )

        assert [group.metric_name for group in grouping.logical_groups] == ["clip_l", "unet"]
        assert [group.lr for group in grouping.logical_groups] == [3e-5, 2e-5]
        assert grouping.execution_groups[0].metadata["param_names"] == ["lora_te1_block.lora_down.weight"]
        assert grouping.execution_groups[1].metadata["param_names"] == ["lora_unet_block.lora_down.weight"]

    def test_build_adapter_grouping_rejects_malformed_text_encoder_component_key(self):
        """Adapter grouping should require explicit repo-owned component identities."""
        te_param = torch.nn.Parameter(torch.randn(2, 2))

        class FakeAdapter:
            def describe_trainable_parameter_refs(self):
                return [
                    AdapterTrainableParameterRef(
                        param=te_param,
                        name="lora_te_bad.lora_down.weight",
                        algorithm="lora",
                        component="clip_l",
                        component_key="text_encoder_bad",
                        target_path="lora_te_bad",
                    )
                ]

        with pytest.raises(ValueError, match="malformed text-encoder component key"):
            build_adapter_grouping(
                adapter=FakeAdapter(),
                learning_rates=LearningRatesConfig(base=1e-5, denoiser=2e-5),
            )

    def test_build_adapter_grouping_requires_repo_owned_trainable_ref_provider(self):
        """Adapter grouping should stay on the repo-owned trainable-ref contract."""

        class LegacyShapedAdapter:
            adapter_resolved_targets = object()

        with pytest.raises(TypeError, match="describe_trainable_parameter_refs"):
            build_adapter_grouping(
                adapter=LegacyShapedAdapter(),
                learning_rates=LearningRatesConfig(base=1e-5, denoiser=2e-5),
            )

    def test_build_finetune_grouping_applies_named_group_overrides_before_component_remainder(self):
        """Named groups should override matched subsets while component LR handles the remaining params."""

        class DummyDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_proj = torch.nn.Linear(4, 4)
                self.time_embed = torch.nn.Linear(4, 4)
                self.other = torch.nn.Linear(4, 4)

        denoiser = DummyDenoiser()
        groups = [
            LearningRateGroupConfig(name="attention", lr=5e-5, match=["denoiser.*attn*"]),
            LearningRateGroupConfig(name="time_embed", lr=1e-4, match=["denoiser.*time_embed.*"]),
        ]

        grouping = build_finetune_grouping(
            denoiser=denoiser,
            train_denoiser=True,
            text_encoders=[],
            te_train_flags=[],
            learning_rates=LearningRatesConfig(base=1e-4, denoiser=1e-5),
            groups=groups,
        )

        assert [group.metric_name for group in grouping.logical_groups] == ["attention", "time_embed", "denoiser"]
        assert [group.lr for group in grouping.logical_groups] == [5e-5, 1e-4, 1e-5]
        assert len({id(param) for group in grouping.execution_groups for param in group.params}) == sum(
            len(group.params) for group in grouping.execution_groups
        )
        assert len(grouping.execution_groups[-1].params) == 2  # only the unmatched "other" layer remains

    def test_build_finetune_grouping_matches_sd3_component_qualified_names(self):
        """Named groups should match SD3-style public component prefixes, not generic placeholders."""

        class DummyMmdit(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_proj = torch.nn.Linear(4, 4)
                self.other = torch.nn.Linear(4, 4)

        denoiser = DummyMmdit()
        grouping = build_finetune_grouping(
            denoiser=denoiser,
            train_denoiser=True,
            text_encoders=[],
            te_train_flags=[],
            learning_rates=LearningRatesConfig(base=1e-4, denoiser=1e-5),
            groups=[LearningRateGroupConfig(name="attention", lr=5e-5, match=["mmdit.*attn*"])],
            component_names=NamedParameterComponentNames(
                text_encoder_names=("clip_l", "clip_g", "t5xxl"),
                vae_name="vae",
                denoiser_name="mmdit",
            ),
        )

        assert [group.metric_name for group in grouping.logical_groups] == ["attention", "denoiser"]
        assert grouping.execution_groups[0].label == "attention"
        assert set(grouping.execution_groups[0].metadata["param_names"]) == {
            "mmdit.attn_proj.weight",
            "mmdit.attn_proj.bias",
        }
        assert "param_names" not in materialize_parameter_groups(grouping.execution_groups)[0]

    def test_resolve_learning_rate_groups_loads_yaml_file(self, tmp_path):
        """Named groups can be loaded from a separate YAML file."""
        groups_file = tmp_path / "groups.yaml"
        groups_file.write_text(
            "- name: attention\n  lr: 5e-5\n  match:\n    - unet.*attn*\n",
            encoding="utf-8",
        )

        groups = resolve_learning_rate_groups(LearningRatesConfig(base=1e-4, groups_file=str(groups_file)))

        assert len(groups) == 1
        assert groups[0].name == "attention"
        assert groups[0].lr == pytest.approx(5e-5)
        assert groups[0].match == ["unet.*attn*"]

    def test_resolve_learning_rate_groups_rejects_inline_and_file_together(self, tmp_path):
        """Inline groups and groups_file should stay mutually exclusive."""
        groups_file = tmp_path / "groups.yaml"
        groups_file.write_text("groups: []\n", encoding="utf-8")

        with pytest.raises(ValueError, match="cannot be set at the same time"):
            resolve_learning_rate_groups(
                LearningRatesConfig(
                    base=1e-4,
                    groups=[LearningRateGroupConfig(name="attention", lr=5e-5, match=["unet.*attn*"])],
                    groups_file=str(groups_file),
                )
            )

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
