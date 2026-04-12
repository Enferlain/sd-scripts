"""
Unit tests for library/training/trainer_utils.py

Tests learning rate logging utilities and accelerator preparation.
"""

import os
import pytest
from unittest.mock import MagicMock, Mock, patch
import torch

from library.config.dataclasses.output import LoggingConfig
from library.config.dataclasses.performance import PrecisionConfig, CompilationConfig, DistributedConfig, DeepSpeedConfig
from library.config.dataclasses.training import TrainingConfig
from library.logging.step_logging import init_trackers, append_lr_to_logs_with_names
from library.optimization.types import LogicalParameterGroup, OptimizationPlan
from library.training.trainer_utils import (
    append_lr_to_logs,
    compute_accelerator_config,
    determine_grad_sync_context,
    log_training_diagnostics,
    prepare_accelerator,
)


@pytest.fixture
def mock_lr_scheduler():
    """Create a mock LR scheduler."""
    scheduler = MagicMock()
    scheduler.get_last_lr.return_value = [1e-4, 1e-5, 1e-6]
    return scheduler


# =============================================================================
# append_lr_to_logs_with_names Tests
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestAppendLrToLogsWithNames:
    """Test append_lr_to_logs_with_names function."""

    def test_adds_lr_entries_to_logs(self, mock_lr_scheduler):
        """Test that LR entries are added to logs dict."""
        logs = {}
        names = ["denoiser", "text_encoder1"]
        mock_lr_scheduler.get_last_lr.return_value = [1e-4, 1e-5]

        append_lr_to_logs_with_names(logs, mock_lr_scheduler, "AdamW", names)

        assert "lr/denoiser" in logs
        assert "lr/text_encoder1" in logs
        assert logs["lr/denoiser"] == 1e-4
        assert logs["lr/text_encoder1"] == 1e-5

    def test_lr_values_are_floats(self, mock_lr_scheduler):
        """Test that LR values are converted to floats."""
        logs = {}
        names = ["model"]
        mock_lr_scheduler.get_last_lr.return_value = [1e-4]

        append_lr_to_logs_with_names(logs, mock_lr_scheduler, "SGD", names)

        assert isinstance(logs["lr/model"], float)

    def test_dadapt_optimizer_adds_d_lr_entry(self, mock_lr_scheduler):
        """Test that DAdapt optimizers add d*lr entries."""
        logs = {}
        names = ["denoiser"]
        mock_lr_scheduler.get_last_lr.return_value = [1e-4]
        mock_lr_scheduler.optimizers = [MagicMock()]
        mock_lr_scheduler.optimizers[-1].param_groups = [{"d": 0.5, "lr": 1e-4}]

        append_lr_to_logs_with_names(logs, mock_lr_scheduler, "DAdaptAdam", names)

        assert "lr/d*lr/denoiser" in logs
        assert logs["lr/d*lr/denoiser"] == 0.5 * 1e-4

    def test_prodigy_optimizer_adds_d_lr_entry(self, mock_lr_scheduler):
        """Test that Prodigy optimizer adds d*lr entries."""
        logs = {}
        names = ["denoiser"]
        mock_lr_scheduler.get_last_lr.return_value = [1e-4]
        mock_lr_scheduler.optimizers = [MagicMock()]
        mock_lr_scheduler.optimizers[-1].param_groups = [{"d": 0.3, "lr": 2e-4}]

        append_lr_to_logs_with_names(logs, mock_lr_scheduler, "Prodigy", names)

        assert "lr/d*lr/denoiser" in logs
        assert logs["lr/d*lr/denoiser"] == 0.3 * 2e-4

    def test_regular_optimizer_does_not_add_d_lr(self, mock_lr_scheduler):
        """Test that regular optimizers don't add d*lr entries."""
        logs = {}
        names = ["denoiser"]
        mock_lr_scheduler.get_last_lr.return_value = [1e-4]

        append_lr_to_logs_with_names(logs, mock_lr_scheduler, "AdamW", names)

        assert "lr/d*lr/denoiser" not in logs


# =============================================================================
# append_lr_to_logs Tests
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestAppendLrToLogs:
    """Test append_lr_to_logs function."""

    def test_includes_denoiser_by_default(self, mock_lr_scheduler):
        """Test that denoiser is included when including_denoiser=True."""
        logs = {}

        append_lr_to_logs(logs, mock_lr_scheduler, "AdamW", including_denoiser=True)

        assert "lr/denoiser" in logs

    def test_excludes_denoiser_when_specified(self, mock_lr_scheduler):
        """Test that denoiser is excluded when including_denoiser=False."""
        logs = {}
        mock_lr_scheduler.get_last_lr.return_value = [1e-4, 1e-5]  # Only 2 values for TE1, TE2

        append_lr_to_logs(logs, mock_lr_scheduler, "AdamW", including_denoiser=False)

        assert "lr/denoiser" not in logs
        assert "lr/text_encoder1" in logs

    def test_always_includes_text_encoders(self, mock_lr_scheduler):
        """Test that text encoders are always included."""
        logs = {}

        append_lr_to_logs(logs, mock_lr_scheduler, "AdamW", including_denoiser=True)

        assert "lr/text_encoder1" in logs
        assert "lr/text_encoder2" in logs


@pytest.mark.training
@pytest.mark.unit
class TestLogTrainingDiagnostics:
    """Test startup diagnostics output."""

    def test_prefers_optimization_plan_group_labels(self):
        """Logical-group metadata should work without legacy lr_descriptions."""
        accelerator = MagicMock()
        accelerator.num_processes = 1
        accelerator.print = MagicMock()

        cfg = MagicMock()
        cfg.performance.precision.mixed_precision = "fp16"
        cfg.performance.memory.gradient_checkpointing = False
        cfg.performance.attention.xformers = False
        cfg.performance.deepspeed.deepspeed = False
        cfg.training.train_batch_size = 1
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.max_train_steps = 10

        module = torch.nn.Linear(2, 2)
        params = list(module.parameters())
        optimizer = MagicMock()
        optimizer.param_groups = [{"params": params, "lr": 1e-4}]
        optimization_plan = OptimizationPlan(
            logical_groups=[
                LogicalParameterGroup(
                    key="denoiser",
                    label="denoiser",
                    params=params,
                    lr=1e-4,
                    execution_group_indices=(0,),
                )
            ]
        )

        log_training_diagnostics(
            accelerator=accelerator,
            cfg=cfg,
            mode=MagicMock(),
            strategies=MagicMock(),
            components=[("denoiser", module)],
            optimizer=optimizer,
            optimizer_name="AdamW",
            lr_descriptions=None,
            optimization_plan=optimization_plan,
        )

        printed_lines = [call.args[0] for call in accelerator.print.call_args_list]
        assert any("denoiser" in line and "params=" in line for line in printed_lines)


# =============================================================================
# Pure Tests - compute_accelerator_config
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestComputeAcceleratorConfig:
    """Test pure accelerator config computation."""

    @pytest.fixture
    def mock_precision_config(self):
        config = Mock(spec=PrecisionConfig)
        config.mixed_precision = "fp16"
        return config

    @pytest.fixture
    def mock_compilation_config(self):
        config = Mock(spec=CompilationConfig)
        config.torch_compile = False
        return config

    @pytest.fixture
    def mock_distributed_config(self):
        config = Mock(spec=DistributedConfig)
        config.ddp_gradient_as_bucket_view = False
        config.ddp_static_graph = False
        return config

    @pytest.fixture
    def mock_deepspeed_config(self):
        return Mock(spec=DeepSpeedConfig)

    @pytest.fixture
    def mock_logging_config(self, tmp_path):
        config = Mock(spec=LoggingConfig)
        config.logging_dir = str(tmp_path / "logs")
        config.log_prefix = "test_"
        config.log_with = None
        config.wandb_api_key = None
        return config

    @pytest.fixture
    def mock_training_config(self):
        config = Mock(spec=TrainingConfig)
        config.gradient_accumulation_steps = 2
        return config

    @patch("library.training.trainer_utils.deepspeed_utils.prepare_deepspeed_plugin")
    @patch("library.training.trainer_utils.time.strftime", return_value="20260325010101")
    def test_defaults_to_tensorboard_when_logging_dir_present(
        self,
        _mock_strftime,
        mock_ds_plugin,
        mock_precision_config,
        mock_compilation_config,
        mock_distributed_config,
        mock_deepspeed_config,
        mock_logging_config,
        mock_training_config,
    ):
        mock_ds_plugin.return_value = None

        result = compute_accelerator_config(
            mock_precision_config,
            mock_compilation_config,
            mock_distributed_config,
            mock_deepspeed_config,
            logging_config=mock_logging_config,
            training_config=mock_training_config,
        )

        assert result.log_with == "tensorboard"
        assert result.project_dir == os.path.join(mock_logging_config.logging_dir, "test_20260325010101")
        assert result.gradient_accumulation_steps == 2
        assert result.configure_wandb is False

    @patch("library.training.trainer_utils.deepspeed_utils.prepare_deepspeed_plugin")
    def test_requires_logging_dir_for_tensorboard(
        self,
        mock_ds_plugin,
        mock_precision_config,
        mock_compilation_config,
        mock_distributed_config,
        mock_deepspeed_config,
        mock_logging_config,
    ):
        mock_ds_plugin.return_value = None
        mock_logging_config.logging_dir = None
        mock_logging_config.log_with = "tensorboard"

        with pytest.raises(ValueError, match="logging_dir is required"):
            compute_accelerator_config(
                mock_precision_config,
                mock_compilation_config,
                mock_distributed_config,
                mock_deepspeed_config,
                logging_config=mock_logging_config,
            )

    @patch("library.training.trainer_utils.deepspeed_utils.prepare_deepspeed_plugin")
    @patch("library.training.trainer_utils.time.strftime", return_value="20260325010101")
    def test_marks_wandb_setup_for_prepare_phase(
        self,
        _mock_strftime,
        mock_ds_plugin,
        mock_precision_config,
        mock_compilation_config,
        mock_distributed_config,
        mock_deepspeed_config,
        mock_logging_config,
    ):
        mock_ds_plugin.return_value = None
        mock_logging_config.log_with = "wandb"
        mock_logging_config.wandb_api_key = "secret"

        result = compute_accelerator_config(
            mock_precision_config,
            mock_compilation_config,
            mock_distributed_config,
            mock_deepspeed_config,
            logging_config=mock_logging_config,
        )

        assert result.log_with == "wandb"
        assert result.configure_wandb is True
        assert result.wandb_api_key == "secret"

    @patch("library.training.trainer_utils.deepspeed_utils.prepare_deepspeed_plugin")
    @patch("library.training.trainer_utils.TorchDynamoPlugin")
    @patch("library.training.trainer_utils.prepare_windows_compiler_env_for_torch_compile")
    def test_torch_compile_bootstraps_windows_compiler_env(
        self,
        mock_prepare_compile_env,
        mock_dynamo_plugin,
        mock_ds_plugin,
        mock_precision_config,
        mock_compilation_config,
        mock_distributed_config,
        mock_deepspeed_config,
    ):
        mock_compilation_config.torch_compile = True
        mock_ds_plugin.return_value = None
        mock_dynamo_plugin.return_value = Mock()

        compute_accelerator_config(
            mock_precision_config,
            mock_compilation_config,
            mock_distributed_config,
            mock_deepspeed_config,
        )

        mock_prepare_compile_env.assert_called_once()
        mock_dynamo_plugin.assert_called_once()


# =============================================================================
# Heavy Mocking Tests - prepare_accelerator
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestPrepareAccelerator:
    """Test prepare_accelerator with mocked Accelerator."""

    @pytest.fixture
    def mock_precision_config(self):
        """Create a mock PrecisionConfig."""
        config = Mock(spec=PrecisionConfig)
        config.mixed_precision = "fp16"
        return config

    @pytest.fixture
    def mock_compilation_config(self):
        """Create a mock CompilationConfig."""
        config = Mock(spec=CompilationConfig)
        config.torch_compile = False
        return config

    @pytest.fixture
    def mock_distributed_config(self):
        """Create a mock DistributedConfig."""
        config = Mock(spec=DistributedConfig)
        config.ddp_gradient_as_bucket_view = False
        config.ddp_static_graph = False
        return config

    @pytest.fixture
    def mock_deepspeed_config(self):
        """Create a mock DeepSpeedConfig."""
        config = Mock(spec=DeepSpeedConfig)
        return config

    @pytest.fixture
    def mock_logging_config(self, tmp_path):
        """Create a mock LoggingConfig."""
        config = Mock(spec=LoggingConfig)
        config.logging_dir = str(tmp_path / "logs")
        config.log_prefix = "test_"
        config.log_with = None
        config.wandb_api_key = None
        return config

    @pytest.fixture
    def mock_training_config(self):
        """Create a mock TrainingConfig."""
        config = Mock(spec=TrainingConfig)
        config.gradient_accumulation_steps = 2
        return config

    @patch("library.training.trainer_utils.Accelerator")
    @patch("library.training.trainer_utils.deepspeed_utils.prepare_deepspeed_plugin")
    def test_creates_accelerator_with_basic_config(
        self,
        mock_ds_plugin,
        mock_accelerator_class,
        mock_precision_config,
        mock_compilation_config,
        mock_distributed_config,
        mock_deepspeed_config,
    ):
        """Test that Accelerator is created with basic config."""
        mock_ds_plugin.return_value = None
        mock_accelerator = Mock()
        mock_accelerator_class.return_value = mock_accelerator

        result = prepare_accelerator(mock_precision_config, mock_compilation_config, mock_distributed_config, mock_deepspeed_config)

        assert result is mock_accelerator
        mock_accelerator_class.assert_called_once()

    @patch("library.training.trainer_utils.Accelerator")
    @patch("library.training.trainer_utils.deepspeed_utils.prepare_deepspeed_plugin")
    def test_uses_gradient_accumulation_steps(
        self,
        mock_ds_plugin,
        mock_accelerator_class,
        mock_precision_config,
        mock_compilation_config,
        mock_distributed_config,
        mock_deepspeed_config,
        mock_training_config,
    ):
        """Test that gradient_accumulation_steps is passed correctly."""
        mock_ds_plugin.return_value = None
        mock_accelerator = Mock()
        mock_accelerator_class.return_value = mock_accelerator

        prepare_accelerator(
            mock_precision_config,
            mock_compilation_config,
            mock_distributed_config,
            mock_deepspeed_config,
            training_config=mock_training_config,
        )

        call_kwargs = mock_accelerator_class.call_args[1]
        assert call_kwargs["gradient_accumulation_steps"] == 2

    @patch("library.training.trainer_utils.Accelerator")
    @patch("library.training.trainer_utils.deepspeed_utils.prepare_deepspeed_plugin")
    def test_uses_mixed_precision(
        self,
        mock_ds_plugin,
        mock_accelerator_class,
        mock_precision_config,
        mock_compilation_config,
        mock_distributed_config,
        mock_deepspeed_config,
    ):
        """Test that mixed_precision is passed correctly."""
        mock_ds_plugin.return_value = None
        mock_accelerator = Mock()
        mock_accelerator_class.return_value = mock_accelerator

        prepare_accelerator(mock_precision_config, mock_compilation_config, mock_distributed_config, mock_deepspeed_config)

        call_kwargs = mock_accelerator_class.call_args[1]
        assert call_kwargs["mixed_precision"] == "fp16"

    @patch("library.training.trainer_utils.Accelerator")
    @patch("library.training.trainer_utils.deepspeed_utils.prepare_deepspeed_plugin")
    @patch("library.training.trainer_utils.TorchDynamoPlugin")
    def test_torch_compile_creates_dynamo_plugin(
        self,
        mock_dynamo,
        mock_ds_plugin,
        mock_accelerator_class,
        mock_precision_config,
        mock_compilation_config,
        mock_distributed_config,
        mock_deepspeed_config,
    ):
        """Test that torch_compile creates dynamo plugin."""
        mock_compilation_config.torch_compile = True
        mock_ds_plugin.return_value = None
        mock_accelerator = Mock()
        mock_accelerator_class.return_value = mock_accelerator

        prepare_accelerator(mock_precision_config, mock_compilation_config, mock_distributed_config, mock_deepspeed_config)

        mock_dynamo.assert_called_once()

    @patch("library.training.trainer_utils.Accelerator")
    @patch("library.training.trainer_utils.deepspeed_utils.prepare_deepspeed_plugin")
    @patch("library.training.trainer_utils.os.makedirs")
    @patch("library.training.trainer_utils.time.strftime", return_value="20260325010101")
    def test_wandb_side_effects_happen_in_prepare_phase(
        self,
        _mock_strftime,
        mock_makedirs,
        mock_ds_plugin,
        mock_accelerator_class,
        mock_precision_config,
        mock_compilation_config,
        mock_distributed_config,
        mock_deepspeed_config,
        mock_logging_config,
    ):
        """Wandb import/login and WANDB_DIR setup should remain in prepare_accelerator()."""
        mock_ds_plugin.return_value = None
        mock_accelerator = Mock()
        mock_accelerator_class.return_value = mock_accelerator
        mock_logging_config.log_with = "wandb"
        mock_logging_config.wandb_api_key = "secret"
        mock_wandb = Mock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}), patch.dict("os.environ", {}, clear=True):
            prepare_accelerator(
                mock_precision_config,
                mock_compilation_config,
                mock_distributed_config,
                mock_deepspeed_config,
                logging_config=mock_logging_config,
            )

            expected_dir = os.path.join(mock_logging_config.logging_dir, "test_20260325010101")
            mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
            assert os.environ["WANDB_DIR"] == expected_dir
            mock_wandb.login.assert_called_once_with(key="secret")


# =============================================================================
# Heavy Mocking Tests - init_trackers
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestInitTrackers:
    """Test init_trackers with mocked Accelerator."""

    @pytest.fixture
    def mock_accelerator(self):
        """Create a mock Accelerator."""
        accel = Mock()
        accel.is_main_process = True
        return accel

    @pytest.fixture
    def mock_cfg(self):
        """Create a mock config object with dataclass fields."""
        from dataclasses import dataclass

        @dataclass
        class MockLogging:
            wandb_run_name: str = None
            log_tracker_config: dict = None
            log_tracker_name: str = None

        @dataclass
        class MockConfig:
            logging: MockLogging = None

        cfg = MockConfig(logging=MockLogging())
        return cfg

    def test_calls_init_trackers_on_main_process(self, mock_accelerator, mock_cfg):
        """Test that init_trackers is called when is_main_process=True."""
        init_trackers(mock_accelerator, mock_cfg.logging, "test_tracker")

        mock_accelerator.init_trackers.assert_called_once()

    def test_skips_on_non_main_process(self, mock_accelerator, mock_cfg):
        """Test that init_trackers is skipped on non-main process."""
        mock_accelerator.is_main_process = False

        init_trackers(mock_accelerator, mock_cfg.logging, "test_tracker")

        mock_accelerator.init_trackers.assert_not_called()

    def test_uses_default_tracker_name(self, mock_accelerator, mock_cfg):
        """Test that default tracker name is used when not specified."""
        init_trackers(mock_accelerator, mock_cfg.logging, "default_name")

        call_args = mock_accelerator.init_trackers.call_args[0]
        assert call_args[0] == "default_name"

    def test_uses_custom_tracker_name(self, mock_accelerator, mock_cfg):
        """Test that custom tracker name is used when specified."""
        mock_cfg.logging.log_tracker_name = "custom_name"

        init_trackers(mock_accelerator, mock_cfg.logging, "default_name")

        call_args = mock_accelerator.init_trackers.call_args[0]
        assert call_args[0] == "custom_name"

    def test_sanitizes_sensitive_keys(self, mock_accelerator):
        """Test that sensitive keys are masked in logged config."""
        from dataclasses import dataclass

        @dataclass
        class MockLogging:
            wandb_run_name: str = None
            log_tracker_config: dict = None
            log_tracker_name: str = None
            wandb_api_key: str = None
            huggingface_token: str = None

        cfg = MockLogging(wandb_api_key="secret_key", huggingface_token="another_secret")

        init_trackers(mock_accelerator, cfg, "test")

        # The config should have masked sensitive keys
        call_kwargs = mock_accelerator.init_trackers.call_args[1]
        logged_config = call_kwargs["config"]
        assert logged_config["wandb_api_key"] == "*****"
        assert logged_config["huggingface_token"] == "*****"


# =============================================================================
# Heavy Mocking Tests - determine_grad_sync_context
# =============================================================================


@pytest.mark.training
@pytest.mark.unit
class TestDetermineGradSyncContext:
    """Test determine_grad_sync_context with mocked Accelerator."""

    @pytest.fixture
    def mock_args(self):
        """Create mock args."""
        args = Mock()
        args.full_bf16 = False
        return args

    @pytest.fixture
    def mock_accelerator(self):
        """Create mock Accelerator."""
        accel = Mock()
        accel.accumulate = Mock(return_value="accumulate_context")
        accel.no_sync = Mock(return_value="no_sync_context")
        accel.num_processes = 1
        return accel

    @pytest.fixture
    def mock_training_model(self):
        """Create mock training model."""
        return Mock()

    def test_returns_accumulate_context(self, mock_args, mock_accelerator, mock_training_model):
        """Test that accumulate context is returned."""
        result = determine_grad_sync_context(mock_args, mock_accelerator, sync_gradients=True, training_model=mock_training_model)

        mock_accelerator.accumulate.assert_called_once_with(mock_training_model)
        assert result == "accumulate_context"

    def test_returns_accumulate_with_auxiliary_model(self, mock_args, mock_accelerator, mock_training_model):
        """Test that accumulate context includes an auxiliary model when provided."""
        auxiliary_model = Mock()

        context = determine_grad_sync_context(
            mock_args, mock_accelerator, sync_gradients=True, training_model=mock_training_model, auxiliary_model=auxiliary_model
        )

        mock_accelerator.accumulate.assert_called_once_with(mock_training_model, auxiliary_model)
        assert context == "accumulate_context"
