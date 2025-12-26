"""
Unit tests for library/training/trainer_utils.py

Tests validation check logic and learning rate logging utilities.
"""

import pytest
from unittest.mock import MagicMock, PropertyMock

from library.training.trainer_utils import (
    calculate_val_loss_check,
    append_lr_to_logs,
    append_lr_to_logs_with_names,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_training_config():
    """Create a mock training config."""
    config = MagicMock()
    config.max_train_steps = 1000
    config.validate_every_n_steps = None
    return config


@pytest.fixture
def mock_train_dataloader():
    """Create a mock train dataloader with fixed length."""
    dataloader = MagicMock()
    dataloader.__len__ = MagicMock(return_value=100)
    return dataloader


@pytest.fixture
def mock_val_dataloader():
    """Create a mock validation dataloader."""
    return MagicMock()


@pytest.fixture
def mock_lr_scheduler():
    """Create a mock LR scheduler."""
    scheduler = MagicMock()
    scheduler.get_last_lr.return_value = [1e-4, 1e-5, 1e-6]
    return scheduler


# =============================================================================
# calculate_val_loss_check Tests
# =============================================================================

@pytest.mark.training
@pytest.mark.unit
class TestCalculateValLossCheck:
    """Test calculate_val_loss_check function."""
    
    def test_returns_false_when_val_dataloader_is_none(self, mock_training_config, mock_train_dataloader):
        """Test that validation is skipped when val_dataloader is None."""
        result = calculate_val_loss_check(
            mock_training_config,
            global_step=100,
            epoch_step=50,
            val_dataloader=None,
            train_dataloader=mock_train_dataloader
        )
        
        assert result is False
        
    def test_returns_true_at_step_zero(self, mock_training_config, mock_val_dataloader, mock_train_dataloader):
        """Test that validation runs at step 0."""
        result = calculate_val_loss_check(
            mock_training_config,
            global_step=0,
            epoch_step=0,
            val_dataloader=mock_val_dataloader,
            train_dataloader=mock_train_dataloader
        )
        
        assert result is True
        
    def test_returns_true_at_max_train_steps(self, mock_training_config, mock_val_dataloader, mock_train_dataloader):
        """Test that validation runs when reaching max_train_steps."""
        result = calculate_val_loss_check(
            mock_training_config,
            global_step=1000,  # at max_train_steps
            epoch_step=50,
            val_dataloader=mock_val_dataloader,
            train_dataloader=mock_train_dataloader
        )
        
        assert result is True
        
    def test_returns_true_at_validate_every_n_steps(self, mock_training_config, mock_val_dataloader, mock_train_dataloader):
        """Test that validation runs at validate_every_n_steps intervals."""
        mock_training_config.validate_every_n_steps = 100
        
        result = calculate_val_loss_check(
            mock_training_config,
            global_step=200,  # divisible by 100
            epoch_step=50,
            val_dataloader=mock_val_dataloader,
            train_dataloader=mock_train_dataloader
        )
        
        assert result is True
        
    def test_returns_false_between_validation_intervals(self, mock_training_config, mock_val_dataloader, mock_train_dataloader):
        """Test that validation is skipped between intervals."""
        mock_training_config.validate_every_n_steps = 100
        
        result = calculate_val_loss_check(
            mock_training_config,
            global_step=150,  # not divisible by 100
            epoch_step=50,
            val_dataloader=mock_val_dataloader,
            train_dataloader=mock_train_dataloader
        )
        
        assert result is False
        
    def test_returns_true_at_epoch_end_when_no_step_interval(self, mock_training_config, mock_val_dataloader, mock_train_dataloader):
        """Test that validation runs at end of epoch when validate_every_n_steps is None."""
        mock_training_config.validate_every_n_steps = None
        
        result = calculate_val_loss_check(
            mock_training_config,
            global_step=100,
            epoch_step=99,  # last step (dataloader length - 1)
            val_dataloader=mock_val_dataloader,
            train_dataloader=mock_train_dataloader
        )
        
        assert result is True
        
    def test_returns_false_mid_epoch_when_no_step_interval(self, mock_training_config, mock_val_dataloader, mock_train_dataloader):
        """Test that validation is skipped mid-epoch when validate_every_n_steps is None."""
        mock_training_config.validate_every_n_steps = None
        
        result = calculate_val_loss_check(
            mock_training_config,
            global_step=100,
            epoch_step=50,  # not at end of epoch
            val_dataloader=mock_val_dataloader,
            train_dataloader=mock_train_dataloader
        )
        
        assert result is False


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
        names = ["unet", "text_encoder1"]
        mock_lr_scheduler.get_last_lr.return_value = [1e-4, 1e-5]
        
        append_lr_to_logs_with_names(logs, mock_lr_scheduler, "AdamW", names)
        
        assert "lr/unet" in logs
        assert "lr/text_encoder1" in logs
        assert logs["lr/unet"] == 1e-4
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
        names = ["unet"]
        mock_lr_scheduler.get_last_lr.return_value = [1e-4]
        mock_lr_scheduler.optimizers = [MagicMock()]
        mock_lr_scheduler.optimizers[-1].param_groups = [{"d": 0.5, "lr": 1e-4}]
        
        append_lr_to_logs_with_names(logs, mock_lr_scheduler, "DAdaptAdam", names)
        
        assert "lr/d*lr/unet" in logs
        assert logs["lr/d*lr/unet"] == 0.5 * 1e-4
        
    def test_prodigy_optimizer_adds_d_lr_entry(self, mock_lr_scheduler):
        """Test that Prodigy optimizer adds d*lr entries."""
        logs = {}
        names = ["unet"]
        mock_lr_scheduler.get_last_lr.return_value = [1e-4]
        mock_lr_scheduler.optimizers = [MagicMock()]
        mock_lr_scheduler.optimizers[-1].param_groups = [{"d": 0.3, "lr": 2e-4}]
        
        append_lr_to_logs_with_names(logs, mock_lr_scheduler, "Prodigy", names)
        
        assert "lr/d*lr/unet" in logs
        assert logs["lr/d*lr/unet"] == 0.3 * 2e-4
        
    def test_regular_optimizer_does_not_add_d_lr(self, mock_lr_scheduler):
        """Test that regular optimizers don't add d*lr entries."""
        logs = {}
        names = ["unet"]
        mock_lr_scheduler.get_last_lr.return_value = [1e-4]
        
        append_lr_to_logs_with_names(logs, mock_lr_scheduler, "AdamW", names)
        
        assert "lr/d*lr/unet" not in logs


# =============================================================================
# append_lr_to_logs Tests
# =============================================================================

@pytest.mark.training
@pytest.mark.unit
class TestAppendLrToLogs:
    """Test append_lr_to_logs function."""
    
    def test_includes_unet_by_default(self, mock_lr_scheduler):
        """Test that unet is included when including_unet=True."""
        logs = {}
        
        append_lr_to_logs(logs, mock_lr_scheduler, "AdamW", including_unet=True)
        
        assert "lr/unet" in logs
        
    def test_excludes_unet_when_specified(self, mock_lr_scheduler):
        """Test that unet is excluded when including_unet=False."""
        logs = {}
        mock_lr_scheduler.get_last_lr.return_value = [1e-4, 1e-5]  # Only 2 values for TE1, TE2
        
        append_lr_to_logs(logs, mock_lr_scheduler, "AdamW", including_unet=False)
        
        assert "lr/unet" not in logs
        assert "lr/text_encoder1" in logs
        
    def test_always_includes_text_encoders(self, mock_lr_scheduler):
        """Test that text encoders are always included."""
        logs = {}
        
        append_lr_to_logs(logs, mock_lr_scheduler, "AdamW", including_unet=True)
        
        assert "lr/text_encoder1" in logs
        assert "lr/text_encoder2" in logs


# =============================================================================
# Heavy Mocking Tests - prepare_accelerator
# =============================================================================

from unittest.mock import Mock, patch, MagicMock
from library.training.trainer_utils import (
    prepare_accelerator,
    init_trackers,
    determine_grad_sync_context,
)
from library.config.dataclasses.performance import PerformanceConfig
from library.config.dataclasses.output import LoggingConfig
from library.config.dataclasses.training import TrainingConfig


@pytest.mark.training
@pytest.mark.unit
class TestPrepareAccelerator:
    """Test prepare_accelerator with mocked Accelerator."""

    @pytest.fixture
    def mock_performance_config(self):
        """Create a mock PerformanceConfig with nested structure."""
        config = Mock(spec=PerformanceConfig)
        # Nested precision config
        config.precision = Mock()
        config.precision.mixed_precision = "fp16"
        # Nested compilation config
        config.compilation = Mock()
        config.compilation.torch_compile = False
        # Nested distributed config
        config.distributed = Mock()
        config.distributed.ddp_gradient_as_bucket_view = False
        config.distributed.ddp_static_graph = False
        # Deepspeed
        config.deepspeed = Mock()
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

    @patch('library.training.trainer_utils.Accelerator')
    @patch('library.training.trainer_utils.deepspeed_utils.prepare_deepspeed_plugin')
    def test_creates_accelerator_with_basic_config(
        self, mock_ds_plugin, mock_accelerator_class, mock_performance_config
    ):
        """Test that Accelerator is created with basic config."""
        mock_ds_plugin.return_value = None
        mock_accelerator = Mock()
        mock_accelerator_class.return_value = mock_accelerator
        
        result = prepare_accelerator(mock_performance_config)
        
        assert result is mock_accelerator
        mock_accelerator_class.assert_called_once()

    @patch('library.training.trainer_utils.Accelerator')
    @patch('library.training.trainer_utils.deepspeed_utils.prepare_deepspeed_plugin')
    def test_uses_gradient_accumulation_steps(
        self, mock_ds_plugin, mock_accelerator_class, 
        mock_performance_config, mock_training_config
    ):
        """Test that gradient_accumulation_steps is passed correctly."""
        mock_ds_plugin.return_value = None
        mock_accelerator = Mock()
        mock_accelerator_class.return_value = mock_accelerator
        
        prepare_accelerator(mock_performance_config, training_config=mock_training_config)
        
        call_kwargs = mock_accelerator_class.call_args[1]
        assert call_kwargs['gradient_accumulation_steps'] == 2

    @patch('library.training.trainer_utils.Accelerator')
    @patch('library.training.trainer_utils.deepspeed_utils.prepare_deepspeed_plugin')
    def test_uses_mixed_precision(
        self, mock_ds_plugin, mock_accelerator_class, mock_performance_config
    ):
        """Test that mixed_precision is passed correctly."""
        mock_ds_plugin.return_value = None
        mock_accelerator = Mock()
        mock_accelerator_class.return_value = mock_accelerator
        
        prepare_accelerator(mock_performance_config)
        
        call_kwargs = mock_accelerator_class.call_args[1]
        assert call_kwargs['mixed_precision'] == "fp16"

    @patch('library.training.trainer_utils.Accelerator')
    @patch('library.training.trainer_utils.deepspeed_utils.prepare_deepspeed_plugin')
    @patch('library.training.trainer_utils.TorchDynamoPlugin')
    def test_torch_compile_creates_dynamo_plugin(
        self, mock_dynamo, mock_ds_plugin, mock_accelerator_class, mock_performance_config
    ):
        """Test that torch_compile creates dynamo plugin."""
        mock_performance_config.compilation.torch_compile = True
        mock_ds_plugin.return_value = None
        mock_accelerator = Mock()
        mock_accelerator_class.return_value = mock_accelerator
        
        prepare_accelerator(mock_performance_config)
        
        mock_dynamo.assert_called_once()


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
        init_trackers(mock_accelerator, mock_cfg, "test_tracker")
        
        mock_accelerator.init_trackers.assert_called_once()

    def test_skips_on_non_main_process(self, mock_accelerator, mock_cfg):
        """Test that init_trackers is skipped on non-main process."""
        mock_accelerator.is_main_process = False
        
        init_trackers(mock_accelerator, mock_cfg, "test_tracker")
        
        mock_accelerator.init_trackers.assert_not_called()

    def test_uses_default_tracker_name(self, mock_accelerator, mock_cfg):
        """Test that default tracker name is used when not specified."""
        init_trackers(mock_accelerator, mock_cfg, "default_name")
        
        call_args = mock_accelerator.init_trackers.call_args[0]
        assert call_args[0] == "default_name"

    def test_uses_custom_tracker_name(self, mock_accelerator, mock_cfg):
        """Test that custom tracker name is used when specified."""
        mock_cfg.output.logging.log_tracker_name = "custom_name"
        
        init_trackers(mock_accelerator, mock_cfg, "default_name")
        
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
        
        @dataclass
        class MockConfig:
            logging: MockLogging = None
            wandb_api_key: str = None
            huggingface_token: str = None
        
        cfg = MockConfig(
            logging=MockLogging(),
            wandb_api_key="secret_key",
            huggingface_token="another_secret"
        )
        
        init_trackers(mock_accelerator, cfg, "test")
        
        # The config should have masked sensitive keys
        call_kwargs = mock_accelerator.init_trackers.call_args[1]
        logged_config = call_kwargs['config']
        assert logged_config['wandb_api_key'] == "*****"
        assert logged_config['huggingface_token'] == "*****"


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
        result = determine_grad_sync_context(
            mock_args, mock_accelerator, 
            sync_gradients=True, training_model=mock_training_model
        )
        
        mock_accelerator.accumulate.assert_called_once_with(mock_training_model)
        assert result == "accumulate_context"

    def test_returns_accumulate_with_edm2_model(
        self, mock_args, mock_accelerator, mock_training_model
    ):
        """Test that accumulate context includes edm2_model when provided."""
        edm2_model = Mock()
        
        result = determine_grad_sync_context(
            mock_args, mock_accelerator,
            sync_gradients=True, training_model=mock_training_model,
            edm2_model=edm2_model
        )
        
        mock_accelerator.accumulate.assert_called_once_with(mock_training_model, edm2_model)

