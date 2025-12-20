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
