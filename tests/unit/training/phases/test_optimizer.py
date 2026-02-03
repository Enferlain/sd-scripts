"""
Unit tests for library/training/phases/optimizer.py

Tests the optimizer phase functions with mocked trainer state.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.training
@pytest.mark.unit
class TestCalculateMaxTrainSteps:
    """Test calculate_max_train_steps function."""

    def test_basic_calculation(self):
        """Test basic step calculation."""
        from library.training.phases.optimizer import calculate_max_train_steps

        result = calculate_max_train_steps(
            max_train_epochs=10,
            num_batches_per_epoch=100,
            num_processes=1,
            gradient_accumulation_steps=1,
        )

        assert result == 1000  # 10 * 100 / 1 / 1

    def test_with_gradient_accumulation(self):
        """Test step calculation with gradient accumulation."""
        from library.training.phases.optimizer import calculate_max_train_steps

        result = calculate_max_train_steps(
            max_train_epochs=10,
            num_batches_per_epoch=100,
            num_processes=1,
            gradient_accumulation_steps=4,
        )

        assert result == 250  # 10 * 100 / 1 / 4

    def test_with_multiple_processes(self):
        """Test step calculation with multi-GPU."""
        from library.training.phases.optimizer import calculate_max_train_steps

        result = calculate_max_train_steps(
            max_train_epochs=10,
            num_batches_per_epoch=100,
            num_processes=2,
            gradient_accumulation_steps=1,
        )

        assert result == 500  # 10 * 100 / 2 / 1

    def test_rounds_up(self):
        """Test that result uses ceiling for batches per epoch."""
        from library.training.phases.optimizer import calculate_max_train_steps

        result = calculate_max_train_steps(
            max_train_epochs=3,
            num_batches_per_epoch=10,
            num_processes=1,
            gradient_accumulation_steps=4,
        )

        # 3 * ceil(10 / 1 / 4) = 3 * ceil(2.5) = 3 * 3 = 9
        assert result == 9
        assert isinstance(result, int)


@pytest.mark.training
@pytest.mark.unit
class TestPrepareOptimizer:
    """Test prepare_optimizer function."""

    def test_creates_optimizer(self, mock_trainer):
        """Test that optimizer is created and assigned."""
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()

        with (
            patch("library.training.phases.optimizer._prepare_optimizer_util") as mock_prep_opt,
            patch("library.training.phases.optimizer.get_scheduler_fix", return_value=mock_scheduler),
            patch("library.training.phases.optimizer._prepare_with_accelerator"),
            patch("library.training.phases.optimizer._setup_gradient_checkpointing"),
            patch("library.training.phases.optimizer.register_adapter_state_hooks") as mock_register,
            patch("library.training.phases.optimizer.resume_from_local_or_hf_if_specified"),
        ):
            mock_prep_opt.return_value = (
                "AdamW",  # optimizer_name
                {},  # optimizer_args
                mock_optimizer,  # optimizer
                MagicMock(),  # train_fn
                MagicMock(),  # eval_fn
                ["unet"],  # lr_descriptions
            )
            mock_register.return_value = MagicMock(return_value=None)

            from library.training.phases.optimizer import prepare_optimizer

            prepare_optimizer(mock_trainer)

            mock_prep_opt.assert_called_once()
            assert mock_trainer.optimizer is mock_optimizer

    def test_creates_lr_scheduler(self, mock_trainer):
        """Test that LR scheduler is created and assigned."""
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()

        with (
            patch("library.training.phases.optimizer._prepare_optimizer_util") as mock_prep_opt,
            patch("library.training.phases.optimizer.get_scheduler_fix", return_value=mock_scheduler) as mock_get_sched,
            patch("library.training.phases.optimizer._prepare_with_accelerator"),
            patch("library.training.phases.optimizer._setup_gradient_checkpointing"),
            patch("library.training.phases.optimizer.register_adapter_state_hooks") as mock_register,
            patch("library.training.phases.optimizer.resume_from_local_or_hf_if_specified"),
        ):
            mock_prep_opt.return_value = (
                "AdamW",
                {},
                mock_optimizer,
                MagicMock(),
                MagicMock(),
                ["unet"],
            )
            mock_register.return_value = MagicMock(return_value=None)

            from library.training.phases.optimizer import prepare_optimizer

            prepare_optimizer(mock_trainer)

            mock_get_sched.assert_called_once()
            assert mock_trainer.lr_scheduler is mock_scheduler


@pytest.mark.training
@pytest.mark.unit
class TestPrepareWithAccelerator:
    """Test _prepare_with_accelerator function."""

    def test_calls_accelerator_prepare(self, mock_trainer):
        """Test that _prepare_with_accelerator calls accelerator.prepare."""
        # Set up config to take non-deepspeed path
        mock_trainer.cfg.performance.deepspeed = False
        mock_trainer._train_unet = False
        mock_trainer._train_text_encoder = False

        from library.training.phases.optimizer import _prepare_with_accelerator

        _prepare_with_accelerator(mock_trainer)

        # Should call accelerator.prepare with adapter, optimizer, lr_scheduler
        mock_trainer.accelerator.prepare.assert_called()


@pytest.mark.training
@pytest.mark.unit
class TestSetupGradientCheckpointing:
    """Test _setup_gradient_checkpointing function."""

    def test_enables_when_configured(self, mock_trainer):
        """Test gradient checkpointing is enabled when config says so."""
        mock_trainer.cfg.performance.memory.gradient_checkpointing = True

        from library.training.phases.optimizer import _setup_gradient_checkpointing

        _setup_gradient_checkpointing(mock_trainer)

        mock_trainer.unet.enable_gradient_checkpointing.assert_called()

    def test_skips_when_disabled(self, mock_trainer):
        """Test gradient checkpointing is skipped when disabled."""
        mock_trainer.cfg.performance.memory.gradient_checkpointing = False

        from library.training.phases.optimizer import _setup_gradient_checkpointing

        _setup_gradient_checkpointing(mock_trainer)

        mock_trainer.unet.enable_gradient_checkpointing.assert_not_called()
