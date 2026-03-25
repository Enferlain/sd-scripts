"""
Unit tests for library/training/phases/optimizer.py

Tests the optimizer phase functions with mocked trainer state.
After the TrainingMode refactor, optimizer params building, accelerator preparation,
and gradient setup are delegated to trainer.mode.* hooks.
"""

import pytest
from unittest.mock import MagicMock, patch

from library.training.checkpointing import ResumeState


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
    """Test prepare_optimizer function.

    After the TrainingMode refactor, optimizer params are built via
    trainer.mode.build_optimizer_params() and accelerator wrapping via
    trainer.mode.prepare_with_accelerator().
    """

    def test_creates_optimizer(self, mock_trainer):
        """Test that optimizer is created via mode and assigned."""
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()

        # Configure mode.build_optimizer_params return
        mock_trainer.mode.build_optimizer_params.return_value = (
            "AdamW",  # optimizer_name
            {},  # optimizer_args
            mock_optimizer,  # optimizer
            MagicMock(),  # train_fn
            MagicMock(),  # eval_fn
            ["denoiser"],  # lr_descriptions
        )
        mock_trainer.mode.register_state_hooks.return_value = ResumeState()

        with (
            patch("library.training.phases.optimizer.get_scheduler_fix", return_value=mock_scheduler),
            patch("library.training.phases.optimizer._setup_gradient_checkpointing"),
            patch("library.training.phases.optimizer.resume_from_local_or_hf_if_specified"),
        ):
            from library.training.phases.optimizer import prepare_optimizer

            prepare_optimizer(mock_trainer)

            mock_trainer.mode.build_optimizer_params.assert_called_once_with(mock_trainer)
            assert mock_trainer.optimizer is mock_optimizer

    def test_creates_lr_scheduler(self, mock_trainer):
        """Test that LR scheduler is created and assigned."""
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()

        mock_trainer.mode.build_optimizer_params.return_value = (
            "AdamW",
            {},
            mock_optimizer,
            MagicMock(),
            MagicMock(),
            ["denoiser"],
        )
        mock_trainer.mode.register_state_hooks.return_value = ResumeState()

        with (
            patch("library.training.phases.optimizer.get_scheduler_fix", return_value=mock_scheduler) as mock_get_sched,
            patch("library.training.phases.optimizer._setup_gradient_checkpointing"),
            patch("library.training.phases.optimizer.resume_from_local_or_hf_if_specified"),
        ):
            from library.training.phases.optimizer import prepare_optimizer

            prepare_optimizer(mock_trainer)

            mock_get_sched.assert_called_once()
            assert mock_trainer.lr_scheduler is mock_scheduler

    def test_delegates_accelerator_prepare_to_mode(self, mock_trainer):
        """Test that accelerator preparation is delegated to mode."""
        mock_trainer.mode.build_optimizer_params.return_value = (
            "AdamW",
            {},
            MagicMock(),
            MagicMock(),
            MagicMock(),
            ["denoiser"],
        )
        mock_trainer.mode.register_state_hooks.return_value = ResumeState()

        with (
            patch("library.training.phases.optimizer.get_scheduler_fix", return_value=MagicMock()),
            patch("library.training.phases.optimizer._setup_gradient_checkpointing"),
            patch("library.training.phases.optimizer.resume_from_local_or_hf_if_specified"),
        ):
            from library.training.phases.optimizer import prepare_optimizer

            prepare_optimizer(mock_trainer)

            mock_trainer.mode.prepare_with_accelerator.assert_called_once_with(mock_trainer)

    def test_restores_resume_state_step(self, mock_trainer):
        """Test that explicit ResumeState.step restores optimizer resume counters."""
        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()

        mock_trainer.mode.build_optimizer_params.return_value = (
            "AdamW",
            {},
            mock_optimizer,
            MagicMock(),
            MagicMock(),
            ["denoiser"],
        )
        mock_trainer.mode.register_state_hooks.return_value = ResumeState(epoch=2, step=15)

        with (
            patch("library.training.phases.optimizer.get_scheduler_fix", return_value=mock_scheduler),
            patch("library.training.phases.optimizer._setup_gradient_checkpointing"),
            patch("library.training.phases.optimizer.resume_from_local_or_hf_if_specified"),
        ):
            from library.training.phases.optimizer import prepare_optimizer

            prepare_optimizer(mock_trainer)

            assert mock_trainer._initial_step == 15
            assert mock_trainer.epoch_to_start == 1
            assert mock_trainer.global_step == 15


@pytest.mark.training
@pytest.mark.unit
class TestSetupGradientCheckpointing:
    """Test _setup_gradient_checkpointing function."""

    def test_enables_when_configured(self, mock_trainer):
        """Test gradient checkpointing is enabled when config says so."""
        mock_trainer.cfg.performance.memory.gradient_checkpointing = True

        from library.training.phases.optimizer import _setup_gradient_checkpointing

        _setup_gradient_checkpointing(mock_trainer)

        mock_trainer.denoiser.enable_gradient_checkpointing.assert_called()

    def test_skips_when_disabled(self, mock_trainer):
        """Test gradient checkpointing is skipped when disabled."""
        mock_trainer.cfg.performance.memory.gradient_checkpointing = False

        from library.training.phases.optimizer import _setup_gradient_checkpointing

        _setup_gradient_checkpointing(mock_trainer)

        mock_trainer.denoiser.enable_gradient_checkpointing.assert_not_called()

    def test_delegates_adapter_gradient_to_mode(self, mock_trainer):
        """Test that adapter-specific gradient setup is delegated to mode."""
        mock_trainer.cfg.performance.memory.gradient_checkpointing = False

        from library.training.phases.optimizer import _setup_gradient_checkpointing

        _setup_gradient_checkpointing(mock_trainer)

        mock_trainer.mode.setup_gradient_training.assert_called_once_with(mock_trainer)
