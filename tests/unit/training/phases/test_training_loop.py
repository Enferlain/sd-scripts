"""
Unit tests for library/training/phases/training_loop.py

Tests the training loop phase function with mocked trainer state.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import torch


@pytest.mark.training
@pytest.mark.unit
class TestRunTrainingLoop:
    """Test run_training_loop function."""

    def test_creates_dataloader_per_epoch(self, mock_trainer):
        """Test that dataloader is created for each epoch."""
        mock_trainer.num_train_epochs = 2
        mock_trainer.epoch_to_start = 0
        mock_trainer.max_train_steps = 5

        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter([]))

        with (
            patch("library.training.phases.training_loop.prepare_epoch") as mock_prep,
            patch("library.training.phases.training_loop.create_training_dataloader", return_value=mock_dataloader) as mock_create,
            patch("library.training.phases.training_loop.CaptionConfig"),
        ):
            mock_prep.return_value = MagicMock()

            from library.training.phases.training_loop import run_training_loop

            run_training_loop(mock_trainer)

            # Should create dataloader for each epoch
            assert mock_create.call_count == 2

    def test_updates_global_step(self, mock_trainer):
        """Test that global_step is incremented on sync_gradients."""
        mock_trainer.num_train_epochs = 1
        mock_trainer.epoch_to_start = 0
        mock_trainer.global_step = 0
        mock_trainer.max_train_steps = 10

        # Create fake batches
        fake_batches = [{"latent": torch.randn(1, 4, 64, 64)} for _ in range(3)]
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter(fake_batches))

        # Ensure sync_gradients is True
        type(mock_trainer.accelerator).sync_gradients = PropertyMock(return_value=True)

        with (
            patch("library.training.phases.training_loop.prepare_epoch"),
            patch("library.training.phases.training_loop.create_training_dataloader", return_value=mock_dataloader),
            patch("library.training.phases.training_loop.CaptionConfig"),
            patch("library.training.phases.training_loop.determine_grad_sync_context") as mock_ctx,
            patch("library.training.phases.training_loop.sample_images_check", return_value=False),
            patch("library.training.phases.training_loop.generate_step_logs", return_value={}),
        ):
            # Context manager mock
            mock_ctx.return_value.__enter__ = MagicMock()
            mock_ctx.return_value.__exit__ = MagicMock()

            from library.training.phases.training_loop import run_training_loop

            run_training_loop(mock_trainer)

            # global_step should have increased
            assert mock_trainer.global_step == 3

    def test_calls_strategies_process_batch(self, mock_trainer):
        """Test that strategies.process_batch is called for each step."""
        mock_trainer.num_train_epochs = 1
        mock_trainer.epoch_to_start = 0
        mock_trainer.global_step = 0
        mock_trainer.max_train_steps = 10

        fake_batches = [{"latent": torch.randn(1, 4, 64, 64)} for _ in range(2)]
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter(fake_batches))

        with (
            patch("library.training.phases.training_loop.prepare_epoch"),
            patch("library.training.phases.training_loop.create_training_dataloader", return_value=mock_dataloader),
            patch("library.training.phases.training_loop.CaptionConfig"),
            patch("library.training.phases.training_loop.determine_grad_sync_context") as mock_ctx,
            patch("library.training.phases.training_loop.sample_images_check", return_value=False),
            patch("library.training.phases.training_loop.generate_step_logs", return_value={}),
        ):
            mock_ctx.return_value.__enter__ = MagicMock()
            mock_ctx.return_value.__exit__ = MagicMock()

            from library.training.phases.training_loop import run_training_loop

            run_training_loop(mock_trainer)

            # process_batch should be called for each batch
            assert mock_trainer.strategies.process_batch.call_count == 2

    def test_calls_accelerator_backward(self, mock_trainer):
        """Test that accelerator.backward is called with loss."""
        mock_trainer.num_train_epochs = 1
        mock_trainer.epoch_to_start = 0
        mock_trainer.global_step = 0
        mock_trainer.max_train_steps = 10

        fake_batches = [{"latent": torch.randn(1, 4, 64, 64)}]
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter(fake_batches))

        with (
            patch("library.training.phases.training_loop.prepare_epoch"),
            patch("library.training.phases.training_loop.create_training_dataloader", return_value=mock_dataloader),
            patch("library.training.phases.training_loop.CaptionConfig"),
            patch("library.training.phases.training_loop.determine_grad_sync_context") as mock_ctx,
            patch("library.training.phases.training_loop.sample_images_check", return_value=False),
            patch("library.training.phases.training_loop.generate_step_logs", return_value={}),
        ):
            mock_ctx.return_value.__enter__ = MagicMock()
            mock_ctx.return_value.__exit__ = MagicMock()

            from library.training.phases.training_loop import run_training_loop

            run_training_loop(mock_trainer)

            mock_trainer.accelerator.backward.assert_called()

    def test_updates_progress_bar_on_sync(self, mock_trainer):
        """Test that progress bar is updated when sync_gradients is True."""
        mock_trainer.num_train_epochs = 1
        mock_trainer.epoch_to_start = 0
        mock_trainer.global_step = 0
        mock_trainer.max_train_steps = 10

        fake_batches = [{"latent": torch.randn(1, 4, 64, 64)} for _ in range(3)]
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter(fake_batches))

        type(mock_trainer.accelerator).sync_gradients = PropertyMock(return_value=True)

        with (
            patch("library.training.phases.training_loop.prepare_epoch"),
            patch("library.training.phases.training_loop.create_training_dataloader", return_value=mock_dataloader),
            patch("library.training.phases.training_loop.CaptionConfig"),
            patch("library.training.phases.training_loop.determine_grad_sync_context") as mock_ctx,
            patch("library.training.phases.training_loop.sample_images_check", return_value=False),
            patch("library.training.phases.training_loop.generate_step_logs", return_value={}),
        ):
            mock_ctx.return_value.__enter__ = MagicMock()
            mock_ctx.return_value.__exit__ = MagicMock()

            from library.training.phases.training_loop import run_training_loop

            run_training_loop(mock_trainer)

            # Progress bar should be updated for each sync
            assert mock_trainer._progress_bar.update.call_count == 3

    @pytest.mark.skip(reason="Requires integration test - mock state tracking insufficient for loop break logic")
    def test_respects_max_train_steps(self, mock_trainer):
        """Test that loop stops at max_train_steps."""
        mock_trainer.num_train_epochs = 10  # Many epochs
        mock_trainer.epoch_to_start = 0
        mock_trainer.global_step = 0
        mock_trainer.max_train_steps = 2  # But only 2 steps

        fake_batches = [{"latent": torch.randn(1, 4, 64, 64)} for _ in range(100)]
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter(fake_batches))

        type(mock_trainer.accelerator).sync_gradients = PropertyMock(return_value=True)

        with (
            patch("library.training.phases.training_loop.prepare_epoch"),
            patch("library.training.phases.training_loop.create_training_dataloader", return_value=mock_dataloader),
            patch("library.training.phases.training_loop.CaptionConfig"),
            patch("library.training.phases.training_loop.determine_grad_sync_context") as mock_ctx,
            patch("library.training.phases.training_loop.sample_images_check", return_value=False),
            patch("library.training.phases.training_loop.generate_step_logs", return_value={}),
        ):
            mock_ctx.return_value.__enter__ = MagicMock()
            mock_ctx.return_value.__exit__ = MagicMock()

            from library.training.phases.training_loop import run_training_loop

            run_training_loop(mock_trainer)

            # Should stop at max_train_steps
            assert mock_trainer.global_step == 2


@pytest.mark.training
@pytest.mark.unit
class TestStepTrackingLogs:
    """Test step-log emission wiring."""

    def test_emit_step_tracking_logs_passes_optimization_plan(self, mock_trainer):
        """Step-log generation should receive the trainer optimization plan."""
        mock_trainer._accumulation_counter = 1
        mock_trainer._current_global_step_loss = 0.5
        mock_trainer.cfg.output.logging.log_every_n_steps = 1
        mock_trainer.optimization_plan = MagicMock()

        with (
            patch("library.training.phases.training_loop.generate_step_logs", return_value={}) as mock_generate_logs,
            patch("library.training.phases.training_loop.step_logging"),
        ):
            from library.training.phases.training_loop import _emit_step_tracking_logs

            _emit_step_tracking_logs(
                mock_trainer,
                timesteps=torch.tensor([10]),
                keys_scaled=None,
                mean_norm=None,
                maximum_norm=None,
            )

        assert mock_generate_logs.call_args.kwargs["optimization_plan"] is mock_trainer.optimization_plan
        assert mock_generate_logs.call_args.kwargs["lr_descriptions"] is None


@pytest.mark.training
@pytest.mark.unit
class TestValidationSamplingDecoupling:
    """Assert that validation and sampling triggers are behaviorally independent."""

    def _run_loop_with_triggers(self, mock_trainer, *, validation_returns, sample_returns):
        """Helper: run one epoch with explicit validation/sampling trigger control.

        Args:
            mock_trainer: Mock trainer fixture.
            validation_returns: Value that scheduler.should_run() returns.
            sample_returns: Value that sample_images_check() returns.
        """
        mock_trainer.num_train_epochs = 1
        mock_trainer.epoch_to_start = 0
        mock_trainer.global_step = 0
        mock_trainer.max_train_steps = 10

        # One batch per epoch
        fake_batches = [{"latent": torch.randn(1, 4, 64, 64)}]
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter(fake_batches))

        type(mock_trainer.accelerator).sync_gradients = PropertyMock(return_value=True)

        # Wire scheduler mock
        mock_trainer._validation_scheduler.should_run = MagicMock(return_value=validation_returns)
        mock_trainer.strategies.calculate_val_loss.return_value = (0.5, 0.5)

        with (
            patch("library.training.phases.training_loop.prepare_epoch"),
            patch("library.training.phases.training_loop.create_training_dataloader", return_value=mock_dataloader),
            patch("library.training.phases.training_loop.CaptionConfig"),
            patch("library.training.phases.training_loop.determine_grad_sync_context") as mock_ctx,
            patch("library.training.phases.training_loop.sample_images_check", return_value=sample_returns),
            patch("library.training.phases.training_loop.generate_step_logs", return_value={}),
        ):
            mock_ctx.return_value.__enter__ = MagicMock()
            mock_ctx.return_value.__exit__ = MagicMock()

            from library.training.phases.training_loop import run_training_loop

            run_training_loop(mock_trainer)

    def test_validation_trigger_does_not_force_sampling(self, mock_trainer):
        """Validation triggering should NOT cause sample_images to be called."""
        self._run_loop_with_triggers(mock_trainer, validation_returns=True, sample_returns=False)

        # Validation ran
        mock_trainer.strategies.calculate_val_loss.assert_called()
        # Sampling did NOT run
        mock_trainer.strategies.sample_images.assert_not_called()

    def test_validation_call_uses_epoch_then_batch_argument_order(self, mock_trainer):
        """Validation call should pass epoch, batch, then train_text_encoder.

        Regression guard for the active shared path: the strategy contract is
        ``(..., cfg, epoch, batch=None, train_text_encoder=True)``.
        """
        self._run_loop_with_triggers(mock_trainer, validation_returns=True, sample_returns=False)

        call_args = mock_trainer.strategies.calculate_val_loss.call_args
        args = call_args.args

        assert args[15] == mock_trainer._current_epoch_state.value
        assert isinstance(args[16], dict)
        assert args[16]["latent"].shape == (1, 4, 64, 64)
        assert args[17] is mock_trainer._train_text_encoder

    def test_sampling_trigger_does_not_force_validation(self, mock_trainer):
        """Sampling triggering should NOT cause calculate_val_loss to be called."""
        self._run_loop_with_triggers(mock_trainer, validation_returns=False, sample_returns=True)

        # Sampling ran
        mock_trainer.strategies.sample_images.assert_called()
        # Validation did NOT run
        mock_trainer.strategies.calculate_val_loss.assert_not_called()

    def test_both_triggers_fire_independently(self, mock_trainer):
        """Both triggers firing should call both actions."""
        self._run_loop_with_triggers(mock_trainer, validation_returns=True, sample_returns=True)

        mock_trainer.strategies.sample_images.assert_called()
        mock_trainer.strategies.calculate_val_loss.assert_called()

    def test_eval_mode_entered_when_validation_only(self, mock_trainer):
        """Eval mode should be entered even if only validation triggers."""
        self._run_loop_with_triggers(mock_trainer, validation_returns=True, sample_returns=False)

        mock_trainer.mode.set_eval.assert_called()
        mock_trainer.mode.set_train.assert_called()

    def test_neither_trigger_skips_eval_mode(self, mock_trainer):
        """Neither trigger firing should NOT enter eval mode."""
        self._run_loop_with_triggers(mock_trainer, validation_returns=False, sample_returns=False)

        mock_trainer.mode.set_eval.assert_not_called()
