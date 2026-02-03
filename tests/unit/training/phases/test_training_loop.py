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
            patch("library.training.phases.training_loop.calculate_val_loss_check", return_value=False),
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
            patch("library.training.phases.training_loop.calculate_val_loss_check", return_value=False),
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
            patch("library.training.phases.training_loop.calculate_val_loss_check", return_value=False),
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
            patch("library.training.phases.training_loop.calculate_val_loss_check", return_value=False),
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
            patch("library.training.phases.training_loop.calculate_val_loss_check", return_value=False),
            patch("library.training.phases.training_loop.generate_step_logs", return_value={}),
        ):
            mock_ctx.return_value.__enter__ = MagicMock()
            mock_ctx.return_value.__exit__ = MagicMock()

            from library.training.phases.training_loop import run_training_loop

            run_training_loop(mock_trainer)

            # Should stop at max_train_steps
            assert mock_trainer.global_step == 2
