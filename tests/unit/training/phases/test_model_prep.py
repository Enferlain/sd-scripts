"""
Unit tests for library/training/phases/model_prep.py

Tests the model preparation phase functions with mocked trainer state.
After the TrainingMode refactor, create_adapter logic moved to PeftMode.prepare_trainables()
and adapter-specific precision logic moved to PeftMode.configure_trainable_precision().
"""

import pytest
from unittest.mock import MagicMock, patch
import torch


@pytest.mark.training
@pytest.mark.unit
class TestPrepareModels:
    """Test prepare_models orchestration function."""

    def test_lazy_loads_denoiser_when_none(self, mock_trainer):
        """Test that the denoiser is lazy loaded when trainer.denoiser is None."""
        mock_trainer.denoiser = None
        mock_denoiser = MagicMock()
        mock_text_encoders = [MagicMock()]
        mock_trainer.strategies.load_denoiser_lazily.return_value = (mock_denoiser, mock_text_encoders)

        with patch("library.training.phases.model_prep.configure_precision"):
            from library.training.phases.model_prep import prepare_models

            prepare_models(mock_trainer)

            mock_trainer.strategies.load_denoiser_lazily.assert_called_once()
            assert mock_trainer.denoiser is mock_denoiser

    def test_skips_lazy_load_when_denoiser_exists(self, mock_trainer):
        """Test that lazy load is skipped when the denoiser is already present."""
        original_denoiser = mock_trainer.denoiser

        with patch("library.training.phases.model_prep.configure_precision"):
            from library.training.phases.model_prep import prepare_models

            prepare_models(mock_trainer)

            mock_trainer.strategies.load_denoiser_lazily.assert_not_called()
            assert mock_trainer.denoiser is original_denoiser

    def test_calls_mode_hooks_and_configure_precision(self, mock_trainer):
        """Test that mode.prepare_trainables, configure_precision, and mode.configure_trainable_precision are called."""
        with patch("library.training.phases.model_prep.configure_precision") as mock_config:
            from library.training.phases.model_prep import prepare_models

            prepare_models(mock_trainer)

            mock_trainer.mode.prepare_trainables.assert_called_once_with(mock_trainer)
            mock_config.assert_called_once_with(mock_trainer)
            mock_trainer.mode.configure_trainable_precision.assert_called_once_with(mock_trainer)


@pytest.mark.training
@pytest.mark.unit
class TestConfigurePrecision:
    """Test configure_precision function (shared concerns only).

    After the TrainingMode refactor, configure_precision only handles:
    - denoiser/TE/VAE dtype casting
    - fp8_base setup
    - Text encoder fp8 preparation
    Adapter-specific casting and gradient disabling moved to mode.configure_trainable_precision().
    """

    def test_fp8_base_sets_denoiser_dtype(self, mock_trainer):
        """Test fp8_base sets denoiser_weight_dtype to float8."""
        mock_trainer.cfg.performance.precision.fp8_base = True

        from library.training.phases.model_prep import configure_precision

        configure_precision(mock_trainer)

        assert mock_trainer.denoiser_weight_dtype == torch.float8_e4m3fn

    def test_denoiser_cast_when_strategy_says(self, mock_trainer):
        """Test the denoiser is cast to denoiser_weight_dtype when strategy says to."""
        mock_trainer.strategies.cast_denoiser.return_value = True
        mock_trainer.denoiser_weight_dtype = torch.float16

        from library.training.phases.model_prep import configure_precision

        configure_precision(mock_trainer)

        mock_trainer.denoiser.to.assert_called_with(dtype=torch.float16)

    def test_te_cast_when_strategy_says(self, mock_trainer):
        """Test text encoders are cast to te_weight_dtype when strategy says to."""
        mock_trainer.strategies.cast_text_encoder.return_value = True
        mock_trainer.te_weight_dtype = torch.float16

        from library.training.phases.model_prep import configure_precision

        configure_precision(mock_trainer)

        for te in mock_trainer.text_encoders:
            te.to.assert_called_with(dtype=torch.float16)

    def test_no_cast_when_strategy_says_no(self, mock_trainer):
        """Test no casting when strategy returns False."""
        mock_trainer.strategies.cast_denoiser.return_value = False
        mock_trainer.strategies.cast_text_encoder.return_value = False

        from library.training.phases.model_prep import configure_precision

        configure_precision(mock_trainer)

        mock_trainer.denoiser.to.assert_not_called()
        for te in mock_trainer.text_encoders:
            te.to.assert_not_called()
