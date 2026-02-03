"""
Unit tests for library/training/phases/model_prep.py

Tests the model preparation phase functions with mocked trainer state.
"""

import pytest
from unittest.mock import MagicMock, patch
import torch


@pytest.mark.training
@pytest.mark.unit
class TestPrepareModels:
    """Test prepare_models orchestration function."""

    def test_lazy_loads_unet_when_none(self, mock_trainer):
        """Test that UNet is lazy loaded when trainer.unet is None."""
        mock_trainer.unet = None
        mock_unet = MagicMock()
        mock_text_encoders = [MagicMock()]
        mock_trainer.strategies.load_unet_lazily.return_value = (mock_unet, mock_text_encoders)

        with patch("library.training.phases.model_prep.create_adapter"), patch("library.training.phases.model_prep.configure_precision"):
            from library.training.phases.model_prep import prepare_models

            prepare_models(mock_trainer)

            mock_trainer.strategies.load_unet_lazily.assert_called_once()
            assert mock_trainer.unet is mock_unet

    def test_skips_lazy_load_when_unet_exists(self, mock_trainer):
        """Test that lazy load is skipped when UNet already present."""
        original_unet = mock_trainer.unet

        with patch("library.training.phases.model_prep.create_adapter"), patch("library.training.phases.model_prep.configure_precision"):
            from library.training.phases.model_prep import prepare_models

            prepare_models(mock_trainer)

            mock_trainer.strategies.load_unet_lazily.assert_not_called()
            assert mock_trainer.unet is original_unet

    def test_calls_create_adapter_and_configure_precision(self, mock_trainer):
        """Test that both sub-functions are called."""
        with (
            patch("library.training.phases.model_prep.create_adapter") as mock_create,
            patch("library.training.phases.model_prep.configure_precision") as mock_config,
        ):
            from library.training.phases.model_prep import prepare_models

            prepare_models(mock_trainer)

            mock_create.assert_called_once_with(mock_trainer)
            mock_config.assert_called_once_with(mock_trainer)


@pytest.mark.training
@pytest.mark.unit
class TestCreateAdapter:
    """Test create_adapter function."""

    def test_imports_adapter_module(self, mock_trainer):
        """Test that adapter module is dynamically imported."""
        mock_adapter = MagicMock()
        mock_module = MagicMock()
        mock_module.create_adapter.return_value = mock_adapter

        with (
            patch("library.training.phases.model_prep.importlib.import_module", return_value=mock_module),
            patch("library.training.phases.model_prep.resolve_adapter_kwargs"),
        ):
            from library.training.phases.model_prep import create_adapter

            create_adapter(mock_trainer)

            assert mock_trainer.adapter is mock_adapter

    def test_applies_adapter_to_models(self, mock_trainer):
        """Test that adapter.apply_to is called with training flags."""
        mock_adapter = MagicMock()
        mock_module = MagicMock()
        mock_module.create_adapter.return_value = mock_adapter

        with (
            patch("library.training.phases.model_prep.importlib.import_module", return_value=mock_module),
            patch("library.training.phases.model_prep.resolve_adapter_kwargs"),
        ):
            from library.training.phases.model_prep import create_adapter

            create_adapter(mock_trainer)

            mock_adapter.apply_to.assert_called_once()
            # Verify training flags were set
            assert hasattr(mock_trainer, "_train_unet")
            assert hasattr(mock_trainer, "_train_text_encoder")

    def test_loads_weights_when_specified(self, mock_trainer):
        """Test that adapter weights are loaded when path specified."""
        mock_trainer.cfg.peft.adapter_weights = "/path/to/weights.safetensors"
        mock_adapter = MagicMock()
        mock_adapter.load_weights.return_value = "loaded"
        mock_module = MagicMock()
        mock_module.create_adapter.return_value = mock_adapter

        with (
            patch("library.training.phases.model_prep.importlib.import_module", return_value=mock_module),
            patch("library.training.phases.model_prep.resolve_adapter_kwargs"),
        ):
            from library.training.phases.model_prep import create_adapter

            create_adapter(mock_trainer)

            mock_adapter.load_weights.assert_called_once_with("/path/to/weights.safetensors")

    def test_merges_base_weights(self, mock_trainer):
        """Test base weights merging when specified."""
        mock_trainer.cfg.peft.base_weights = ["/path/to/base.safetensors"]
        mock_trainer.cfg.peft.base_weights_multiplier = [0.5]
        mock_adapter = MagicMock()
        mock_merge_module = MagicMock()
        mock_merge_module.merge_to = MagicMock()
        mock_module = MagicMock()
        mock_module.create_adapter.return_value = mock_adapter
        mock_module.create_adapter_from_weights.return_value = (mock_merge_module, {})

        with (
            patch("library.training.phases.model_prep.importlib.import_module", return_value=mock_module),
            patch("library.training.phases.model_prep.resolve_adapter_kwargs"),
        ):
            from library.training.phases.model_prep import create_adapter

            create_adapter(mock_trainer)

            mock_module.create_adapter_from_weights.assert_called()
            mock_merge_module.merge_to.assert_called_once()


@pytest.mark.training
@pytest.mark.unit
class TestConfigurePrecision:
    """Test configure_precision function."""

    def test_full_fp16_casts_adapter(self, mock_trainer):
        """Test adapter is cast to weight_dtype when full_fp16=True."""
        mock_trainer.cfg.performance.precision.full_fp16 = True
        mock_trainer.weight_dtype = torch.float16

        from library.training.phases.model_prep import configure_precision

        configure_precision(mock_trainer)

        mock_trainer.adapter.to.assert_called_with(torch.float16)

    def test_full_bf16_casts_adapter(self, mock_trainer):
        """Test adapter is cast to weight_dtype when full_bf16=True."""
        mock_trainer.cfg.performance.precision.full_bf16 = True
        mock_trainer.weight_dtype = torch.bfloat16

        from library.training.phases.model_prep import configure_precision

        configure_precision(mock_trainer)

        mock_trainer.adapter.to.assert_called_with(torch.bfloat16)

    def test_fp8_base_sets_unet_dtype(self, mock_trainer):
        """Test fp8_base sets unet_weight_dtype to float8."""
        mock_trainer.cfg.performance.precision.fp8_base = True

        from library.training.phases.model_prep import configure_precision

        configure_precision(mock_trainer)

        assert mock_trainer.unet_weight_dtype == torch.float8_e4m3fn

    def test_disables_unet_gradients(self, mock_trainer):
        """Test that UNet gradients are disabled."""
        from library.training.phases.model_prep import configure_precision

        configure_precision(mock_trainer)

        mock_trainer.unet.requires_grad_.assert_called_with(False)

    def test_disables_text_encoder_gradients(self, mock_trainer):
        """Test that text encoder gradients are disabled."""
        from library.training.phases.model_prep import configure_precision

        configure_precision(mock_trainer)

        for te in mock_trainer.text_encoders:
            te.requires_grad_.assert_called_with(False)
