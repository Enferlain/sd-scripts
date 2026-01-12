"""
Unit tests for library/training/sdxl_checkpointing.py

Tests the SDXL-specific checkpointing functions which wrap the common checkpointing logic.
"""

import pytest
from unittest.mock import patch, MagicMock
import torch

from library.config.dataclasses.output import SavingConfig
from library.config.dataclasses.output import MetadataConfig
from library.config.dataclasses.loss import LossConfig
from library.training.sdxl_checkpointing import (
    save_sd_model_on_train_end,
    save_sd_model_on_epoch_end_or_stepwise,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def saving_config():
    """Basic SavingConfig for testing."""
    return SavingConfig(
        output_dir="./output",
        output_name="test_model",
        save_model_as="safetensors",
        save_every_n_epochs=1,
        save_every_n_steps=None,
    )


@pytest.fixture
def loss_config():
    """LossConfig for testing."""
    return LossConfig(v_parameterization=False)


@pytest.fixture
def metadata_config():
    """Basic MetadataConfig for testing."""
    return MetadataConfig()


@pytest.fixture
def mock_models():
    """Mock model objects."""
    return {
        "text_encoder1": MagicMock(name="text_encoder1"),
        "text_encoder2": MagicMock(name="text_encoder2"),
        "unet": MagicMock(name="unet"),
        "vae": MagicMock(name="vae"),
        "logit_scale": MagicMock(name="logit_scale"),
        "ckpt_info": MagicMock(name="ckpt_info"),
    }


@pytest.fixture
def mock_accelerator():
    """Mock Accelerator object."""
    accelerator = MagicMock()
    accelerator.is_main_process = True
    return accelerator


# =============================================================================
# Tests: save_sd_model_on_train_end
# =============================================================================


class TestSaveSDModelOnTrainEnd:
    """Tests for save_sd_model_on_train_end function."""

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_train_end_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_calls_common_function(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, loss_config, mock_models
    ):
        """Should call save_sd_model_on_train_end_common with correct args."""
        save_sd_model_on_train_end(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_config,
            src_path="/path/to/model",
            save_stable_diffusion_format=True,
            use_safetensors=True,
            save_dtype=torch.float16,
            epoch=10,
            global_step=1000,
            **mock_models,
        )

        mock_common.assert_called_once()
        call_args = mock_common.call_args

        # Verify key arguments
        assert call_args.args[0] is saving_config
        assert call_args.args[1] is True  # save_stable_diffusion_format
        assert call_args.args[2] is True  # use_safetensors
        assert call_args.args[3] == 10  # epoch
        assert call_args.args[4] == 1000  # global_step

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_train_end_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_sd_saver_callback_calls_sdxl_util(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, loss_config, mock_models
    ):
        """The sd_saver callback should call sdxl_model_util.save_stable_diffusion_checkpoint."""
        save_sd_model_on_train_end(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_config,
            src_path="/path/to/model",
            save_stable_diffusion_format=True,
            use_safetensors=True,
            save_dtype=torch.float16,
            epoch=10,
            global_step=1000,
            **mock_models,
        )

        # Get the sd_saver callback (5th positional arg)
        sd_saver = mock_common.call_args.args[5]

        # Call the callback
        sd_saver("/output/model.safetensors", 10, 1000)

        # Should call sdxl save function
        mock_sdxl_util.save_stable_diffusion_checkpoint.assert_called_once()

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_train_end_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_sd_saver_gets_sai_metadata(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, loss_config, mock_models
    ):
        """The sd_saver callback should get SAI metadata with SDXL flags."""
        mock_sai_spec.get_model_metadata_from_config.return_value = {"mock": "metadata"}

        save_sd_model_on_train_end(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_config,
            src_path="/path/to/model",
            save_stable_diffusion_format=True,
            use_safetensors=True,
            save_dtype=torch.float16,
            epoch=10,
            global_step=1000,
            **mock_models,
        )

        # Get and call the sd_saver callback
        sd_saver = mock_common.call_args.args[5]
        sd_saver("/output/model.safetensors", 10, 1000)

        # Check SAI spec was called with SDXL flags
        mock_sai_spec.get_model_metadata_from_config.assert_called_once()
        call_kwargs = mock_sai_spec.get_model_metadata_from_config.call_args.kwargs
        assert call_kwargs["is_sdxl"] is True
        assert call_kwargs["is_v2"] is False
        assert call_kwargs["is_stable_diffusion_ckpt"] is True

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_train_end_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_diffusers_saver_callback_calls_sdxl_util(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, loss_config, mock_models
    ):
        """The diffusers_saver callback should call sdxl_model_util.save_diffusers_checkpoint."""
        save_sd_model_on_train_end(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_config,
            src_path="/path/to/model",
            save_stable_diffusion_format=False,
            use_safetensors=True,
            save_dtype=torch.float16,
            epoch=10,
            global_step=1000,
            **mock_models,
        )

        # Get the diffusers_saver callback (6th positional arg)
        diffusers_saver = mock_common.call_args.args[6]

        # Call the callback
        diffusers_saver("/output/diffusers_model")

        # Should call diffusers save function
        mock_sdxl_util.save_diffusers_checkpoint.assert_called_once()

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_train_end_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_diffusers_saver_passes_models_and_config(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, loss_config, mock_models
    ):
        """The diffusers_saver should pass correct model objects."""
        save_sd_model_on_train_end(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_config,
            src_path="/path/to/source",
            save_stable_diffusion_format=False,
            use_safetensors=True,
            save_dtype=torch.float16,
            epoch=10,
            global_step=1000,
            **mock_models,
        )

        diffusers_saver = mock_common.call_args.args[6]
        diffusers_saver("/output/diffusers_model")

        call_args = mock_sdxl_util.save_diffusers_checkpoint.call_args
        assert call_args.args[0] == "/output/diffusers_model"
        assert call_args.args[1] is mock_models["text_encoder1"]
        assert call_args.args[2] is mock_models["text_encoder2"]
        assert call_args.args[3] is mock_models["unet"]
        assert call_args.args[4] == "/path/to/source"
        assert call_args.args[5] is mock_models["vae"]


# =============================================================================
# Tests: save_sd_model_on_epoch_end_or_stepwise
# =============================================================================


class TestSaveSDModelOnEpochEndOrStepwise:
    """Tests for save_sd_model_on_epoch_end_or_stepwise function."""

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_epoch_end_or_stepwise_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_calls_common_function(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, loss_config, mock_models, mock_accelerator
    ):
        """Should call save_sd_model_on_epoch_end_or_stepwise_common."""
        save_sd_model_on_epoch_end_or_stepwise(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_config,
            on_epoch_end=True,
            accelerator=mock_accelerator,
            src_path="/path/to/model",
            save_stable_diffusion_format=True,
            use_safetensors=True,
            save_dtype=torch.float16,
            epoch=5,
            num_train_epochs=10,
            global_step=500,
            **mock_models,
        )

        mock_common.assert_called_once()

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_epoch_end_or_stepwise_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_passes_on_epoch_end_flag(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, loss_config, mock_models, mock_accelerator
    ):
        """on_epoch_end flag should be passed to common function."""
        save_sd_model_on_epoch_end_or_stepwise(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_config,
            on_epoch_end=True,
            accelerator=mock_accelerator,
            src_path="/path/to/model",
            save_stable_diffusion_format=True,
            use_safetensors=True,
            save_dtype=torch.float16,
            epoch=5,
            num_train_epochs=10,
            global_step=500,
            **mock_models,
        )

        call_args = mock_common.call_args
        assert call_args.args[1] is True  # on_epoch_end

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_epoch_end_or_stepwise_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_passes_accelerator(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, loss_config, mock_models, mock_accelerator
    ):
        """Accelerator should be passed to common function."""
        save_sd_model_on_epoch_end_or_stepwise(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_config,
            on_epoch_end=True,
            accelerator=mock_accelerator,
            src_path="/path/to/model",
            save_stable_diffusion_format=True,
            use_safetensors=True,
            save_dtype=torch.float16,
            epoch=5,
            num_train_epochs=10,
            global_step=500,
            **mock_models,
        )

        call_args = mock_common.call_args
        assert call_args.args[2] is mock_accelerator

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_epoch_end_or_stepwise_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_passes_epochs_and_steps(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, loss_config, mock_models, mock_accelerator
    ):
        """Epoch and step values should be passed correctly."""
        save_sd_model_on_epoch_end_or_stepwise(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_config,
            on_epoch_end=False,  # stepwise save
            accelerator=mock_accelerator,
            src_path="/path/to/model",
            save_stable_diffusion_format=True,
            use_safetensors=True,
            save_dtype=torch.float16,
            epoch=3,
            num_train_epochs=20,
            global_step=750,
            **mock_models,
        )

        call_args = mock_common.call_args
        assert call_args.args[5] == 3  # epoch
        assert call_args.args[6] == 20  # num_train_epochs
        assert call_args.args[7] == 750  # global_step

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_epoch_end_or_stepwise_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_sd_saver_callback_saves_checkpoint(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, loss_config, mock_models, mock_accelerator
    ):
        """sd_saver callback should save checkpoint via sdxl_model_util."""
        save_sd_model_on_epoch_end_or_stepwise(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_config,
            on_epoch_end=True,
            accelerator=mock_accelerator,
            src_path="/path/to/model",
            save_stable_diffusion_format=True,
            use_safetensors=True,
            save_dtype=torch.float16,
            epoch=5,
            num_train_epochs=10,
            global_step=500,
            **mock_models,
        )

        # Get sd_saver (8th positional arg)
        sd_saver = mock_common.call_args.args[8]
        sd_saver("/output/epoch_5.safetensors", 5, 500)

        mock_sdxl_util.save_stable_diffusion_checkpoint.assert_called_once()

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_epoch_end_or_stepwise_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_diffusers_saver_callback_saves_checkpoint(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, loss_config, mock_models, mock_accelerator
    ):
        """diffusers_saver callback should save via sdxl_model_util."""
        save_sd_model_on_epoch_end_or_stepwise(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_config,
            on_epoch_end=True,
            accelerator=mock_accelerator,
            src_path="/path/to/model",
            save_stable_diffusion_format=False,
            use_safetensors=True,
            save_dtype=torch.float16,
            epoch=5,
            num_train_epochs=10,
            global_step=500,
            **mock_models,
        )

        # Get diffusers_saver (9th positional arg)
        diffusers_saver = mock_common.call_args.args[9]
        diffusers_saver("/output/diffusers_epoch_5")

        mock_sdxl_util.save_diffusers_checkpoint.assert_called_once()

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_epoch_end_or_stepwise_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_v_parameterization_passed_to_metadata(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, mock_models, mock_accelerator
    ):
        """v_parameterization from loss_config should be in metadata."""
        v_param_loss_config = LossConfig(v_parameterization=True)

        save_sd_model_on_epoch_end_or_stepwise(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=v_param_loss_config,
            on_epoch_end=True,
            accelerator=mock_accelerator,
            src_path="/path/to/model",
            save_stable_diffusion_format=True,
            use_safetensors=True,
            save_dtype=torch.float16,
            epoch=5,
            num_train_epochs=10,
            global_step=500,
            **mock_models,
        )

        sd_saver = mock_common.call_args.args[8]
        sd_saver("/output/model.safetensors", 5, 500)

        call_kwargs = mock_sai_spec.get_model_metadata_from_config.call_args.kwargs
        assert call_kwargs["v_parameterization"] is True


# =============================================================================
# Tests: Common Parameters Verification
# =============================================================================


class TestCommonParameters:
    """Tests verifying common patterns across both functions."""

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_train_end_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_train_end_is_sdxl_true(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, loss_config, mock_models
    ):
        """SAI metadata should always have is_sdxl=True for SDXL checkpointing."""
        save_sd_model_on_train_end(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_config,
            src_path="/path/to/model",
            save_stable_diffusion_format=True,
            use_safetensors=True,
            save_dtype=torch.float16,
            epoch=10,
            global_step=1000,
            **mock_models,
        )

        sd_saver = mock_common.call_args.args[5]
        sd_saver("/output/model.safetensors", 10, 1000)

        call_kwargs = mock_sai_spec.get_model_metadata_from_config.call_args.kwargs
        assert call_kwargs["is_sdxl"] is True
        assert call_kwargs["is_v2"] is False
        assert call_kwargs["is_lora"] is False
        assert call_kwargs["is_textual_inversion"] is False

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_train_end_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_save_dtype_passed_to_checkpoint(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, loss_config, mock_models
    ):
        """save_dtype should be passed to the checkpoint saving function."""
        save_sd_model_on_train_end(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_config,
            src_path="/path/to/model",
            save_stable_diffusion_format=True,
            use_safetensors=True,
            save_dtype=torch.bfloat16,  # Different dtype
            epoch=10,
            global_step=1000,
            **mock_models,
        )

        sd_saver = mock_common.call_args.args[5]
        sd_saver("/output/model.safetensors", 10, 1000)

        # Check save_dtype was passed (last argument)
        call_args = mock_sdxl_util.save_stable_diffusion_checkpoint.call_args.args
        assert call_args[-1] == torch.bfloat16

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_train_end_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_use_safetensors_passed_to_diffusers(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, loss_config, mock_models
    ):
        """use_safetensors should be passed to diffusers saver."""
        save_sd_model_on_train_end(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_config,
            src_path="/path/to/model",
            save_stable_diffusion_format=False,
            use_safetensors=False,  # Explicitly False
            save_dtype=torch.float16,
            epoch=10,
            global_step=1000,
            **mock_models,
        )

        diffusers_saver = mock_common.call_args.args[6]
        diffusers_saver("/output/diffusers_model")

        call_kwargs = mock_sdxl_util.save_diffusers_checkpoint.call_args.kwargs
        assert call_kwargs["use_safetensors"] is False

    @patch("library.training.sdxl_checkpointing.save_sd_model_on_train_end_common")
    @patch("library.training.sdxl_checkpointing.model_metadata")
    @patch("library.training.sdxl_checkpointing.conversion")
    def test_v_parameterization_from_loss_config(
        self, mock_sdxl_util, mock_sai_spec, mock_common, saving_config, metadata_config, mock_models
    ):
        """v_parameterization should come from LossConfig, not TrainingConfig."""
        # Use real LossConfig to verify it reads v_parameterization correctly
        loss_cfg = LossConfig(v_parameterization=True)

        save_sd_model_on_train_end(
            saving_config=saving_config,
            metadata_config=metadata_config,
            loss_config=loss_cfg,
            src_path="/path/to/model",
            save_stable_diffusion_format=True,
            use_safetensors=True,
            save_dtype=torch.float16,
            epoch=10,
            global_step=1000,
            **mock_models,
        )

        sd_saver = mock_common.call_args.args[5]
        sd_saver("/output/model.safetensors", 10, 1000)

        call_kwargs = mock_sai_spec.get_model_metadata_from_config.call_args.kwargs
        assert call_kwargs["v_parameterization"] is True
