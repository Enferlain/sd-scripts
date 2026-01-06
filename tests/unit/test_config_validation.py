"""
Unit tests for the centralized config validation module.

Tests prepare_config auto-fixups, validate_config errors/warnings,
and script-specific validators.
"""

import pytest
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

from library.config.config_validation import (
    prepare_config,
    validate_config,
    validate_sd_peft,
    validate_sdxl_peft,
    validate_sd_textual_inversion,
    validate_sdxl_textual_inversion,
)


# =============================================================================
# prepare_config Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.config
class TestPrepareConfig:
    """Test prepare_config auto-fixups."""

    def test_cache_latents_to_disk_enables_cache_latents(self):
        """cache_latents_to_disk should automatically enable cache_latents."""
        cfg = OmegaConf.create(
            {
                "data": {"caching": {"cache_latents": False, "cache_latents_to_disk": True}, "caption": {"caption_extention": None}},
                "optimizer": {"use_8bit_adam": False, "use_lion_optimizer": False, "optimizer_type": ""},
                "output": {"sampling": {"sample_every_n_epochs": None, "sample_every_n_steps": None}},
            }
        )
        prepare_config(cfg)
        assert cfg.data.caching.cache_latents is True

    def test_cache_latents_unchanged_if_disk_false(self):
        """cache_latents should remain False if cache_latents_to_disk is False."""
        cfg = OmegaConf.create(
            {
                "data": {"caching": {"cache_latents": False, "cache_latents_to_disk": False}, "caption": {"caption_extention": None}},
                "optimizer": {"use_8bit_adam": False, "use_lion_optimizer": False, "optimizer_type": ""},
                "output": {"sampling": {"sample_every_n_epochs": None, "sample_every_n_steps": None}},
            }
        )
        prepare_config(cfg)
        assert cfg.data.caching.cache_latents is False

    def test_caption_extention_backward_compat(self):
        """caption_extention (typo) should copy to caption_extension."""
        cfg = OmegaConf.create(
            {
                "data": {
                    "caching": {"cache_latents": False, "cache_latents_to_disk": False},
                    "caption": {"caption_extention": ".txt", "caption_extension": ".caption"},
                },
                "optimizer": {"use_8bit_adam": False, "use_lion_optimizer": False, "optimizer_type": ""},
                "output": {"sampling": {"sample_every_n_epochs": None, "sample_every_n_steps": None}},
            }
        )
        prepare_config(cfg)
        assert cfg.data.caption.caption_extension == ".txt"

    def test_sdxl_cache_text_encoder_outputs_to_disk_enables_parent(self):
        """SDXL cache_text_encoder_outputs_to_disk enables cache_text_encoder_outputs."""
        cfg = OmegaConf.create(
            {
                "data": {
                    "caching": {
                        "cache_latents": False,
                        "cache_latents_to_disk": False,
                        "cache_text_encoder_outputs": False,
                        "cache_text_encoder_outputs_to_disk": True,
                    },
                    "caption": {"caption_extention": None},
                },
                "optimizer": {"use_8bit_adam": False, "use_lion_optimizer": False, "optimizer_type": ""},
                "output": {"sampling": {"sample_every_n_epochs": None, "sample_every_n_steps": None}},
                # performance key required for prepare_config to run TE cache fixup (line 46 guard)
                "performance": {},
            }
        )
        prepare_config(cfg)
        assert cfg.data.caching.cache_text_encoder_outputs is True

    def test_use_8bit_adam_sets_optimizer_type(self):
        """use_8bit_adam should set optimizer_type to AdamW8bit."""
        cfg = OmegaConf.create(
            {
                "data": {"caching": {"cache_latents": False, "cache_latents_to_disk": False}, "caption": {"caption_extention": None}},
                "optimizer": {"use_8bit_adam": True, "use_lion_optimizer": False, "optimizer_type": ""},
                "output": {"sampling": {"sample_every_n_epochs": None, "sample_every_n_steps": None}},
            }
        )
        prepare_config(cfg)
        assert cfg.optimizer.optimizer_type == "AdamW8bit"

    def test_use_lion_optimizer_sets_optimizer_type(self):
        """use_lion_optimizer should set optimizer_type to Lion."""
        cfg = OmegaConf.create(
            {
                "data": {"caching": {"cache_latents": False, "cache_latents_to_disk": False}, "caption": {"caption_extention": None}},
                "optimizer": {"use_8bit_adam": False, "use_lion_optimizer": True, "optimizer_type": ""},
                "output": {"sampling": {"sample_every_n_epochs": None, "sample_every_n_steps": None}},
            }
        )
        prepare_config(cfg)
        assert cfg.optimizer.optimizer_type == "Lion"

    def test_sample_every_n_epochs_zero_becomes_none(self):
        """sample_every_n_epochs <= 0 should become None."""
        cfg = OmegaConf.create(
            {
                "data": {"caching": {"cache_latents": False, "cache_latents_to_disk": False}, "caption": {"caption_extention": None}},
                "optimizer": {"use_8bit_adam": False, "use_lion_optimizer": False, "optimizer_type": ""},
                "output": {"sampling": {"sample_every_n_epochs": 0, "sample_every_n_steps": None}},
            }
        )
        prepare_config(cfg)
        assert cfg.output.sampling.sample_every_n_epochs is None

    def test_sample_every_n_steps_negative_becomes_none(self):
        """sample_every_n_steps <= 0 should become None."""
        cfg = OmegaConf.create(
            {
                "data": {"caching": {"cache_latents": False, "cache_latents_to_disk": False}, "caption": {"caption_extention": None}},
                "optimizer": {"use_8bit_adam": False, "use_lion_optimizer": False, "optimizer_type": ""},
                "output": {"sampling": {"sample_every_n_epochs": None, "sample_every_n_steps": -1}},
            }
        )
        prepare_config(cfg)
        assert cfg.output.sampling.sample_every_n_steps is None


# =============================================================================
# validate_config Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.config
class TestValidateConfig:
    """Test validate_config errors and warnings."""

    def test_adaptive_noise_scale_requires_noise_offset(self):
        """adaptive_noise_scale without noise_offset should raise ValueError."""
        cfg = OmegaConf.create(
            {
                "loss": {
                    "regularization": {"adaptive_noise_scale": 0.1, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd1"},
                "training": {"clip_skip": None},
            }
        )
        with pytest.raises(ValueError, match="adaptive_noise_scale requires noise_offset"):
            validate_config(cfg)

    def test_adaptive_noise_scale_with_noise_offset_passes(self):
        """adaptive_noise_scale with noise_offset should not raise."""
        cfg = OmegaConf.create(
            {
                "loss": {
                    "regularization": {"adaptive_noise_scale": 0.1, "noise_offset": 0.1, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd1"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"blocks": None}},
            }
        )
        validate_config(cfg)  # Should not raise

    def test_scale_v_pred_loss_requires_v_parameterization(self):
        """scale_v_pred_loss_like_noise_pred requires v_parameterization."""
        cfg = OmegaConf.create(
            {
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": True, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd1"},
                "training": {"clip_skip": None},
            }
        )
        with pytest.raises(ValueError, match="scale_v_pred_loss_like_noise_pred requires v_parameterization"):
            validate_config(cfg)

    def test_v_pred_like_loss_conflicts_with_v_parameterization(self):
        """v_pred_like_loss with v_parameterization should raise ValueError."""
        cfg = OmegaConf.create(
            {
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": 0.5},
                    "v_parameterization": True,
                },
                "model": {"model_type": "sd1"},
                "training": {"clip_skip": None},
            }
        )
        with pytest.raises(ValueError, match="v_pred_like_loss conflicts with v_parameterization"):
            validate_config(cfg)

    def test_v2_with_clip_skip_warns(self):
        """v2 model with clip_skip should log a warning."""
        cfg = OmegaConf.create(
            {
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd2"},
                "training": {"clip_skip": 2},
                "optimizer": {"learning_rates": {"blocks": None}},
            }
        )
        with patch("library.config.config_validation.logger") as mock_logger:
            validate_config(cfg)
            mock_logger.warning.assert_called_once()
            assert "v2 with clip_skip" in str(mock_logger.warning.call_args)

    def test_zero_terminal_snr_without_v_param_warns(self):
        """zero_terminal_snr without v_parameterization should log a warning."""
        cfg = OmegaConf.create(
            {
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": True},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd1"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"blocks": None}},
            }
        )
        with patch("library.config.config_validation.logger") as mock_logger:
            validate_config(cfg)
            mock_logger.warning.assert_called_once()
            assert "zero_terminal_snr" in str(mock_logger.warning.call_args)

    def test_full_fp16_requires_fp16_mixed_precision(self):
        """full_fp16 without mixed_precision='fp16' should raise ValueError."""
        cfg = OmegaConf.create(
            {
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd1"},
                "training": {"clip_skip": None},
                "performance": {"precision": {"full_fp16": True, "full_bf16": False, "mixed_precision": "bf16"}},
            }
        )
        with pytest.raises(ValueError, match="full_fp16 requires mixed_precision='fp16'"):
            validate_config(cfg)

    def test_full_bf16_requires_bf16_mixed_precision(self):
        """full_bf16 without mixed_precision='bf16' should raise ValueError."""
        cfg = OmegaConf.create(
            {
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd1"},
                "training": {"clip_skip": None},
                "performance": {"precision": {"full_fp16": False, "full_bf16": True, "mixed_precision": "fp16"}},
            }
        )
        with pytest.raises(ValueError, match="full_bf16 requires mixed_precision='bf16'"):
            validate_config(cfg)

    @pytest.mark.skip(reason="Block LR validation is model-specific, currently disabled pending refactor")
    def test_sdxl_block_lr_wrong_count_raises(self):
        """SDXL block_lr with wrong count should raise ValueError."""
        cfg = OmegaConf.create(
            {
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd1"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"blocks": "0.1,0.2,0.3"}},  # Only 3 values, need 23
            }
        )
        with pytest.raises(ValueError, match="block_lr must have 23 values"):
            validate_config(cfg)

    @pytest.mark.skip(reason="Block LR validation is model-specific, currently disabled pending refactor")
    def test_sdxl_block_lr_correct_count_passes(self):
        """SDXL block_lr with 23 values should not raise."""
        block_lrs = ",".join(["0.1"] * 23)
        cfg = OmegaConf.create(
            {
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd1"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"blocks": block_lrs}},
            }
        )
        validate_config(cfg)  # Should not raise


# =============================================================================
# Script-specific validator Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.config
class TestScriptSpecificValidators:
    """Test script-specific validation functions."""

    def test_validate_sd_peft_calls_verify_bucket_reso(self):
        """validate_sd_peft should verify bucket reso with 64 steps."""
        cfg = MagicMock()
        train_ds = MagicMock()
        val_ds = MagicMock()

        validate_sd_peft(cfg, train_ds, val_ds)

        train_ds.verify_bucket_reso_steps.assert_called_once_with(64)
        val_ds.verify_bucket_reso_steps.assert_called_once_with(64)

    def test_validate_sd_peft_no_val_dataset(self):
        """validate_sd_peft should handle None val_dataset."""
        cfg = MagicMock()
        train_ds = MagicMock()

        validate_sd_peft(cfg, train_ds, None)

        train_ds.verify_bucket_reso_steps.assert_called_once_with(64)

    def test_validate_sdxl_peft_calls_verify_bucket_reso_32(self):
        """validate_sdxl_peft should verify bucket reso with 32 steps."""
        cfg = MagicMock()
        cfg.data.caching.cache_text_encoder_outputs = False
        # Set TE LR to 0 (not training TE)
        cfg.optimizer.learning_rates.text_encoders = 0
        train_ds = MagicMock()
        val_ds = MagicMock()

        validate_sdxl_peft(cfg, train_ds, val_ds)

        train_ds.verify_bucket_reso_steps.assert_called_once_with(32)
        val_ds.verify_bucket_reso_steps.assert_called_once_with(32)

    def test_validate_sdxl_peft_cache_te_requires_cacheable(self):
        """SDXL cache_text_encoder_outputs requires dataset to be cacheable."""
        cfg = MagicMock()
        cfg.data.caching.cache_text_encoder_outputs = True
        # Set TE LR to 0 (not training TE, so caching is allowed)
        cfg.optimizer.learning_rates.text_encoders = 0
        train_ds = MagicMock()
        train_ds.is_text_encoder_output_cacheable.return_value = False

        with pytest.raises(AssertionError):
            validate_sdxl_peft(cfg, train_ds, None)

    def test_validate_sdxl_peft_cache_te_conflicts_with_te_training(self):
        """Cannot cache TE outputs while training TE peft."""
        cfg = MagicMock()
        cfg.data.caching.cache_text_encoder_outputs = True
        # TE LR > 0 means training TE, which conflicts with caching
        cfg.optimizer.learning_rates.text_encoders = 1e-5
        train_ds = MagicMock()
        train_ds.is_text_encoder_output_cacheable.return_value = True

        with pytest.raises(AssertionError):
            validate_sdxl_peft(cfg, train_ds, None)

    def test_validate_sd_textual_inversion_bucket_64(self):
        """SD textual inversion uses 64-step buckets."""
        cfg = MagicMock()
        train_ds = MagicMock()

        validate_sd_textual_inversion(cfg, train_ds, None)

        train_ds.verify_bucket_reso_steps.assert_called_once_with(64)

    def test_validate_sdxl_textual_inversion_bucket_32(self):
        """SDXL textual inversion uses 32-step buckets."""
        cfg = MagicMock()
        train_ds = MagicMock()

        validate_sdxl_textual_inversion(cfg, train_ds, None)

        train_ds.verify_bucket_reso_steps.assert_called_once_with(32)
