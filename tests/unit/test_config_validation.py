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
    validate_dataset_groups,
)


def make_prepare_cfg(overrides: dict | None = None):
    """Build a minimal config shape suitable for prepare_config tests."""
    base = {
        "data": {
            "caching": {
                "cache_latents": False,
                "cache_latents_to_disk": False,
                "cache_text_encoder_outputs": False,
                "cache_text_encoder_outputs_to_disk": False,
                "cache_dir": None,
            },
            "caption": {"caption_extention": None},
            "source": {"train_data_dir": None},
        },
        "optimizer": {
            "use_8bit_adam": False,
            "use_lion_optimizer": False,
            "optimizer_type": "",
            "learning_rates": {"base": 1e-4, "denoiser": None, "text_encoders": None},
        },
        "output": {
            "sampling": {"sample_every_n_epochs": None, "sample_every_n_steps": None},
            "logging": {
                "log_every_n_steps": 1,
                "resource_monitor": {
                    "enabled": True,
                    "mode": "basic",
                    "log_every_n_steps": 0,
                    "sample_interval_sec": 1.0,
                    "jsonl_flush_every_n_events": 50,
                    "queue_maxsize": 1024,
                    "max_collection_ms": 0.0,
                    "deep_window_steps": 0,
                    "deep_window_seconds": 0.0,
                },
            },
        },
        "validation": {"validate_every_n_steps": None, "validate_every_n_epochs": None},
        "performance": {},
    }
    if overrides is None:
        return OmegaConf.create(base)
    return OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overrides))


def make_validate_cfg(overrides: dict | None = None):
    """Build a minimally valid config suitable for validate_config tests."""
    base = {
        "mode": "finetune",
        "loss": {
            "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
            "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
            "v_parameterization": False,
        },
        "model": {"model_type": "sd15"},
        "training": {"clip_skip": None},
        "optimizer": {
            "learning_rates": {"blocks": None, "text_encoders": 0, "denoiser": 1e-4, "base": 1e-4},
        },
        "data": {
            "caching": {"cache_text_encoder_outputs": False},
            "bucketing": {"bucket_reso_steps": 64},
            "caption": {
                "shuffle_caption": False,
                "caption_dropout_rate": 0.0,
                "token_warmup_step": 0.0,
                "caption_tag_dropout_rate": 0.0,
            },
        },
        "performance": {
            "memory": {"offload_text_encoders": False},
            "precision": {"full_fp16": False, "full_bf16": False, "mixed_precision": "fp16", "fp8_base": False},
        },
        "output": {
            "sampling": {"sample_every_n_steps": None, "sample_every_n_epochs": None},
            "logging": {
                "resource_monitor": {
                    "mode": "basic",
                    "rank_scope": "main",
                    "device_scope": "local",
                    "jsonl_flush_mode": "auto",
                    "drop_policy": "drop_oldest",
                }
            },
        },
    }
    if overrides is None:
        return OmegaConf.create(base)
    return OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overrides))


# =============================================================================
# prepare_config Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.config
class TestPrepareConfig:
    """Test prepare_config auto-fixups."""

    def test_cache_latents_to_disk_enables_cache_latents(self):
        """cache_latents_to_disk should automatically enable cache_latents."""
        cfg = make_prepare_cfg({"data": {"caching": {"cache_latents_to_disk": True}}})
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

    def test_resource_monitor_disabled_forces_off_mode(self):
        """resource_monitor.enabled=False should force mode=off."""
        cfg = OmegaConf.create(
            {
                "data": {"caching": {"cache_latents": False, "cache_latents_to_disk": False}, "caption": {"caption_extention": None}},
                "optimizer": {"use_8bit_adam": False, "use_lion_optimizer": False, "optimizer_type": ""},
                "output": {
                    "sampling": {"sample_every_n_epochs": None, "sample_every_n_steps": None},
                    "logging": {"log_every_n_steps": 1, "resource_monitor": {"enabled": False, "mode": "basic"}},
                },
            }
        )
        prepare_config(cfg)
        assert cfg.output.logging.resource_monitor.mode == "off"

    def test_resource_monitor_negative_log_steps_clamped_to_zero(self):
        """resource monitor step cadence below zero should normalize to 0."""
        cfg = OmegaConf.create(
            {
                "data": {"caching": {"cache_latents": False, "cache_latents_to_disk": False}, "caption": {"caption_extention": None}},
                "optimizer": {"use_8bit_adam": False, "use_lion_optimizer": False, "optimizer_type": ""},
                "output": {
                    "sampling": {"sample_every_n_epochs": None, "sample_every_n_steps": None},
                    "logging": {
                        "log_every_n_steps": 1,
                        "resource_monitor": {
                            "enabled": True,
                            "mode": "basic",
                            "log_every_n_steps": -3,
                            "sample_interval_sec": 1.0,
                            "jsonl_flush_every_n_events": 50,
                            "queue_maxsize": 1024,
                            "max_collection_ms": 0.0,
                            "deep_window_steps": 0,
                            "deep_window_seconds": 0.0,
                        },
                    },
                },
            }
        )
        prepare_config(cfg)
        assert cfg.output.logging.resource_monitor.log_every_n_steps == 0

    def test_cache_dir_defaults_to_train_data_dir(self):
        """cache_dir should default to train_data_dir when caching omits it."""
        cfg = make_prepare_cfg({"data": {"source": {"train_data_dir": "/tmp/train-data"}}})
        prepare_config(cfg)
        assert cfg.data.caching.cache_dir == "/tmp/train-data"

    def test_learning_rates_default_to_base_lr(self):
        """Missing denoiser and text-encoder LRs should inherit the base LR."""
        cfg = make_prepare_cfg({"optimizer": {"learning_rates": {"base": 2e-4, "denoiser": None, "text_encoders": None}}})
        prepare_config(cfg)
        assert cfg.optimizer.learning_rates.denoiser == 2e-4
        assert cfg.optimizer.learning_rates.text_encoders == 2e-4

    def test_log_every_n_steps_zero_defaults_to_one(self):
        """Tracker cadence at zero should normalize back to 1."""
        cfg = make_prepare_cfg({"output": {"logging": {"log_every_n_steps": 0}}})
        prepare_config(cfg)
        assert cfg.output.logging.log_every_n_steps == 1

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("sample_interval_sec", 0.0, 1.0),
            ("jsonl_flush_every_n_events", 0, 1),
            ("queue_maxsize", 0, 1),
            ("max_collection_ms", -1.0, 0.0),
            ("deep_window_steps", -1, 0),
            ("deep_window_seconds", -1.0, 0.0),
        ],
    )
    def test_resource_monitor_invalid_values_are_normalized(self, field, value, expected):
        """Resource monitor numeric settings should clamp invalid values."""
        cfg = make_prepare_cfg({"output": {"logging": {"resource_monitor": {field: value}}}})
        prepare_config(cfg)
        assert getattr(cfg.output.logging.resource_monitor, field) == expected

    def test_validation_cadence_zero_or_negative_disables_both(self):
        """Validation cadence should disable non-positive step and epoch schedules."""
        cfg = make_prepare_cfg({"validation": {"validate_every_n_steps": 0, "validate_every_n_epochs": -2}})
        prepare_config(cfg)
        assert cfg.validation.validate_every_n_steps is None
        assert cfg.validation.validate_every_n_epochs is None

    def test_prepare_config_tolerates_missing_logging_fields(self):
        """Partial configs without output.logging should not raise."""
        cfg = OmegaConf.create(
            {
                "data": {"caching": {"cache_latents": False, "cache_latents_to_disk": False}, "caption": {"caption_extention": None}},
                "optimizer": {"use_8bit_adam": False, "use_lion_optimizer": False, "optimizer_type": ""},
                "output": {"sampling": {"sample_every_n_epochs": None, "sample_every_n_steps": None}},
            }
        )
        prepare_config(cfg)


# =============================================================================
# validate_config Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.config
class TestValidateConfig:
    """Test validate_config errors and warnings."""

    def test_missing_model_type_raises(self):
        """Null or omitted model_type should raise a clear config error."""
        cfg = OmegaConf.create(
            {
                "mode": "peft",
                "peft": {"adapter_rank": 16},
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": None},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"blocks": None, "text_encoders": 0}},
                "data": {"caching": {"cache_text_encoder_outputs": False}},
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
            }
        )
        with pytest.raises(ValueError, match="model\\.model_type is required"):
            validate_config(cfg)

    def test_invalid_mode_raises(self):
        """Unknown top-level mode should raise ValueError when present."""
        cfg = make_validate_cfg({"mode": "mystery", "model": {"model_type": "sd1"}})
        with pytest.raises(ValueError, match="mode must be one of"):
            validate_config(cfg)

    def test_mode_peft_requires_peft_section(self):
        """PEFT mode should fail when the PEFT section is missing."""
        cfg = OmegaConf.create(
            {
                "mode": "peft",
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sdxl"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"blocks": None, "text_encoders": 0}},
                "data": {"caching": {"cache_text_encoder_outputs": False}},
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
            }
        )
        with pytest.raises(ValueError, match="mode=peft requires a `peft` section"):
            validate_config(cfg)

    def test_mode_finetune_forbids_peft_section(self):
        """Fine-tune mode should reject PEFT-specific config."""
        cfg = OmegaConf.create(
            {
                "mode": "finetune",
                "peft": {"adapter_rank": 16},
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd15"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"blocks": None, "text_encoders": 0}},
                "data": {"caching": {"cache_text_encoder_outputs": False}},
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
            }
        )
        with pytest.raises(ValueError, match="mode=finetune cannot be used with `peft` or `textual_inversion` sections"):
            validate_config(cfg)

    def test_mode_textual_inversion_forbids_peft_section(self):
        """Textual inversion mode should reject mixed mode sections."""
        cfg = OmegaConf.create(
            {
                "mode": "textual_inversion",
                "peft": {"adapter_rank": 16},
                "textual_inversion": {"token_string": "test"},
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sdxl"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"blocks": None, "text_encoders": 0}},
                "data": {"caching": {"cache_text_encoder_outputs": False}},
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
            }
        )
        with pytest.raises(ValueError, match="`peft` and `textual_inversion` sections cannot both be active"):
            validate_config(cfg)

    def test_mode_textual_inversion_requires_section(self):
        """Textual inversion mode should fail when its section is missing."""
        cfg = make_validate_cfg({"mode": "textual_inversion", "model": {"model_type": "sdxl"}})
        with pytest.raises(ValueError, match="mode=textual_inversion requires a `textual_inversion` section"):
            validate_config(cfg)

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
                "optimizer": {"learning_rates": {"blocks": None, "text_encoders": 0}},
                "data": {"caching": {"cache_text_encoder_outputs": False}},
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
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
                "optimizer": {"learning_rates": {"blocks": None, "text_encoders": 0}},
                "data": {"caching": {"cache_text_encoder_outputs": False}},
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
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
                "optimizer": {"learning_rates": {"blocks": None, "text_encoders": 0}},
                "data": {"caching": {"cache_text_encoder_outputs": False}},
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
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

    def test_invalid_resource_monitor_mode_raises(self):
        """resource_monitor.mode must be one of off/basic/sampled/deep."""
        cfg = OmegaConf.create(
            {
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd1"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"blocks": None, "text_encoders": 0}},
                "data": {"caching": {"cache_text_encoder_outputs": False}},
                "performance": {
                    "memory": {"offload_text_encoders": False},
                    "precision": {"full_fp16": False, "full_bf16": False, "mixed_precision": "fp16"},
                },
                "output": {
                    "logging": {
                        "resource_monitor": {
                            "mode": "unknown",
                            "rank_scope": "main",
                            "device_scope": "local",
                            "jsonl_flush_mode": "auto",
                            "drop_policy": "drop_oldest",
                        },
                    }
                },
            }
        )
        with pytest.raises(ValueError, match="resource_monitor.mode must be one of"):
            validate_config(cfg)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("rank_scope", "world", "resource_monitor.rank_scope must be one of"),
            ("device_scope", "remote", "resource_monitor.device_scope must be one of"),
            ("jsonl_flush_mode", "immediate", "resource_monitor.jsonl_flush_mode must be one of"),
            ("drop_policy", "discard", "resource_monitor.drop_policy must be one of"),
        ],
    )
    def test_invalid_resource_monitor_enum_fields_raise(self, field, value, message):
        """Each resource monitor enum field should reject unknown values."""
        cfg = make_validate_cfg({"output": {"logging": {"resource_monitor": {field: value}}}})
        with pytest.raises(ValueError, match=message):
            validate_config(cfg)

    @pytest.mark.parametrize("fp8_field", ["fp8_base", "fp8_base_unet"])
    def test_fp8_base_requires_mixed_precision(self, fp8_field):
        """FP8 precision flags should reject configs that disable mixed precision entirely."""
        cfg = make_validate_cfg({"performance": {"precision": {fp8_field: True, "mixed_precision": "no"}}})
        with pytest.raises(ValueError, match="fp8_base requires mixed_precision='fp16' or 'bf16'"):
            validate_config(cfg)

    def test_sampling_cadence_conflict_raises(self):
        """Sampling should reject configs that enable both step and epoch cadence."""
        cfg = OmegaConf.create(
            {
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sdxl"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"blocks": None, "text_encoders": 0}},
                "data": {
                    "caching": {"cache_text_encoder_outputs": False},
                    "bucketing": {"bucket_reso_steps": 32},
                    "caption": {
                        "shuffle_caption": False,
                        "caption_dropout_rate": 0.0,
                        "token_warmup_step": 0.0,
                        "caption_tag_dropout_rate": 0.0,
                    },
                },
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
                "output": {"sampling": {"sample_every_n_steps": 100, "sample_every_n_epochs": 1}},
            }
        )
        with pytest.raises(ValueError, match="sample_every_n_steps and sample_every_n_epochs cannot both be set"):
            validate_config(cfg)

    def test_sdxl_bucket_reso_steps_must_be_divisible_by_32(self):
        """SDXL configs should reject bucket step sizes incompatible with the model family."""
        cfg = OmegaConf.create(
            {
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sdxl"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"blocks": None, "text_encoders": 0}},
                "data": {
                    "caching": {"cache_text_encoder_outputs": False},
                    "bucketing": {"bucket_reso_steps": 48},
                    "caption": {
                        "shuffle_caption": False,
                        "caption_dropout_rate": 0.0,
                        "token_warmup_step": 0.0,
                        "caption_tag_dropout_rate": 0.0,
                    },
                },
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
            }
        )
        with pytest.raises(ValueError, match="bucket_reso_steps=48 must be divisible by 32"):
            validate_config(cfg)

    def test_cache_te_outputs_rejects_caption_mutation_settings(self):
        """TE-output caching should reject caption settings that change TE outputs over time."""
        cfg = OmegaConf.create(
            {
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sdxl"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"blocks": None, "text_encoders": 0}},
                "data": {
                    "caching": {"cache_text_encoder_outputs": True},
                    "bucketing": {"bucket_reso_steps": 32},
                    "caption": {
                        "shuffle_caption": True,
                        "caption_dropout_rate": 0.0,
                        "token_warmup_step": 0.0,
                        "caption_tag_dropout_rate": 0.0,
                    },
                },
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
            }
        )
        with pytest.raises(ValueError, match="cache_text_encoder_outputs cannot be used with"):
            validate_config(cfg)

    def test_offload_text_encoders_conflicts_with_te_output_caching(self):
        """Config validation should reject TE offloading together with TE output caching."""
        cfg = make_validate_cfg(
            {
                "data": {"caching": {"cache_text_encoder_outputs": True}},
                "performance": {"memory": {"offload_text_encoders": True}},
            }
        )
        with pytest.raises(ValueError, match="Cannot use both offload_text_encoders and cache_text_encoder_outputs"):
            validate_config(cfg)

    def test_offload_text_encoders_conflicts_with_te_training(self):
        """Offloading text encoders should reject configs that still train them."""
        cfg = make_validate_cfg(
            {
                "optimizer": {"learning_rates": {"text_encoders": 1e-5}},
                "performance": {"memory": {"offload_text_encoders": True}},
            }
        )
        with pytest.raises(ValueError, match="Cannot train text encoder while offloading to CPU"):
            validate_config(cfg)

    def test_te_output_caching_conflicts_with_te_training(self):
        """TE-output caching should reject configs that still train text encoders."""
        cfg = make_validate_cfg(
            {
                "optimizer": {"learning_rates": {"text_encoders": 1e-5}},
                "data": {"caching": {"cache_text_encoder_outputs": True}},
            }
        )
        with pytest.raises(ValueError, match="Cannot train text encoder while TE output caching is enabled"):
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
# Dataset-group validator Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.config
class TestDatasetGroupValidation:
    """Test dataset-group validation derived from the active config."""

    def test_sd_family_uses_64_step_buckets(self):
        """SD family configs should verify bucket reso with 64 steps."""
        cfg = OmegaConf.create({"model": {"model_type": "sd15"}, "data": {"caching": {"cache_text_encoder_outputs": False}}})
        train_ds = MagicMock()
        val_ds = MagicMock()

        validate_dataset_groups(cfg, train_ds, val_ds)

        train_ds.verify_bucket_reso_steps.assert_called_once_with(64)
        val_ds.verify_bucket_reso_steps.assert_called_once_with(64)

    def test_no_val_dataset_is_allowed(self):
        """Dataset-group validation should handle missing validation datasets."""
        cfg = OmegaConf.create({"model": {"model_type": "sd15"}, "data": {"caching": {"cache_text_encoder_outputs": False}}})
        train_ds = MagicMock()

        validate_dataset_groups(cfg, train_ds, None)

        train_ds.verify_bucket_reso_steps.assert_called_once_with(64)

    def test_sdxl_uses_32_step_buckets(self):
        """SDXL configs should verify bucket reso with 32 steps."""
        cfg = OmegaConf.create({"model": {"model_type": "sdxl"}, "data": {"caching": {"cache_text_encoder_outputs": False}}})
        train_ds = MagicMock()
        val_ds = MagicMock()

        validate_dataset_groups(cfg, train_ds, val_ds)

        train_ds.verify_bucket_reso_steps.assert_called_once_with(32)
        val_ds.verify_bucket_reso_steps.assert_called_once_with(32)

    def test_cache_te_requires_dataset_group_cacheability(self):
        """Dataset-group validation should reject TE caching for non-cacheable datasets."""
        cfg = OmegaConf.create({"model": {"model_type": "sdxl"}, "data": {"caching": {"cache_text_encoder_outputs": True}}})
        train_ds = MagicMock()
        train_ds.is_text_encoder_output_cacheable.return_value = False

        with pytest.raises(ValueError, match="cache_text_encoder_outputs"):
            validate_dataset_groups(cfg, train_ds, None)

    def test_cache_te_without_cacheability_probe_is_allowed(self):
        """Dataset-group validation should tolerate dataset groups without the optional probe."""
        cfg = OmegaConf.create({"model": {"model_type": "sdxl"}, "data": {"caching": {"cache_text_encoder_outputs": True}}})
        train_ds = MagicMock(spec=["verify_bucket_reso_steps"])

        validate_dataset_groups(cfg, train_ds, None)

        train_ds.verify_bucket_reso_steps.assert_called_once_with(32)

    def test_unknown_model_type_skips_bucket_validation(self):
        """Unknown or future model families should not force bucket-step validation here."""
        cfg = OmegaConf.create({"model": {"model_type": "flux"}, "data": {"caching": {"cache_text_encoder_outputs": False}}})
        train_ds = MagicMock()

        validate_dataset_groups(cfg, train_ds, None)

        train_ds.verify_bucket_reso_steps.assert_not_called()
