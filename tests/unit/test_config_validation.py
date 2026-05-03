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
        "objective": {"path": "ddpm", "prediction": "epsilon"},
        "loss": {
            "edm2": {
                "enabled": False,
                "importance": {"enabled": False, "safety_override": False},
            },
            "snr": {"debiased_estimation_loss": False, "min_snr_gamma": None},
        },
        "performance": {},
    }
    if overrides is None:
        return OmegaConf.create(base)
    return OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overrides))


def make_validate_cfg(overrides: dict | None = None):
    """Build a minimally valid config suitable for validate_config tests."""
    base = {
        "mode": "finetune",
        "objective": {"path": "ddpm", "prediction": "epsilon"},
        "loss": {
            "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
            "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
            "edm2": {"laplace_timestep_sampling": False},
            "v_parameterization": False,
        },
        "model": {"model_type": "sd15"},
        "training": {"clip_skip": None},
        "timestep": {
            "timestep_sampling": "uniform",
            "adaptive_log_snr": {
                "bins": 32,
                "ema_beta": 0.9,
                "temperature": 0.5,
                "prior_weight": 0.25,
                "min_prob": 1e-4,
                "warmup_steps": 2000,
                "entropy_floor": 0.7,
                "uniform_mix_when_low_entropy": 0.1,
            },
        },
        "optimizer": {
            "learning_rates": {"text_encoders": 0, "denoiser": 1e-4, "base": 1e-4},
        },
        "data": {
            "source": {"val_data_dir": None},
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
        "validation": {
            "validation_split": 0.0,
            "validation_seed": None,
            "run_at_start": False,
            "run_at_end": False,
            "validate_every_n_steps": None,
            "validate_every_n_epochs": None,
            "max_validation_steps": None,
            "validation_timesteps": "[50, 350, 500, 650, 950]",
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

    def test_prepare_config_silently_syncs_legacy_v_parameterization_for_rf(self):
        """RF configs should not warn when syncing the legacy DDPM-only mirror flag."""
        cfg = make_prepare_cfg(
            {
                "objective": {"path": "rectified_flow", "prediction": "flow"},
                "loss": {"v_parameterization": True},
            }
        )
        with patch("library.config.config_validation.logger") as mock_logger:
            prepare_config(cfg)
            assert cfg.loss.v_parameterization is False
            mock_logger.warning.assert_not_called()

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

    def test_learning_rates_stay_null_when_base_is_null(self):
        """Inherited component LRs should remain null when the shared fallback is disabled."""
        cfg = make_prepare_cfg({"optimizer": {"learning_rates": {"base": None, "denoiser": None, "text_encoders": None}}})
        prepare_config(cfg)
        assert cfg.optimizer.learning_rates.denoiser is None
        assert cfg.optimizer.learning_rates.text_encoders is None

    def test_learning_rates_preserve_explicit_zero_as_frozen(self):
        """Explicit zero LRs should remain zero instead of being rewritten as inherited values."""
        cfg = make_prepare_cfg({"optimizer": {"learning_rates": {"base": 2e-4, "denoiser": 0.0, "text_encoders": 0.0}}})
        prepare_config(cfg)
        assert cfg.optimizer.learning_rates.denoiser == 0.0
        assert cfg.optimizer.learning_rates.text_encoders == 0.0

    def test_objective_prediction_syncs_legacy_v_parameterization(self):
        """The legacy boolean should mirror the explicit objective prediction."""
        cfg = make_prepare_cfg(
            {
                "objective": {"path": "ddpm", "prediction": "v_prediction"},
                "loss": {"v_parameterization": False},
            }
        )
        with patch("library.config.config_validation.logger") as mock_logger:
            prepare_config(cfg)
        assert cfg.objective.path == "ddpm"
        assert cfg.objective.prediction == "v_prediction"
        assert cfg.loss.v_parameterization is True
        mock_logger.warning.assert_called()

    def test_objective_prediction_keeps_legacy_mirror_in_sync(self):
        """An explicit objective prediction should win and sync the legacy mirror."""
        cfg = make_prepare_cfg(
            {
                "objective": {"path": "ddpm", "prediction": "epsilon"},
                "loss": {"v_parameterization": True},
            }
        )
        with patch("library.config.config_validation.logger") as mock_logger:
            prepare_config(cfg)
        assert cfg.objective.prediction == "epsilon"
        assert cfg.loss.v_parameterization is False
        mock_logger.warning.assert_called()

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

    def test_edm2_importance_weighting_disables_conflicting_snr_settings(self):
        """EDM2 importance weighting should normalize conflicting nested SNR settings."""
        cfg = make_prepare_cfg(
            {
                "loss": {
                    "edm2": {
                        "enabled": True,
                        "importance": {"enabled": True},
                    },
                    "snr": {"debiased_estimation_loss": True, "min_snr_gamma": 5.0},
                }
            }
        )

        prepare_config(cfg)

        assert cfg.loss.snr.debiased_estimation_loss is False
        assert cfg.loss.snr.min_snr_gamma is None


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
                "mode": "adapter",
                "adapter": {"peft": {"lora": {"rank": 16}}},
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": None},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"text_encoders": 0}},
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

    def test_mode_adapter_requires_peft_section(self):
        """Adapter mode should fail when the PEFT section is missing."""
        cfg = OmegaConf.create(
            {
                "mode": "adapter",
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sdxl"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"text_encoders": 0}},
                "data": {"caching": {"cache_text_encoder_outputs": False}},
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
            }
        )
        with pytest.raises(ValueError, match="mode=adapter requires an `adapter\\.peft` section"):
            validate_config(cfg)

    def test_mode_finetune_forbids_adapter_section(self):
        """Fine-tune mode should reject adapter-specific config."""
        cfg = OmegaConf.create(
            {
                "mode": "finetune",
                "adapter": {"peft": {"lora": {"rank": 16}}},
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd15"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"text_encoders": 0}},
                "data": {"caching": {"cache_text_encoder_outputs": False}},
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
            }
        )
        with pytest.raises(ValueError, match="mode=finetune cannot be used with `adapter` or `textual_inversion` sections"):
            validate_config(cfg)

    def test_mode_textual_inversion_forbids_adapter_section(self):
        """Textual inversion mode should reject mixed mode sections."""
        cfg = OmegaConf.create(
            {
                "mode": "textual_inversion",
                "adapter": {"peft": {"lora": {"rank": 16}}},
                "textual_inversion": {"token_string": "test"},
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sdxl"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"text_encoders": 0}},
                "data": {"caching": {"cache_text_encoder_outputs": False}},
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
            }
        )
        with pytest.raises(ValueError, match="`adapter` and `textual_inversion` sections cannot both be active"):
            validate_config(cfg)

    def test_mode_textual_inversion_requires_section(self):
        """Textual inversion mode should fail when its section is missing."""
        cfg = make_validate_cfg({"mode": "textual_inversion", "model": {"model_type": "sdxl"}})
        with pytest.raises(ValueError, match="mode=textual_inversion requires a `textual_inversion` section"):
            validate_config(cfg)

    def test_peft_strict_continue_rejects_active_method_settings(self):
        """Strict continuation should reject config-defined method settings."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "continue_from": "adapter.safetensors",
                        "continue_mode": "strict",
                        "lora": {"rank": 16},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.continue_mode='strict' treats the artifact as authoritative"):
            validate_config(cfg)

    def test_peft_continue_from_defaults_to_strict_mode(self):
        """Continuation defaults to strict mode when no override is requested."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "continue_from": "adapter.safetensors",
                        "lora": {},
                    }
                },
            }
        )

        validate_config(cfg)

    def test_peft_rejects_loha_without_rank(self):
        """LoHa should require an explicit method-level rank setting."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "loha": {},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.loha\\.rank must be set to a positive integer"):
            validate_config(cfg)

    def test_peft_rejects_legacy_adapter_args_as_forward_surface(self):
        """Dynamic adapter args should not be a method-settings path."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "loha": {},
                        "adapter_args": ["use_scalar=True"],
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.adapter_args is no longer part of the forward adapter config surface"):
            validate_config(cfg)

    def test_peft_rejects_non_positive_loha_rank(self):
        """LoHa rank should fail fast when set to a non-positive value."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "loha": {"rank": 0},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.loha\\.rank must be a positive integer when set"):
            validate_config(cfg)

    def test_peft_rejects_loha_plain_dropout(self):
        """LoHa plain dropout is intentionally unsupported in the repo-owned path."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "loha": {"rank": 8, "dropout": 0.1},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.loha\\.dropout is not supported"):
            validate_config(cfg)

    def test_peft_rejects_unknown_loha_init_mode(self):
        """LoHa init mode must be one of the repo-owned supported options."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "loha": {"rank": 8, "init_mode": "mystery_mode"},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.loha\\.init_mode must be one of"):
            validate_config(cfg)

    def test_peft_rejects_loha_bypass_mode_with_weight_decompose(self):
        """LoHa bypass mode should not coexist with weight decomposition."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "loha": {"weight_decompose": True, "bypass_mode": True},
                    }
                },
            }
        )

        with pytest.raises(
            ValueError,
            match="adapter\\.peft\\.loha\\.bypass_mode cannot be enabled when adapter\\.peft\\.loha\\.weight_decompose is true",
        ):
            validate_config(cfg)

    def test_peft_rejects_locon_without_rank(self):
        """LoCon should require an explicit method-level rank setting."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "locon": {},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.locon\\.rank must be set to a positive integer"):
            validate_config(cfg)

    def test_peft_rejects_non_positive_locon_rank(self):
        """LoCon rank should fail fast when set to a non-positive value."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "locon": {"rank": 0},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.locon\\.rank must be a positive integer when set"):
            validate_config(cfg)

    def test_peft_rejects_unknown_locon_init_mode(self):
        """LoCon init mode must be one of the repo-owned supported options."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "locon": {"rank": 8, "init_mode": "mystery_mode"},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.locon\\.init_mode must be one of"):
            validate_config(cfg)

    def test_peft_rejects_locon_bypass_mode_with_weight_decompose(self):
        """LoCon bypass mode should not coexist with weight decomposition."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "locon": {"weight_decompose": True, "bypass_mode": True},
                    }
                },
            }
        )

        with pytest.raises(
            ValueError,
            match="adapter\\.peft\\.locon\\.bypass_mode cannot be enabled when adapter\\.peft\\.locon\\.weight_decompose is true",
        ):
            validate_config(cfg)

    def test_peft_rejects_lokr_without_rank(self):
        """LoKr should require an explicit method-level rank setting."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "lokr": {},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.lokr\\.rank must be set to a positive integer"):
            validate_config(cfg)

    def test_peft_rejects_non_positive_lokr_rank(self):
        """LoKr rank should fail fast when set to a non-positive value."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "lokr": {"rank": 0},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.lokr\\.rank must be a positive integer when set"):
            validate_config(cfg)

    def test_peft_rejects_lokr_plain_dropout(self):
        """LoKr plain dropout is intentionally unsupported in the repo-owned path."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "lokr": {"rank": 8, "dropout": 0.1},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.lokr\\.dropout is not supported"):
            validate_config(cfg)

    def test_peft_rejects_lokr_bypass_mode_with_weight_decompose(self):
        """LoKr bypass mode should not coexist with weight decomposition."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "lokr": {"weight_decompose": True, "bypass_mode": True},
                    }
                },
            }
        )

        with pytest.raises(
            ValueError,
            match="adapter\\.peft\\.lokr\\.bypass_mode cannot be enabled when adapter\\.peft\\.lokr\\.weight_decompose is true",
        ):
            validate_config(cfg)

    def test_peft_rejects_oft_without_factor(self):
        """OFT should require an explicit method-level factor setting."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "oft": {},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.oft\\.factor must be set to a positive integer"):
            validate_config(cfg)

    def test_peft_rejects_boft_without_factor(self):
        """BOFT should require an explicit method-level factor setting."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "boft": {},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.boft\\.factor must be set to a positive integer"):
            validate_config(cfg)

    def test_peft_rejects_dylora_without_rank(self):
        """DyLoRA should require an explicit method-level rank setting."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "dylora": {},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.dylora\\.rank must be set to a positive integer"):
            validate_config(cfg)

    def test_peft_rejects_glora_without_rank(self):
        """GLoRA should require an explicit method-level rank setting."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "glora": {},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.glora\\.rank must be set to a positive integer"):
            validate_config(cfg)

    def test_peft_rejects_abba_without_rank(self):
        """ABBA should require an explicit method-level rank setting."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "abba": {},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.abba\\.rank must be set to an integer greater than or equal to 2"):
            validate_config(cfg)

    def test_peft_rejects_non_positive_oft_factor(self):
        """OFT factor should fail fast when set to a non-positive value."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "oft": {"factor": 0},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.oft\\.factor must be a positive integer when set"):
            validate_config(cfg)

    def test_peft_rejects_non_positive_boft_factor(self):
        """BOFT factor should fail fast when set to a non-positive value."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "boft": {"factor": 0},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.boft\\.factor must be a positive integer when set"):
            validate_config(cfg)

    def test_peft_rejects_non_positive_dylora_rank(self):
        """DyLoRA rank should fail fast when set to a non-positive value."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "dylora": {"rank": 0},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.dylora\\.rank must be a positive integer when set"):
            validate_config(cfg)

    def test_peft_rejects_non_positive_glora_rank(self):
        """GLoRA rank should fail fast when set to a non-positive value."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "glora": {"rank": 0},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.glora\\.rank must be a positive integer when set"):
            validate_config(cfg)

    def test_peft_rejects_abba_rank_below_two(self):
        """ABBA rank should fail fast when it cannot populate both factor pairs."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "abba": {"rank": 1},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.abba\\.rank must be greater than or equal to 2 when set"):
            validate_config(cfg)

    def test_peft_rejects_dylora_block_size_that_does_not_divide_rank(self):
        """DyLoRA block_size should divide rank exactly."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "dylora": {"rank": 8, "block_size": 3},
                    }
                },
            }
        )

        with pytest.raises(
            ValueError,
            match="adapter\\.peft\\.dylora\\.block_size must divide adapter\\.peft\\.dylora\\.rank exactly",
        ):
            validate_config(cfg)

    def test_peft_rejects_negative_oft_constraint(self):
        """OFT constraint should fail fast when set negative."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "oft": {"factor": 4, "constraint": -0.1},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.oft\\.constraint must be non-negative"):
            validate_config(cfg)

    def test_peft_rejects_negative_boft_constraint(self):
        """BOFT constraint should fail fast when set negative."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "boft": {"factor": 4, "constraint": -0.1},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.boft\\.constraint must be non-negative"):
            validate_config(cfg)

    def test_peft_rejects_non_positive_boft_num_stages(self):
        """BOFT partial stage count should fail fast when set non-positive."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "boft": {"factor": 4, "num_stages": 0},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.boft\\.num_stages must be a positive integer when set"):
            validate_config(cfg)

    def test_peft_rejects_out_of_range_oft_dropout(self):
        """OFT dropout probabilities should stay within [0, 1]."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "oft": {"factor": 4, "dropout": 1.5},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.oft\\.dropout must be between 0.0 and 1.0 inclusive"):
            validate_config(cfg)

    def test_peft_rejects_out_of_range_boft_dropout(self):
        """BOFT dropout probabilities should stay within [0, 1]."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "boft": {"factor": 4, "dropout": 1.5},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.boft\\.dropout must be between 0.0 and 1.0 inclusive"):
            validate_config(cfg)

    def test_peft_rejects_out_of_range_dylora_module_dropout(self):
        """DyLoRA module dropout should stay within [0, 1]."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "dylora": {"rank": 8, "module_dropout": 1.5},
                    }
                },
            }
        )

        with pytest.raises(
            ValueError,
            match="adapter\\.peft\\.dylora\\.module_dropout must be between 0.0 and 1.0 inclusive",
        ):
            validate_config(cfg)

    def test_peft_rejects_out_of_range_ia3_module_dropout(self):
        """IA3 module dropout should stay within [0, 1]."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "ia3": {"module_dropout": 1.5},
                    }
                },
            }
        )

        with pytest.raises(
            ValueError,
            match="adapter\\.peft\\.ia3\\.module_dropout must be between 0.0 and 1.0 inclusive",
        ):
            validate_config(cfg)

    def test_peft_rejects_out_of_range_glora_dropout(self):
        """GLoRA dropout probabilities should stay within [0, 1]."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "glora": {"rank": 8, "dropout": 1.5},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.glora\\.dropout must be between 0.0 and 1.0 inclusive"):
            validate_config(cfg)

    def test_peft_rejects_out_of_range_abba_dropout(self):
        """ABBA dropout probabilities should stay within [0, 1]."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "abba": {"rank": 4, "dropout": 1.5},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="adapter\\.peft\\.abba\\.dropout must be between 0.0 and 1.0 inclusive"):
            validate_config(cfg)

    def test_peft_rejects_abba_bypass_mode_with_weight_decompose(self):
        """ABBA bypass mode should not coexist with weight decomposition."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "abba": {"rank": 4, "weight_decompose": True, "bypass_mode": True},
                    }
                },
            }
        )

        with pytest.raises(
            ValueError,
            match="adapter\\.peft\\.abba\\.bypass_mode cannot be enabled when adapter\\.peft\\.abba\\.weight_decompose is true",
        ):
            validate_config(cfg)

    def test_peft_rejects_legacy_method_without_active_branch(self):
        """Legacy method alone should not replace branch-presence selection."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "method": "lora",
                        "lora": None,
                        "loha": None,
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="Legacy peft\\.method='lora' is set, but no method branch is configured"):
            validate_config(cfg)

    def test_peft_initialize_from_artifact_allows_active_method_settings(self):
        """Non-strict continuation should allow config-defined method settings."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "continue_from": "adapter.safetensors",
                        "continue_mode": "initialize_from_artifact",
                        "lora": {"rank": 16},
                    }
                },
            }
        )

        validate_config(cfg)

    def test_peft_rejects_nondefault_inactive_method_subtree(self):
        """Only the selected method subtree should carry active settings."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {
                    "peft": {
                        "lora": {"rank": 16},
                        "loha": {"use_scalar": True},
                    }
                },
            }
        )

        with pytest.raises(ValueError, match="exactly one method branch"):
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
                "optimizer": {"learning_rates": {"text_encoders": 0}},
                "data": {"caching": {"cache_text_encoder_outputs": False}},
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
            }
        )
        validate_config(cfg)  # Should not raise

    def test_scale_v_pred_loss_requires_v_prediction(self):
        """scale_v_pred_loss_like_noise_pred requires objective.prediction='v_prediction'."""
        cfg = OmegaConf.create(
            {
                "objective": {"path": "ddpm", "prediction": "epsilon"},
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": True, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd1"},
                "training": {"clip_skip": None},
            }
        )
        with pytest.raises(ValueError, match="scale_v_pred_loss_like_noise_pred requires objective.prediction='v_prediction'"):
            validate_config(cfg)

    def test_v_pred_like_loss_conflicts_with_v_prediction(self):
        """v_pred_like_loss with objective.prediction='v_prediction' should raise ValueError."""
        cfg = OmegaConf.create(
            {
                "objective": {"path": "ddpm", "prediction": "v_prediction"},
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": False},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": 0.5},
                    "v_parameterization": True,
                },
                "model": {"model_type": "sd1"},
                "training": {"clip_skip": None},
            }
        )
        with pytest.raises(ValueError, match="v_pred_like_loss conflicts with objective.prediction='v_prediction'"):
            validate_config(cfg)

    def test_ddpm_path_rejects_flow_prediction(self):
        """DDPM should reject the RF-native prediction label."""
        cfg = make_validate_cfg({"objective": {"path": "ddpm", "prediction": "flow"}})

        with pytest.raises(ValueError, match="objective\\.path='ddpm' requires objective\\.prediction to be 'epsilon' or 'v_prediction'"):
            validate_config(cfg)

    def test_rectified_flow_path_requires_flow_prediction(self):
        """RF should reject DDPM-style prediction labels."""
        cfg = make_validate_cfg(
            {
                "model": {"model_type": "sd3"},
                "objective": {"path": "rectified_flow", "prediction": "epsilon"},
            }
        )

        with pytest.raises(ValueError, match="objective\\.path='rectified_flow' requires objective\\.prediction='flow'"):
            validate_config(cfg)

    def test_rectified_flow_path_rejects_edm2(self):
        """RF should reject DDPM-only EDM2 weighting early in validation."""
        cfg = make_validate_cfg(
            {
                "model": {"model_type": "sd3"},
                "objective": {"path": "rectified_flow", "prediction": "flow"},
                "loss": {"edm2": {"enabled": True}},
            }
        )

        with pytest.raises(ValueError, match="loss\\.edm2 is only supported with objective\\.path='ddpm'"):
            validate_config(cfg)

    def test_sdxl_rectified_flow_path_is_allowed(self):
        """SDXL should be allowed to opt into the active RF path."""
        cfg = make_validate_cfg(
            {
                "model": {"model_type": "sdxl"},
                "objective": {"path": "rectified_flow", "prediction": "flow"},
            }
        )

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
                "optimizer": {"learning_rates": {"text_encoders": 0}},
                "data": {"caching": {"cache_text_encoder_outputs": False}},
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
            }
        )
        with patch("library.config.config_validation.logger") as mock_logger:
            validate_config(cfg)
            mock_logger.warning.assert_called_once()
            assert "v2 with clip_skip" in str(mock_logger.warning.call_args)

    def test_zero_terminal_snr_without_v_prediction_warns(self):
        """zero_terminal_snr without v_prediction should log a warning."""
        cfg = OmegaConf.create(
            {
                "objective": {"path": "ddpm", "prediction": "epsilon"},
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": True},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd1"},
                "training": {"clip_skip": None},
                "optimizer": {"learning_rates": {"text_encoders": 0}},
                "data": {"caching": {"cache_text_encoder_outputs": False}},
                "performance": {"memory": {"offload_text_encoders": False}, "precision": {"full_fp16": False, "full_bf16": False}},
            }
        )
        with patch("library.config.config_validation.logger") as mock_logger:
            validate_config(cfg)
            mock_logger.warning.assert_called_once()
            assert "objective.prediction" in str(mock_logger.warning.call_args)

    def test_zero_terminal_snr_on_rectified_flow_does_not_warn(self):
        """DDPM-only zero_terminal_snr warning should not fire on RF configs."""
        cfg = OmegaConf.create(
            {
                "objective": {"path": "rectified_flow", "prediction": "flow"},
                "loss": {
                    "regularization": {"adaptive_noise_scale": None, "noise_offset": None, "zero_terminal_snr": True},
                    "snr": {"scale_v_pred_loss_like_noise_pred": False, "v_pred_like_loss": None},
                    "edm2": {"laplace_timestep_sampling": False},
                    "v_parameterization": False,
                },
                "model": {"model_type": "sd3"},
                "training": {"clip_skip": None},
                "timestep": {
                    "timestep_sampling": "uniform",
                    "adaptive_log_snr": {
                        "bins": 32,
                        "ema_beta": 0.9,
                        "temperature": 0.5,
                        "prior_weight": 0.25,
                        "min_prob": 1e-4,
                        "warmup_steps": 2000,
                        "entropy_floor": 0.7,
                        "uniform_mix_when_low_entropy": 0.1,
                    },
                },
                "optimizer": {"learning_rates": {"text_encoders": 0, "denoiser": 1e-4, "base": 1e-4}},
                "data": {"source": {"val_data_dir": None}, "caching": {"cache_text_encoder_outputs": False}},
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
                "validation": {
                    "validation_split": 0.0,
                    "validation_seed": None,
                    "run_at_start": False,
                    "run_at_end": False,
                    "validate_every_n_steps": None,
                    "validate_every_n_epochs": None,
                    "max_validation_steps": None,
                    "validation_timesteps": "[50, 350, 500, 650, 950]",
                },
                "mode": "finetune",
            }
        )
        with patch("library.config.config_validation.logger") as mock_logger:
            validate_config(cfg)
            mock_logger.warning.assert_not_called()

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
                "optimizer": {"learning_rates": {"text_encoders": 0}},
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
                "optimizer": {"learning_rates": {"text_encoders": 0}},
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
                "optimizer": {"learning_rates": {"text_encoders": 0}},
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
                "optimizer": {"learning_rates": {"text_encoders": 0}},
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
        with pytest.raises(ValueError, match="Cannot train text encoder parameters while offloading to CPU"):
            validate_config(cfg)

    def test_offload_text_encoders_conflicts_with_grouped_te_training(self):
        """Offloading text encoders should also reject explicit TE-targeting groups."""
        cfg = make_validate_cfg(
            {
                "optimizer": {
                    "learning_rates": {
                        "text_encoders": 0.0,
                        "groups": [{"name": "clip_probe", "lr": 1e-5, "match": ["clip_l.*"]}],
                    }
                },
                "performance": {"memory": {"offload_text_encoders": True}},
                "model": {"model_type": "sdxl"},
            }
        )
        with pytest.raises(ValueError, match="Cannot train text encoder parameters while offloading to CPU"):
            validate_config(cfg)

    def test_te_output_caching_conflicts_with_te_training(self):
        """TE-output caching should reject configs that still train text encoders."""
        cfg = make_validate_cfg(
            {
                "optimizer": {"learning_rates": {"text_encoders": 1e-5}},
                "data": {"caching": {"cache_text_encoder_outputs": True}},
            }
        )
        with pytest.raises(ValueError, match="Cannot train text encoder parameters while TE output caching is enabled"):
            validate_config(cfg)

    def test_te_output_caching_conflicts_with_grouped_te_training(self):
        """TE-output caching should also reject explicit TE-targeting groups."""
        cfg = make_validate_cfg(
            {
                "optimizer": {
                    "learning_rates": {
                        "text_encoders": 0.0,
                        "groups": [{"name": "clip_probe", "lr": 1e-5, "match": ["clip_l.*"]}],
                    }
                },
                "data": {"caching": {"cache_text_encoder_outputs": True}},
                "model": {"model_type": "sdxl"},
            }
        )
        with pytest.raises(ValueError, match="Cannot train text encoder parameters while TE output caching is enabled"):
            validate_config(cfg)

    def test_adapter_mode_rejects_learning_rate_groups(self):
        """Adapter mode should fail early when fine-tune-only LR groups are configured."""
        cfg = make_validate_cfg(
            {
                "mode": "adapter",
                "adapter": {"peft": {"lora": {}}},
                "optimizer": {
                    "learning_rates": {
                        "groups": [{"name": "unet_probe", "lr": 1e-5, "match": ["unet.*"]}],
                    }
                },
            }
        )
        with pytest.raises(ValueError, match="currently supported only for fine-tune mode"):
            validate_config(cfg)

    def test_edm2_laplace_flag_fails_fast(self):
        """Dormant EDM2 Laplace timestep weighting should fail fast instead of silently doing nothing."""
        cfg = make_validate_cfg({"loss": {"edm2": {"laplace_timestep_sampling": True}}})

        with pytest.raises(ValueError, match="laplace_timestep_sampling is not implemented"):
            validate_config(cfg)

    def test_timestep_sampling_rejects_removed_experimental_modes(self):
        """Removed one-off adaptive samplers should fail fast in the active config surface."""
        cfg = make_validate_cfg({"timestep": {"timestep_sampling": "snr_windowed"}})

        with pytest.raises(ValueError, match="timestep\\.timestep_sampling must be one of"):
            validate_config(cfg)

    def test_timestep_sampling_rejects_removed_shift_alias(self):
        """The old shift alias should fail fast now that logit_normal is canonical."""
        cfg = make_validate_cfg({"timestep": {"timestep_sampling": "shift"}})

        with pytest.raises(ValueError, match="timestep\\.timestep_sampling must be one of"):
            validate_config(cfg)

    def test_adaptive_log_snr_prior_weight_must_be_in_range(self):
        """Adaptive log-SNR config should validate its probability-mixing bounds."""
        cfg = make_validate_cfg({"timestep": {"adaptive_log_snr": {"prior_weight": 1.5}}})

        with pytest.raises(ValueError, match="timestep\\.adaptive_log_snr\\.prior_weight must be between 0\\.0 and 1\\.0 inclusive"):
            validate_config(cfg)

    def test_logit_normal_timestep_mode_is_available_outside_sd3(self):
        """The shared logit-normal sampler should validate on non-SD3 model types too."""
        cfg = make_validate_cfg({"model": {"model_type": "sd15"}, "timestep": {"timestep_sampling": "logit_normal"}})

        validate_config(cfg)

    def test_shared_log_snr_modes_reject_active_sd3_path(self):
        """Shared DDPM timestep samplers should fail fast on the current RF path until implemented there."""
        cfg = make_validate_cfg(
            {
                "model": {"model_type": "sd3"},
                "objective": {"path": "rectified_flow", "prediction": "flow"},
                "timestep": {"timestep_sampling": "adaptive_log_snr"},
            }
        )

        with pytest.raises(ValueError, match="is not implemented for the active SD3/RF timestep path yet"):
            validate_config(cfg)

    def test_validation_split_must_be_between_zero_and_one(self):
        """Validation split should fail fast when outside the scanner-supported range."""
        cfg = make_validate_cfg({"validation": {"validation_split": 1.5}})

        with pytest.raises(ValueError, match="validation\\.validation_split must be between 0\\.0 and 1\\.0 inclusive"):
            validate_config(cfg)

    def test_max_validation_steps_must_be_positive(self):
        """Zero validation steps should fail instead of dividing by zero in the validation loop."""
        cfg = make_validate_cfg({"validation": {"max_validation_steps": 0}})

        with pytest.raises(ValueError, match="validation\\.max_validation_steps must be a positive integer"):
            validate_config(cfg)

    def test_validation_timesteps_literal_must_parse(self):
        """Malformed validation timestep literals should fail before strategy runtime."""
        cfg = make_validate_cfg({"validation": {"validation_timesteps": "[50, bad, 950]"}})

        with pytest.raises(ValueError, match="validation\\.validation_timesteps must be a valid Python list/tuple literal"):
            validate_config(cfg)

    def test_validation_timesteps_must_not_be_empty(self):
        """An empty timestep list would produce a divide-by-zero in validation averaging."""
        cfg = make_validate_cfg({"validation": {"validation_timesteps": "[]"}})

        with pytest.raises(ValueError, match="validation\\.validation_timesteps must contain at least one timestep"):
            validate_config(cfg)

    def test_validation_timesteps_must_be_non_negative_ints(self):
        """Only non-negative integer validation timesteps should be accepted."""
        cfg = make_validate_cfg({"validation": {"validation_timesteps": "[50, -1, 950]"}})

        with pytest.raises(ValueError, match="validation\\.validation_timesteps must contain only non-negative integers"):
            validate_config(cfg)

    def test_validation_split_conflicts_with_separate_validation_dir(self):
        """Separate validation directories should not be combined with train-data splitting."""
        cfg = make_validate_cfg(
            {
                "data": {"source": {"val_data_dir": "/tmp/val-data"}},
                "validation": {"validation_split": 0.2},
            }
        )

        with pytest.raises(
            ValueError,
            match="data\\.source\\.val_data_dir and validation\\.validation_split cannot both be set",
        ):
            validate_config(cfg)

    def test_validation_schedule_requires_validation_data_source(self):
        """Explicit validation scheduling should fail fast when no validation data exists."""
        cfg = make_validate_cfg({"validation": {"run_at_start": True}})

        with pytest.raises(
            ValueError,
            match="Validation is scheduled but no validation data source is configured",
        ):
            validate_config(cfg)

    def test_validation_schedule_with_split_is_allowed(self):
        """Validation scheduling should be allowed when train-data splitting provides validation data."""
        cfg = make_validate_cfg(
            {
                "validation": {
                    "validation_split": 0.2,
                    "validate_every_n_steps": 100,
                }
            }
        )

        validate_config(cfg)

    def test_validation_schedule_with_separate_val_dir_is_allowed(self):
        """Validation scheduling should be allowed when a separate validation directory exists."""
        cfg = make_validate_cfg(
            {
                "data": {"source": {"val_data_dir": "/tmp/val-data"}},
                "validation": {"validate_every_n_epochs": 1},
            }
        )

        validate_config(cfg)


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
