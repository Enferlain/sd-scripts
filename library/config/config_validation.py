"""
Centralized config validation for all training scripts.

This module consolidates all validation logic from scattered __post_init__ methods
and script-specific checks into a single location for discoverability.

Usage:
    from library.config.validation import prepare_config, validate_config

    @hydra.main(...)
    def main(cfg):
        prepare_config(cfg)
        validate_config(cfg)
        # ... training code
"""

import logging

from library.optimizers.optimizer_utils import should_train_text_encoder

logger = logging.getLogger(__name__)


def _is_non_bool_number(value: object) -> bool:
    """Return True for int/float values, excluding bool."""
    return isinstance(value, int | float) and not isinstance(value, bool)


# =============================================================================
# Auto-fixups (mutate config)
# =============================================================================


def prepare_config(cfg) -> None:
    """
    Apply auto-fixups to config. Call immediately after Hydra loads.

    These are non-error modifications that enable dependent flags or handle
    backward compatibility.
    """
    # Dataset: cache_latents_to_disk implies cache_latents
    if cfg.data.caching.cache_latents_to_disk and not cfg.data.caching.cache_latents:
        cfg.data.caching.cache_latents = True

    # Dataset: backward compat for old typo
    if cfg.data.caption.caption_extention is not None:
        cfg.data.caption.caption_extension = cfg.data.caption.caption_extention

    # Performance: cache_text_encoder_outputs_to_disk implies cache_text_encoder_outputs
    if (
        hasattr(cfg, "performance")
        and cfg.performance is not None
        and cfg.data.caching.cache_text_encoder_outputs_to_disk
        and not cfg.data.caching.cache_text_encoder_outputs
    ):
        cfg.data.caching.cache_text_encoder_outputs = True

    # Optimizer: shortcut flags
    if cfg.optimizer.use_8bit_adam:
        cfg.optimizer.optimizer_type = "AdamW8bit"
    if cfg.optimizer.use_lion_optimizer:
        cfg.optimizer.optimizer_type = "Lion"

    # Learning rates: default denoiser/text_encoders to base if not set
    if hasattr(cfg.optimizer, "learning_rates"):
        lr_cfg = cfg.optimizer.learning_rates
        if lr_cfg.denoiser is None:
            lr_cfg.denoiser = lr_cfg.base
        if lr_cfg.text_encoders is None:
            lr_cfg.text_encoders = lr_cfg.base

    # Data: cache_dir defaults to train_data_dir if not set
    if (
        hasattr(cfg.data, "caching")
        and getattr(cfg.data.caching, "cache_dir", None) is None
        and hasattr(cfg.data, "source")
        and cfg.data.source.train_data_dir
    ):
        cfg.data.caching.cache_dir = cfg.data.source.train_data_dir

    # Sampling: disable if <= 0
    if _is_non_bool_number(cfg.output.sampling.sample_every_n_epochs) and cfg.output.sampling.sample_every_n_epochs <= 0:
        logger.warning("sample_every_n_epochs <= 0, disabling")
        cfg.output.sampling.sample_every_n_epochs = None
    if _is_non_bool_number(cfg.output.sampling.sample_every_n_steps) and cfg.output.sampling.sample_every_n_steps <= 0:
        logger.warning("sample_every_n_steps <= 0, disabling")
        cfg.output.sampling.sample_every_n_steps = None

    # Logging: normalize invalid tracker emission cadence
    try:
        if _is_non_bool_number(cfg.output.logging.log_every_n_steps) and cfg.output.logging.log_every_n_steps <= 0:
            logger.warning(f"log_every_n_steps={cfg.output.logging.log_every_n_steps} <= 0, defaulting to 1")
            cfg.output.logging.log_every_n_steps = 1
    except AttributeError:
        pass  # Field not present in partial config

    # Logging: resource monitor normalization
    try:
        logging_cfg = cfg.output.logging
        resource_monitor_cfg = logging_cfg.resource_monitor

        if isinstance(resource_monitor_cfg.enabled, bool) and not resource_monitor_cfg.enabled:
            resource_monitor_cfg.mode = "off"

        if _is_non_bool_number(resource_monitor_cfg.log_every_n_steps) and resource_monitor_cfg.log_every_n_steps < 0:
            logger.warning(f"resource_monitor.log_every_n_steps={resource_monitor_cfg.log_every_n_steps} < 0, defaulting to 0")
            resource_monitor_cfg.log_every_n_steps = 0

        if _is_non_bool_number(resource_monitor_cfg.sample_interval_sec) and resource_monitor_cfg.sample_interval_sec <= 0:
            logger.warning(f"resource_monitor.sample_interval_sec={resource_monitor_cfg.sample_interval_sec} <= 0, defaulting to 1.0")
            resource_monitor_cfg.sample_interval_sec = 1.0

        if _is_non_bool_number(resource_monitor_cfg.jsonl_flush_every_n_events) and resource_monitor_cfg.jsonl_flush_every_n_events < 1:
            logger.warning(
                f"resource_monitor.jsonl_flush_every_n_events={resource_monitor_cfg.jsonl_flush_every_n_events} < 1, defaulting to 1"
            )
            resource_monitor_cfg.jsonl_flush_every_n_events = 1

        if _is_non_bool_number(resource_monitor_cfg.queue_maxsize) and resource_monitor_cfg.queue_maxsize < 1:
            logger.warning(f"resource_monitor.queue_maxsize={resource_monitor_cfg.queue_maxsize} < 1, defaulting to 1")
            resource_monitor_cfg.queue_maxsize = 1

        if _is_non_bool_number(resource_monitor_cfg.max_collection_ms) and resource_monitor_cfg.max_collection_ms < 0:
            logger.warning(f"resource_monitor.max_collection_ms={resource_monitor_cfg.max_collection_ms} < 0, defaulting to 0.0")
            resource_monitor_cfg.max_collection_ms = 0.0

        if _is_non_bool_number(resource_monitor_cfg.deep_window_steps) and resource_monitor_cfg.deep_window_steps < 0:
            logger.warning(f"resource_monitor.deep_window_steps={resource_monitor_cfg.deep_window_steps} < 0, defaulting to 0")
            resource_monitor_cfg.deep_window_steps = 0

        if _is_non_bool_number(resource_monitor_cfg.deep_window_seconds) and resource_monitor_cfg.deep_window_seconds < 0:
            logger.warning(f"resource_monitor.deep_window_seconds={resource_monitor_cfg.deep_window_seconds} < 0, defaulting to 0.0")
            resource_monitor_cfg.deep_window_seconds = 0.0
    except AttributeError:
        pass

    # Validation: normalize invalid cadence values
    if hasattr(cfg, "validation") and cfg.validation is not None:
        if _is_non_bool_number(cfg.validation.validate_every_n_steps) and cfg.validation.validate_every_n_steps <= 0:
            logger.warning(f"validate_every_n_steps={cfg.validation.validate_every_n_steps} <= 0, disabling")
            cfg.validation.validate_every_n_steps = None
        if _is_non_bool_number(cfg.validation.validate_every_n_epochs) and cfg.validation.validate_every_n_epochs <= 0:
            logger.warning(f"validate_every_n_epochs={cfg.validation.validate_every_n_epochs} <= 0, disabling")
            cfg.validation.validate_every_n_epochs = None


# =============================================================================
# Cross-config validation (errors and warnings)
# =============================================================================


def validate_config(cfg) -> None:
    """
    Validate config for conflicting settings. Call after prepare_config().

    Raises ValueError for hard errors, logs warnings for soft issues.
    """
    # Resource monitor validation
    try:
        resource_monitor_cfg = cfg.output.logging.resource_monitor
    except AttributeError:
        resource_monitor_cfg = None

    if resource_monitor_cfg is not None:
        valid_modes = {"off", "basic", "sampled", "deep"}
        if resource_monitor_cfg.mode not in valid_modes:
            raise ValueError(f"resource_monitor.mode must be one of {sorted(valid_modes)}, got {resource_monitor_cfg.mode}")

        valid_rank_scopes = {"main", "all"}
        if resource_monitor_cfg.rank_scope not in valid_rank_scopes:
            raise ValueError(
                f"resource_monitor.rank_scope must be one of {sorted(valid_rank_scopes)}, got {resource_monitor_cfg.rank_scope}"
            )

        valid_device_scopes = {"local", "all_visible"}
        if resource_monitor_cfg.device_scope not in valid_device_scopes:
            raise ValueError(
                f"resource_monitor.device_scope must be one of {sorted(valid_device_scopes)}, got {resource_monitor_cfg.device_scope}"
            )

        valid_flush_modes = {"auto", "line", "batch"}
        if resource_monitor_cfg.jsonl_flush_mode not in valid_flush_modes:
            raise ValueError(
                f"resource_monitor.jsonl_flush_mode must be one of {sorted(valid_flush_modes)}, got {resource_monitor_cfg.jsonl_flush_mode}"
            )

        valid_drop_policies = {"drop_oldest", "drop_newest", "block"}
        if resource_monitor_cfg.drop_policy not in valid_drop_policies:
            raise ValueError(
                f"resource_monitor.drop_policy must be one of {sorted(valid_drop_policies)}, got {resource_monitor_cfg.drop_policy}"
            )

    # === Errors ===

    # Regularization: adaptive_noise_scale requires noise_offset
    if cfg.loss.regularization.adaptive_noise_scale is not None and cfg.loss.regularization.noise_offset is None:
        raise ValueError("adaptive_noise_scale requires noise_offset")

    # Loss: scale_v_pred_loss_like_noise_pred requires v_parameterization
    if cfg.loss.snr.scale_v_pred_loss_like_noise_pred and not cfg.loss.v_parameterization:
        raise ValueError("scale_v_pred_loss_like_noise_pred requires v_parameterization")

    # Loss: v_pred_like_loss conflicts with v_parameterization
    if cfg.loss.snr.v_pred_like_loss is not None and cfg.loss.v_parameterization:
        raise ValueError("v_pred_like_loss conflicts with v_parameterization")

    # Precision: full_fp16 requires mixed_precision='fp16'
    if hasattr(cfg, "performance") and cfg.performance is not None:
        if cfg.performance.precision.full_fp16 and cfg.performance.precision.mixed_precision != "fp16":
            raise ValueError("full_fp16 requires mixed_precision='fp16'")
        if cfg.performance.precision.full_bf16 and cfg.performance.precision.mixed_precision != "bf16":
            raise ValueError("full_bf16 requires mixed_precision='bf16'")
        # fp8_base requires mixed precision enabled
        if (
            hasattr(cfg.performance, "fp8_base")
            and (cfg.performance.precision.fp8_base or getattr(cfg.performance, "fp8_base_unet", False))
            and cfg.performance.precision.mixed_precision == "no"
        ):
            raise ValueError("fp8_base requires mixed_precision='fp16' or 'bf16'")

        # TE offloading + caching conflict (can't use both)
        if cfg.performance.memory.offload_text_encoders and cfg.data.caching.cache_text_encoder_outputs:
            raise ValueError(
                "Cannot use both offload_text_encoders and cache_text_encoder_outputs. "
                "Choose one: offloading (allows caption augmentation) or caching (faster, no augmentation)."
            )

        # TE offloading + TE training conflict
        # TODO: Could support granular offloading (e.g., [1e-5, 0] trains TE1 on GPU, offloads TE2 to CPU)
        #       Would require per-TE device placement and mixed-device encoding in _get_text_cond
        if cfg.performance.memory.offload_text_encoders and should_train_text_encoder(cfg.optimizer.learning_rates):
            raise ValueError(
                "Cannot train text encoder while offloading to CPU. Text encoder training requires TEs on GPU. "
                "Either set text_encoders LR to 0, or disable offload_text_encoders."
            )

    # TE caching + TE training conflict
    if cfg.data.caching.cache_text_encoder_outputs and should_train_text_encoder(cfg.optimizer.learning_rates):
        raise ValueError(
            "Cannot train text encoder while caching TE outputs. Set text_encoders LR to 0, or disable cache_text_encoder_outputs."
        )

    # TODO: Revisit when model-agnostic block/layer granular LR is implemented
    # Currently SDXL-specific and assumes 23 blocks - not widely used
    # if hasattr(cfg.optimizer, 'learning_rates') and cfg.optimizer.learning_rates.blocks:
    #     block_lr_count = len(cfg.optimizer.learning_rates.blocks.split(","))
    #     if block_lr_count != 23:
    #         raise ValueError(f"block_lr must have 23 values, got {block_lr_count}")

    # === Warnings ===

    # Model: v2 with clip_skip is unexpected
    if cfg.model.model_type == "sd2" and cfg.training.clip_skip is not None:
        logger.warning("v2 with clip_skip is unexpected")

    # Regularization: zero_terminal_snr without v_parameterization
    if cfg.loss.regularization.zero_terminal_snr and not cfg.loss.v_parameterization:
        logger.warning("zero_terminal_snr is enabled but v_parameterization is not. Training results may be unexpected.")


# =============================================================================
# Script-specific validators
# =============================================================================


def validate_sd_peft(cfg, train_dataset_group, val_dataset_group) -> None:
    """SD 1.5/2.0 PEFT-specific validation."""
    train_dataset_group.verify_bucket_reso_steps(64)
    if val_dataset_group is not None:
        val_dataset_group.verify_bucket_reso_steps(64)


def validate_sdxl_peft(cfg, train_dataset_group, val_dataset_group) -> None:
    """SDXL PEFT-specific validation."""
    train_dataset_group.verify_bucket_reso_steps(32)
    if val_dataset_group is not None:
        val_dataset_group.verify_bucket_reso_steps(32)

    # SDXL caching constraints
    if cfg.data.caching.cache_text_encoder_outputs:
        assert train_dataset_group.is_text_encoder_output_cacheable(), (
            "when caching Text Encoder output, caption_dropout_rate, shuffle_caption, "
            "token_warmup_step, or caption_tag_dropout_rate cannot be used"
        )

    # Cannot train TE peft while caching TE outputs
    train_te = should_train_text_encoder(cfg.optimizer.learning_rates)
    assert not train_te or not cfg.data.caching.cache_text_encoder_outputs, (
        "Adapter for Text Encoder cannot be trained with caching Text Encoder outputs"
    )


def validate_sd_textual_inversion(cfg, train_dataset_group, val_dataset_group) -> None:
    """SD 1.5/2.0 Textual Inversion-specific validation."""
    train_dataset_group.verify_bucket_reso_steps(64)
    if val_dataset_group is not None:
        val_dataset_group.verify_bucket_reso_steps(64)


def validate_sdxl_textual_inversion(cfg, train_dataset_group, val_dataset_group) -> None:
    """SDXL Textual Inversion-specific validation."""
    train_dataset_group.verify_bucket_reso_steps(32)
    if val_dataset_group is not None:
        val_dataset_group.verify_bucket_reso_steps(32)
