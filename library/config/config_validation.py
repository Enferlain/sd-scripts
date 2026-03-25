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

VALID_MODES = {"finetune", "peft", "textual_inversion"}


def _is_non_bool_number(value: object) -> bool:
    """Return True for int/float values, excluding bool."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _get_optional_attr(root, *path, default=None):
    """Return a nested attribute or ``default`` when a partial config omits it."""
    current = root
    for attr in path:
        try:
            current = getattr(current, attr)
        except Exception:
            return default
        if current is None:
            return default
    return current


def _get_required_bucket_reso_steps(model_type: str | None) -> int | None:
    """Return the bucket-step requirement for known active model families."""
    if model_type == "sdxl":
        return 32
    if model_type in {"sd1", "sd15", "sd2"}:
        return 64
    return None


def _is_text_encoder_output_cacheable_config(cfg) -> bool:
    """Return whether the current caption settings permit TE-output caching."""
    caption_cfg = _get_optional_attr(cfg, "data", "caption")
    if caption_cfg is None:
        return True
    return not (
        caption_cfg.caption_dropout_rate > 0
        or caption_cfg.shuffle_caption
        or caption_cfg.token_warmup_step > 0
        or caption_cfg.caption_tag_dropout_rate > 0
    )


def _validate_model_profile_config(cfg) -> None:
    """Validate model-family rules expressible directly from active config."""
    model_type = _get_optional_attr(cfg, "model", "model_type")
    if model_type is None or (isinstance(model_type, str) and model_type.strip() == ""):
        raise ValueError("model.model_type is required. Set it in the config to a value like sd15, sd2, sdxl, or flux.")

    required_steps = _get_required_bucket_reso_steps(model_type)
    bucket_reso_steps = _get_optional_attr(cfg, "data", "bucketing", "bucket_reso_steps")
    if required_steps is not None and bucket_reso_steps is not None and bucket_reso_steps % required_steps != 0:
        raise ValueError(f"bucket_reso_steps={bucket_reso_steps} must be divisible by {required_steps} for model_type={model_type}")

    cache_te_outputs = _get_optional_attr(cfg, "data", "caching", "cache_text_encoder_outputs", default=False)
    if cache_te_outputs and not _is_text_encoder_output_cacheable_config(cfg):
        raise ValueError(
            "cache_text_encoder_outputs cannot be used with caption_dropout_rate, "
            "shuffle_caption, token_warmup_step, or caption_tag_dropout_rate because "
            "those settings change text conditioning between steps."
        )


def _normalize_edm2_loss_config(cfg) -> None:
    """Apply EDM2/SNR auto-fixups that depend on the nested loss config shape."""
    edm2_cfg = _get_optional_attr(cfg, "loss", "edm2")
    snr_cfg = _get_optional_attr(cfg, "loss", "snr")
    if edm2_cfg is None or snr_cfg is None:
        return

    if (
        edm2_cfg.enabled
        and edm2_cfg.importance.enabled
        and not edm2_cfg.importance.safety_override
    ):
        if getattr(snr_cfg, "debiased_estimation_loss", False):
            snr_cfg.debiased_estimation_loss = False
            logger.warning(
                "Debiased estimation loss AND EDM2 loss weighting with importance weighting are enabled. "
                "It is not advised to use both, as there is a possibility of loss curving to 0 as SNR approaches 0, "
                "as such, debiased estimation loss has been disabled. "
                "You may override this behavior by setting "
                "loss.edm2.importance.safety_override=true."
            )

        if getattr(snr_cfg, "min_snr_gamma", None):
            snr_cfg.min_snr_gamma = None
            logger.warning(
                "Min SNR gamma AND EDM2 loss weighting with importance weighting are enabled. "
                "It is not advised to use both, as there is a possibility of loss curving to 0 as SNR approaches 0, "
                "as such, min_snr_gamma has been disabled. "
                "You may override this behavior by setting "
                "loss.edm2.importance.safety_override=true."
            )


def _validate_mode_config(cfg) -> None:
    """Validate the top-level training mode when present on the config."""
    mode = getattr(cfg, "mode", None)
    if mode is None:
        return

    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode}")

    has_peft = _get_optional_attr(cfg, "peft") is not None
    has_textual_inversion = _get_optional_attr(cfg, "textual_inversion") is not None

    if has_peft and has_textual_inversion:
        raise ValueError("`peft` and `textual_inversion` sections cannot both be active in the same config.")

    if mode == "peft":
        if not has_peft:
            raise ValueError("mode=peft requires a `peft` section.")
        if has_textual_inversion:
            raise ValueError("mode=peft cannot be used with a `textual_inversion` section.")

    if mode == "textual_inversion":
        if not has_textual_inversion:
            raise ValueError("mode=textual_inversion requires a `textual_inversion` section.")
        if has_peft:
            raise ValueError("mode=textual_inversion cannot be used with a `peft` section.")

    if mode == "finetune" and (has_peft or has_textual_inversion):
        raise ValueError("mode=finetune cannot be used with `peft` or `textual_inversion` sections.")


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

    _normalize_edm2_loss_config(cfg)

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

    sample_every_n_steps = _get_optional_attr(cfg, "output", "sampling", "sample_every_n_steps")
    sample_every_n_epochs = _get_optional_attr(cfg, "output", "sampling", "sample_every_n_epochs")
    if sample_every_n_steps is not None and sample_every_n_epochs is not None:
        raise ValueError(
            "sample_every_n_steps and sample_every_n_epochs cannot both be set. "
            "Choose either step-based or epoch-based sampling cadence."
        )

    # Regularization: adaptive_noise_scale requires noise_offset
    if cfg.loss.regularization.adaptive_noise_scale is not None and cfg.loss.regularization.noise_offset is None:
        raise ValueError("adaptive_noise_scale requires noise_offset")

    # Loss: scale_v_pred_loss_like_noise_pred requires v_parameterization
    if cfg.loss.snr.scale_v_pred_loss_like_noise_pred and not cfg.loss.v_parameterization:
        raise ValueError("scale_v_pred_loss_like_noise_pred requires v_parameterization")

    # Loss: v_pred_like_loss conflicts with v_parameterization
    if cfg.loss.snr.v_pred_like_loss is not None and cfg.loss.v_parameterization:
        raise ValueError("v_pred_like_loss conflicts with v_parameterization")

    if _get_optional_attr(cfg, "loss", "edm2", "laplace_timestep_sampling", default=False):
        raise ValueError(
            "loss.edm2.laplace_timestep_sampling is not implemented in the active training path yet. "
            "Disable it until the timestep-sampling wiring is implemented."
        )

    # Precision: full_fp16 requires mixed_precision='fp16'
    if hasattr(cfg, "performance") and cfg.performance is not None:
        if cfg.performance.precision.full_fp16 and cfg.performance.precision.mixed_precision != "fp16":
            raise ValueError("full_fp16 requires mixed_precision='fp16'")
        if cfg.performance.precision.full_bf16 and cfg.performance.precision.mixed_precision != "bf16":
            raise ValueError("full_bf16 requires mixed_precision='bf16'")
        # fp8_base requires mixed precision enabled
        fp8_base = _get_optional_attr(cfg, "performance", "precision", "fp8_base", default=False)
        fp8_base_unet = _get_optional_attr(cfg, "performance", "precision", "fp8_base_unet", default=False)
        mixed_precision = _get_optional_attr(cfg, "performance", "precision", "mixed_precision")
        if (fp8_base or fp8_base_unet) and mixed_precision == "no":
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
            "Cannot train text encoder while TE output caching is enabled. Disable TE output caching, or set text_encoders LR to 0."
        )

    _validate_mode_config(cfg)
    _validate_model_profile_config(cfg)

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


def _validate_dataset_group_bucket_steps(train_dataset_group, val_dataset_group, min_steps: int) -> None:
    """Apply dataset-group bucket-step validation for script-owned dataset paths."""
    train_dataset_group.verify_bucket_reso_steps(min_steps)
    if val_dataset_group is not None:
        val_dataset_group.verify_bucket_reso_steps(min_steps)


def validate_dataset_groups(cfg, train_dataset_group, val_dataset_group) -> None:
    """Validate dataset-group constraints that depend on the active config."""
    required_steps = _get_required_bucket_reso_steps(_get_optional_attr(cfg, "model", "model_type"))
    if required_steps is not None:
        _validate_dataset_group_bucket_steps(train_dataset_group, val_dataset_group, required_steps)

    if _get_optional_attr(cfg, "data", "caching", "cache_text_encoder_outputs", default=False):
        is_cacheable = getattr(train_dataset_group, "is_text_encoder_output_cacheable", None)
        if callable(is_cacheable) and not is_cacheable():
            raise ValueError(
                "cache_text_encoder_outputs cannot be used with dataset/caption settings that change text conditioning "
                "between steps."
            )
