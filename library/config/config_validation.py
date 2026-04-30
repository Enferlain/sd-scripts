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

import ast
from contextlib import suppress
import fnmatch
import logging
import re

from library.adapters.method_configs import (
    build_adapter_runtime_spec,
    get_adapter_peft_config,
    get_inactive_method_config_values,
    get_nondefault_method_config_values,
    resolve_adapter_method_registration,
)
from library.config.dataclasses.peft import VALID_PEFT_CONTINUE_MODES
from library.models.parameter_dump import resolve_component_names
from library.optimization.grouping import resolve_learning_rate_groups
from library.optimization.optimizer_utils import should_train_text_encoder

logger = logging.getLogger(__name__)

VALID_MODES = {"finetune", "adapter", "textual_inversion"}
VALID_TIMESTEP_SAMPLERS = {"uniform", "log_snr_uniform", "adaptive_log_snr", "logit_normal", "cosine_shaped"}
VALID_OBJECTIVE_PATHS = {"ddpm", "rectified_flow"}
VALID_OBJECTIVE_PREDICTIONS = {"epsilon", "v_prediction", "flow"}


def _is_non_bool_number(value: object) -> bool:
    """Return True for int/float values, excluding bool."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_non_bool_int(value: object) -> bool:
    """Return True for int values, excluding bool."""
    return isinstance(value, int) and not isinstance(value, bool)


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


def _pattern_targets_selector(pattern: str, selector_name: str) -> bool:
    """Return whether a group pattern can target the provided selector probe."""
    if pattern.startswith("re:"):
        return re.search(pattern[3:], selector_name) is not None
    return fnmatch.fnmatchcase(selector_name, pattern)


def _has_positive_text_encoder_groups(cfg) -> bool:
    """Return whether explicit positive-LR groups target any known TE selector namespace."""
    learning_rates = _get_optional_attr(cfg, "optimizer", "learning_rates")
    if learning_rates is None:
        return False

    has_inline_groups = bool(_get_optional_attr(learning_rates, "groups", default=[]))
    has_groups_file = _get_optional_attr(learning_rates, "groups_file") is not None
    if not has_inline_groups and not has_groups_file:
        return False

    groups = resolve_learning_rate_groups(learning_rates)
    if not groups:
        return False

    model_type = _get_optional_attr(cfg, "model", "model_type")
    component_names = resolve_component_names(model_type)
    selector_prefixes = list(component_names.text_encoder_names) if component_names is not None else []
    selector_prefixes.extend(f"text_encoder{i + 1}" for i in range(len(selector_prefixes)))
    if not selector_prefixes:
        selector_prefixes.extend(["text_encoder", "text_encoder1", "text_encoder2", "text_encoder3"])

    selector_probes = [probe for prefix in selector_prefixes for probe in (prefix, f"{prefix}.__probe__")]

    for group in groups:
        if group.lr <= 0:
            continue
        if any(_pattern_targets_selector(pattern, selector_probe) for pattern in group.match for selector_probe in selector_probes):
            return True
    return False


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


def _parse_validation_timesteps(raw_timesteps: object) -> list[int]:
    """Parse validation timestep config into a validated list of non-negative ints."""
    if isinstance(raw_timesteps, str):
        try:
            parsed_timesteps = ast.literal_eval(raw_timesteps)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("validation.validation_timesteps must be a valid Python list/tuple literal of integers.") from exc
    elif isinstance(raw_timesteps, list | tuple):
        parsed_timesteps = raw_timesteps
    else:
        raise ValueError("validation.validation_timesteps must be provided as a list/tuple or Python list literal string.")

    if not isinstance(parsed_timesteps, list | tuple) or len(parsed_timesteps) == 0:
        raise ValueError("validation.validation_timesteps must contain at least one timestep.")

    validated_timesteps: list[int] = []
    for timestep in parsed_timesteps:
        if not _is_non_bool_int(timestep):
            raise ValueError("validation.validation_timesteps must contain only integers.")
        if timestep < 0:
            raise ValueError("validation.validation_timesteps must contain only non-negative integers.")
        validated_timesteps.append(int(timestep))

    return validated_timesteps


def _validate_validation_config(cfg) -> None:
    """Validate the active validation config before the runtime loop consumes it."""
    validation_cfg = _get_optional_attr(cfg, "validation")
    if validation_cfg is None:
        return

    validation_split = getattr(validation_cfg, "validation_split", None)
    if validation_split is not None and not 0.0 <= float(validation_split) <= 1.0:
        raise ValueError("validation.validation_split must be between 0.0 and 1.0 inclusive.")

    val_data_dir = _get_optional_attr(cfg, "data", "source", "val_data_dir")
    has_separate_val_dir = isinstance(val_data_dir, str) and val_data_dir.strip() != ""
    has_validation_split = validation_split is not None and float(validation_split) > 0.0
    if has_separate_val_dir and has_validation_split:
        raise ValueError(
            "data.source.val_data_dir and validation.validation_split cannot both be set. "
            "Choose either a separate validation directory or a split from the training data."
        )

    max_validation_steps = getattr(validation_cfg, "max_validation_steps", None)
    if max_validation_steps is not None and (not _is_non_bool_int(max_validation_steps) or max_validation_steps < 1):
        raise ValueError("validation.max_validation_steps must be a positive integer when set.")

    validation_timesteps = getattr(validation_cfg, "validation_timesteps", None)
    if validation_timesteps is not None:
        _parse_validation_timesteps(validation_timesteps)

    explicit_validation_requested = any(
        (
            bool(getattr(validation_cfg, "run_at_start", False)),
            bool(getattr(validation_cfg, "run_at_end", False)),
            getattr(validation_cfg, "validate_every_n_steps", None) is not None,
            getattr(validation_cfg, "validate_every_n_epochs", None) is not None,
        )
    )
    if explicit_validation_requested and not (has_separate_val_dir or has_validation_split):
        raise ValueError(
            "Validation is scheduled but no validation data source is configured. "
            "Set data.source.val_data_dir or validation.validation_split."
        )


def _validate_timestep_config(cfg) -> None:
    """Validate the active timestep sampler surface and adaptive sampler settings."""
    timestep_cfg = _get_optional_attr(cfg, "timestep")
    if timestep_cfg is None:
        return

    timestep_sampling = getattr(timestep_cfg, "timestep_sampling", "uniform") or "uniform"
    if timestep_sampling not in VALID_TIMESTEP_SAMPLERS:
        raise ValueError(f"timestep.timestep_sampling must be one of {sorted(VALID_TIMESTEP_SAMPLERS)}, got {timestep_sampling!r}")

    model_type = _get_optional_attr(cfg, "model", "model_type")
    if timestep_sampling in {"log_snr_uniform", "adaptive_log_snr"} and model_type == "sd3":
        raise ValueError(f"timestep.timestep_sampling={timestep_sampling!r} is not implemented for the active SD3/RF timestep path yet.")

    adaptive_cfg = getattr(timestep_cfg, "adaptive_log_snr", None)
    if adaptive_cfg is None:
        return

    if not _is_non_bool_int(adaptive_cfg.bins) or adaptive_cfg.bins < 2:
        raise ValueError("timestep.adaptive_log_snr.bins must be an integer >= 2.")
    if not _is_non_bool_number(adaptive_cfg.ema_beta) or not 0.0 <= float(adaptive_cfg.ema_beta) < 1.0:
        raise ValueError("timestep.adaptive_log_snr.ema_beta must be in [0.0, 1.0).")
    if not _is_non_bool_number(adaptive_cfg.temperature) or float(adaptive_cfg.temperature) <= 0.0:
        raise ValueError("timestep.adaptive_log_snr.temperature must be > 0.")
    if not _is_non_bool_number(adaptive_cfg.prior_weight) or not 0.0 <= float(adaptive_cfg.prior_weight) <= 1.0:
        raise ValueError("timestep.adaptive_log_snr.prior_weight must be between 0.0 and 1.0 inclusive.")
    if not _is_non_bool_number(adaptive_cfg.min_prob) or float(adaptive_cfg.min_prob) < 0.0:
        raise ValueError("timestep.adaptive_log_snr.min_prob must be >= 0.")
    if not _is_non_bool_int(adaptive_cfg.warmup_steps) or adaptive_cfg.warmup_steps < 0:
        raise ValueError("timestep.adaptive_log_snr.warmup_steps must be a non-negative integer.")
    if not _is_non_bool_number(adaptive_cfg.entropy_floor) or not 0.0 <= float(adaptive_cfg.entropy_floor) <= 1.0:
        raise ValueError("timestep.adaptive_log_snr.entropy_floor must be between 0.0 and 1.0 inclusive.")
    if (
        not _is_non_bool_number(adaptive_cfg.uniform_mix_when_low_entropy)
        or not 0.0 <= float(adaptive_cfg.uniform_mix_when_low_entropy) <= 1.0
    ):
        raise ValueError("timestep.adaptive_log_snr.uniform_mix_when_low_entropy must be between 0.0 and 1.0 inclusive.")


def _normalize_edm2_loss_config(cfg) -> None:
    """Apply EDM2/SNR auto-fixups that depend on the nested loss config shape."""
    edm2_cfg = _get_optional_attr(cfg, "loss", "edm2")
    snr_cfg = _get_optional_attr(cfg, "loss", "snr")
    if edm2_cfg is None or snr_cfg is None:
        return

    if edm2_cfg.enabled and edm2_cfg.importance.enabled and not edm2_cfg.importance.safety_override:
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

    has_adapter = _get_optional_attr(cfg, "adapter") is not None
    has_root_peft = _get_optional_attr(cfg, "peft") is not None
    has_peft = get_adapter_peft_config(cfg) is not None
    has_textual_inversion = _get_optional_attr(cfg, "textual_inversion") is not None

    if has_root_peft:
        raise ValueError("Root `peft` config has moved to `adapter.peft`.")

    if has_adapter and has_textual_inversion:
        raise ValueError("`adapter` and `textual_inversion` sections cannot both be active in the same config.")

    if mode == "adapter":
        if not has_peft:
            raise ValueError("mode=adapter requires an `adapter.peft` section.")
        if has_textual_inversion:
            raise ValueError("mode=adapter cannot be used with a `textual_inversion` section.")

    if mode == "textual_inversion":
        if not has_textual_inversion:
            raise ValueError("mode=textual_inversion requires a `textual_inversion` section.")
        if has_adapter:
            raise ValueError("mode=textual_inversion cannot be used with an `adapter` section.")

    if mode == "finetune" and (has_adapter or has_textual_inversion):
        raise ValueError("mode=finetune cannot be used with `adapter` or `textual_inversion` sections.")


def _validate_peft_config(cfg) -> None:
    """Validate the active PEFT config surface and continuation intent."""

    peft_cfg = get_adapter_peft_config(cfg)
    if peft_cfg is None:
        return

    try:
        registration = resolve_adapter_method_registration(peft_cfg)
    except (KeyError, ValueError) as exc:
        raise ValueError(str(exc)) from exc

    continue_mode = getattr(peft_cfg, "continue_mode", None)
    if continue_mode is not None and continue_mode not in VALID_PEFT_CONTINUE_MODES:
        raise ValueError(
            f"adapter.peft.continue_mode must be one of {list(VALID_PEFT_CONTINUE_MODES)}, got {continue_mode!r}"
        )

    continue_from = getattr(peft_cfg, "continue_from", None)
    if continue_from is None and continue_mode is not None:
        raise ValueError("adapter.peft.continue_mode can only be set explicitly when adapter.peft.continue_from is also set.")
    effective_continue_mode = continue_mode or "strict"

    legacy_adapter_args = getattr(peft_cfg, "adapter_args", None)
    if legacy_adapter_args:
        raise ValueError(
            "adapter.peft.adapter_args is no longer part of the forward adapter config surface. "
            f"Move those settings under adapter.peft.{registration.name}."
        )

    inactive_method_values = get_inactive_method_config_values(peft_cfg, active_method=registration.name)
    if inactive_method_values:
        inactive_methods = ", ".join(sorted(inactive_method_values))
        raise ValueError(
            f"adapter.peft.{registration.name} is active, but other method config branches also have values: {inactive_methods}."
        )

    build_adapter_runtime_spec(peft_cfg)

    if continue_from is not None and effective_continue_mode == "strict":
        active_method_values = get_nondefault_method_config_values(peft_cfg, registration.name)
        if active_method_values:
            raise ValueError(
                f"adapter.peft.continue_mode='strict' treats the artifact as authoritative. "
                f"Remove active adapter.peft.{registration.name} settings or use continue_mode='initialize_from_artifact'."
            )


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

    # Learning rates: inherit denoiser/text_encoders from base when one exists.
    # If base is null, inherited component fields remain null as well.
    if hasattr(cfg.optimizer, "learning_rates"):
        lr_cfg = cfg.optimizer.learning_rates
        if lr_cfg.denoiser is None:
            lr_cfg.denoiser = lr_cfg.base
        if lr_cfg.text_encoders is None:
            lr_cfg.text_encoders = lr_cfg.base

    peft_cfg = get_adapter_peft_config(cfg)
    if peft_cfg is not None:
        legacy_module = getattr(peft_cfg, "adapter_module", None)
        if getattr(peft_cfg, "method", None) is None and legacy_module:
            with suppress(KeyError, ValueError):
                peft_cfg.method = resolve_adapter_method_registration(peft_cfg).name

        legacy_continue_from = getattr(peft_cfg, "adapter_weights", None)
        if getattr(peft_cfg, "continue_from", None) is None and legacy_continue_from is not None:
            peft_cfg.continue_from = legacy_continue_from
            peft_cfg.continue_mode = "strict" if getattr(peft_cfg, "adapter_rank_from_weights", False) else "initialize_from_artifact"

        legacy_training_comment = getattr(peft_cfg, "training_comment", None)
        metadata_cfg = _get_optional_attr(cfg, "output", "metadata")
        if metadata_cfg is not None and getattr(metadata_cfg, "training_comment", None) is None and legacy_training_comment is not None:
            metadata_cfg.training_comment = legacy_training_comment

    _normalize_edm2_loss_config(cfg)

    objective_cfg = _get_optional_attr(cfg, "objective")
    if objective_cfg is not None:
        objective_path = getattr(objective_cfg, "path", None)
        configured_prediction = getattr(objective_cfg, "prediction", None)
        current_v_parameterization = _get_optional_attr(cfg, "loss", "v_parameterization", default=False)
        if configured_prediction is not None and current_v_parameterization != (configured_prediction == "v_prediction"):
            if _get_optional_attr(cfg, "loss") is not None:
                cfg.loss.v_parameterization = configured_prediction == "v_prediction"
            if objective_path in {None, "ddpm"}:
                logger.warning(
                    "objective.prediction overrides legacy loss.v_parameterization; the boolean is being synchronized for compatibility."
                )

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
            "sample_every_n_steps and sample_every_n_epochs cannot both be set. Choose either step-based or epoch-based sampling cadence."
        )

    # Regularization: adaptive_noise_scale requires noise_offset
    if cfg.loss.regularization.adaptive_noise_scale is not None and cfg.loss.regularization.noise_offset is None:
        raise ValueError("adaptive_noise_scale requires noise_offset")

    objective_path = _get_optional_attr(cfg, "objective", "path", default="ddpm")
    objective_prediction = _get_optional_attr(cfg, "objective", "prediction", default="epsilon")

    if objective_path not in VALID_OBJECTIVE_PATHS:
        raise ValueError(f"objective.path must be one of {sorted(VALID_OBJECTIVE_PATHS)}, got {objective_path}")
    if objective_prediction not in VALID_OBJECTIVE_PREDICTIONS:
        raise ValueError(f"objective.prediction must be one of {sorted(VALID_OBJECTIVE_PREDICTIONS)}, got {objective_prediction}")

    model_type = _get_optional_attr(cfg, "model", "model_type")
    if model_type == "sd3" and objective_path != "rectified_flow":
        raise ValueError("model.model_type=sd3 requires objective.path='rectified_flow'")
    if model_type in {"sd1", "sd15", "sd2"} and objective_path != "ddpm":
        raise ValueError(f"model.model_type={model_type} requires objective.path='ddpm'")

    if objective_path == "ddpm" and objective_prediction not in {"epsilon", "v_prediction"}:
        raise ValueError("objective.path='ddpm' requires objective.prediction to be 'epsilon' or 'v_prediction'")
    if objective_path == "rectified_flow" and objective_prediction != "flow":
        raise ValueError("objective.path='rectified_flow' requires objective.prediction='flow'")
    if objective_path == "rectified_flow" and _get_optional_attr(cfg, "loss", "edm2", "enabled", default=False):
        raise ValueError("loss.edm2 is only supported with objective.path='ddpm'")

    is_v_prediction = objective_prediction == "v_prediction"

    # Loss: scale_v_pred_loss_like_noise_pred requires v_prediction
    if cfg.loss.snr.scale_v_pred_loss_like_noise_pred and not is_v_prediction:
        raise ValueError("scale_v_pred_loss_like_noise_pred requires objective.prediction='v_prediction'")

    # Loss: v_pred_like_loss conflicts with v_prediction
    if cfg.loss.snr.v_pred_like_loss is not None and is_v_prediction:
        raise ValueError("v_pred_like_loss conflicts with objective.prediction='v_prediction'")

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
        trains_text_encoder = should_train_text_encoder(cfg.optimizer.learning_rates) or _has_positive_text_encoder_groups(cfg)
        if cfg.performance.memory.offload_text_encoders and trains_text_encoder:
            raise ValueError(
                "Cannot train text encoder parameters while offloading to CPU. Text encoder training requires TEs on GPU. "
                "Set text_encoders LR to 0 and remove TE-targeting groups, or disable offload_text_encoders."
            )

    # TE caching + TE training conflict
    trains_text_encoder = should_train_text_encoder(cfg.optimizer.learning_rates) or _has_positive_text_encoder_groups(cfg)
    if cfg.data.caching.cache_text_encoder_outputs and trains_text_encoder:
        raise ValueError(
            "Cannot train text encoder parameters while TE output caching is enabled. "
            "Disable TE output caching, or set text_encoders LR to 0 and remove TE-targeting groups."
        )

    if _get_optional_attr(cfg, "mode") == "adapter" and resolve_learning_rate_groups(cfg.optimizer.learning_rates):
        raise ValueError(
            "optimizer.learning_rates.groups and groups_file are currently supported only for fine-tune mode. "
            "Remove groups/groups_file or switch mode to finetune."
        )

    _validate_validation_config(cfg)
    _validate_mode_config(cfg)
    _validate_peft_config(cfg)
    _validate_model_profile_config(cfg)
    _validate_timestep_config(cfg)

    # === Warnings ===

    # Model: v2 with clip_skip is unexpected
    if cfg.model.model_type == "sd2" and cfg.training.clip_skip is not None:
        logger.warning("v2 with clip_skip is unexpected")

    # DDPM scheduler shaping: zero_terminal_snr without v_prediction
    if objective_path == "ddpm" and cfg.loss.regularization.zero_terminal_snr and not is_v_prediction:
        logger.warning("zero_terminal_snr is enabled but objective.prediction is not 'v_prediction'. Training results may be unexpected.")


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
                "cache_text_encoder_outputs cannot be used with dataset/caption settings that change text conditioning between steps."
            )
