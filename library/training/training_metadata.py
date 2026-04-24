"""
Training metadata generation for model saving.

Creates metadata dicts that are embedded in trained model files (e.g., safetensors).
This includes training configuration, dataset statistics, and provenance information.
"""

import json
import os

from library.adapters.method_configs import get_method_config, resolve_adapter_method_registration
from library.constants import SS_METADATA_MINIMUM_KEYS
from library.data import DatasetManifest, compute_tag_frequency
from library.objectives import ObjectiveDefinition, build_objective
from library.utils.hash_utils import get_git_revision_hash, model_hash, calculate_hash


def append_objective_metadata(metadata: dict[str, object], cfg, objective_name: str) -> None:
    """Append objective/runtime metadata that belongs to the shared metadata builder."""
    if objective_name == "rectified_flow":
        metadata["ss_timestep_sampling"] = cfg.timestep.timestep_sampling
        metadata["ss_rf_loss_weighting_scheme"] = cfg.timestep.rf_loss_weighting_scheme
        metadata["ss_training_shift"] = cfg.timestep.training_shift
        metadata["ss_logit_mean"] = cfg.timestep.logit_mean
        metadata["ss_logit_std"] = cfg.timestep.logit_std
        metadata["ss_cosine_shape_scale"] = cfg.timestep.cosine_shape_scale


def create_training_metadata(
    cfg,
    manifest: DatasetManifest,
    val_manifest: DatasetManifest | None,
    session_id: int,
    training_started_at: float,
    model_version: str,
    num_train_epochs: int,
    optimizer_name: str,
    optimizer_args: str,
    net_kwargs: dict,
    num_batches_per_epoch: int,
    total_batch_size: int,
    use_dreambooth_method: bool,  # TODO Parameter 'use_dreambooth_method' value is not used
    objective: ObjectiveDefinition | None = None,
) -> tuple:
    """
    Create training metadata dict for model saving.

    Args:
        cfg: Configuration object.
        manifest: DatasetManifest for training data.
        val_manifest: DatasetManifest for validation data (optional).
        session_id: Session ID.
        training_started_at: Timestamp when training started.
        model_version: Version of the base model.
        num_train_epochs: Number of training epochs.
        optimizer_name: Name of the optimizer.
        optimizer_args: Arguments for the optimizer.
        net_kwargs: Network keyword arguments (e.g., for LoRA).
        num_batches_per_epoch: Number of batches per epoch.
        total_batch_size: Total batch size.
        use_dreambooth_method: Whether DreamBooth method is used.

    Returns:
        tuple: (metadata dict, minimum_metadata dict)
    """
    # Compute stats from manifest
    # Note: Training images exclude regularization images (is_reg=True)
    num_train_images = sum(e.num_repeats for e in manifest.entries.values() if not e.is_reg)
    num_reg_images = sum(e.num_repeats for e in manifest.entries.values() if e.is_reg)
    num_val_images = sum(e.num_repeats for e in val_manifest.entries.values()) if val_manifest else 0

    metadata = {
        "ss_session_id": session_id,
        "ss_training_started_at": training_started_at,
        "ss_output_name": cfg.output.saving.output_name,
        "ss_learning_rate": cfg.optimizer.learning_rates.base,
        "ss_text_encoder_lr": cfg.optimizer.learning_rates.text_encoders,
        "ss_unet_lr": cfg.optimizer.learning_rates.denoiser,
        "ss_num_train_images": num_train_images,
        "ss_num_validation_images": num_val_images,
        "ss_num_reg_images": num_reg_images,
        "ss_num_batches_per_epoch": num_batches_per_epoch,
        "ss_num_epochs": num_train_epochs,
        "ss_gradient_checkpointing": cfg.performance.memory.gradient_checkpointing,
        "ss_gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "ss_max_train_steps": cfg.training.max_train_steps,
        "ss_lr_warmup_steps": cfg.optimizer.scheduler.lr_warmup_steps,
        "ss_lr_scheduler": cfg.optimizer.scheduler.lr_scheduler,
        "ss_mixed_precision": cfg.performance.precision.mixed_precision,
        "ss_full_fp16": bool(cfg.performance.precision.full_fp16),
        "ss_v2": bool(cfg.model.model_type == "sd2"),
        "ss_base_model_version": model_version,
        "ss_clip_skip": cfg.training.clip_skip,
        "ss_max_token_length": cfg.training.max_token_length,
        "ss_cache_latents": bool(cfg.data.caching.cache_latents),
        "ss_seed": cfg.training.seed,
        "ss_lowram": cfg.performance.memory.lowram,
        "ss_noise_offset": cfg.loss.regularization.noise_offset,
        "ss_multires_noise_iterations": cfg.loss.regularization.multires_noise_iterations,
        "ss_multires_noise_discount": cfg.loss.regularization.multires_noise_discount,
        "ss_adaptive_noise_scale": cfg.loss.regularization.adaptive_noise_scale,
        "ss_zero_terminal_snr": cfg.loss.regularization.zero_terminal_snr,
        "ss_sd_scripts_commit_hash": get_git_revision_hash(),
        "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
        "ss_max_grad_norm": cfg.optimizer.max_grad_norm,
        "ss_caption_dropout_rate": cfg.data.caption.caption_dropout_rate,
        "ss_caption_dropout_every_n_epochs": cfg.data.caption.caption_dropout_every_n_epochs,
        "ss_caption_tag_dropout_rate": cfg.data.caption.caption_tag_dropout_rate,
        "ss_face_crop_aug_range": cfg.data.preprocessing.face_crop_aug_range,
        "ss_prior_loss_weight": cfg.loss.prior_loss_weight,
        "ss_min_snr_gamma": cfg.loss.snr.min_snr_gamma,
        "ss_ip_noise_gamma": cfg.loss.regularization.ip_noise_gamma,
        "ss_debiased_estimation": bool(cfg.loss.snr.debiased_estimation_loss),
        "ss_noise_offset_random_strength": cfg.loss.regularization.noise_offset_random_strength,
        "ss_ip_noise_gamma_random_strength": cfg.loss.regularization.ip_noise_gamma_random_strength,
        "ss_loss_type": cfg.loss.loss_type,
        "ss_huber_schedule": cfg.loss.huber.huber_schedule,
        "ss_huber_scale": cfg.loss.huber.huber_scale,
        "ss_huber_c": cfg.loss.huber.huber_c,
        "ss_fp8_base": bool(cfg.performance.precision.fp8_base),
        "ss_fp8_base_unet": bool(cfg.performance.precision.fp8_base_unet),
        "ss_validation_seed": cfg.validation.validation_seed,
        "ss_validation_split": float(cfg.validation.validation_split),
        "ss_max_validation_steps": cfg.validation.max_validation_steps,
        "ss_validate_every_n_epochs": cfg.validation.validate_every_n_epochs,
        "ss_validate_every_n_steps": cfg.validation.validate_every_n_steps,
        "ss_run_validation_at_start": cfg.validation.run_at_start,
        "ss_run_validation_at_end": cfg.validation.run_at_end,
        "ss_resize_interpolation": cfg.data.preprocessing.resize_interpolation,
    }

    # PEFT-specific metadata (omitted entirely for non-PEFT modes)
    if hasattr(cfg, "peft") and cfg.peft is not None:
        registration = resolve_adapter_method_registration(cfg.peft)
        _registration, method_config = get_method_config(cfg.peft)
        metadata_config = getattr(cfg.output, "metadata", None)
        metadata["ss_adapter_module"] = registration.legacy_module_path
        metadata["ss_adapter_rank"] = getattr(method_config, "rank", None)
        metadata["ss_adapter_alpha"] = getattr(method_config, "alpha", None)
        metadata["ss_adapter_neuron_dropout"] = getattr(method_config, "dropout", None)
        metadata["ss_training_comment"] = getattr(metadata_config, "training_comment", None)
        metadata["ss_scale_weight_norms"] = cfg.peft.scale_weight_norms

    # Compute tag frequency from manifest
    tag_frequency = compute_tag_frequency(manifest, cfg.data.caption.caption_separator)

    # Build bucket info from manifest
    bucket_info = {}
    for bucket_key, bucket in manifest.buckets.items():
        bucket_info[bucket_key] = {
            "resolution": bucket.resolution,
            "count": len(bucket.image_ids),
        }

    # Build dataset dirs info from manifest entries
    dataset_dirs_info = {}
    reg_dataset_dirs_info = {}
    for entry in manifest.entries.values():
        dir_name = os.path.basename(os.path.dirname(entry.image_path))
        info_dict = reg_dataset_dirs_info if entry.is_reg else dataset_dirs_info
        if dir_name not in info_dict:
            info_dict[dir_name] = {"n_repeats": entry.num_repeats, "img_count": 0}
        else:
            # Use max n_repeats if entries in same directory have different values
            info_dict[dir_name]["n_repeats"] = max(info_dict[dir_name]["n_repeats"], entry.num_repeats)
        info_dict[dir_name]["img_count"] += 1

    # Add dataset-level metadata
    metadata.update(
        {
            "ss_batch_size_per_device": cfg.training.train_batch_size,
            "ss_total_batch_size": total_batch_size,
            "ss_resolution": cfg.data.preprocessing.resolution,
            "ss_color_aug": bool(cfg.data.preprocessing.color_aug),
            "ss_flip_aug": bool(cfg.data.preprocessing.flip_aug),
            "ss_random_crop": bool(cfg.data.preprocessing.random_crop),
            "ss_shuffle_caption": bool(cfg.data.caption.shuffle_caption),
            "ss_enable_bucket": bool(cfg.data.bucketing.enable_bucket),
            "ss_bucket_no_upscale": bool(cfg.data.bucketing.bucket_no_upscale),
            "ss_min_bucket_reso": cfg.data.bucketing.min_bucket_reso,
            "ss_max_bucket_reso": cfg.data.bucketing.max_bucket_reso,
            "ss_keep_tokens": cfg.data.caption.keep_tokens,
            "ss_dataset_dirs": json.dumps(dataset_dirs_info),
            "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
            "ss_tag_frequency": json.dumps(tag_frequency),
            "ss_bucket_info": json.dumps(bucket_info),
        }
    )

    # Adapter args (PEFT only)
    if hasattr(cfg, "peft") and cfg.peft is not None and cfg.peft.adapter_args:
        metadata["ss_adapter_args"] = json.dumps(net_kwargs)

    # Model name and hash
    hash_algorithm = cfg.output.saving.hash_algorithm
    if cfg.model.pretrained_model_name_or_path is not None:
        sd_model_name = cfg.model.pretrained_model_name_or_path
        if os.path.exists(sd_model_name):
            metadata["ss_sd_model_hash"] = model_hash(sd_model_name)
            metadata["ss_new_sd_model_hash"] = calculate_hash(sd_model_name, hash_algorithm)
            sd_model_name = os.path.basename(sd_model_name)
        metadata["ss_sd_model_name"] = sd_model_name

    if cfg.model.vae is not None:
        vae_name = cfg.model.vae
        if os.path.exists(vae_name):
            metadata["ss_vae_hash"] = model_hash(vae_name)
            metadata["ss_new_vae_hash"] = calculate_hash(vae_name, hash_algorithm)
            vae_name = os.path.basename(vae_name)
        metadata["ss_vae_name"] = vae_name

    if objective is None:
        objective = build_objective(cfg)
    append_objective_metadata(metadata, cfg, objective.name)

    # Convert all values to strings
    metadata = {k: str(v) for k, v in metadata.items()}

    # Create minimum metadata for filtering
    minimum_metadata = {}
    for key in SS_METADATA_MINIMUM_KEYS:
        if key in metadata:
            minimum_metadata[key] = metadata[key]

    return metadata, minimum_metadata
