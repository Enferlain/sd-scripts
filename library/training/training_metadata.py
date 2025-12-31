import json
import os

from library.constants import SS_METADATA_MINIMUM_KEYS
from library.utils.hash_utils import get_git_revision_hash, model_hash, calculate_hash


def create_training_metadata(
    cfg,
    session_id: int,
    training_started_at: float,
    model_version: str,
    train_dataset_group,
    val_dataset_group,
    num_train_epochs: int,
    optimizer_name: str,
    optimizer_args: str,
    net_kwargs: dict,
    train_dataloader,
    total_batch_size: int,
    use_user_config: bool,
    use_dreambooth_method: bool,
) -> tuple:
    """
    Create training metadata dict for model saving.

    Returns:
        tuple: (metadata dict, minimum_metadata dict)
    """
    metadata = {
        "ss_session_id": session_id,
        "ss_training_started_at": training_started_at,
        "ss_output_name": cfg.output.saving.output_name,
        "ss_learning_rate": cfg.optimizer.learning_rates.base,
        "ss_text_encoder_lr": cfg.optimizer.learning_rates.text_encoders,
        "ss_unet_lr": cfg.optimizer.learning_rates.unet,
        "ss_num_train_images": train_dataset_group.num_train_images,
        "ss_num_validation_images": val_dataset_group.num_train_images if val_dataset_group is not None else 0,
        "ss_num_reg_images": train_dataset_group.num_reg_images,
        "ss_num_batches_per_epoch": len(train_dataloader),
        "ss_num_epochs": num_train_epochs,
        "ss_gradient_checkpointing": cfg.performance.memory.gradient_checkpointing,
        "ss_gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "ss_max_train_steps": cfg.training.max_train_steps,
        "ss_lr_warmup_steps": cfg.optimizer.scheduler.lr_warmup_steps,
        "ss_lr_scheduler": cfg.optimizer.scheduler.lr_scheduler,
        "ss_adapter_module": cfg.peft.module,  # adapter REFACTOR
        "ss_adapter_rank": cfg.peft.adapter_rank,
        "ss_adapter_alpha": cfg.peft.adapter_alpha,
        "ss_adapter_neuron_dropout": cfg.peft.neuron_dropout,
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
        "ss_training_comment": cfg.peft.training_comment,
        "ss_sd_scripts_commit_hash": get_git_revision_hash(),
        "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
        "ss_max_grad_norm": cfg.optimizer.max_grad_norm,
        "ss_caption_dropout_rate": cfg.data.caption.caption_dropout_rate,
        "ss_caption_dropout_every_n_epochs": cfg.data.caption.caption_dropout_every_n_epochs,
        "ss_caption_tag_dropout_rate": cfg.data.caption.caption_tag_dropout_rate,
        "ss_face_crop_aug_range": cfg.data.preprocessing.face_crop_aug_range,
        "ss_prior_loss_weight": cfg.loss.prior_loss_weight,
        "ss_min_snr_gamma": cfg.loss.snr.min_snr_gamma,
        "ss_scale_weight_norms": cfg.peft.scale_weight_norms,
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
        "ss_resize_interpolation": cfg.data.preprocessing.resize_interpolation,
    }

    # Dataset-specific metadata
    if use_user_config:
        datasets_metadata = []
        tag_frequency = {}
        dataset_dirs_info = {}

        for dataset in train_dataset_group.datasets:
            # TODO: DATA REFACTOR - Use proper isinstance check once data module circular import is fixed
            # Currently using duck-typing: DreamBooth subsets have 'is_reg' attribute, FineTuning subsets don't
            is_dreambooth_dataset = len(dataset.subsets) > 0 and hasattr(dataset.subsets[0], 'is_reg')
            dataset_metadata = {
                "is_dreambooth": is_dreambooth_dataset,
                "batch_size_per_device": dataset.batch_size,
                "num_train_images": dataset.num_train_images,
                "num_reg_images": dataset.num_reg_images,
                "resolution": (dataset.width, dataset.height),
                "enable_bucket": bool(dataset.enable_bucket),
                "min_bucket_reso": dataset.min_bucket_reso,
                "max_bucket_reso": dataset.max_bucket_reso,
                "tag_frequency": dataset.tag_frequency,
                "bucket_info": dataset.bucket_info,
                "resize_interpolation": dataset.resize_interpolation,
            }

            subsets_metadata = []
            for subset in dataset.subsets:
                subset_metadata = {
                    "img_count": subset.img_count,
                    "num_repeats": subset.num_repeats,
                    "color_aug": bool(subset.color_aug),
                    "flip_aug": bool(subset.flip_aug),
                    "random_crop": bool(subset.random_crop),
                    "random_crop_padding_percent": float(getattr(subset, "random_crop_padding_percent", 0.05)),
                    "shuffle_caption": bool(subset.shuffle_caption),
                    "keep_tokens": subset.keep_tokens,
                    "keep_tokens_separator": subset.keep_tokens_separator,
                    "secondary_separator": subset.secondary_separator,
                    "enable_wildcard": bool(subset.enable_wildcard),
                    "caption_prefix": subset.caption_prefix,
                    "caption_suffix": subset.caption_suffix,
                    "resize_interpolation": subset.resize_interpolation,
                }

                image_dir_or_metadata_file = None
                if subset.image_dir:
                    image_dir = os.path.basename(subset.image_dir)
                    subset_metadata["image_dir"] = image_dir
                    image_dir_or_metadata_file = image_dir

                if is_dreambooth_dataset:
                    subset_metadata["class_tokens"] = subset.class_tokens
                    subset_metadata["is_reg"] = subset.is_reg
                    if subset.is_reg:
                        image_dir_or_metadata_file = None
                else:
                    metadata_file = os.path.basename(subset.metadata_file)
                    subset_metadata["metadata_file"] = metadata_file
                    image_dir_or_metadata_file = metadata_file

                subsets_metadata.append(subset_metadata)

                if image_dir_or_metadata_file is not None:
                    v = image_dir_or_metadata_file
                    i = 2
                    while v in dataset_dirs_info:
                        v = image_dir_or_metadata_file + f" ({i})"
                        i += 1
                    image_dir_or_metadata_file = v

                    dataset_dirs_info[image_dir_or_metadata_file] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }

            dataset_metadata["subsets"] = subsets_metadata
            datasets_metadata.append(dataset_metadata)

            for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                if ds_dir_name in tag_frequency:
                    continue
                tag_frequency[ds_dir_name] = ds_freq_for_dir

        metadata["ss_datasets"] = json.dumps(datasets_metadata)
        metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
        metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
    else:
        assert (
            len(train_dataset_group.datasets) == 1
        ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found."

        dataset = train_dataset_group.datasets[0]

        dataset_dirs_info = {}
        reg_dataset_dirs_info = {}
        if use_dreambooth_method:
            for subset in dataset.subsets:
                info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                info[os.path.basename(subset.image_dir)] = {
                    "n_repeats": subset.num_repeats,
                    "img_count": subset.img_count
                }
        else:
            for subset in dataset.subsets:
                dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                    "n_repeats": subset.num_repeats,
                    "img_count": subset.img_count,
                }

        metadata.update({
            "ss_batch_size_per_device": cfg.training.train_batch_size,
            "ss_total_batch_size": total_batch_size,
            "ss_resolution": cfg.data.preprocessing.resolution,
            "ss_color_aug": bool(cfg.data.preprocessing.color_aug),
            "ss_flip_aug": bool(cfg.data.preprocessing.flip_aug),
            "ss_random_crop": bool(cfg.data.preprocessing.random_crop),
            "ss_random_crop_padding_percent": float(getattr(cfg.dataset, "random_crop_padding_percent", 0.05)),
            "ss_shuffle_caption": bool(cfg.data.caption.shuffle_caption),
            "ss_enable_bucket": bool(dataset.enable_bucket),
            "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
            "ss_min_bucket_reso": dataset.min_bucket_reso,
            "ss_max_bucket_reso": dataset.max_bucket_reso,
            "ss_keep_tokens": cfg.data.caption.keep_tokens,
            "ss_dataset_dirs": json.dumps(dataset_dirs_info),
            "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
            "ss_tag_frequency": json.dumps(dataset.tag_frequency),
            "ss_bucket_info": json.dumps(dataset.bucket_info),
        })

    # Adapter args
    if cfg.peft.args:
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

    # Convert all values to strings
    metadata = {k: str(v) for k, v in metadata.items()}

    # Create minimum metadata for filtering
    minimum_metadata = {}
    for key in SS_METADATA_MINIMUM_KEYS:
        if key in metadata:
            minimum_metadata[key] = metadata[key]

    return metadata, minimum_metadata
