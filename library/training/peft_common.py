# PEFT Training Common Utilities
# Shared utility functions for PEFT training that are model-agnostic

import logging
import os
from typing import Optional

import torch
from accelerate import Accelerator

from library.utils.common_utils import setup_logging

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

setup_logging()
logger = logging.getLogger(__name__)


def generate_step_logs(
    cfg,
    current_loss,
    avr_loss,
    lr_scheduler,
    lr_descriptions,
    la_sampler=None,
    optimizer=None,
    keys_scaled=None,
    mean_norm=None,
    maximum_norm=None,
    mean_grad_norm=None,
    mean_combined_norm=None,
    edm2_lr_scheduler=None,
    current_loss_scaled=None,
    average_loss_scaled=None,
    current_val_loss=None,
    average_val_loss=None,
    timesteps: Optional[torch.Tensor] = None,
):
    """Generate step logs for training progress tracking."""
    logs = {"loss/current": current_loss, "loss/average": avr_loss}

    if current_loss_scaled is not None:
        logs["loss/current_scaled"] = current_loss_scaled
        logs["loss/average_scaled"] = average_loss_scaled

    if keys_scaled is not None:
        logs["max_norm/keys_scaled"] = keys_scaled
        logs["max_norm/max_key_norm"] = maximum_norm
    if mean_norm is not None:
        logs["norm/avg_key_norm"] = mean_norm
    if mean_grad_norm is not None:
        logs["norm/avg_grad_norm"] = mean_grad_norm
    if mean_combined_norm is not None:
        logs["norm/avg_combined_norm"] = mean_combined_norm

    if current_val_loss is not None:
        logs["loss/current_val_loss"] = current_val_loss
        logs["loss/average_val_loss"] = average_val_loss

    lrs = lr_scheduler.get_last_lr()
    for i, lr in enumerate(lrs):
        if lr_descriptions is not None:
            lr_desc = lr_descriptions[i]
        else:
            idx = i - (0 if cfg.network.network_train_unet_only else -1)
            if idx == -1:
                lr_desc = "textencoder"
            else:
                if len(lrs) > 2:
                    lr_desc = f"group{idx}"
                else:
                    lr_desc = "unet"

        logs[f"lr/{lr_desc}"] = lr

        if cfg.optimizer.optimizer_type.lower().startswith("DAdapt".lower()) or cfg.optimizer.optimizer_type.lower() == "Prodigy".lower():
            logs[f"lr/d*lr/{lr_desc}"] = (
                lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
            )
        if cfg.optimizer.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None:
            logs["lr/d*lr"] = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
    else:
        idx = 0
        if not cfg.network.network_train_unet_only:
            logs["lr/textencoder"] = float(lrs[0])
            idx = 1

        for i in range(idx, len(lrs)):
            logs[f"lr/group{i}"] = float(lrs[i])
            if cfg.optimizer.optimizer_type.lower().startswith("DAdapt".lower()) or cfg.optimizer.optimizer_type.lower() == "Prodigy".lower():
                logs[f"lr/d*lr/group{i}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )
            if cfg.optimizer.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None:
                logs[f"lr/d*lr/group{i}"] = optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["lr"]

    if edm2_lr_scheduler is not None:
        logs["lr/edm2"] = edm2_lr_scheduler.get_last_lr()[0]

    if cfg.timestep.timestep_sampling == "mix_adaptive" and la_sampler is not None and timesteps is not None:
        if hasattr(la_sampler, "last_mix_p"):
            logs["sampler/mix_p"] = la_sampler.last_mix_p
        if hasattr(la_sampler, "last_small_t_frac"):
            logs["sampler/small_t_frac"] = la_sampler.last_small_t_frac

        if hasattr(la_sampler, "bin_loss_ema"):
            logs["sampler/ema_loss_mean"] = la_sampler.bin_loss_ema.mean().item()
            logs["sampler/ema_loss_std"] = la_sampler.bin_loss_ema.std().item()

            for i, loss_val in enumerate(la_sampler.bin_loss_ema):
                logs[f"sampler_ema_loss_bins/bin_{i}"] = loss_val.item()

        if hasattr(la_sampler, "num_bins") and hasattr(la_sampler, "T"):
            hist = torch.histogram(
                timesteps.float().cpu(),
                bins=la_sampler.num_bins,
                range=(0, la_sampler.T),
            )
            for i, count in enumerate(hist.hist):
                logs[f"sampler_timestep_hist/bin_{i}"] = count.item()

    return logs


def step_logging(accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
    """Log metrics at each step."""
    accelerator_logging(accelerator, logs, global_step, global_step, epoch)


def epoch_logging(accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
    """Log metrics at epoch end."""
    accelerator_logging(accelerator, logs, epoch, global_step, epoch)


def accelerator_logging(accelerator: Accelerator, logs: dict, step_value: int, global_step: int, epoch: int):
    """
    Log metrics to trackers.
    step_value is for tensorboard, other values are for wandb.
    """
    tensorboard_tracker = None
    wandb_tracker = None
    other_trackers = []
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tensorboard_tracker = accelerator.get_tracker("tensorboard")
        elif tracker.name == "wandb":
            wandb_tracker = accelerator.get_tracker("wandb")
        else:
            other_trackers.append(accelerator.get_tracker(tracker.name))

    if tensorboard_tracker is not None:
        tensorboard_tracker.log(logs, step=step_value)

    if wandb_tracker is not None:
        logs["global_step"] = global_step
        logs["epoch"] = epoch
        wandb_tracker.log(logs)

    for tracker in other_trackers:
        tracker.log(logs, step=step_value)


def save_timestep_distribution_plot(cfg, global_step, timestep_counts, settings_dict=None):
    """Save timestep distribution plot to disk."""
    if plt is None:
        logger.warning("Matplotlib is not installed. Cannot save timestep distribution plot.")
        return

    output_dir = os.path.join(cfg.saving.output_dir, "timestep_plots")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(15, 7))
    plt.bar(range(len(timestep_counts)), timestep_counts, width=1.0)
    plt.title(f"Timestep Distribution at Step {global_step}")
    plt.xlabel("Timestep")
    plt.ylabel("Accumulated Count")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    if settings_dict:
        settings_text = "\n".join([f"{key}: {value}" for key, value in settings_dict.items() if value is not None])
        plt.figtext(0.01, 0.01, settings_text, wrap=True, horizontalalignment='left', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.1))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    filename = os.path.join(output_dir, f"step_{global_step:06d}.png")
    plt.savefig(filename)
    plt.close()


def close_live_plotter(live_plotter_process):
    """Clean up live plotter subprocess."""
    import subprocess
    
    if live_plotter_process is not None:
        logger.info("Shutting down live plotter server...")
        try:
            if live_plotter_process.stdin:
                live_plotter_process.stdin.close()
            if live_plotter_process.poll() is None:
                live_plotter_process.terminate()
                live_plotter_process.wait(timeout=5)
            logger.info("Live plotter server shut down.")
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Could not shut down live plotter server cleanly, killing: {e}")
            live_plotter_process.kill()


def init_timestep_sampler(cfg, noise_scheduler, accelerator):
    """
    Initialize the appropriate timestep sampler based on config.
    
    Returns the sampler instance and potentially modifies cfg.timestep.timestep_sampling.
    
    Args:
        cfg: Training configuration
        noise_scheduler: Diffusers noise scheduler
        accelerator: HuggingFace Accelerator
        
    Returns:
        Timestep sampler instance or None for uniform/shift sampling
    """
    from library.timestep_samplers.loss_aware_sampler import LossAwareTimestepSampler
    from library.timestep_samplers.log_snr_sampler import LogSNRUniformSampler
    from library.timestep_samplers.tempered_adaptive_sampler import TemperedAdaptiveSampler
    from library.timestep_samplers.gaussian_mid_snr_sampler import GaussianMidSNRAdaptiveSampler
    from library.timestep_samplers.snr_windowed_loss_aware_sampler import SNRWindowedLossAwareSampler
    
    la_sampler = None
    
    if not cfg.timestep.timestep_sampling:
        return None
    
    sampling_type = cfg.timestep.timestep_sampling
    
    if sampling_type == "log_snr_uniform":
        accelerator.print("Initializing LogSNRUniformSampler.")
        la_sampler = LogSNRUniformSampler(noise_scheduler, noise_scheduler.config.num_train_timesteps)
        cfg.timestep.timestep_sampling = "mix_adaptive"
        
    elif sampling_type == "tempered_adaptive":
        accelerator.print("Initializing TemperedAdaptiveSampler.")
        la_sampler = TemperedAdaptiveSampler(
            noise_scheduler,
            num_bins=cfg.timestep.mix_adaptive_bins,
            ema_beta=cfg.timestep.mix_adaptive_ema_beta,
            temperature=cfg.timestep.mix_adaptive_temperature,
            prior_weight=cfg.timestep.mix_adaptive_prior_weight,
            min_prob=cfg.timestep.mix_adaptive_min_prob,
            warmup_steps=cfg.timestep.mix_adaptive_warmup_steps,
            prior_bias=cfg.timestep.mix_adaptive_prior_bias,
            entropy_floor=cfg.timestep.mix_adaptive_entropy_floor_ratio,
        )
        cfg.timestep.timestep_sampling = "mix_adaptive"
        
    elif sampling_type == "gaussian_mid_snr":
        accelerator.print("Initializing GaussianMidSNRSampler.")
        la_sampler = GaussianMidSNRAdaptiveSampler(
            noise_scheduler,
            num_bins=cfg.timestep.mix_adaptive_bins,
            ema_beta=cfg.timestep.mix_adaptive_ema_beta,
            temperature=cfg.timestep.mix_adaptive_temperature,
            min_prob=cfg.timestep.mix_adaptive_min_prob,
            entropy_floor=cfg.timestep.mix_adaptive_entropy_floor_ratio,
            prior_mu=cfg.timestep.mix_adaptive_prior_mu,
            prior_sigma=cfg.timestep.mix_adaptive_prior_sigma,
            prior_weight=cfg.timestep.mix_adaptive_prior_weight,
            warmup_steps=cfg.timestep.mix_adaptive_warmup_steps,
        )
        cfg.timestep.timestep_sampling = "mix_adaptive"
        
    elif sampling_type == "snr_windowed":
        accelerator.print("Initializing SNRWindowedSampler.")
        la_sampler = SNRWindowedLossAwareSampler(
            noise_scheduler,
            num_bins=cfg.timestep.mix_adaptive_bins,
            ema_beta=cfg.timestep.mix_adaptive_ema_beta,
            temperature=cfg.timestep.mix_adaptive_temperature,
            min_prob=cfg.timestep.mix_adaptive_min_prob,
            entropy_floor=cfg.timestep.mix_adaptive_entropy_floor_ratio,
            center_mu=cfg.timestep.mix_adaptive_center_mu,
            half_width=cfg.timestep.mix_adaptive_half_width,
            widen_to=cfg.timestep.mix_adaptive_widen_to,
            total_widen_steps=cfg.timestep.mix_adaptive_max_train_steps,
            cap_max_t=cfg.timestep.mix_adaptive_cap_max_t,
        )
        cfg.timestep.timestep_sampling = "mix_adaptive"
        
    elif sampling_type == "mix_adaptive":
        accelerator.print("Initializing LossAwareTimestepSampler.")
        la_sampler = LossAwareTimestepSampler(
            num_train_timesteps=noise_scheduler.config.num_train_timesteps,
            num_bins=cfg.timestep.mix_adaptive_bins,
            ema_beta=cfg.timestep.mix_adaptive_ema_beta,
            small_t_frac=cfg.timestep.mix_adaptive_small_t_frac,
            small_t_cap=cfg.timestep.mix_adaptive_small_t_cap,
            start_p=cfg.timestep.mix_adaptive_start_p,
            end_p=cfg.timestep.mix_adaptive_end_p,
            anneal=cfg.timestep.mix_adaptive_anneal,
            fixed_p=cfg.timestep.mix_adaptive_fixed_p,
        )
        
    elif sampling_type in ("sigma", "uniform"):
        la_sampler = None
        cfg.timestep.timestep_sampling = "uniform"
        if sampling_type == "sigma":
            logger.warning("sigma sampling is not supported yet, using uniform sampling")
            
    elif sampling_type == "shift":
        la_sampler = None
        # shift sampling is handled in get_noise_noisy_latents_and_timesteps
    
    return la_sampler


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
    text_encoder_lr,
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
    import json
    import os
    from library.training.checkpointing import get_git_revision_hash, model_hash, calculate_sha256
    from library.constants import SS_METADATA_MINIMUM_KEYS
    from library.data.dataset import DreamBoothDataset
    
    metadata = {
        "ss_session_id": session_id,
        "ss_training_started_at": training_started_at,
        "ss_output_name": cfg.saving.output_name,
        "ss_learning_rate": cfg.optimizer.learning_rate,
        "ss_text_encoder_lr": text_encoder_lr,
        "ss_unet_lr": cfg.network.unet_lr,
        "ss_num_train_images": train_dataset_group.num_train_images,
        "ss_num_validation_images": val_dataset_group.num_train_images if val_dataset_group is not None else 0,
        "ss_num_reg_images": train_dataset_group.num_reg_images,
        "ss_num_batches_per_epoch": len(train_dataloader),
        "ss_num_epochs": num_train_epochs,
        "ss_gradient_checkpointing": cfg.performance.gradient_checkpointing,
        "ss_gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "ss_max_train_steps": cfg.training.max_train_steps,
        "ss_lr_warmup_steps": cfg.optimizer.lr_warmup_steps,
        "ss_lr_scheduler": cfg.optimizer.lr_scheduler,
        "ss_network_module": cfg.network.network_module,
        "ss_network_dim": cfg.network.network_dim,
        "ss_network_alpha": cfg.network.network_alpha,
        "ss_network_dropout": cfg.network.network_dropout,
        "ss_mixed_precision": cfg.performance.mixed_precision,
        "ss_full_fp16": bool(cfg.performance.full_fp16),
        "ss_v2": bool(cfg.model.v2),
        "ss_base_model_version": model_version,
        "ss_clip_skip": cfg.training.clip_skip,
        "ss_max_token_length": cfg.training.max_token_length,
        "ss_cache_latents": bool(cfg.dataset.cache_latents),
        "ss_seed": cfg.training.seed,
        "ss_lowram": cfg.performance.lowram,
        "ss_noise_offset": cfg.regularization.noise_offset,
        "ss_multires_noise_iterations": cfg.regularization.multires_noise_iterations,
        "ss_multires_noise_discount": cfg.regularization.multires_noise_discount,
        "ss_adaptive_noise_scale": cfg.regularization.adaptive_noise_scale,
        "ss_zero_terminal_snr": cfg.regularization.zero_terminal_snr,
        "ss_training_comment": cfg.network.training_comment,
        "ss_sd_scripts_commit_hash": get_git_revision_hash(),
        "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
        "ss_max_grad_norm": cfg.optimizer.max_grad_norm,
        "ss_caption_dropout_rate": cfg.dataset.caption_dropout_rate,
        "ss_caption_dropout_every_n_epochs": cfg.dataset.caption_dropout_every_n_epochs,
        "ss_caption_tag_dropout_rate": cfg.dataset.caption_tag_dropout_rate,
        "ss_face_crop_aug_range": cfg.dataset.face_crop_aug_range,
        "ss_prior_loss_weight": cfg.loss.prior_loss_weight,
        "ss_min_snr_gamma": cfg.loss.min_snr_gamma,
        "ss_scale_weight_norms": cfg.network.scale_weight_norms,
        "ss_ip_noise_gamma": cfg.regularization.ip_noise_gamma,
        "ss_debiased_estimation": bool(cfg.loss.debiased_estimation_loss),
        "ss_noise_offset_random_strength": cfg.regularization.noise_offset_random_strength,
        "ss_ip_noise_gamma_random_strength": cfg.regularization.ip_noise_gamma_random_strength,
        "ss_loss_type": cfg.loss.loss_type,
        "ss_huber_schedule": cfg.loss.huber_schedule,
        "ss_huber_scale": cfg.loss.huber_scale,
        "ss_huber_c": cfg.loss.huber_c,
        "ss_fp8_base": bool(cfg.performance.fp8_base),
        "ss_fp8_base_unet": bool(cfg.performance.fp8_base_unet),
        "ss_validation_seed": cfg.dataset.validation_seed,
        "ss_validation_split": float(cfg.dataset.validation_split),
        "ss_max_validation_steps": cfg.training.max_validation_steps,
        "ss_validate_every_n_epochs": cfg.training.validate_every_n_epochs,
        "ss_validate_every_n_steps": cfg.training.validate_every_n_steps,
        "ss_resize_interpolation": cfg.dataset.resize_interpolation,
    }
    
    # Dataset-specific metadata
    if use_user_config:
        datasets_metadata = []
        tag_frequency = {}
        dataset_dirs_info = {}

        for dataset in train_dataset_group.datasets:
            is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
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
            "ss_resolution": cfg.dataset.resolution,
            "ss_color_aug": bool(cfg.dataset.color_aug),
            "ss_flip_aug": bool(cfg.dataset.flip_aug),
            "ss_random_crop": bool(cfg.dataset.random_crop),
            "ss_random_crop_padding_percent": float(getattr(cfg.dataset, "random_crop_padding_percent", 0.05)),
            "ss_shuffle_caption": bool(cfg.dataset.shuffle_caption),
            "ss_enable_bucket": bool(dataset.enable_bucket),
            "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
            "ss_min_bucket_reso": dataset.min_bucket_reso,
            "ss_max_bucket_reso": dataset.max_bucket_reso,
            "ss_keep_tokens": cfg.dataset.keep_tokens,
            "ss_dataset_dirs": json.dumps(dataset_dirs_info),
            "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
            "ss_tag_frequency": json.dumps(dataset.tag_frequency),
            "ss_bucket_info": json.dumps(dataset.bucket_info),
        })

    # Network args
    if cfg.network.network_args:
        metadata["ss_network_args"] = json.dumps(net_kwargs)

    # Model name and hash
    if cfg.model.pretrained_model_name_or_path is not None:
        sd_model_name = cfg.model.pretrained_model_name_or_path
        if os.path.exists(sd_model_name):
            metadata["ss_sd_model_hash"] = model_hash(sd_model_name)
            metadata["ss_new_sd_model_hash"] = calculate_sha256(sd_model_name)
            sd_model_name = os.path.basename(sd_model_name)
        metadata["ss_sd_model_name"] = sd_model_name

    if cfg.model.vae is not None:
        vae_name = cfg.model.vae
        if os.path.exists(vae_name):
            metadata["ss_vae_hash"] = model_hash(vae_name)
            metadata["ss_new_vae_hash"] = calculate_sha256(vae_name)
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


def get_plotter_settings(cfg, la_sampler) -> dict:
    """
    Gather plotter settings based on the sampler type and config.
    
    Returns a dict of settings for display in the live plotter.
    """
    # Import sampler types for isinstance checks
    from library.timestep_samplers.log_snr_sampler import LogSNRUniformSampler
    from library.timestep_samplers.tempered_adaptive_sampler import TemperedAdaptiveSampler
    
    # Determine the actual sampler being used
    sampler_type = cfg.timestep.timestep_sampling
    if la_sampler is not None:
        if isinstance(la_sampler, LogSNRUniformSampler):
            sampler_type = "log_snr_uniform"
        elif isinstance(la_sampler, TemperedAdaptiveSampler):
            sampler_type = "tempered_adaptive"

    plotter_settings = {
        "Timestep Sampler": sampler_type,
        "Dynamic Schedule": "Enabled" if cfg.timestep.dynamic_timestep_schedule else "Disabled",
        "Min Timestep": cfg.timestep.min_timestep,
        "Max Timestep": cfg.timestep.max_timestep,
    }

    # Add sampler-specific settings
    if sampler_type == "mix_adaptive":
        plotter_settings.update({
            "Anneal": cfg.timestep.mix_adaptive_anneal,
            "Start/End P": f"{cfg.timestep.mix_adaptive_start_p} -> {cfg.timestep.mix_adaptive_end_p}",
            "Fixed P": cfg.timestep.mix_adaptive_fixed_p,
            "Num Bins": cfg.timestep.mix_adaptive_bins,
            "EMA Beta": cfg.timestep.mix_adaptive_ema_beta,
            "Small T Frac/Cap": f"{cfg.timestep.mix_adaptive_small_t_frac} / {cfg.timestep.mix_adaptive_small_t_cap}",
        })
    elif sampler_type == "tempered_adaptive":
        plotter_settings.update({
            "Num Bins": cfg.timestep.mix_adaptive_bins,
            "EMA Beta": cfg.timestep.mix_adaptive_ema_beta,
            "Temperature": cfg.timestep.mix_adaptive_temperature,
            "Prior Weight": cfg.timestep.mix_adaptive_prior_weight,
            "Min Prob": cfg.timestep.mix_adaptive_min_prob,
            "Warmup Steps": cfg.timestep.mix_adaptive_warmup_steps,
            "Prior Bias": cfg.timestep.mix_adaptive_prior_bias,
            "Entropy Floor": cfg.timestep.mix_adaptive_entropy_floor_ratio,
        })
    elif sampler_type == "gaussian_mid_snr":
        plotter_settings.update({
            "Num Bins": cfg.timestep.mix_adaptive_bins,
            "EMA Beta": cfg.timestep.mix_adaptive_ema_beta,
            "Temperature": cfg.timestep.mix_adaptive_temperature,
            "Min Prob": cfg.timestep.mix_adaptive_min_prob,
            "Entropy Floor": cfg.timestep.mix_adaptive_entropy_floor_ratio,
            "Uniform Mix When Low Entropy": cfg.timestep.mix_adaptive_uniform_mix_when_low_entropy,
            "Prior_Mu": cfg.timestep.mix_adaptive_prior_mu,
            "Prior Sigma": cfg.timestep.mix_adaptive_prior_sigma,
            "Prior Weight": cfg.timestep.mix_adaptive_prior_weight,
            "Warmup Steps": cfg.timestep.mix_adaptive_warmup_steps,
        })
    elif sampler_type == "snr_windowed":
        plotter_settings.update({
            "Num Bins": cfg.timestep.mix_adaptive_bins,
            "EMA Beta": cfg.timestep.mix_adaptive_ema_beta,
            "Temperature": cfg.timestep.mix_adaptive_temperature,
            "Min Prob": cfg.timestep.mix_adaptive_min_prob,
            "Entropy Floor": cfg.timestep.mix_adaptive_entropy_floor_ratio,
            "Uniform Mix": cfg.timestep.mix_adaptive_uniform_mix_when_low_entropy,
            "Center Mu": cfg.timestep.mix_adaptive_center_mu,
            "Half Width": cfg.timestep.mix_adaptive_half_width,
            "Widen To": cfg.timestep.mix_adaptive_widen_to,
            "Total Widen Steps": cfg.timestep.mix_adaptive_max_train_steps,
            "Cap Max T": cfg.timestep.mix_adaptive_cap_max_t,
        })
    elif sampler_type not in ["uniform", "log_snr_uniform"]:
        plotter_settings.update({
            "Shift": cfg.timestep.discrete_flow_shift,
            "Sigmoid Scale": cfg.timestep.sigmoid_scale,
        })

    return plotter_settings


def setup_live_plotter(cfg, noise_scheduler, la_sampler, strategy):
    """
    Setup live plotter subprocess and static plot timestep tracking.
    
    Args:
        cfg: Training configuration
        noise_scheduler: Diffusers noise scheduler
        la_sampler: Loss-aware timestep sampler (or None)
        strategy: Training strategy (used to store plotter process)
        
    Returns:
        tuple: (timestep_counts array or None, plotter_settings dict or None)
    """
    import subprocess
    import sys
    import json
    import numpy as np
    
    timestep_counts = None
    plotter_settings = None
    
    # Get plotter settings
    plotter_settings = get_plotter_settings(cfg, la_sampler)
    
    # Setup for the live interactive plotter
    if cfg.logging.live_plot_port is not None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plotter_script_path = os.path.join(project_root, "tools", "visualization", "live_plotter.py")

        if not os.path.exists(plotter_script_path):
            logger.error(f"live_plotter.py not found at {plotter_script_path}. Live plotter disabled.")
        else:
            if strategy.live_plotter_process is None or strategy.live_plotter_process.poll() is not None:
                logger.info(f"Launching live plotter server on port {cfg.logging.live_plot_port}")
                strategy.live_plotter_process = subprocess.Popen(
                    [sys.executable, plotter_script_path, "--port", str(cfg.logging.live_plot_port)],
                    stdin=subprocess.PIPE,
                )

            # Send the initial "handshake" data
            reset_str = "RESET::\n"
            alphas_cumprod_np = noise_scheduler.alphas_cumprod.cpu().numpy()
            schedule_str = f"SCHEDULE::{','.join(map(str, alphas_cumprod_np))}\n"
            settings_str = f"SETTINGS::{json.dumps(plotter_settings)}\n"

            try:
                strategy.live_plotter_process.stdin.write(reset_str.encode('utf-8'))
                strategy.live_plotter_process.stdin.write(schedule_str.encode('utf-8'))
                strategy.live_plotter_process.stdin.write(settings_str.encode('utf-8'))
                strategy.live_plotter_process.stdin.flush()
            except (BrokenPipeError, OSError):
                logger.error("Failed to send data to live plotter. It may have crashed.")
                strategy.live_plotter_process = None

    # Setup for saving static plot images
    if cfg.logging.log_timestep_distribution_every_n_steps is not None:
        timestep_counts = np.zeros(noise_scheduler.config.num_train_timesteps, dtype=np.int64)

    return timestep_counts, plotter_settings
