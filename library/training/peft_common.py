# PEFT Training Common Utilities
# Shared utility functions for PEFT training that are model-agnostic

import json
import logging
import os
import torch
import math
import ast
import subprocess
import sys
import numpy as np

from multiprocessing import Value
from typing import Optional
from accelerate import Accelerator

import library.config.config_util as config_util

from library.config.config_util import BlueprintGenerator
from library.data.dataset import load_arbitrary_dataset, collator_class, debug_dataset
from library.utils.common_utils import setup_logging
from library.config.dataclasses.peft import PeftConfig
from library.training.optimizer import should_train_text_encoder
from library.training.checkpointing import get_git_revision_hash, model_hash, calculate_sha256
from library.constants import SS_METADATA_MINIMUM_KEYS
from library.data.dataset import DreamBoothDataset


setup_logging()
logger = logging.getLogger(__name__)


def prepare_datasets(cfg, strategies):
    """
    Prepare train and validation dataset groups.
    
    Handles:
    - User config generation from train_data_dir/reg_data_dir
    - Blueprint generation
    - Dataset group creation
    - Collator setup
    - Latent cacheability validation
    
    Args:
        cfg: Training configuration (SDPeftConfig or SDXLPeftConfig)
        strategies: PEFT strategy instance for validation
        
    Returns:
        Tuple of (train_dataset_group, val_dataset_group, collator, current_epoch, current_step)
        Returns None for the tuple if debug_dataset mode is active or no data found.
    """
    cache_latents = cfg.data.caching.cache_latents
    
    # Prepare datasets
    if cfg.data.source.dataset_class is None:
        # Check if we have manually provided subsets via train_data_dir/reg_data_dir
        if (cfg.data.source.train_data_dir is not None or cfg.data.source.reg_data_dir is not None) and len(
                cfg.data.source.subsets) == 0:
            # Generate subsets config from dirs
            user_config = config_util.generate_user_config_from_dataset(cfg)
            # We need to inject this into cfg.data.source.subsets
            # cfg.data.source.subsets is a List[dict] (or ListConfig)
            # user_config['datasets'][0]['subsets'] is the list we want
            if user_config['datasets']:
                cfg.data.source.subsets = user_config['datasets'][0]['subsets']

        blueprint_generator = BlueprintGenerator()
        blueprint = blueprint_generator.generate(cfg)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(
            blueprint.dataset_group)
    else:
        # use arbitrary dataset class
        train_dataset_group = load_arbitrary_dataset(cfg)
        val_dataset_group = None  # placeholder until validation dataset supported for arbitrary

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if cfg.training.max_data_loader_n_workers == 0 else None
    collator = collator_class(current_epoch, current_step, ds_for_collator)

    if cfg.data.preprocessing.debug_dataset:
        train_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
        debug_dataset(train_dataset_group)

        if val_dataset_group is not None:
            val_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
            debug_dataset(val_dataset_group)
        return None  # Signal to caller to exit early
        
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
        )
        return None  # Signal to caller to exit early

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
        if val_dataset_group is not None:
            assert (
                val_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    strategies.validate_extra_config(cfg, train_dataset_group, val_dataset_group)

    return train_dataset_group, val_dataset_group, collator, current_epoch, current_step


def calculate_initial_step(cfg, train_dataloader, accelerator, steps_from_state):
    """
    Calculate initial step and epoch for training start/resume.
    
    Handles:
    - initial_epoch/initial_step from config
    - steps_from_state from resume
    - skip_until_initial_step logic
    
    Args:
        cfg: Training configuration
        train_dataloader: Training data loader
        accelerator: HuggingFace Accelerator
        steps_from_state: Steps loaded from saved state (or None)
        
    Returns:
        Tuple of (initial_step, epoch_to_start)
    """
    initial_step = 0
    if cfg.training.initial_epoch is not None or cfg.training.initial_step is not None:
        # if initial_epoch or initial_step is specified, steps_from_state is ignored even when resuming
        if steps_from_state is not None:
            logger.warning(
                "steps from the state is ignored because initial_step is specified / initial_stepが指定されているため、stateからのステップ数は無視されます"
            )
        if cfg.training.initial_step is not None:
            initial_step = cfg.training.initial_step
        else:
            # num steps per epoch is calculated by num_processes and gradient_accumulation_steps
            initial_step = (cfg.training.initial_epoch - 1) * math.ceil(
                len(train_dataloader) / accelerator.num_processes / cfg.training.gradient_accumulation_steps
            )
    else:
        # if initial_epoch and initial_step are not specified, steps_from_state is used when resuming
        if steps_from_state is not None:
            initial_step = steps_from_state

    if initial_step > 0:
        assert (
                cfg.training.max_train_steps > initial_step
        ), f"max_train_steps should be greater than initial step / max_train_stepsは初期ステップより大きい必要があります: {cfg.training.max_train_steps} vs {initial_step}"

    epoch_to_start = 0
    if initial_step > 0:
        if cfg.training.skip_until_initial_step:
            # if skip_until_initial_step is specified, load data and discard it to ensure the same data is used
            if not cfg.output.saving.resume:
                logger.info(
                    f"initial_step is specified but not resuming. lr scheduler will be started from the beginning / initial_stepが指定されていますがresumeしていないため、lr schedulerは最初から始まります"
                )
            logger.info(f"skipping {initial_step} steps / {initial_step}ステップをスキップします")
            initial_step *= cfg.training.gradient_accumulation_steps

            # set epoch to start to make initial_step less than len(train_dataloader)
            epoch_to_start = initial_step // math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
        else:
            # if not, only epoch no is skipped for informative purpose
            epoch_to_start = initial_step // math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
            initial_step = 0  # do not skip


    return initial_step, epoch_to_start


def register_adapter_state_hooks(accelerator, adapter, cfg, current_epoch, current_step):
    """
    Register save/load hooks for peft-only checkpointing.
    
    These hooks ensure that only the PEFT peft weights (LoRA/LyCORIS) are saved/loaded
    during checkpointing, not the full base model weights.
    
    Args:
        accelerator: HuggingFace Accelerator
        adapter: The PEFT peft to save/load
        cfg: Training configuration (needs cfg.performance.deepspeed)
        current_epoch: Shared Value for current epoch tracking
        current_step: Shared Value for current step tracking
        
    Returns:
        Callable that returns steps_from_state (or None if not resumed)
    """
    # Container for steps loaded from state (nonlocal workaround)
    state_container = {"steps_from_state": None}
    
    def save_model_hook(models, weights, output_dir):
        # pop weights of other models than peft to save only peft weights
        # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
        if accelerator.is_main_process or cfg.performance.deepspeed:
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(adapter))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                if len(weights) > i:
                    weights.pop(i)

        # save current epoch and step
        train_state_file = os.path.join(output_dir, "train_state.json")
        # +1 is needed because the state is saved before current_step is set from global_step
        logger.info(
            f"save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value + 1}")
        with open(train_state_file, "w", encoding="utf-8") as f:
            json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)

    def load_model_hook(models, input_dir):
        # remove models except peft
        remove_indices = []
        for i, model in enumerate(models):
            if not isinstance(model, type(accelerator.unwrap_model(adapter))):
                remove_indices.append(i)
        for i in reversed(remove_indices):
            models.pop(i)

        # load current epoch and step
        train_state_file = os.path.join(input_dir, "train_state.json")
        if os.path.exists(train_state_file):
            with open(train_state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            state_container["steps_from_state"] = data["current_step"]
            logger.info(f"load train state from {train_state_file}: {data}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    def get_steps_from_state():
        return state_container["steps_from_state"]
    
    return get_steps_from_state


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
    
    # Check if TE is being trained (LR-based)
    train_te = should_train_text_encoder(cfg.optimizer)
    
    for i, lr in enumerate(lrs):
        if lr_descriptions is not None:
            lr_desc = lr_descriptions[i]
        else:
            idx = i - (0 if not train_te else -1)
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
        if train_te:
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



def resolve_adapter_kwargs(cfg: PeftConfig, net_kwargs: dict):
    """
    Populate net_kwargs with explicit LoRA fields from PeftConfig if they are set.
    """
    # Mapping explicit config fields to peft kwargs
    fields = [
        "conv_dim", "conv_alpha", "rank_dropout", "module_dropout",
        "block_dims", "block_alphas", "conv_block_dims", "conv_block_alphas",
        "down_lr_weight", "mid_lr_weight", "up_lr_weight", "block_lr_zero_threshold",
        "loraplus_lr_ratio", "loraplus_unet_lr_ratio", "loraplus_text_encoder_lr_ratio"
    ]
    
    for field_name in fields:
        value = getattr(cfg, field_name, None)
        if value is not None:
             net_kwargs[field_name] = value
