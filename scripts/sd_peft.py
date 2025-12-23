import gc
import importlib
import math
import os
import typing
import subprocess
import sys
import random
import time
import json
import numpy as np
import ast
import itertools
import atexit
import torch
import torch.nn as nn
import logging
import hydra

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from typing import Any, List, Union, Optional
from multiprocessing import Value
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from ramtorch.helpers import replace_linear_with_ramtorch

import library.config.config_util as config_util
import library.utils.huggingface_util as huggingface_util

from library.config.validation import prepare_config, validate_config, validate_sd_peft
from library.constants import SS_METADATA_MINIMUM_KEYS
from library.strategies import strategy_sd, strategy_base
from library.optimizations import deepspeed_utils
from library.models import model_util
from library.utils import sai_model_spec
from library.utils.common_utils import setup_logging
from library.utils.device_utils import init_ipex, clean_memory_on_device
from library.utils.torch_utils import set_torch_cuda_reduced_precision, set_seed_from_config, prepare_dtype
from library.training.diffusion import get_noise_noisy_latents_and_timesteps
from library.training.model_prep import load_target_model, replace_unet_modules, patch_accelerator_for_fp16_training
from library.training.optimizer import prepare_optimizer, get_scheduler_fix
from library.training.sample_generation import sample_images, sample_images_check
from library.losses.loss import get_huber_threshold_if_needed, conditional_loss, EMARecorder
from library.config.dataclasses.sd_peft import SDPeftConfig

from library.timestep_samplers.loss_aware_sampler import LossAwareTimestepSampler
from library.timestep_samplers.log_snr_sampler import LogSNRUniformSampler
from library.timestep_samplers.tempered_adaptive_sampler import TemperedAdaptiveSampler
from library.timestep_samplers.gaussian_mid_snr_sampler import GaussianMidSNRAdaptiveSampler
from library.timestep_samplers.snr_windowed_loss_aware_sampler import SNRWindowedLossAwareSampler

from library.config.config_util import (
    BlueprintGenerator,
)

from library.training.checkpointing import (
    get_sai_model_spec,
    resume_from_local_or_hf_if_specified,
    get_git_revision_hash,
    model_hash, calculate_sha256,
    get_step_ckpt_name,
    save_and_remove_state_stepwise, get_remove_step_no,
    get_epoch_ckpt_name,
    get_remove_epoch_no,
    save_and_remove_state_on_epoch_end,
    get_last_ckpt_name,
    save_state_on_train_end
)

from library.data.dataset import (
    DatasetGroup,
    MinimalDataset,
    load_arbitrary_dataset,
    collator_class,
    debug_dataset,
    DreamBoothDataset
)

from library.training.trainer_utils import (
    calculate_val_loss_check,
    prepare_accelerator,
    init_trackers,
    determine_grad_sync_context
)

from library.training.noise_utils import (
    prepare_scheduler_for_custom_training,
    fix_noise_scheduler_betas_for_zero_terminal_snr
)

from library.losses.edm2_loss_utils import (
    prepare_edm2_loss_weighting,
    plot_edm2_loss_weighting_check,
    plot_edm2_loss_weighting
)

from library.losses.loss_weighting import (
    apply_masked_loss, apply_snr_weight,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

init_ipex()

setup_logging()
logger = logging.getLogger(__name__)


class SDPeftTrainer:
    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False
        self.live_plotter_process = None
        atexit.register(self.close)

    def __del__(self):
        self.close()

    def close(self):
        if self.live_plotter_process is not None:
            logger.info("Shutting down live plotter server...")
            try:
                if self.live_plotter_process.stdin:
                    self.live_plotter_process.stdin.close()
                if self.live_plotter_process.poll() is None:
                    self.live_plotter_process.terminate()
                    self.live_plotter_process.wait(timeout=5)
                logger.info("Live plotter server shut down.")
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as e:
                logger.warning(f"Could not shut down live plotter server cleanly, killing: {e}")
                self.live_plotter_process.kill()
            finally:
                self.live_plotter_process = None

    def generate_step_logs(
        self,
        cfg,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
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
                # tracking d*lr value
                logs[f"lr/d*lr/{lr_desc}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )
            if (
                cfg.optimizer.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None
            ):  # tracking d*lr value of unet.
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
            logs[f"lr/edm2"] = edm2_lr_scheduler.get_last_lr()[0]

        if cfg.timestep.timestep_sampling == "mix_adaptive" and self.la_sampler is not None and timesteps is not None:
            if hasattr(self.la_sampler, "last_mix_p"):
                logs["sampler/mix_p"] = self.la_sampler.last_mix_p
            if hasattr(self.la_sampler, "last_small_t_frac"):
                logs["sampler/small_t_frac"] = self.la_sampler.last_small_t_frac

            # Add mean and std of ema_loss
            if hasattr(self.la_sampler, "bin_loss_ema"):
                logs["sampler/ema_loss_mean"] = self.la_sampler.bin_loss_ema.mean().item()
                logs["sampler/ema_loss_std"] = self.la_sampler.bin_loss_ema.std().item()

                # EMA loss per bin (in a separate category for clarity in TensorBoard)
                for i, loss_val in enumerate(self.la_sampler.bin_loss_ema):
                    logs[f"sampler_ema_loss_bins/bin_{i}"] = loss_val.item()

            # Timestep histogram for the current batch
            if hasattr(self.la_sampler, "num_bins") and hasattr(self.la_sampler, "T"):
                hist = torch.histogram(
                    timesteps.float().cpu(),
                    bins=self.la_sampler.num_bins,
                    range=(0, self.la_sampler.T),
                )
                for i, count in enumerate(hist.hist):
                    logs[f"sampler_timestep_hist/bin_{i}"] = count.item()

        return logs

    def step_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
        self.accelerator_logging(accelerator, logs, global_step, global_step, epoch)

    def epoch_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
        self.accelerator_logging(accelerator, logs, epoch, global_step, epoch)

    def accelerator_logging(
        self, accelerator: Accelerator, logs: dict, step_value: int, global_step: int, epoch: int):
        """
        step_value is for tensorboard, other values are for wandb
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

    def save_timestep_distribution_plot(self, cfg, global_step, timestep_counts, settings_dict=None):
        if plt is None:
            logger.warning("Matplotlib is not installed. Cannot save timestep distribution plot.")
            return

        output_dir = os.path.join(cfg.saving.output_dir, "timestep_plots")
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 7)) # Make figure wider
        plt.bar(range(len(timestep_counts)), timestep_counts, width=1.0)
        plt.title(f"Timestep Distribution at Step {global_step}")
        plt.xlabel("Timestep")
        plt.ylabel("Accumulated Count")
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        # --- START MODIFICATION: Add settings text to the plot ---
        if settings_dict:
            settings_text = "\n".join([f"{key}: {value}" for key, value in settings_dict.items() if value is not None])
            plt.figtext(0.01, 0.01, settings_text, wrap=True, horizontalalignment='left', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.1))
        
        # Adjust layout to make room for the text
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        # --- END MODIFICATION ---

        filename = os.path.join(output_dir, f"step_{global_step:06d}.png")
        plt.savefig(filename)
        plt.close()

    def validate_extra_config(
        self,
        cfg,
        train_dataset_group: Union[DatasetGroup, MinimalDataset],
        val_dataset_group: Optional[DatasetGroup],
    ):
        validate_sd_peft(cfg, train_dataset_group, val_dataset_group)

    def load_target_model(self, cfg, weight_dtype, accelerator) -> tuple[str, nn.Module, nn.Module, Optional[nn.Module]]:
        text_encoder, vae, unet, _ = load_target_model(cfg.model, cfg.performance, weight_dtype, accelerator)

        if cfg.performance.use_ramtorch:
            logger.info("Applying RamTorch to SD UNet, VAE, and Clip-L.")
            if isinstance(unet, torch.nn.Module):
                unet = replace_linear_with_ramtorch(unet, accelerator.device)
                logger.info("RamTorch applied to SD unet.")

            if isinstance(text_encoder, torch.nn.Module):
                text_encoder = replace_linear_with_ramtorch(text_encoder, accelerator.device)
                logger.info("RamTorch applied to SD Clip-L.")

            if isinstance(vae, torch.nn.Module):
                vae = replace_linear_with_ramtorch(vae, accelerator.device)
                logger.info("RamTorch applied to SD VAE.")

        # モデルに xformers とか memory efficient attention を組み込む
        replace_unet_modules(unet, cfg.performance.mem_eff_attn, cfg.performance.xformers, cfg.performance.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(cfg.performance.xformers)

        return model_util.get_model_version_str_for_sd1_sd2(cfg.model.v2, cfg.loss.v_parameterization), text_encoder, vae, unet

    def load_unet_lazily(self, cfg, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, List[nn.Module]]:
        raise NotImplementedError()

    def get_tokenize_strategy(self, cfg):
        return strategy_sd.SdTokenizeStrategy(cfg.model.v2, cfg.training.max_token_length, cfg.model.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sd.SdTokenizeStrategy) -> List[Any]:
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, cfg):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            True, cfg.dataset.cache_latents_to_disk, cfg.dataset.vae_batch_size, cfg.dataset.skip_cache_check
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, cfg):
        return strategy_sd.SdTextEncodingStrategy(cfg.training.clip_skip)

    def get_text_encoder_outputs_caching_strategy(self, cfg):
        return None

    def get_models_for_text_encoding(self, cfg, accelerator, text_encoders):
        """
        Returns a list of models that will be used for text encoding. SDXL uses wrapped and unwrapped models.
        FLUX.1 and SD3 may cache some outputs of the text encoder, so return the models that will be used for encoding (not cached).
        """
        return text_encoders

    # returns a list of bool values indicating whether each text encoder should be trained
    def get_text_encoders_train_flags(self, cfg, text_encoders):
        return [True] * len(text_encoders) if self.is_train_text_encoder(cfg) else [False] * len(text_encoders)

    def is_train_text_encoder(self, cfg):
        return not cfg.network.network_train_unet_only

    def cache_text_encoder_outputs_if_needed(self, cfg, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device, dtype=weight_dtype)

    def call_unet(self, cfg, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype, **kwargs):
        noise_pred = unet(noisy_latents, timesteps, text_conds[0]).sample
        return noise_pred

    def all_reduce_network(self, accelerator, network):
        for param in network.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def sample_images(self, accelerator, cfg, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        sample_images(accelerator, cfg.sampling, cfg.training, cfg.saving, epoch, global_step, device, vae, tokenizers[0], text_encoder, unet)

    # region SD/SDXL

    def post_process_network(self, cfg, accelerator, network, text_encoders, unet):
        pass

    def get_noise_scheduler(self, cfg, device: torch.device) -> Any:
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )

        if cfg.regularization.zero_terminal_snr:
            fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

        prepare_scheduler_for_custom_training(noise_scheduler, device)
        return noise_scheduler

    def encode_images_to_latents(self, cfg, vae: AutoencoderKL, images: torch.FloatTensor) -> torch.FloatTensor:
        return vae.encode(images).latent_dist.sample()

    def shift_scale_latents(self, cfg, latents: torch.FloatTensor) -> torch.FloatTensor:
        return latents * self.vae_scale_factor

    def get_noise_pred_and_target(
        self,
        cfg,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        fixed_timesteps=None,
        is_train=True,
        min_timestep_override=None,
        max_timestep_override=None,
        global_step=0,
    ):
        # Sample noise, sample a random timestep for each image, and add noise to the latents,
        # with noise offset and/or multires noise if specified
        noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(
            cfg.regularization,
            cfg.timestep,
            cfg.training,
            noise_scheduler, 
            latents,
            la_sampler=self.la_sampler,
            global_step=global_step,
            fixed_timesteps=fixed_timesteps, 
            is_train=is_train, 
            min_timestep_override=min_timestep_override,
            max_timestep_override=max_timestep_override
        )

        # ensure the hidden state will require grad
        if is_train and cfg.performance.gradient_checkpointing:
            for x in noisy_latents:
                x.requires_grad_(True)
            for t in text_encoder_conds:
                t.requires_grad_(True)

        # Predict the noise residual
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            noise_pred = self.call_unet(
                cfg,
                accelerator,
                unet,
                noisy_latents.requires_grad_(train_unet),
                timesteps,
                text_encoder_conds,
                batch,
                weight_dtype,
            )

        if cfg.loss.v_parameterization:
            # v-parameterization training
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        # differential output preservation
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                with torch.no_grad(), accelerator.autocast():
                    noise_pred_prior = self.call_unet(
                        cfg,
                        accelerator,
                        unet,
                        noisy_latents,
                        timesteps,
                        text_encoder_conds,
                        batch,
                        weight_dtype,
                        indices=diff_output_pr_indices,
                    )
                network.set_multiplier(1.0)  # may be overwritten by "network_multipliers" in the next step
                target[diff_output_pr_indices] = noise_pred_prior.to(target.dtype)

        return noise_pred, target, timesteps, None

    def post_process_loss(self, loss, cfg, timesteps: torch.IntTensor, noise_scheduler) -> torch.FloatTensor:
        if cfg.loss.min_snr_gamma:
            loss = apply_snr_weight(loss, timesteps, noise_scheduler, cfg.loss.min_snr_gamma, cfg.loss.v_parameterization)
        if cfg.loss.scale_v_pred_loss_like_noise_pred:
            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
        if cfg.loss.v_pred_like_loss:
            loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, cfg.loss.v_pred_like_loss)
        if cfg.loss.debiased_estimation_loss:
            loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, cfg.loss.v_parameterization)
        return loss

    def get_sai_model_spec(self, cfg):
        return get_sai_model_spec(None, cfg, self.is_sdxl, True, False)  # HYDRA RELATED? Expected type 'dict', got 'None' instead?

    def update_metadata(self, metadata, cfg):
        pass

    def is_text_encoder_not_needed_for_training(self, cfg):
        return False  # use for sample images

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # set top parameter requires_grad = True for gradient checkpointing works
        text_encoder.text_model.embeddings.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)

    def prepare_unet_with_accelerator(
        self, cfg, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        return accelerator.prepare(unet)

    def on_step_start(self, cfg, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train: bool = True):
        pass

    def on_validation_step_end(self, cfg, accelerator, network, text_encoders, unet, batch, weight_dtype):
        pass

    # endregion

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        cfg,
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
        edm2_model=None,
        min_timestep_override=None,
        max_timestep_override=None,
        global_step=0,
    ) -> tuple:
        """
        Process a batch for the network
        """
        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(torch.FloatTensor, batch["latents"].to(accelerator.device))
            else:
                # latentに変換
                if cfg.dataset.vae_batch_size is None or len(batch["images"]) <= cfg.dataset.vae_batch_size:
                    latents = self.encode_images_to_latents(cfg, vae, batch["images"].to(accelerator.device, dtype=vae_dtype))
                else:
                    chunks = [
                        batch["images"][i : i + cfg.dataset.vae_batch_size] for i in range(0, len(batch["images"]), cfg.dataset.vae_batch_size)
                    ]
                    list_latents = []
                    for chunk in chunks:
                        with torch.no_grad():
                            chunk = self.encode_images_to_latents(cfg, vae, chunk.to(accelerator.device, dtype=vae_dtype))
                            list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                # NaNが含まれていれば警告を表示し0に置き換える
                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents))

            latents = self.shift_scale_latents(cfg, latents)

        text_encoder_conds = []
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoder_conds = text_encoder_outputs_list  # List of text encoder outputs

        if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
            # TODO this does not work if 'some text_encoders are trained' and 'some are not and not cached'
            with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
                # Get the text embedding for conditioning
                if cfg.dataset.weighted_captions:
                    input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                        tokenize_strategy,
                        self.get_models_for_text_encoding(cfg, accelerator, text_encoders),
                        input_ids_list,
                        weights_list,
                    )
                else:
                    input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                        tokenize_strategy,
                        self.get_models_for_text_encoding(cfg, accelerator, text_encoders),
                        input_ids,
                    )
                if cfg.performance.full_fp16:
                    encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

            # if text_encoder_conds is not cached, use encoded_text_encoder_conds
            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                # if encoded_text_encoder_conds is not None, update cached text_encoder_conds
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        # sample noise, call unet, get target
        noise_pred, target, timesteps, weighting = self.get_noise_pred_and_target(
            cfg,
            accelerator,
            noise_scheduler,
            latents,
            batch,
            text_encoder_conds,
            unet,
            network,
            weight_dtype,
            train_unet,
            is_train=is_train,
            min_timestep_override=min_timestep_override,
            max_timestep_override=max_timestep_override,
            global_step=global_step,
        )

        if is_train:
            huber_c = get_huber_threshold_if_needed(cfg.loss, timesteps, noise_scheduler)
            loss = conditional_loss(noise_pred.float(), target.float(), cfg.loss.loss_type, "none", huber_c, scale=float(cfg.loss.loss_scale))
            if weighting is not None:
                loss = loss * weighting
            if cfg.masked_loss.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                loss = apply_masked_loss(loss, batch)
        else:
                loss = conditional_loss(noise_pred.float(), target.float(), "l2", "none", None)

        per_sample_loss = loss.mean([1, 2, 3])

        # Feed the timesteps and their corresponding per-sample loss back to the sampler for its EMA update.
        if is_train and self.la_sampler is not None and hasattr(self.la_sampler, "update"):
            # We detach to ensure this operation doesn't affect the gradients for backpropagation.
            self.la_sampler.update(timesteps.detach(), per_sample_loss.detach())

        loss = per_sample_loss

        if is_train:
            loss_weights = batch["loss_weights"]  # 各sampleごとのweight
            loss = loss * loss_weights
            loss = self.post_process_loss(loss, cfg, timesteps, noise_scheduler)

        if is_train and cfg.loss.loss_multiplier:
            loss.mul_(float(cfg.loss.loss_multiplier) if cfg.loss.loss_multiplier is not None else 1.0)

        # For logging
        pre_scaling_loss = loss.mean()

        if is_train and cfg.loss.edm2_loss_weighting:
            loss, loss_scaled = edm2_model(loss, timesteps)
            loss_scaled = loss_scaled.mean()
        else:
            loss_scaled = None

        return loss.mean(), pre_scaling_loss, loss_scaled, timesteps
    
    def process_val_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        cfg,
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
        train_text_encoder=True,
        train_unet=True,
        timesteps_list: list = [50, 350, 500, 650, 950]
    ) -> torch.Tensor:
        """
        Process a batch for the network to determine val loss
        """
        total_loss = 0.0 
        with torch.autograd.grad_mode.inference_mode(mode=True):
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(torch.FloatTensor, batch["latents"].to(accelerator.device))
            else:
                # latentに変換
                if cfg.dataset.vae_batch_size is None or len(batch["images"]) <= cfg.dataset.vae_batch_size:
                    latents = self.encode_images_to_latents(cfg, vae, batch["images"].to(accelerator.device, dtype=vae_dtype))
                else:
                    chunks = [
                        batch["images"][i : i + cfg.dataset.vae_batch_size] for i in range(0, len(batch["images"]), cfg.dataset.vae_batch_size)
                    ]
                    list_latents = []
                    for chunk in chunks:
                        with torch.no_grad():
                            chunk = self.encode_images_to_latents(cfg, vae, chunk.to(accelerator.device, dtype=vae_dtype))
                            list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                # NaNが含まれていれば警告を表示し0に置き換える
                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents))

            latents = self.shift_scale_latents(cfg, latents)

            text_encoder_conds = []
            text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
            if text_encoder_outputs_list is not None:
                text_encoder_conds = text_encoder_outputs_list  # List of text encoder outputs

            if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
                # TODO this does not work if 'some text_encoders are trained' and 'some are not and not cached'
                with torch.set_grad_enabled(False and train_text_encoder), accelerator.autocast():
                    # Get the text embedding for conditioning
                    if cfg.dataset.weighted_captions:
                        input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                        encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                            tokenize_strategy,
                            self.get_models_for_text_encoding(cfg, accelerator, text_encoders),
                            input_ids_list,
                            weights_list,
                        )
                    else:
                        input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                        encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                            tokenize_strategy,
                            self.get_models_for_text_encoding(cfg, accelerator, text_encoders),
                            input_ids,
                        )
                    if cfg.performance.full_fp16:
                        encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

                # if text_encoder_conds is not cached, use encoded_text_encoder_conds
                if len(text_encoder_conds) == 0:
                    text_encoder_conds = encoded_text_encoder_conds
                else:
                    # if encoded_text_encoder_conds is not None, update cached text_encoder_conds
                    for i in range(len(encoded_text_encoder_conds)):
                        if encoded_text_encoder_conds[i] is not None:
                            text_encoder_conds[i] = encoded_text_encoder_conds[i]

            batch_size = latents.shape[0]
            for fixed_timesteps in timesteps_list:
                timesteps = torch.full((batch_size,), fixed_timesteps, dtype=torch.long, device=latents.device)

                # sample noise, call unet, get target
                noise_pred, target, _, _ = self.get_noise_pred_and_target(
                    cfg,
                    accelerator,
                    noise_scheduler,
                    latents,
                    batch,
                    text_encoder_conds,
                    unet,
                    network,
                    weight_dtype,
                    train_unet,
                    timesteps,
                    is_train=False,
                )

                loss = conditional_loss(noise_pred.float(), target.float(), "l2", "none", None)
                loss = loss.mean([1, 2, 3])
                loss = loss.mean()
                total_loss += loss

        average_loss = total_loss / len(timesteps_list)    

        return average_loss

    def cast_text_encoder(self, cfg):
        return True  # default for other than HunyuanImage

    def cast_vae(self, cfg):
        return True  # default for other than HunyuanImage

    def cast_unet(self, cfg):
        return True  # default for other than HunyuanImage

    def switch_rng_state(self, val_seed: int, accelerator):
        # Store current RNG states
        cpu_rng_state = torch.get_rng_state()
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        
        gpu_rng_state = None
        if accelerator.device.type == "cuda":
            gpu_rng_state = torch.cuda.get_rng_state()
        elif accelerator.device.type == "xpu":
            gpu_rng_state = torch.xpu.get_rng_state()

        # Set new seed for validation
        random.seed(val_seed)
        np.random.seed(val_seed)
        torch.manual_seed(val_seed)
        if accelerator.device.type == "cuda":
            torch.cuda.manual_seed_all(val_seed)

        return (cpu_rng_state, gpu_rng_state, python_rng_state, numpy_rng_state)

    def restore_rng_state(self, rng_states, accelerator):
        cpu_rng_state, gpu_rng_state, python_rng_state, numpy_rng_state = rng_states
        
        # Restore RNG states
        torch.set_rng_state(cpu_rng_state)
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        
        if gpu_rng_state is not None:
            if accelerator.device.type == "cuda":
                torch.cuda.set_rng_state(gpu_rng_state)
            elif accelerator.device.type == "xpu":
                torch.xpu.set_rng_state(gpu_rng_state)

    def calculate_val_loss(self, 
                           global_step,
                           epoch_step,
                           train_dataloader,
                           val_loss_recorder,
                           val_dataloader,
                           cyclic_val_dataloader,
                           network, 
                           tokenize_strategy, 
                           text_encoders, 
                           text_encoding_strategy, 
                           unet, 
                           vae, 
                           noise_scheduler, 
                           vae_dtype, 
                           weight_dtype, 
                           accelerator, 
                           cfg, 
                           epoch,
                           batch=None,
                           train_text_encoder=True):

        # Pass training config directly instead of legacy ArgsAdapter
        if not calculate_val_loss_check(cfg.training, global_step, epoch_step, val_dataloader, train_dataloader):
            return None, None, None
        
        if batch is not None:
            self.on_step_start(cfg, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=False)
   
        rng_states = self.switch_rng_state(int(cfg.dataset.validation_seed) if cfg.dataset.validation_seed else 23, accelerator)

        timesteps_list = ast.literal_eval(cfg.training.validation_timesteps)
              
        accelerator.print("") 
        accelerator.print("Validating バリデーション処理...")
        total_loss = 0.0
        with torch.no_grad():
            validation_steps = min(int(cfg.training.max_validation_steps), len(val_dataloader)) if cfg.training.max_validation_steps is not None else len(val_dataloader)
            val_dataloader_seed = random.randint(global_step, 0x7FFFFFFF)
            val_dataloader_state = random.Random(val_dataloader_seed).getstate()
            for val_step in tqdm(range(validation_steps), desc='Validation Steps'):
                val_original_state = random.getstate()
                random.setstate(val_dataloader_state)
                batch = next(cyclic_val_dataloader)
                val_dataloader_state = random.getstate()
                random.setstate(val_original_state)
                loss = self.process_val_batch(batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype, 
                                              weight_dtype, accelerator, cfg, text_encoding_strategy, tokenize_strategy, 
                                              train_text_encoder=train_text_encoder,
                                              timesteps_list=timesteps_list)
                total_loss += loss.detach().item()
            current_val_loss = total_loss / validation_steps
            val_loss_recorder.add(current_val_loss)   
                     
        average_val_loss: float = val_loss_recorder.average
        logs = {"loss/current_val_loss": current_val_loss, "loss/average_val_loss": average_val_loss}

        self.restore_rng_state(rng_states, accelerator)

        return current_val_loss, average_val_loss, logs


# train moved to library/training/peft_trainer.py


# Register the structure config with Hydra
cs = ConfigStore.instance()
cs.store(name="sd_peft", node=SDPeftConfig)

@hydra.main(version_base=None, config_path="../configs", config_name="sd_peft")
def main(cfg: SDPeftConfig):
    prepare_config(cfg)
    validate_config(cfg)
    trainer = SDPeftTrainer()
    trainer.train(cfg)

if __name__ == "__main__":
    main()
