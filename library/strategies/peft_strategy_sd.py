# SD1.5/2 PEFT Training Strategy implementation

import ast
import logging
import random
import typing
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from torch import nn
from diffusers import DDPMScheduler
from tqdm import tqdm
from ramtorch.helpers import replace_linear_with_ramtorch

from library.strategies import strategy_sd, strategy_base
from library.strategies.peft_strategy_base import PeftTrainingStrategy
from library.models import model_util
from library.training.model_prep import replace_unet_modules
from library.training.sd_model_prep import load_target_model
from library.training.sd_sample_generation import sample_images
from library.utils.sai_model_spec import get_sai_model_spec_from_config
from library.training.diffusion import get_noise_noisy_latents_and_timesteps
from library.training.trainer_utils import calculate_val_loss_check
from library.training.noise_utils import (
    prepare_scheduler_for_custom_training,
    fix_noise_scheduler_betas_for_zero_terminal_snr
)
from library.losses.loss import get_huber_threshold_if_needed, conditional_loss
from library.losses.loss_weighting import (
    apply_masked_loss, apply_snr_weight,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation
)
from library.config.validation import validate_sd_peft
from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class SdPeftStrategy(PeftTrainingStrategy):
    """
    SD1.5/2 implementation of PEFT training strategy.
    
    Extracted from SDPeftTrainer class methods.
    """
    
    vae_scale_factor: float = 0.18215
    is_sdxl: bool = False
    
    def load_target_model(self, cfg, weight_dtype, accelerator) -> tuple[str, nn.Module, nn.Module, Optional[nn.Module]]:
        """Load SD1.5/2 model components."""
        text_encoder, vae, unet, _ = load_target_model(cfg.model, cfg.performance, weight_dtype, accelerator)

        if cfg.performance.memory.use_ramtorch:
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

        # Apply xformers / memory efficient attention
        replace_unet_modules(unet, cfg.performance.attention.mem_eff_attn, cfg.performance.attention.xformers, cfg.performance.attention.sdpa)
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(cfg.performance.attention.xformers)

        return model_util.get_model_version_str_for_sd1_sd2(cfg.model.v2, cfg.loss.v_parameterization), text_encoder, vae, unet

    def get_tokenize_strategy(self, cfg):
        """Return SD1.5/2 tokenize strategy."""
        return strategy_sd.SdTokenizeStrategy(cfg.model.v2, cfg.training.max_token_length, cfg.model.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sd.SdTokenizeStrategy) -> List[Any]:
        """Return single tokenizer for SD1.5/2."""
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, cfg):
        """Return SD latents caching strategy."""
        return strategy_sd.SdSdxlLatentsCachingStrategy(
            True, cfg.dataset.cache_latents_to_disk, cfg.dataset.vae_batch_size, cfg.dataset.skip_cache_check
        )

    def get_text_encoding_strategy(self, cfg):
        """Return SD text encoding strategy."""
        return strategy_sd.SdTextEncodingStrategy(cfg.training.clip_skip)

    def get_text_encoder_outputs_caching_strategy(self, cfg):
        """SD doesn't cache text encoder outputs by default."""
        return None

    def cache_text_encoder_outputs_if_needed(self, cfg, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        """Move text encoders to device for SD."""
        for t_enc in text_encoders:
            t_enc.to(accelerator.device, dtype=weight_dtype)

    def get_models_for_text_encoding(self, cfg, accelerator, text_encoders) -> List:
        """Return text encoders for encoding (SD uses single encoder)."""
        return text_encoders

    def call_unet(self, cfg, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype, **kwargs):
        """Call SD UNet with simple signature."""
        noise_pred = unet(noisy_latents, timesteps, text_conds[0]).sample
        return noise_pred

    def sample_images(self, accelerator, cfg, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        """Generate sample images for SD."""
        sample_images(accelerator, cfg.sampling, cfg.training, cfg.saving, epoch, global_step, device, vae, tokenizers[0], text_encoder, unet)

    def validate_extra_config(self, cfg, train_dataset_group, val_dataset_group):
        """Run SD-specific config validation."""
        validate_sd_peft(cfg, train_dataset_group, val_dataset_group)

    def update_metadata(self, metadata: dict, cfg):
        """SD doesn't add extra metadata."""
        pass

    def get_sai_model_spec(self, cfg) -> dict:
        """Get SAI model spec for SD."""
        return get_sai_model_spec_from_config(
            state_dict=None,
            metadata_config=cfg.metadata,
            is_sdxl=self.is_sdxl,
            is_v2=cfg.model.v2,
            v_parameterization=cfg.loss.v_parameterization,
            is_lora=True,
            is_textual_inversion=False,
            resolution=cfg.dataset.resolution,
            min_timestep=cfg.timestep.min_timestep,
            max_timestep=cfg.timestep.max_timestep,
            clip_skip=cfg.training.clip_skip,
        )

    def get_noise_scheduler(self, cfg, device: torch.device) -> Any:
        """Create noise scheduler for SD."""
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
            num_train_timesteps=1000, clip_sample=False
        )

        if cfg.regularization.zero_terminal_snr:
            fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

        prepare_scheduler_for_custom_training(noise_scheduler, device)
        return noise_scheduler

    def encode_images_to_latents(self, cfg, vae, images: torch.FloatTensor) -> torch.FloatTensor:
        """Encode images to latents using VAE."""
        return vae.encode(images).latent_dist.sample()

    def shift_scale_latents(self, cfg, latents: torch.FloatTensor) -> torch.FloatTensor:
        """Apply VAE scale factor to latents."""
        return latents * self.vae_scale_factor

    # region Training batch processing methods

    def get_noise_pred_and_target(
        self, cfg, accelerator, noise_scheduler, latents, batch, text_encoder_conds,
        unet, network, weight_dtype, train_unet, fixed_timesteps=None, is_train=True,
        min_timestep_override=None, max_timestep_override=None, global_step=0,
    ):
        """Sample noise, call UNet, get noise prediction target."""
        noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(
            cfg.regularization, cfg.timestep, cfg.training, noise_scheduler, latents,
            la_sampler=self.la_sampler, global_step=global_step, fixed_timesteps=fixed_timesteps,
            is_train=is_train, min_timestep_override=min_timestep_override, max_timestep_override=max_timestep_override
        )

        if is_train and cfg.performance.memory.gradient_checkpointing:
            for x in noisy_latents:
                x.requires_grad_(True)
            for t in text_encoder_conds:
                t.requires_grad_(True)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            noise_pred = self.call_unet(cfg, accelerator, unet, noisy_latents.requires_grad_(train_unet),
                                        timesteps, text_encoder_conds, batch, weight_dtype)

        if cfg.loss.v_parameterization:
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
                    noise_pred_prior = self.call_unet(cfg, accelerator, unet, noisy_latents, timesteps,
                                                       text_encoder_conds, batch, weight_dtype, indices=diff_output_pr_indices)
                network.set_multiplier(1.0)
                target[diff_output_pr_indices] = noise_pred_prior.to(target.dtype)

        return noise_pred, target, timesteps, None

    def post_process_loss(self, loss, cfg, timesteps: torch.IntTensor, noise_scheduler) -> torch.FloatTensor:
        """Apply SNR weighting, v-pred scaling, debiased estimation etc."""
        if cfg.loss.min_snr_gamma:
            loss = apply_snr_weight(loss, timesteps, noise_scheduler, cfg.loss.min_snr_gamma, cfg.loss.v_parameterization)
        if cfg.loss.scale_v_pred_loss_like_noise_pred:
            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
        if cfg.loss.v_pred_like_loss:
            loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, cfg.loss.v_pred_like_loss)
        if cfg.loss.debiased_estimation_loss:
            loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, cfg.loss.v_parameterization)
        return loss

    def process_batch(
        self, batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype, weight_dtype,
        accelerator, cfg, text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy, is_train=True, train_text_encoder=True,
        train_unet=True, edm2_model=None, min_timestep_override=None, max_timestep_override=None, global_step=0,
    ) -> tuple:
        """Process a batch for training."""
        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(torch.FloatTensor, batch["latents"].to(accelerator.device))
            else:
                if cfg.dataset.vae_batch_size is None or len(batch["images"]) <= cfg.dataset.vae_batch_size:
                    latents = self.encode_images_to_latents(cfg, vae, batch["images"].to(accelerator.device, dtype=vae_dtype))
                else:
                    chunks = [batch["images"][i : i + cfg.dataset.vae_batch_size] for i in range(0, len(batch["images"]), cfg.dataset.vae_batch_size)]
                    list_latents = []
                    for chunk in chunks:
                        with torch.no_grad():
                            chunk = self.encode_images_to_latents(cfg, vae, chunk.to(accelerator.device, dtype=vae_dtype))
                            list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents))

            latents = self.shift_scale_latents(cfg, latents)

        text_encoder_conds = []
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoder_conds = text_encoder_outputs_list

        if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
            with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
                if cfg.dataset.weighted_captions:
                    input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                        tokenize_strategy, self.get_models_for_text_encoding(cfg, accelerator, text_encoders), input_ids_list, weights_list)
                else:
                    input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                        tokenize_strategy, self.get_models_for_text_encoding(cfg, accelerator, text_encoders), input_ids)
                if cfg.performance.precision.full_fp16:
                    encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        noise_pred, target, timesteps, weighting = self.get_noise_pred_and_target(
            cfg, accelerator, noise_scheduler, latents, batch, text_encoder_conds, unet, network,
            weight_dtype, train_unet, is_train=is_train, min_timestep_override=min_timestep_override,
            max_timestep_override=max_timestep_override, global_step=global_step)

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

        if is_train and self.la_sampler is not None and hasattr(self.la_sampler, "update"):
            self.la_sampler.update(timesteps.detach(), per_sample_loss.detach())

        loss = per_sample_loss
        if is_train:
            loss = loss * batch["loss_weights"]
            loss = self.post_process_loss(loss, cfg, timesteps, noise_scheduler)

        if is_train and cfg.loss.loss_multiplier:
            loss.mul_(float(cfg.loss.loss_multiplier) if cfg.loss.loss_multiplier is not None else 1.0)

        pre_scaling_loss = loss.mean()

        if is_train and cfg.loss.edm2_loss_weighting:
            loss, loss_scaled = edm2_model(loss, timesteps)
            loss_scaled = loss_scaled.mean()
        else:
            loss_scaled = None

        return loss.mean(), pre_scaling_loss, loss_scaled, timesteps

    def process_val_batch(
        self, batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype, weight_dtype,
        accelerator, cfg, text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy, train_text_encoder=True, train_unet=True,
        timesteps_list: list = [50, 350, 500, 650, 950]
    ) -> torch.Tensor:
        """Process a batch for validation loss."""
        total_loss = 0.0
        with torch.autograd.grad_mode.inference_mode(mode=True):
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(torch.FloatTensor, batch["latents"].to(accelerator.device))
            else:
                if cfg.dataset.vae_batch_size is None or len(batch["images"]) <= cfg.dataset.vae_batch_size:
                    latents = self.encode_images_to_latents(cfg, vae, batch["images"].to(accelerator.device, dtype=vae_dtype))
                else:
                    chunks = [batch["images"][i : i + cfg.dataset.vae_batch_size] for i in range(0, len(batch["images"]), cfg.dataset.vae_batch_size)]
                    list_latents = []
                    for chunk in chunks:
                        with torch.no_grad():
                            chunk = self.encode_images_to_latents(cfg, vae, chunk.to(accelerator.device, dtype=vae_dtype))
                            list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents))

            latents = self.shift_scale_latents(cfg, latents)

            text_encoder_conds = []
            text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
            if text_encoder_outputs_list is not None:
                text_encoder_conds = text_encoder_outputs_list

            if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
                with torch.set_grad_enabled(False and train_text_encoder), accelerator.autocast():
                    if cfg.dataset.weighted_captions:
                        input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                        encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                            tokenize_strategy, self.get_models_for_text_encoding(cfg, accelerator, text_encoders), input_ids_list, weights_list)
                    else:
                        input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                        encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                            tokenize_strategy, self.get_models_for_text_encoding(cfg, accelerator, text_encoders), input_ids)
                    if cfg.performance.precision.full_fp16:
                        encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

                if len(text_encoder_conds) == 0:
                    text_encoder_conds = encoded_text_encoder_conds
                else:
                    for i in range(len(encoded_text_encoder_conds)):
                        if encoded_text_encoder_conds[i] is not None:
                            text_encoder_conds[i] = encoded_text_encoder_conds[i]

            batch_size = latents.shape[0]
            for fixed_timesteps in timesteps_list:
                timesteps = torch.full((batch_size,), fixed_timesteps, dtype=torch.long, device=latents.device)
                noise_pred, target, _, _ = self.get_noise_pred_and_target(
                    cfg, accelerator, noise_scheduler, latents, batch, text_encoder_conds, unet, network,
                    weight_dtype, train_unet, fixed_timesteps, is_train=False)

                loss = conditional_loss(noise_pred.float(), target.float(), "l2", "none", None)
                loss = loss.mean([1, 2, 3]).mean()
                total_loss += loss

        return total_loss / len(timesteps_list)

    def calculate_val_loss(
        self, global_step, epoch_step, train_dataloader, val_loss_recorder, val_dataloader,
        cyclic_val_dataloader, network, tokenize_strategy, text_encoders, text_encoding_strategy,
        unet, vae, noise_scheduler, vae_dtype, weight_dtype, accelerator, cfg, epoch, batch=None, train_text_encoder=True
    ):
        """Calculate validation loss."""
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
                                              train_text_encoder=train_text_encoder, timesteps_list=timesteps_list)
                total_loss += loss.detach().item()
            current_val_loss = total_loss / validation_steps
            val_loss_recorder.add(current_val_loss)

        average_val_loss: float = val_loss_recorder.average
        logs = {"loss/current_val_loss": current_val_loss, "loss/average_val_loss": average_val_loss}

        self.restore_rng_state(rng_states, accelerator)

        return current_val_loss, average_val_loss, logs

    # endregion
