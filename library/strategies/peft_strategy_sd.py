# SD1.5/2 PEFT Training Strategy implementation

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from torch import nn
from diffusers import DDPMScheduler
from ramtorch.helpers import replace_linear_with_ramtorch

from library.strategies import strategy_sd
from library.strategies.peft_strategy_base import PeftTrainingStrategy
from library.models import model_util
from library.training.model_prep import load_target_model, replace_unet_modules
from library.training.sample_generation import sample_images
from library.training.checkpointing import get_sai_model_spec
from library.training.noise_utils import (
    prepare_scheduler_for_custom_training,
    fix_noise_scheduler_betas_for_zero_terminal_snr
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

        # Apply xformers / memory efficient attention
        replace_unet_modules(unet, cfg.performance.mem_eff_attn, cfg.performance.xformers, cfg.performance.sdpa)
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(cfg.performance.xformers)

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
        return get_sai_model_spec(None, cfg, self.is_sdxl, True, False)

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
