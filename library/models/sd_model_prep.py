"""
SD1.5/2 Model Loading Functions

This module contains SD1.5/2-specific model loading functions.
Generic utilities are in model_prep.py.
"""

import os
import logging

from diffusers import StableDiffusionPipeline

import library.models.sd_model_util
from library.models import model_util
from library.models.sd_original_unet import UNet2DConditionModel
from library.utils.device_utils import clean_memory_on_device
from library.config.dataclasses.model import ModelConfig
from library.config.dataclasses.performance import MemoryConfig
from library.models.model_prep import set_padding_mode_for_vae_conv2d_modules

logger = logging.getLogger(__name__)


def _load_target_model(
    model_config: ModelConfig,
    v2: bool,
    weight_dtype,
    device="cpu",
    unet_use_linear_projection_in_v2=False,
):
    """
    Internal function to load SD1.5/2 model from checkpoint or diffusers.
    """
    name_or_path = model_config.pretrained_model_name_or_path
    name_or_path = (
        os.path.realpath(name_or_path) if os.path.islink(name_or_path) else name_or_path
    )
    load_stable_diffusion_format = os.path.isfile(
        name_or_path
    )  # determine SD or Diffusers
    if load_stable_diffusion_format:
        logger.info(f"load StableDiffusion checkpoint: {name_or_path}")
        text_encoder, vae, unet = (
            library.models.sd_model_util.load_models_from_stable_diffusion_checkpoint(
                v2,
                name_or_path,
                device,
                unet_use_linear_projection_in_v2=unet_use_linear_projection_in_v2,
            )
        )
    else:
        # Diffusers model is loaded to CPU
        logger.info(f"load Diffusers pretrained models: {name_or_path}")
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                name_or_path, tokenizer=None, safety_checker=None
            )
        except EnvironmentError as ex:
            logger.error(
                f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path}"
            )
            raise ex
        text_encoder = pipe.text_encoder
        vae = pipe.vae
        unet = pipe.unet
        del pipe

        # Diffusers U-Net to original U-Net
        original_unet = UNet2DConditionModel(
            unet.config.sample_size,
            unet.config.attention_head_dim,
            unet.config.cross_attention_dim,
            unet.config.use_linear_projection,
            unet.config.upcast_attention,
        )
        original_unet.load_state_dict(unet.state_dict())
        unet = original_unet
        logger.info("U-Net converted to original U-Net")

    # VAEを読み込む
    if model_config.vae is not None:
        vae = model_util.load_vae(model_config.vae, weight_dtype)
        logger.info("additional VAE loaded")

    if (
        model_config.vae_conv2d_padding_mode is not None
        and model_config.vae_conv2d_padding_mode.lower() != "zeros"
    ):
        logger.info(
            f"Loaded VAE with padding mode: {model_config.vae_conv2d_padding_mode}"
        )
        set_padding_mode_for_vae_conv2d_modules(
            vae, model_config.vae_conv2d_padding_mode
        )

    return text_encoder, vae, unet, load_stable_diffusion_format


def load_target_model(model_config: ModelConfig, memory_config: MemoryConfig, weight_dtype, accelerator,
                      unet_use_linear_projection_in_v2=False):
    """
    Load SD1.5/2 model components.

    Args:
        model_config: Model configuration (pretrained path, VAE, etc.)
        memory_config: Memory configuration (lowram, etc.)
        weight_dtype: Weight data type
        accelerator: Accelerator instance
        unet_use_linear_projection_in_v2: Whether to use linear projection in V2 U-Net

    Returns:
        Tuple of (text_encoder, vae, unet, load_stable_diffusion_format)
    """
    is_v2 = model_config.model_type == "sd2"
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.info(
                f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}"
            )

            text_encoder, vae, unet, load_stable_diffusion_format = _load_target_model(
                model_config,
                is_v2,
                weight_dtype,
                accelerator.device if memory_config.lowram else "cpu",
                unet_use_linear_projection_in_v2=unet_use_linear_projection_in_v2,
            )
            # work on low-ram device
            if memory_config.lowram:
                text_encoder.to(accelerator.device)
                unet.to(accelerator.device)
                vae.to(accelerator.device)

            clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()
    return text_encoder, vae, unet, load_stable_diffusion_format  # TODO: Local variables might be referenced before assignment
