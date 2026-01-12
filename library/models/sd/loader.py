"""
SD1.5/2 Model Loading Functions

This module contains SD1.5/2-specific model loading functions.
Generic utilities are in runtime_utils.py.
"""

import os
import logging
from typing import Literal, cast

from diffusers import StableDiffusionPipeline

import library.models.sd.conversion
from library.models.sd.unet import UNet2DConditionModel
from library.utils.device_utils import clean_memory_on_device
from library.config.dataclasses.model import ModelConfig
from library.config.dataclasses.performance import MemoryConfig
from library.models.runtime_utils import set_padding_mode_for_vae_conv2d_modules

logger = logging.getLogger(__name__)


def _load_target_model(
    model_config: ModelConfig,
    v2: bool,
    weight_dtype,
    device="cpu",
    unet_use_linear_projection_in_v2=False,
):
    """
    Internal function to load the target Stable Diffusion model (v1.5 or v2).

    Args:
        model_config (ModelConfig): Configuration for the model.
        v2 (bool): Whether the model is Stable Diffusion v2.
        weight_dtype: The data type for the model weights.
        device (str, optional): The device to load the model on. Defaults to "cpu".
        unet_use_linear_projection_in_v2 (bool, optional): Whether to use linear projection in v2 U-Net. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - text_encoder: The text encoder model.
            - vae: The VAE model.
            - unet: The U-Net model.
            - load_stable_diffusion_format (bool): Whether the model was loaded from a Stable Diffusion checkpoint.
    """
    name_or_path = model_config.pretrained_model_name_or_path
    assert name_or_path is not None, "pretrained_model_name_or_path must be specified"
    name_or_path = os.path.realpath(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers
    if load_stable_diffusion_format:
        logger.info(f"load StableDiffusion checkpoint: {name_or_path}")
        text_encoder, vae, unet = library.models.sd.conversion.load_models_from_stable_diffusion_checkpoint(
            v2,
            name_or_path,
            device,
            unet_use_linear_projection_in_v2=unet_use_linear_projection_in_v2,
        )
    else:
        # Diffusers model is loaded to CPU
        logger.info(f"load Diffusers pretrained models: {name_or_path}")
        try:
            pipe = StableDiffusionPipeline.from_pretrained(name_or_path, tokenizer=None, safety_checker=None)
        except OSError as ex:
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
        vae = vae.load_vae(model_config.vae, weight_dtype)
        logger.info("additional VAE loaded")

    if model_config.vae_conv2d_padding_mode is not None and model_config.vae_conv2d_padding_mode.lower() != "zeros":
        logger.info(f"Loaded VAE with padding mode: {model_config.vae_conv2d_padding_mode}")
        padding_mode = cast(Literal["zeros", "reflect", "replicate", "circular"], model_config.vae_conv2d_padding_mode)
        set_padding_mode_for_vae_conv2d_modules(vae, padding_mode)

    return text_encoder, vae, unet, load_stable_diffusion_format


def load_target_model(
    model_config: ModelConfig, memory_config: MemoryConfig, weight_dtype, accelerator, unet_use_linear_projection_in_v2=False
):
    """
    Load the target Stable Diffusion model, handling distributed loading and memory configurations.

    Args:
        model_config (ModelConfig): Configuration for the model.
        memory_config (MemoryConfig): Configuration for memory usage.
        weight_dtype: The data type for the model weights.
        accelerator: The Accelerator instance.
        unet_use_linear_projection_in_v2 (bool, optional): Whether to use linear projection in v2 U-Net. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - text_encoder: The text encoder model.
            - vae: The VAE model.
            - unet: The U-Net model.
            - load_stable_diffusion_format (bool): Whether the model was loaded from a Stable Diffusion checkpoint.
    """
    is_v2 = model_config.model_type == "sd2"
    # Initialize variables before loop to satisfy type checker
    text_encoder = None
    vae = None
    unet = None
    load_stable_diffusion_format = False

    assert accelerator.state.num_processes > 0, "num_processes must be greater than 0"
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.info(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

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
    assert text_encoder is not None and vae is not None and unet is not None, "Model loading failed"
    return text_encoder, vae, unet, load_stable_diffusion_format
