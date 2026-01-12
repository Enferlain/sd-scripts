import os
import logging
import torch
from typing import Literal, cast

from accelerate import init_empty_weights

from library.utils.common_utils import setup_logging
from library.utils.device_utils import clean_memory_on_device
from library.utils.torch_utils import match_mixed_precision
from library.models.model_prep import set_padding_mode_for_vae_conv2d_modules
from library.models.sdxl import conversion, unet as sdxl_unet
from library.models.sd import vae as vae_util
from library.config.dataclasses.model import ModelConfig
from library.config.dataclasses.performance import MemoryConfig, PrecisionConfig
from library.config.dataclasses.data import CachingConfig

setup_logging()
logger = logging.getLogger(__name__)


def load_target_model(
    model_config: ModelConfig,
    memory_config: MemoryConfig,
    caching_config: CachingConfig,
    precision_config: PrecisionConfig,
    accelerator,
    model_version: str,
    weight_dtype,
):
    """
    Load SDXL model components.

    Args:
        model_config (ModelConfig): Model configuration (pretrained path, VAE, etc.)
        memory_config (MemoryConfig): Memory configuration (lowram, etc.)
        caching_config (CachingConfig): Caching configuration (disable_mmap, etc.)
        precision_config (PrecisionConfig): Precision configuration (mixed_precision, etc.)
        accelerator: Accelerator instance
        model_version (str): Model version string
        weight_dtype: Weight data type

    Returns:
        Tuple of (load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info)
    """
    model_dtype = match_mixed_precision(precision_config, weight_dtype)

    # Initialize variables before loop to satisfy type checker
    load_stable_diffusion_format = False
    text_encoder1 = None
    text_encoder2 = None
    vae = None
    unet = None
    logit_scale = None
    ckpt_info = None

    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.info(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

            (
                load_stable_diffusion_format,
                text_encoder1,
                text_encoder2,
                vae,
                unet,
                logit_scale,
                ckpt_info,
            ) = _load_target_model(
                model_config,
                model_config.pretrained_model_name_or_path or "",
                model_config.vae,
                model_version,
                weight_dtype,
                accelerator.device if memory_config.lowram else "cpu",
                model_dtype,
                caching_config.disable_mmap_load_safetensors,
            )

            if memory_config.lowram:
                text_encoder1.to(accelerator.device)
                text_encoder2.to(accelerator.device)
                unet.to(accelerator.device)
                vae.to(accelerator.device)

            clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    assert text_encoder1 is not None and text_encoder2 is not None and vae is not None and unet is not None, "Model loading failed"
    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info


def _load_target_model(
    model_config: ModelConfig,
    name_or_path: str,
    vae_path: str | None,
    model_version: str,
    weight_dtype,
    device="cpu",
    model_dtype=None,
    disable_mmap=False,
):
    """
    Internal function to load SDXL model from checkpoint or diffusers.

    Args:
        model_config (ModelConfig): Model configuration (for vae_conv2d_padding_mode)
        name_or_path (str): Path to model checkpoint or HuggingFace model name
        vae_path (Optional[str]): Optional path to separate VAE
        model_version (str): Model version string
        weight_dtype: Weight data type
        device (str, optional): Device to load model to. Defaults to "cpu".
        model_dtype: Model data type
        disable_mmap (bool, optional): Whether to disable memory mapping for safetensors. Defaults to False.

    Returns:
        Tuple of (load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info)
    """
    name_or_path = os.readlink(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)

    if load_stable_diffusion_format:
        logger.info(f"load StableDiffusion checkpoint: {name_or_path}")
        (
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = conversion.load_models_from_sdxl_checkpoint(model_version, name_or_path, device, model_dtype, disable_mmap)
    else:
        from diffusers import StableDiffusionXLPipeline

        variant = "fp16" if weight_dtype == torch.float16 else None
        logger.info(f"load Diffusers pretrained models: {name_or_path}, variant={variant}")
        try:
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(name_or_path, torch_dtype=model_dtype, variant=variant, tokenizer=None)
            except OSError as ex:
                if variant is not None:
                    logger.info("try to load fp32 model")
                    pipe = StableDiffusionXLPipeline.from_pretrained(name_or_path, variant=None, tokenizer=None)
                else:
                    raise ex
        except OSError as ex:
            logger.error(f"model is not found as a file or in Hugging Face, perhaps file name is wrong?: {name_or_path}")
            raise ex

        text_encoder1 = pipe.text_encoder
        text_encoder2 = pipe.text_encoder_2

        if text_encoder1.dtype != torch.float32:
            text_encoder1 = text_encoder1.to(dtype=torch.float32)
        if text_encoder2.dtype != torch.float32:
            text_encoder2 = text_encoder2.to(dtype=torch.float32)

        vae = pipe.vae
        unet = pipe.unet
        del pipe

        state_dict = conversion.convert_diffusers_unet_state_dict_to_sdxl(unet.state_dict())
        with init_empty_weights():
            unet = sdxl_unet.SdxlUNet2DConditionModel()
        conversion._load_state_dict_on_device(unet, state_dict, device=device, dtype=model_dtype)
        logger.info("U-Net converted to original U-Net")

        logit_scale = None
        ckpt_info = None

    if vae_path is not None:
        vae = vae_util.load_vae(vae_path, weight_dtype)
        logger.info("additional VAE loaded")

    if model_config.vae_conv2d_padding_mode is not None and model_config.vae_conv2d_padding_mode.lower() != "zeros":
        logger.info(f"Loading VAE with padding mode: {model_config.vae_conv2d_padding_mode}")
        padding_mode = cast(Literal["zeros", "reflect", "replicate", "circular"], model_config.vae_conv2d_padding_mode)
        set_padding_mode_for_vae_conv2d_modules(vae, padding_mode)

    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info
