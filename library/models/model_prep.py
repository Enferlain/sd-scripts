"""
Generic Model Preparation Utilities

This module contains model preparation utilities that are used by both SD1.5/2 and SDXL.
SD-specific model loading is in sd_model_prep.py.
SDXL-specific model loading is in sdxl_model_prep.py.
"""

import logging
import torch
from typing import Literal

from library.models.sd_original_unet import UNet2DConditionModel

logger = logging.getLogger(__name__)


def replace_unet_modules(unet: UNet2DConditionModel, mem_eff_attn, xformers, sdpa):
    """
    Replace U-Net modules with specific attention mechanisms.

    Args:
        unet (UNet2DConditionModel): The U-Net model to modify.
        mem_eff_attn (bool): Whether to enable memory efficient attention.
        xformers (bool): Whether to enable xformers.
        sdpa (bool): Whether to enable SDPA (Scaled Dot Product Attention).
    """
    if mem_eff_attn:
        logger.info("Enable memory efficient attention for U-Net")
        unet.set_use_memory_efficient_attention(False, True)
    elif xformers:
        logger.info("Enable xformers for U-Net")
        try:
            import xformers.ops
        except ImportError as err:
            raise ImportError("No xformers / xformersがインストールされていないようです") from err

        unet.set_use_memory_efficient_attention(True, False)
    elif sdpa:
        logger.info("Enable SDPA for U-Net")
        unet.set_use_sdpa(True)


def set_padding_mode_for_vae_conv2d_modules(
    vae: torch.nn.Module, padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros"
):
    """
    Apply padding mode only to Conv2d modules with non-zero padding (for EQ VAE).

    Args:
        vae (torch.nn.Module): The VAE model.
        padding_mode (Literal["zeros", "reflect", "replicate", "circular"]): The padding mode to apply.
    """
    logger.info(f"VAE padding mode set to: {padding_mode}")
    for _name, module in vae.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            pad = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
            if pad[0] > 0 or pad[1] > 0:
                module.padding_mode = padding_mode  # type: ignore[assignment]  # Literal is compatible with str


# NOTE: SD-specific load_target_model and _load_target_model moved to sd_model_prep.py


def patch_accelerator_for_fp16_training(accelerator):
    """
    Patch the accelerator to handle gradient unscaling for FP16 training.

    This function modifies the accelerator's scaler to allow unscaling gradients
    even if inf/nan values are found, which is necessary for some training setups.

    Args:
        accelerator: The Accelerator instance to patch.
    """
    from accelerate import DistributedType

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        return

    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer
