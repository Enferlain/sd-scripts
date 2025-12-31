"""
Generic Model Preparation Utilities

This module contains model preparation utilities that are used by both SD1.5/2 and SDXL.
SD-specific model loading is in sd_model_prep.py.
SDXL-specific model loading is in sdxl_model_prep.py.
"""
import logging
import torch

from library.models.sd_original_unet import UNet2DConditionModel

logger = logging.getLogger(__name__)


def replace_unet_modules(unet: UNet2DConditionModel, mem_eff_attn, xformers, sdpa):
    if mem_eff_attn:
        logger.info("Enable memory efficient attention for U-Net")
        unet.set_use_memory_efficient_attention(False, True)
    elif xformers:
        logger.info("Enable xformers for U-Net")
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("No xformers / xformersがインストールされていないようです")

        unet.set_use_memory_efficient_attention(True, False)
    elif sdpa:
        logger.info("Enable SDPA for U-Net")
        unet.set_use_sdpa(True)


def set_padding_mode_for_vae_conv2d_modules(vae: torch.nn.Module, padding_mode: str = 'zeros'):
    """Apply padding mode only to Conv2d modules with non-zero padding (for EQ VAE)"""
    logger.info(f"VAE padding mode set to: {padding_mode}")
    for name, module in vae.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            pad = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
            if pad[0] > 0 or pad[1] > 0:
                module.padding_mode = padding_mode  # TODO: Expected type 'Literal["zeros", "reflect", "replicate", "circular"]', got 'str' instead


# NOTE: SD-specific load_target_model and _load_target_model moved to sd_model_prep.py


def patch_accelerator_for_fp16_training(accelerator):
    from accelerate import DistributedType

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        return

    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer
