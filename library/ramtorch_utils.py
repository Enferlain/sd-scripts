import torch
import torch.nn as nn
from library.utils import setup_logging

try:
    from ramtorch.helpers import replace_linear_with_ramtorch

    RAMTORCH_AVAILABLE = True
except ImportError:
    RAMTORCH_AVAILABLE = False

setup_logging()
import logging

logger = logging.getLogger(__name__)

if args.use_ramtorch:
    logger.info("Applying RamTorch to FLUX models (DiT, T5-XXL, CLIP-L, AE).")
    model = replace_linear_with_ramtorch(model, accelerator.device)
    clip_l = replace_linear_with_ramtorch(clip_l, accelerator.device)
    t5xxl = replace_linear_with_ramtorch(t5xxl, accelerator.device)
    ae = replace_linear_with_ramtorch(ae, accelerator.device)


def apply_ramtorch(args, unet, text_encoders, accelerator):
    # Apply ramtorch
    if args.use_ramtorch:
        logger.info("Applying RamTorch to U-Net and Text Encoders for memory efficiency...")
        replace_linear_with_ramtorch(unet, accelerator.device)
        for text_encoder in text_encoders:
            replace_linear_with_ramtorch(text_encoder, accelerator.device)
        logger.info("RamTorch applied successfully.")
