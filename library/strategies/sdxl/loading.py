import logging
from typing import Any

import torch
from torch import nn

from library.constants import MODEL_VERSION_SDXL_BASE_V1_0
from library.models.runtime_utils import replace_unet_modules
from library.models.sdxl.loader import load_target_model as load_sdxl_target_model
from library.strategies.base.contracts import ModelLoadingStrategy

try:
    from ramtorch.helpers import replace_linear_with_ramtorch
except (ImportError, AssertionError):
    replace_linear_with_ramtorch = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class SdxlModelLoadingStrategy(ModelLoadingStrategy):
    """Model-loading facet for SDXL training strategies."""

    def load_target_model(
        self, cfg: Any, weight_dtype: torch.dtype, accelerator: Any
    ) -> tuple[str, list[nn.Module], nn.Module, nn.Module | None]:
        """
        Load SDXL model components (dual text encoders, VAE, UNet).

        Args:
            cfg: Configuration object.
            weight_dtype: Weight data type.
            accelerator: Accelerator instance.

        Returns:
            Tuple of (model_version, text_encoders, vae, unet).
        """
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = load_sdxl_target_model(
            cfg.model,
            cfg.performance.memory,
            cfg.data.caching,
            cfg.performance.precision,
            accelerator,
            MODEL_VERSION_SDXL_BASE_V1_0,
            weight_dtype,
        )

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        if cfg.performance.memory.use_ramtorch:
            if replace_linear_with_ramtorch is None:
                raise ImportError("RamTorch is not available. Please install it or set use_ramtorch to False.")
            logger.info("Applying RamTorch to SDXL UNet, VAE, and Text Encoders.")
            if isinstance(unet, torch.nn.Module):
                unet = replace_linear_with_ramtorch(unet, accelerator.device)
                logger.info("RamTorch applied to SDXL unet.")

            if isinstance(vae, torch.nn.Module):
                vae = replace_linear_with_ramtorch(vae, accelerator.device)
                logger.info("RamTorch applied to SDXL vae.")

            if isinstance(text_encoder1, torch.nn.Module):
                text_encoder1 = replace_linear_with_ramtorch(text_encoder1, accelerator.device)
                logger.info("RamTorch applied to SDXL Clip-L.")

            if isinstance(text_encoder2, torch.nn.Module):
                text_encoder2 = replace_linear_with_ramtorch(text_encoder2, accelerator.device)
                logger.info("RamTorch applied to SDXL Clip-G.")

        replace_unet_modules(
            unet, cfg.performance.attention.mem_eff_attn, cfg.performance.attention.xformers, cfg.performance.attention.sdpa
        )
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(cfg.performance.attention.xformers)

        return MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet
