import logging
from typing import Any

import torch

import library.models.sd.conversion
from library.models import LoadedModelComponent, build_loaded_components
from library.models.runtime_utils import replace_unet_modules
from library.models.sd.loader import load_target_model
from library.objectives.ddpm import DDPM_PREDICTION_TYPE_V, resolve_ddpm_prediction_type
from library.strategies.base.contracts import ModelLoadingStrategy

try:
    from ramtorch.helpers import replace_linear_with_ramtorch
except (ImportError, AssertionError):
    replace_linear_with_ramtorch = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class SdModelLoadingStrategy(ModelLoadingStrategy):
    """Model-loading facet for SD 1.5/2.0 training strategies."""

    def load_target_model(
        self, cfg: Any, weight_dtype: torch.dtype, accelerator: Any
    ) -> tuple[str, tuple[LoadedModelComponent, ...]]:
        """Load SD1.5/2 model components."""
        text_encoder, vae, unet, _ = load_target_model(cfg.model, cfg.performance.memory, weight_dtype, accelerator)

        if cfg.performance.memory.use_ramtorch:
            if replace_linear_with_ramtorch is None:
                raise ImportError("RamTorch is not available. Please install it or set use_ramtorch to False.")
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

        replace_unet_modules(
            unet, cfg.performance.attention.mem_eff_attn, cfg.performance.attention.xformers, cfg.performance.attention.sdpa
        )
        if torch.__version__ >= "2.0.0":
            vae.set_use_memory_efficient_attention_xformers(cfg.performance.attention.xformers)

        model_version = library.models.sd.conversion.get_model_version_str_for_sd1_sd2(
            cfg.model.model_type == "sd2",
            resolve_ddpm_prediction_type(cfg.objective.prediction) == DDPM_PREDICTION_TYPE_V,
        )
        loaded_components = build_loaded_components(
            cfg.model.model_type,
            {"text_encoder1": text_encoder, "vae": vae, "denoiser": unet},
        )
        return model_version, loaded_components
