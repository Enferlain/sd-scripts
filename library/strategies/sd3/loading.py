import logging
from typing import Any

import torch
from torch import nn

from library.models.sd3.loader import load_target_model as load_sd3_target_model
from library.strategies.base.contracts import ModelLoadingStrategy

try:
    from ramtorch.helpers import replace_linear_with_ramtorch
except (ImportError, AssertionError):
    replace_linear_with_ramtorch = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class Sd3ModelLoadingStrategy(ModelLoadingStrategy):
    """Model-loading facet for SD3 training strategies."""

    def load_target_model(
        self,
        cfg: Any,
        weight_dtype: torch.dtype,
        accelerator: Any,
    ) -> tuple[str, list[nn.Module | None], nn.Module, nn.Module | None]:
        """Load SD3 text encoders, VAE, and MMDiT."""
        model_version, text_encoders, vae, denoiser = load_sd3_target_model(
            cfg.model,
            cfg.performance.memory,
            cfg.data.caching,
            cfg.performance.precision,
            accelerator,
            weight_dtype,
            resolutions=getattr(self, "resolutions", None),
        )
        self._model_version = model_version

        if cfg.performance.memory.use_ramtorch:
            if replace_linear_with_ramtorch is None:
                raise ImportError("RamTorch is not available. Please install it or set use_ramtorch to False.")

            logger.info("Applying RamTorch to SD3 MMDiT, VAE, and text encoders.")
            if isinstance(denoiser, torch.nn.Module):
                denoiser = replace_linear_with_ramtorch(denoiser, accelerator.device)
                logger.info("RamTorch applied to SD3 MMDiT.")

            if isinstance(vae, torch.nn.Module):
                vae = replace_linear_with_ramtorch(vae, accelerator.device)
                logger.info("RamTorch applied to SD3 VAE.")

            for index, text_encoder in enumerate(text_encoders):
                if not isinstance(text_encoder, torch.nn.Module):
                    continue
                text_encoders[index] = replace_linear_with_ramtorch(text_encoder, accelerator.device)
                logger.info("RamTorch applied to SD3 text encoder index %s.", index)

        return model_version, text_encoders, vae, denoiser


__all__ = ["Sd3ModelLoadingStrategy"]
