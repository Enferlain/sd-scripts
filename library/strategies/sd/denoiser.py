from typing import Any

import torch

from library.strategies.base.training import DenoiserCallingStrategy


class SdDenoiserCallingStrategy(DenoiserCallingStrategy):
    """Denoiser-calling facet for SD 1.5/2.0 training strategies."""

    def call_denoiser(
        self,
        cfg: Any,
        accelerator: Any,
        denoiser: Any,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_conds: list[torch.Tensor],
        batch: Any,
        weight_dtype: torch.dtype,
        **kwargs,
    ) -> torch.Tensor:
        """Call the SD UNet through the generic denoiser seam."""
        return denoiser(noisy_latents, timesteps, text_conds[0]).sample
