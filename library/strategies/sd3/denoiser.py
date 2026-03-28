from typing import Any

import torch

from library.strategies.base.contracts import DenoiserCallingStrategy
from library.strategies.sd3.encoding import concat_sd3_encodings


class Sd3DenoiserCallingStrategy(DenoiserCallingStrategy):
    """Denoiser-calling facet for SD3 MMDiT training."""

    def call_denoiser(
        self,
        cfg: Any,
        accelerator: Any,
        denoiser: Any,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_conds: Any,
        batch: Any,
        weight_dtype: torch.dtype,
        **kwargs,
    ) -> torch.Tensor:
        """Call the SD3 MMDiT using concatenated CLIP/T5 conditioning."""
        del cfg, accelerator, batch, weight_dtype

        indices = kwargs.get("indices")
        lg_out, t5_out, lg_pooled, _l_attn_mask, _g_attn_mask, _t5_attn_mask = text_conds
        context, pooled = concat_sd3_encodings(lg_out, t5_out, lg_pooled)

        if indices is not None and len(indices) > 0:
            noisy_latents = noisy_latents[indices]
            timesteps = timesteps[indices]
            context = context[indices]
            pooled = pooled[indices]

        return denoiser(noisy_latents, timesteps, context=context, y=pooled)


__all__ = ["Sd3DenoiserCallingStrategy"]
