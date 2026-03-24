from typing import Any

import torch

from library.models.sdxl.conversion import get_size_embeddings
from library.strategies.base.contracts import DenoiserCallingStrategy
from library.strategies.sdxl.caching import SdxlConditioning


class SdxlDenoiserCallingStrategy(DenoiserCallingStrategy):
    """Denoiser-calling facet for SDXL training strategies."""

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
        """
        Call the SDXL UNet with micro-conditioning and text-conditioning.

        Args:
            cfg: Configuration object.
            accelerator: Accelerator instance.
            denoiser: Denoiser model.
            noisy_latents: Noisy latents tensor.
            timesteps: Timesteps tensor.
            text_conds: Tuple of text conditioning (encoder_hidden_states1, encoder_hidden_states2, pool2).
            batch: Batch data.
            weight_dtype: Weight data type.
            **kwargs: Additional arguments.

        Returns:
            Noise prediction tensor.
        """
        indices = kwargs.get("indices")

        conditionings = batch["conditionings"]
        orig_size, crop_size, target_size = self._extract_conditioning_tensors(conditionings, accelerator.device, weight_dtype)
        embs = get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds

        if pool2.shape[0] != embs.shape[0]:
            raise RuntimeError(
                f"Batch size mismatch in call_denoiser: pool2 has {pool2.shape[0]} samples, "
                f"but conditionings has {len(conditionings)} items (embs shape: {embs.shape}). "
                f"batch latents shape: {batch['latents'].shape if 'latents' in batch else 'N/A'}, "
                f"captions: {len(batch.get('captions', []))}"
            )

        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        if indices is not None and len(indices) > 0:
            noisy_latents = noisy_latents[indices]
            timesteps = timesteps[indices]
            text_embedding = text_embedding[indices]
            vector_embedding = vector_embedding[indices]

        noise_pred = denoiser(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def _extract_conditioning_tensors(
        self,
        conditionings: list[SdxlConditioning],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract SDXL micro-conditioning tensors from batch conditionings.

        Args:
            conditionings: List of SdxlConditioning objects from batch.
            device: Target device for tensors.
            dtype: Target dtype for tensors.

        Returns:
            Tuple of (original_sizes, crop_top_lefts, target_sizes) tensors.
        """
        orig_sizes = []
        crop_top_lefts = []
        target_sizes = []

        for cond in conditionings:
            orig_sizes.append(cond.original_size_hw)
            crop_top_lefts.append(cond.crop_top_left)
            target_sizes.append(cond.target_size_hw)

        return (
            torch.tensor(orig_sizes, device=device, dtype=dtype),
            torch.tensor(crop_top_lefts, device=device, dtype=dtype),
            torch.tensor(target_sizes, device=device, dtype=dtype),
        )
