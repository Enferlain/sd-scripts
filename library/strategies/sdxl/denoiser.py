import torch

from library.models.sdxl.conversion import get_size_embeddings
from library.strategies.base.context import DenoiserContext, StrategyContext, TrainingContext, publish_strategy_context
from library.strategies.base.contracts import DenoiserCallingStrategy
from library.strategies.sdxl.conditioning import SdxlConditioning


class SdxlDenoiserCallingStrategy(DenoiserCallingStrategy):
    """Denoiser-calling facet for SDXL training strategies."""

    def call_denoiser(
        self,
        cfg,
        accelerator,
        denoiser,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
        *,
        phase,
        global_step,
        is_train,
        train_denoiser=True,
        sample_indices=None,
        enable_grad=None,
    ) -> torch.Tensor:
        """
        Call the SDXL UNet with micro-conditioning and text-conditioning.

        Returns:
            Noise prediction tensor.
        """
        index_list: list[int] | None = None
        published_timesteps = timesteps
        batch_size = int(noisy_latents.shape[0]) if noisy_latents.ndim > 0 else None

        if sample_indices is not None and len(sample_indices) > 0:
            index_list = list(sample_indices)
            published_timesteps = timesteps[index_list]
            batch_size = len(index_list)

        strategy_context = StrategyContext(
            phase=phase,
            model_family=getattr(getattr(cfg, "model", None), "model_type", None),
            training=TrainingContext(global_step=global_step, is_train=is_train),
            denoiser=DenoiserContext(
                timesteps=published_timesteps,
                sample_indices=sample_indices,
                batch_size=batch_size,
            ),
        )

        conditionings = batch["conditionings"]
        orig_size, crop_size, target_size = self._extract_conditioning_tensors(
            conditionings,
            accelerator.device,
            weight_dtype,
        )
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
        grad_enabled = is_train if enable_grad is None else enable_grad
        model_input = noisy_latents.requires_grad_(train_denoiser)

        with publish_strategy_context(strategy_context), torch.set_grad_enabled(grad_enabled), accelerator.autocast():
            if index_list is not None:
                model_input = model_input[index_list]
                timesteps = timesteps[index_list]
                text_embedding = text_embedding[index_list]
                vector_embedding = vector_embedding[index_list]

            return denoiser(model_input, timesteps, text_embedding, vector_embedding)

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
