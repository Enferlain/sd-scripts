import torch

from library.strategies.base.context import DenoiserContext, StrategyContext, TrainingContext, publish_strategy_context
from library.strategies.base.contracts import DenoiserCallingStrategy
from library.strategies.sd3.encoding import Sd3TextConditioning, concat_sd3_encodings


class Sd3DenoiserCallingStrategy(DenoiserCallingStrategy):
    """Denoiser-calling facet for SD3 MMDiT training."""

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
        """Call the SD3 MMDiT using concatenated CLIP/T5 conditioning."""
        del batch, weight_dtype
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

        if isinstance(text_conds, list):
            text_conds = Sd3TextConditioning.from_tensor_list(text_conds)

        context, pooled = concat_sd3_encodings(text_conds)
        grad_enabled = is_train if enable_grad is None else enable_grad
        model_input = noisy_latents.requires_grad_(train_denoiser)

        with publish_strategy_context(strategy_context), torch.set_grad_enabled(grad_enabled), accelerator.autocast():
            if index_list is not None:
                model_input = model_input[index_list]
                timesteps = timesteps[index_list]
                context = context[index_list]
                pooled = pooled[index_list]

            return denoiser(model_input, timesteps, context=context, y=pooled)


__all__ = ["Sd3DenoiserCallingStrategy"]
