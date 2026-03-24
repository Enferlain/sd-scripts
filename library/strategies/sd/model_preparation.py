from typing import Any

import torch

from library.strategies.base.contracts import ModelPreparationStrategy


class SdModelPreparationStrategy(ModelPreparationStrategy):
    """Model-preparation facet for SD 1.5/2.0 training strategies."""

    def prepare_text_encoder_grad_ckpt_workaround(self, index: int, text_encoder: Any) -> None:
        """Enable grad on CLIP embeddings so gradient checkpointing works."""
        text_encoder.text_model.embeddings.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index: int, text_encoder: Any, te_weight_dtype: torch.dtype, weight_dtype: torch.dtype) -> None:
        """Cast CLIP embeddings back from FP8; nn.Embedding does not support FP8."""
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)

    def cast_text_encoder(self, cfg: Any) -> bool:
        """SD casts text encoders during shared precision setup."""
        return True

    def cast_vae(self, cfg: Any) -> bool:
        """SD casts the VAE during trainer setup."""
        return True

    def cast_denoiser(self, cfg: Any) -> bool:
        """SD casts the denoiser during shared precision setup."""
        return True

    def post_process_trainable(
        self,
        cfg: Any,
        accelerator: Any,
        trainable_model: Any,
        text_encoders: list[Any],
        denoiser: Any,
    ) -> None:
        """SD currently requires no post-processing after trainable setup."""
        return None
