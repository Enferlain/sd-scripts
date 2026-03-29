from typing import Any

import torch

from library.strategies.base.contracts import ModelPreparationStrategy
from library.strategies.shared.clip.model_preparation import (
    prepare_clip_text_encoder_fp8,
    prepare_clip_text_encoder_grad_ckpt_workaround,
)


class SdxlModelPreparationStrategy(ModelPreparationStrategy):
    """Model-preparation facet for SDXL training strategies."""

    def prepare_text_encoder_grad_ckpt_workaround(self, index: int, text_encoder: Any) -> None:
        """Enable grad on CLIP embeddings so gradient checkpointing works."""
        del index
        prepare_clip_text_encoder_grad_ckpt_workaround(text_encoder)

    def prepare_text_encoder_fp8(self, index: int, text_encoder: Any, te_weight_dtype: torch.dtype, weight_dtype: torch.dtype) -> None:
        """Cast CLIP embeddings back from FP8 because ``nn.Embedding`` does not support it."""
        del index, te_weight_dtype
        prepare_clip_text_encoder_fp8(text_encoder, weight_dtype)

    def cast_text_encoder(self, cfg: Any) -> bool:
        """SDXL casts text encoders during shared precision setup."""
        return True

    def cast_vae(self, cfg: Any) -> bool:
        """SDXL casts the VAE during trainer setup."""
        return True

    def cast_denoiser(self, cfg: Any) -> bool:
        """SDXL casts the denoiser during shared precision setup."""
        return True

    def post_process_trainable(self, cfg: Any, accelerator: Any, trainable_model: Any, text_encoders: list[Any], unet: Any) -> None:
        """Freeze the unstable tail of SDXL TE1 when that encoder is trainable."""
        if len(text_encoders) < 1:
            return

        te1 = text_encoders[0]

        if not any(p.requires_grad for p in te1.parameters()):
            return

        if hasattr(te1, "text_model"):
            text_model = te1.text_model
            if hasattr(text_model, "encoder") and hasattr(text_model.encoder, "layers"):
                last_layer = text_model.encoder.layers[-1]
                last_layer.requires_grad_(False)
            if hasattr(text_model, "final_layer_norm"):
                text_model.final_layer_norm.requires_grad_(False)
