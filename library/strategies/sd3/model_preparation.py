from typing import Any

import torch

from library.optimizers.optimizer_utils import get_text_encoders_train_flags
from library.strategies.base.contracts import ModelPreparationStrategy
from library.strategies.shared.clip.model_preparation import (
    prepare_clip_text_encoder_fp8,
    prepare_clip_text_encoder_grad_ckpt_workaround,
)


class Sd3ModelPreparationStrategy(ModelPreparationStrategy):
    """Model-preparation facet for SD3 training strategies."""

    train_clip: bool = False
    train_t5xxl: bool = False

    def prepare_text_encoder_grad_ckpt_workaround(self, index: int, text_encoder: Any) -> None:
        """Enable grad on embedding layers for CLIP via the shared helper and keep T5 local."""
        if index in (0, 1):
            prepare_clip_text_encoder_grad_ckpt_workaround(text_encoder)
            return

        if hasattr(text_encoder, "encoder") and hasattr(text_encoder.encoder, "embed_tokens"):
            text_encoder.encoder.embed_tokens.requires_grad_(True)
        if hasattr(text_encoder, "shared"):
            text_encoder.shared.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index: int, text_encoder: Any, te_weight_dtype: torch.dtype, weight_dtype: torch.dtype) -> None:
        """Restore CLIP embeddings from FP8 via the shared helper; T5 stays unsupported for now."""
        del te_weight_dtype
        if index in (0, 1):
            prepare_clip_text_encoder_fp8(text_encoder, weight_dtype)
            return

        raise NotImplementedError("SD3 T5-XXL FP8 preparation is not implemented in the active strategy path yet")

    def cast_text_encoder(self, cfg: Any) -> bool:
        """SD3 casts text encoders during shared precision setup."""
        del cfg
        return True

    def cast_vae(self, cfg: Any) -> bool:
        """SD3 casts the VAE during trainer setup."""
        del cfg
        return True

    def cast_denoiser(self, cfg: Any) -> bool:
        """SD3 casts the denoiser during shared precision setup."""
        del cfg
        return True

    def post_process_trainable(
        self,
        cfg: Any,
        accelerator: Any,
        trainable_model: Any,
        text_encoders: list[Any],
        denoiser: Any,
    ) -> None:
        """Capture SD3 text-encoder train flags and guard unsupported cache combinations."""
        del accelerator, denoiser

        te_train_flags = get_text_encoders_train_flags(cfg.optimizer.learning_rates, text_encoders)
        self.train_clip = any(te_train_flags[:2])
        self.train_t5xxl = bool(getattr(trainable_model, "train_t5xxl", False))

        if self.train_t5xxl and cfg.data.caching.cache_text_encoder_outputs:
            raise ValueError("SD3 T5-XXL training is incompatible with cache_text_encoder_outputs in the current strategy path")


__all__ = ["Sd3ModelPreparationStrategy"]
