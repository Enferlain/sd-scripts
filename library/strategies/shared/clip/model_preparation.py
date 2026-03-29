"""Shared CLIP-family model-preparation helpers."""

from typing import Any

import torch


def prepare_clip_text_encoder_grad_ckpt_workaround(text_encoder: Any) -> None:
    """Enable grad on CLIP embeddings so gradient checkpointing works."""
    text_encoder.text_model.embeddings.requires_grad_(True)


def prepare_clip_text_encoder_fp8(text_encoder: Any, weight_dtype: torch.dtype) -> None:
    """Cast CLIP embeddings back from FP8 because ``nn.Embedding`` does not support it."""
    text_encoder.text_model.embeddings.to(dtype=weight_dtype)


__all__ = [
    "prepare_clip_text_encoder_fp8",
    "prepare_clip_text_encoder_grad_ckpt_workaround",
]
