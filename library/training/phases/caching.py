"""
Caching Phase - Latent and Text Encoder caching orchestration.

These functions handle the caching workflow, delegating actual
encoding to CachingEngine and model-specific strategies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator
    from library.data import DatasetManifest

logger = logging.getLogger(__name__)


def run_latent_caching(
    manifest: DatasetManifest,
    vae: Any,
    accelerator: Accelerator,
    cache_dir: str,
    flip_aug: bool,
    vae_batch_size: int,
    num_workers: int,
    latent_dtype: str,
) -> DatasetManifest:
    """
    Cache VAE latents for the dataset.

    Args:
        manifest: Dataset manifest to cache
        vae: VAE model for encoding
        accelerator: Accelerator instance
        cache_dir: Directory to store cache files
        flip_aug: Whether to cache flipped augmentations
        vae_batch_size: Batch size for VAE encoding
        num_workers: Number of worker threads
        latent_dtype: Data type for latents ("fp16" or "fp32")

    Returns:
        Updated manifest with cache paths
    """
    # TODO: Extract from sdxl_peft.py lines ~225-275
    raise NotImplementedError("run_latent_caching() not yet implemented")


def run_te_caching(
    manifest: DatasetManifest,
    text_encoders: list[Any],
    tokenizers: list[Any],
    accelerator: Accelerator,
    cache_dir: str,
    max_token_length: int,
    te_batch_size: int,
    cache_to_disk: bool,
) -> DatasetManifest:
    """
    Cache text encoder outputs for the dataset.

    Args:
        manifest: Dataset manifest to cache
        text_encoders: List of text encoder models
        tokenizers: List of tokenizer instances
        accelerator: Accelerator instance
        cache_dir: Directory to store cache files
        max_token_length: Maximum token length
        te_batch_size: Batch size for TE encoding
        cache_to_disk: If True, save to disk; if False, store in memory

    Returns:
        Updated manifest with TE outputs (on disk or in memory)
    """
    # TODO: Extract from sdxl_peft.py lines ~285-395
    raise NotImplementedError("run_te_caching() not yet implemented")
