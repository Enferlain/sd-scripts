"""
Caching Phase - Latent and Text Encoder caching orchestration.

These functions handle the caching workflow, delegating actual
encoding to CachingEngine and model-specific strategies.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

from library.data import CachingEngine
from library.strategies.sdxl.caching import SdxlLatentsPipelineStrategy, SdxlTextEncoderPipelineStrategy
from library.utils.device_utils import clean_memory_on_device

if TYPE_CHECKING:
    from accelerate import Accelerator
    from library.data import DatasetManifest

logger = logging.getLogger(__name__)


def run_latent_caching(
    train_manifest: DatasetManifest,
    val_manifest: DatasetManifest | None,
    vae: Any,
    accelerator: Accelerator,
    cache_dir: str | None,
    flip_aug: bool,
    vae_batch_size: int,
    num_workers: int,
    latent_dtype: str,
    vae_dtype: torch.dtype,
) -> tuple[DatasetManifest, DatasetManifest | None, SdxlLatentsPipelineStrategy]:
    """
    Cache VAE latents for the dataset.

    Args:
        train_manifest: Training dataset manifest to cache
        val_manifest: Validation dataset manifest (or None)
        vae: VAE model for encoding
        accelerator: Accelerator instance
        cache_dir: Directory to store cache files
        flip_aug: Whether to cache flipped augmentations
        vae_batch_size: Batch size for VAE encoding
        num_workers: Number of worker threads
        latent_dtype: Data type for latents ("fp16" or "fp32")
        vae_dtype: Torch dtype for VAE computation

    Returns:
        Tuple of (train_manifest, val_manifest, latent_strategy)
    """
    latent_strategy = SdxlLatentsPipelineStrategy(
        flip_aug=flip_aug,
        dtype=latent_dtype,
    )

    vae.to(accelerator.device, dtype=vae_dtype)
    vae.requires_grad_(False)
    vae.eval()

    latent_caching_engine = CachingEngine(
        strategy=latent_strategy,
        batch_size=vae_batch_size,
        num_workers=num_workers,
    )

    # RESOURCE TRACKER START
    resource_tracker = None
    if os.environ.get("BENCHMARK_RESOURCES", "").lower() in ("1", "true", "yes"):
        from library.utils.resource_tracker import ResourceTracker

        resource_tracker = ResourceTracker("Latent Caching")
        resource_tracker.start()
    # RESOURCE TRACKER END

    train_manifest = latent_caching_engine.cache_dataset(
        manifest=train_manifest,
        model=vae,
        accelerator=accelerator,
        cache_dir=cache_dir,
        flip_aug=flip_aug,
        cache_type="Latent Caching",
    )
    if val_manifest is not None:
        val_manifest = latent_caching_engine.cache_dataset(
            manifest=val_manifest,
            model=vae,
            accelerator=accelerator,
            cache_dir=cache_dir,
            flip_aug=False,  # No flip aug for validation
            cache_type="Latent Caching",
        )

    # RESOURCE TRACKER END
    if resource_tracker:
        stats = resource_tracker.stop()
        logger.info(f"\n{stats.summary()}")

    vae.to("cpu")
    clean_memory_on_device(accelerator.device)
    accelerator.wait_for_everyone()

    return train_manifest, val_manifest, latent_strategy


def run_te_caching(
    train_manifest: DatasetManifest,
    val_manifest: DatasetManifest | None,
    text_encoders: list[Any],
    tokenizers: list[Any],
    accelerator: Accelerator,
    cache_dir: str | None,
    max_token_length: int | None,
    te_batch_size: int | None,
    cache_to_disk: bool,
) -> tuple[DatasetManifest, DatasetManifest | None, Any]:
    """
    Cache text encoder outputs for the dataset.

    Args:
        train_manifest: Training dataset manifest to cache
        val_manifest: Validation dataset manifest (or None)
        text_encoders: List of text encoder models
        tokenizers: List of tokenizer instances
        accelerator: Accelerator instance
        cache_dir: Directory to store cache files
        max_token_length: Maximum token length
        te_batch_size: Batch size for TE encoding
        cache_to_disk: If True, save to disk; if False, store in memory

    Returns:
        Tuple of (train_manifest, val_manifest, te_strategy)
    """
    # Move text encoders to GPU for caching
    for t_enc in text_encoders:
        t_enc.to(accelerator.device)
        t_enc.requires_grad_(False)
        t_enc.eval()

    te_strategy = None

    if cache_to_disk:
        # Disk-based TE caching: use CachingEngine
        te_strategy = SdxlTextEncoderPipelineStrategy(
            max_token_length=max_token_length,
        )
        te_caching_engine = CachingEngine(
            strategy=te_strategy,
            batch_size=te_batch_size,
        )

        # RESOURCE TRACKER START
        te_resource_tracker = None
        if os.environ.get("BENCHMARK_RESOURCES", "").lower() in ("1", "true", "yes"):
            from library.utils.resource_tracker import ResourceTracker

            te_resource_tracker = ResourceTracker("TE Caching")
            te_resource_tracker.start()
        # RESOURCE TRACKER END

        train_manifest = te_caching_engine.cache_dataset(
            manifest=train_manifest,
            model=(*text_encoders, *tokenizers),  # SDXL: (clip_l_enc, clip_g_enc, clip_l_tok, clip_g_tok)
            accelerator=accelerator,
            cache_dir=cache_dir,
            cache_type="TE Caching",
        )
        if val_manifest is not None:
            val_manifest = te_caching_engine.cache_dataset(
                manifest=val_manifest,
                model=(*text_encoders, *tokenizers),
                accelerator=accelerator,
                cache_dir=cache_dir,
                cache_type="TE Caching",
            )

        # RESOURCE TRACKER END
        if te_resource_tracker:
            stats = te_resource_tracker.stop()
            logger.info(f"\n{stats.summary()}")

    else:
        # In-memory TE caching: compute and store in entry.te_outputs
        from library.strategies.sdxl.training import tokenize_sdxl_captions
        from library.models.sdxl.text_encoder import get_hidden_states_sdxl

        def _cache_te_in_memory(manifest: DatasetManifest, desc: str) -> None:
            """Cache TE outputs in memory for a manifest."""
            for entry in tqdm(
                manifest.entries.values(),
                desc=desc,
                disable=accelerator.process_index != 0,
            ):
                input_ids1, input_ids2 = tokenize_sdxl_captions(tokenizers[0], tokenizers[1], [entry.caption], max_token_length)
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)

                with torch.no_grad():
                    hidden_state1, hidden_state2, pool2 = get_hidden_states_sdxl(
                        max_token_length,
                        input_ids1,
                        input_ids2,
                        tokenizers[0],
                        tokenizers[1],
                        text_encoders[0],
                        text_encoders[1],
                    )
                    entry.te_outputs = {
                        "hidden_state1": hidden_state1.squeeze(0).cpu(),
                        "hidden_state2": hidden_state2.squeeze(0).cpu(),
                        "pool2": pool2.squeeze(0).cpu(),
                    }

        logger.info("Computing text encoder outputs in memory...")
        _cache_te_in_memory(train_manifest, "TE caching (memory)")
        if val_manifest is not None:
            _cache_te_in_memory(val_manifest, "TE caching val (memory)")

    # Move text encoders back to CPU to save VRAM
    for t_enc in text_encoders:
        t_enc.to("cpu")
    clean_memory_on_device(accelerator.device)
    accelerator.wait_for_everyone()

    return train_manifest, val_manifest, te_strategy
