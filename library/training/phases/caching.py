"""
Caching Phase - Latent and Text Encoder caching orchestration.

These functions handle the caching workflow, delegating actual
encoding to CachingEngine and model-specific strategies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tqdm import tqdm

from library.data import CachingEngine
from library.logging.phase_tags import PHASE_CACHE_LATENTS, PHASE_CACHE_TEXT_ENCODER
from library.training.phases.orchestration_helpers import monitored_phase
from library.utils.device_utils import clean_memory_on_device

if TYPE_CHECKING:
    from library.training.runners.trainer import Trainer


logger = logging.getLogger(__name__)


def run_caching(trainer: Trainer) -> None:
    """Phase 2: Cache latents and optionally text encoder outputs.

    Args:
        trainer: Trainer instance containing cfg, vae, accelerator, manifests, etc.
    """
    run_latent_caching(trainer)
    run_te_caching(trainer)

    # Handle TE offloading for on-the-fly encoding (when TE outputs are NOT cached)
    # If TE caching was done, TEs are already moved to CPU in run_te_caching()
    if not trainer.cfg.data.caching.cache_text_encoder_outputs and trainer.cfg.performance.memory.offload_text_encoders:
        logger.info("Offloading text encoders to CPU (on-the-fly encoding enabled)")
        for t_enc in trainer.text_encoders:
            t_enc.to("cpu")
        clean_memory_on_device(trainer.accelerator.device)


def run_latent_caching(trainer: Trainer) -> None:
    """Cache VAE latents for the dataset.

    Updates trainer.train_manifest, trainer.val_manifest, and trainer.latent_cache_backend.

    Args:
        trainer: Trainer instance
    """
    if not trainer.cfg.data.caching.cache_latents:
        return

    # Extract config values
    cache_dir = trainer.cfg.data.caching.cache_dir or trainer.cfg.data.source.train_data_dir

    trainer.latent_cache_backend = trainer.strategies.create_latent_caching_strategy(trainer.cfg)

    trainer.vae.to(trainer.accelerator.device, dtype=trainer.vae_dtype)
    trainer.vae.requires_grad_(False)
    trainer.vae.eval()

    latent_caching_engine = CachingEngine(
        backend=trainer.latent_cache_backend,
        batch_size=trainer.cfg.data.caching.vae_batch_size,
        num_workers=trainer.cfg.data.caching.num_workers,
    )

    with monitored_phase(trainer, PHASE_CACHE_LATENTS):
        trainer.train_manifest = latent_caching_engine.cache_dataset(
            manifest=trainer.train_manifest,
            model=trainer.vae,
            accelerator=trainer.accelerator,
            cache_dir=cache_dir,
            flip_aug=trainer.cfg.data.preprocessing.flip_aug,
            cache_type="Latent Caching",
        )
        if trainer.val_manifest is not None:
            trainer.val_manifest = latent_caching_engine.cache_dataset(
                manifest=trainer.val_manifest,
                model=trainer.vae,
                accelerator=trainer.accelerator,
                cache_dir=cache_dir,
                flip_aug=False,  # No flip aug for validation
                cache_type="Latent Caching",
            )
        trainer.vae.to("cpu")
        clean_memory_on_device(trainer.accelerator.device)
        trainer.accelerator.wait_for_everyone()


def run_te_caching(trainer: Trainer) -> None:
    """Cache text encoder outputs for the dataset.

    Updates trainer.train_manifest, trainer.val_manifest, and trainer.te_cache_backend.

    Args:
        trainer: Trainer instance
    """
    if not trainer.cfg.data.caching.cache_text_encoder_outputs:
        return

    cache_dir = trainer.cfg.data.caching.cache_dir or trainer.cfg.data.source.train_data_dir

    # Move text encoders to GPU for caching
    for t_enc in trainer.text_encoders:
        t_enc.to(trainer.accelerator.device)
        t_enc.requires_grad_(False)
        t_enc.eval()

    if trainer.cfg.data.caching.cache_text_encoder_outputs_to_disk:
        # Disk-based TE caching: use CachingEngine
        trainer.te_cache_backend = trainer.strategies.create_te_caching_strategy(trainer.cfg)
        te_cache_model_bundle = trainer.strategies.build_te_cache_model_bundle(
            trainer.cfg,
            trainer.accelerator,
            trainer.text_encoders,
            trainer.tokenizers,
        )
        te_caching_engine = CachingEngine(
            backend=trainer.te_cache_backend,
            batch_size=trainer.cfg.data.caching.te_batch_size,
        )

        with monitored_phase(trainer, PHASE_CACHE_TEXT_ENCODER):
            trainer.train_manifest = te_caching_engine.cache_dataset(
                manifest=trainer.train_manifest,
                model=te_cache_model_bundle,
                accelerator=trainer.accelerator,
                cache_dir=cache_dir,
                cache_type="TE Caching",
            )
            if trainer.val_manifest is not None:
                trainer.val_manifest = te_caching_engine.cache_dataset(
                    manifest=trainer.val_manifest,
                    model=te_cache_model_bundle,
                    accelerator=trainer.accelerator,
                    cache_dir=cache_dir,
                    cache_type="TE Caching",
                )
    else:
        # In-memory TE caching: compute and store in entry.te_outputs
        with monitored_phase(trainer, PHASE_CACHE_TEXT_ENCODER):

            def _cache_te_in_memory(manifest, desc: str) -> None:
                """Cache TE outputs in memory for a manifest."""
                for entry in tqdm(
                    manifest.entries.values(),
                    desc=desc,
                    disable=trainer.accelerator.process_index != 0,
                ):
                    entry.te_outputs = trainer.strategies.encode_te_outputs_in_memory(
                        text_encoders=trainer.text_encoders,
                        tokenizers=trainer.tokenizers,
                        caption=entry.caption,
                        max_token_length=trainer.cfg.training.max_token_length,
                        device=trainer.accelerator.device,
                    )

            logger.info("Computing text encoder outputs in memory...")
            _cache_te_in_memory(trainer.train_manifest, "TE caching (memory)")
            if trainer.val_manifest is not None:
                _cache_te_in_memory(trainer.val_manifest, "TE caching val (memory)")

    # Move text encoders back to CPU to save VRAM
    for t_enc in trainer.text_encoders:
        t_enc.to("cpu")
    clean_memory_on_device(trainer.accelerator.device)
    trainer.accelerator.wait_for_everyone()
