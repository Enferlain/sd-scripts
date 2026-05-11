"""
Model Preparation Phase - Adapter creation and precision configuration.

These functions handle model loading, adapter creation, and precision
settings for training.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from library.training.runners.trainer import Trainer


logger = logging.getLogger(__name__)


def prepare_models(trainer: Trainer) -> None:
    """Phase 3: Create trainable model and configure precision.

    Delegates to ``trainer.mode`` for mode-specific setup. Updates
    trainer.net_kwargs, trainer.denoiser_weight_dtype, and trainer.te_weight_dtype.

    Args:
        trainer: Trainer instance
    """
    # Lazy load denoiser if it was deferred during setup (memory optimization)
    # This allows VAE/TE caching to complete before loading the large denoiser
    if trainer.denoiser is None:
        trainer.loaded_components = trainer.strategies.load_denoiser_lazily(
            trainer.cfg, trainer.weight_dtype, trainer.accelerator, trainer.loaded_components
        )
        trainer.sync_component_views()

    # Mode-specific: create and configure the trainable model (adapter)
    trainer.mode.prepare_trainables(trainer)

    # Shared precision setup (FP8, dtype computation, denoiser/TE casting)
    configure_precision(trainer)

    # Mode-specific: cast adapter, freeze base model
    trainer.mode.configure_trainable_precision(trainer)


def configure_precision(trainer: Trainer) -> None:
    """Configure shared precision settings for denoiser and text encoders.

    Mode-specific precision (adapter casting, freezing base model) is
    handled by ``trainer.mode.configure_trainable_precision()``.

    Updates trainer.denoiser_weight_dtype and trainer.te_weight_dtype.

    Args:
        trainer: Trainer instance
    """
    cfg = trainer.cfg
    denoiser = trainer.denoiser
    text_encoders = trainer.text_encoders
    weight_dtype = trainer.weight_dtype
    accelerator = trainer.accelerator
    strategies = trainer.strategies

    denoiser_weight_dtype = te_weight_dtype = weight_dtype

    # FP8 support for base model
    if cfg.performance.precision.fp8_base or cfg.performance.precision.fp8_base_unet:  # TODO unet
        assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0"
        accelerator.print("enable fp8 training for denoiser.")
        denoiser_weight_dtype = torch.float8_e4m3fn

        if not cfg.performance.precision.fp8_base_unet:  # TODO unet
            accelerator.print("enable fp8 training for Text Encoder.")
        te_weight_dtype = weight_dtype if cfg.performance.precision.fp8_base_unet else torch.float8_e4m3fn  # TODO unet

        logger.info(f"set denoiser weight dtype to {denoiser_weight_dtype}")
        denoiser.to(dtype=denoiser_weight_dtype)

    # Configure denoiser casting
    if strategies.cast_denoiser(cfg):
        denoiser.to(dtype=denoiser_weight_dtype)

    # Configure text encoder casting
    for i, t_enc in enumerate(text_encoders):
        if t_enc.device.type != "cpu" and strategies.cast_text_encoder(cfg):
            t_enc.to(dtype=te_weight_dtype)

            # nn.Embedding doesn't support FP8
            if te_weight_dtype != weight_dtype:
                strategies.prepare_text_encoder_fp8(i, t_enc, te_weight_dtype, weight_dtype)

    trainer.denoiser_weight_dtype = denoiser_weight_dtype
    trainer.te_weight_dtype = te_weight_dtype
