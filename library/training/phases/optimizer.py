"""
Optimizer Phase - Optimizer and scheduler setup.

These functions handle optimizer creation, LR scheduler configuration,
and training step calculations.
"""

from __future__ import annotations

import itertools
import logging
import math
import os
from typing import TYPE_CHECKING

from library.data import create_training_dataloader, prepare_validation_epoch
from library.models.runtime_utils import patch_accelerator_for_fp16_training
from library.optimization.optimizer_utils import get_text_encoders_train_flags
from library.optimization.scheduler import get_scheduler_fix
from library.training.checkpointing import resume_from_local_or_hf_if_specified

if TYPE_CHECKING:
    from library.training.runners.trainer import Trainer


logger = logging.getLogger(__name__)


def prepare_optimizer(trainer: Trainer) -> None:
    """Phase 4: Create optimizer, LR scheduler, accelerator.prepare, and training state.

    Updates trainer.optimizer, trainer.optimizer_train_fn, trainer.optimizer_eval_fn,
    trainer.lr_descriptions, trainer.max_train_steps, trainer.lr_scheduler,
    trainer._val_dataloader, trainer._cyclic_val_dataloader, trainer.num_train_epochs, etc.

    Args:
        trainer: Trainer instance
    """
    cfg = trainer.cfg

    # Create optimizer (delegated to mode)
    (
        trainer.optimizer_name,
        trainer.optimizer_args,
        trainer.optimizer,
        trainer.optimizer_train_fn,
        trainer.optimizer_eval_fn,
        trainer.lr_descriptions,
    ) = trainer.mode.build_optimizer_params(trainer)

    # NOTE: trainer._train_denoiser and trainer._train_text_encoder are set in
    # prepare_models() -> create_adapter() as single source of truth

    # Create validation dataloader (once, deterministic)
    trainer._n_workers = min(cfg.data.loader.num_workers, os.cpu_count() or 1)
    trainer._val_dataloader = None
    trainer._cyclic_val_dataloader = None

    if trainer.val_manifest is not None:
        val_epoch_manifest = prepare_validation_epoch(
            manifest=trainer.val_manifest,
            batch_size=cfg.training.train_batch_size,
            seed=cfg.validation.validation_seed,
        )
        trainer._val_dataloader = create_training_dataloader(
            dataset_manifest=trainer.val_manifest,
            epoch_manifest=val_epoch_manifest,
            latent_cache_backend=trainer.latent_cache_backend,
            te_cache_backend=trainer.te_cache_backend,
            flip_aug=False,  # No flip aug for validation
            prior_loss_weight=cfg.loss.prior_loss_weight,
            rank=trainer.accelerator.process_index,
            world_size=trainer.accelerator.num_processes,
            num_workers=trainer._n_workers,
            prefetch_factor=cfg.data.loader.prefetch_factor,
            pin_memory=cfg.data.loader.pin_memory,
            persistent_workers=cfg.data.loader.persistent_workers,
        )
        trainer._cyclic_val_dataloader = itertools.cycle(trainer._val_dataloader)

    # Calculate training steps if epochs specified
    if cfg.training.max_train_epochs is not None:
        trainer.max_train_steps = calculate_max_train_steps(
            max_train_epochs=cfg.training.max_train_epochs,
            num_batches_per_epoch=trainer.num_batches_per_epoch,
            num_processes=trainer.accelerator.num_processes,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        )
        trainer.accelerator.print(f"override steps. steps for {cfg.training.max_train_epochs} epochs is: {trainer.max_train_steps}")
    else:
        trainer.max_train_steps = cfg.training.max_train_steps

    # Create LR scheduler
    trainer.lr_scheduler = get_scheduler_fix(
        cfg.optimizer.scheduler,
        cfg.optimizer,
        cfg.training,
        trainer.optimizer,
        trainer.accelerator.num_processes,
    )

    # Accelerator.prepare - handles distributed training setup (delegated to mode)
    trainer.mode.prepare_with_accelerator(trainer)

    # Gradient checkpointing setup (shared denoiser/TE parts + mode-specific adapter parts)
    # NOTE: This happens AFTER accelerator.prepare(), matching legacy behavior.
    # Risk: DDP with cpu_offload_checkpointing=True may have issues if hooks
    # are registered after wrapping. Requires manual verification in distributed
    # environments. See AUDIT/1_phase_ordering_dependencies.md for details.
    _setup_gradient_checkpointing(trainer)

    # Calculate number of epochs and setup epoch-related config
    num_update_steps_per_epoch = math.ceil(trainer.num_batches_per_epoch / cfg.training.gradient_accumulation_steps)
    trainer.num_train_epochs = math.ceil(trainer.max_train_steps / num_update_steps_per_epoch)

    if cfg.output.saving.save_n_epoch_ratio is not None and cfg.output.saving.save_n_epoch_ratio > 0:
        cfg.output.saving.save_every_n_epochs = math.floor(trainer.num_train_epochs / cfg.output.saving.save_n_epoch_ratio) or 1

    # FP16 training patch
    if cfg.performance.precision.full_fp16:
        patch_accelerator_for_fp16_training(trainer.accelerator)

    # Register state hooks for checkpointing (delegated to mode)
    resume_state = trainer.mode.register_state_hooks(trainer)

    # Resume from checkpoint if specified
    resume_from_local_or_hf_if_specified(trainer.accelerator, cfg.output.saving, cfg.output.huggingface)

    # Calculate initial step for resuming
    trainer._initial_step = 0
    trainer.epoch_to_start = 0
    trainer.global_step = 0
    if resume_state.step is not None:
        trainer._initial_step = resume_state.step
        trainer.epoch_to_start = trainer._initial_step // trainer.num_batches_per_epoch
        trainer.global_step = resume_state.step  # Restore global_step for correct logging/checkpointing


def _setup_gradient_checkpointing(trainer: Trainer) -> None:
    """Setup gradient checkpointing for shared models + mode-specific adapter."""
    cfg = trainer.cfg
    te_train_flags = get_text_encoders_train_flags(cfg.optimizer.learning_rates, trainer.text_encoders)

    if cfg.performance.memory.gradient_checkpointing:
        if cfg.performance.memory.cpu_offload_checkpointing:
            trainer.denoiser.enable_gradient_checkpointing(cpu_offload=True)
        else:
            trainer.denoiser.enable_gradient_checkpointing()

        for t_enc, flag in zip(trainer.text_encoders, te_train_flags):
            if flag and t_enc.supports_gradient_checkpointing:
                t_enc.gradient_checkpointing_enable()

        # Train mode for gradient checkpointing
        trainer.denoiser.train()
        for i, (t_enc, flag) in enumerate(zip(trainer.text_encoders, te_train_flags)):
            t_enc.train()
            if flag:
                trainer.strategies.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)
    else:
        trainer.denoiser.eval()
        for t_enc in trainer.text_encoders:
            t_enc.eval()

    # Mode-specific: adapter gradient checkpointing + prepare_grad_etc
    trainer.mode.setup_gradient_training(trainer)

    # VAE setup if not caching latents
    if not trainer._cache_latents:
        trainer.vae.requires_grad_(False)
        trainer.vae.eval()
        trainer.vae.to(trainer.accelerator.device, dtype=trainer.vae_dtype)


def calculate_max_train_steps(
    max_train_epochs: int,
    num_batches_per_epoch: int,
    num_processes: int,
    gradient_accumulation_steps: int,
) -> int:
    """Calculate the total number of training steps from epoch count.

    Args:
        max_train_epochs: Number of epochs to train
        num_batches_per_epoch: Batches per epoch (based on dataset size / batch_size)
        num_processes: Number of distributed processes (accelerator.num_processes)
        gradient_accumulation_steps: Gradient accumulation steps

    Returns:
        Total number of optimization steps
    """
    return max_train_epochs * math.ceil(num_batches_per_epoch / num_processes / gradient_accumulation_steps)
