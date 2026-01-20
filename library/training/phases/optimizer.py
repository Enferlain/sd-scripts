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
from library.optimizers.optimizer_utils import prepare_optimizer as _prepare_optimizer_util
from library.optimizers.scheduler import get_scheduler_fix
from library.performance import deepspeed_utils
from library.training.checkpointing import register_adapter_state_hooks, resume_from_local_or_hf_if_specified

if TYPE_CHECKING:
    from library.training.trainers.peft_trainer import PeftTrainer

from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def prepare_optimizer(trainer: PeftTrainer) -> None:
    """Phase 4: Create optimizer, LR scheduler, accelerator.prepare, and training state.

    Updates trainer.optimizer, trainer.optimizer_train_fn, trainer.optimizer_eval_fn,
    trainer.lr_descriptions, trainer.max_train_steps, trainer.lr_scheduler,
    trainer._val_dataloader, trainer._cyclic_val_dataloader, trainer.num_train_epochs, etc.

    Args:
        trainer: PeftTrainer instance
    """
    cfg = trainer.cfg

    # Create optimizer
    (
        trainer.optimizer_name,
        trainer.optimizer_args,
        trainer.optimizer,
        trainer.optimizer_train_fn,
        trainer.optimizer_eval_fn,
        trainer.lr_descriptions,
    ) = _prepare_optimizer_util(
        cfg.optimizer,
        cfg.optimizer.learning_rates,
        cfg.peft,
        trainer.adapter,
    )

    # Get training flags
    trainer._train_unet = trainer.strategies.is_train_unet(cfg)
    trainer._train_text_encoder = trainer.strategies.is_train_text_encoder(cfg)

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
            latent_strategy=trainer.latent_strategy,
            te_strategy=trainer.te_strategy,
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
        trainer.accelerator.print(
            f"override steps. steps for {cfg.training.max_train_epochs} epochs is / ステップ数: {trainer.max_train_steps}"
        )
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

    # Configure precision (from model_prep phase, but needs to happen after optimizer)
    from library.training.phases.model_prep import configure_precision as _configure_precision

    _configure_precision(trainer)

    # Accelerator.prepare - handles distributed training setup
    _prepare_with_accelerator(trainer)

    # Gradient checkpointing setup
    _setup_gradient_checkpointing(trainer)

    # Calculate number of epochs and setup epoch-related config
    num_update_steps_per_epoch = math.ceil(trainer.num_batches_per_epoch / cfg.training.gradient_accumulation_steps)
    trainer.num_train_epochs = math.ceil(trainer.max_train_steps / num_update_steps_per_epoch)

    if cfg.output.saving.save_n_epoch_ratio is not None and cfg.output.saving.save_n_epoch_ratio > 0:
        cfg.output.saving.save_every_n_epochs = math.floor(trainer.num_train_epochs / cfg.output.saving.save_n_epoch_ratio) or 1

    # FP16 training patch
    if cfg.performance.precision.full_fp16:
        patch_accelerator_for_fp16_training(trainer.accelerator)

    # Register adapter state hooks for checkpointing
    get_steps_from_state = register_adapter_state_hooks(
        trainer.accelerator, trainer.adapter, cfg, trainer._current_epoch_state, trainer._current_step_state
    )

    # Resume from checkpoint if specified
    resume_from_local_or_hf_if_specified(trainer.accelerator, cfg.output.saving, cfg.output.huggingface)
    steps_from_state = get_steps_from_state()

    # Calculate initial step for resuming
    trainer._initial_step = 0
    trainer.epoch_to_start = 0
    if steps_from_state is not None:
        trainer._initial_step = steps_from_state
        trainer.epoch_to_start = trainer._initial_step // trainer.num_batches_per_epoch

    trainer.global_step = 0
    if trainer._initial_step > 0:
        trainer.global_step = trainer._initial_step


def _prepare_with_accelerator(trainer: PeftTrainer) -> None:
    """Handle accelerator.prepare for optimizer, adapter, scheduler, and models."""
    cfg = trainer.cfg

    if cfg.performance.deepspeed:
        flags = trainer.strategies.get_text_encoders_train_flags(cfg, trainer.text_encoders)
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            cfg.performance.precision,
            unet=trainer.unet if trainer._train_unet else None,
            text_encoder1=trainer.text_encoders[0] if flags[0] else None,
            text_encoder2=(trainer.text_encoders[1] if flags[1] else None) if len(trainer.text_encoders) > 1 else None,
            adapter=trainer.adapter,
        )
        ds_model, trainer.optimizer, trainer.lr_scheduler = trainer.accelerator.prepare(ds_model, trainer.optimizer, trainer.lr_scheduler)
        trainer._training_model = ds_model
    else:
        if trainer._train_unet:
            trainer.unet = trainer.strategies.prepare_unet_with_accelerator(cfg, trainer.accelerator, trainer.unet)
        else:
            trainer.unet.to(trainer.accelerator.device, dtype=trainer.unet_weight_dtype if trainer.strategies.cast_unet(cfg) else None)

        if trainer._train_text_encoder:
            trainer.text_encoders = [
                (trainer.accelerator.prepare(t_enc) if flag else t_enc)
                for t_enc, flag in zip(trainer.text_encoders, trainer.strategies.get_text_encoders_train_flags(cfg, trainer.text_encoders))
            ]
            trainer._text_encoder = trainer.text_encoders if len(trainer.text_encoders) > 1 else trainer.text_encoders[0]

        trainer.adapter, trainer.optimizer, trainer.lr_scheduler = trainer.accelerator.prepare(
            trainer.adapter, trainer.optimizer, trainer.lr_scheduler
        )
        trainer._training_model = trainer.adapter


def _setup_gradient_checkpointing(trainer: PeftTrainer) -> None:
    """Setup gradient checkpointing for models if enabled."""
    cfg = trainer.cfg

    if cfg.performance.memory.gradient_checkpointing:
        if cfg.performance.memory.cpu_offload_checkpointing:
            trainer.unet.enable_gradient_checkpointing(cpu_offload=True)
        else:
            trainer.unet.enable_gradient_checkpointing()

        for t_enc, flag in zip(trainer.text_encoders, trainer.strategies.get_text_encoders_train_flags(cfg, trainer.text_encoders)):
            if flag and t_enc.supports_gradient_checkpointing:
                t_enc.gradient_checkpointing_enable()

        trainer.adapter.enable_gradient_checkpointing()

        # Train mode for gradient checkpointing
        trainer.unet.train()
        for i, (t_enc, flag) in enumerate(
            zip(trainer.text_encoders, trainer.strategies.get_text_encoders_train_flags(cfg, trainer.text_encoders))
        ):
            t_enc.train()
            if flag:
                trainer.strategies.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)
    else:
        trainer.unet.eval()
        for t_enc in trainer.text_encoders:
            t_enc.eval()

    # Prepare grad etc
    trainer.accelerator.unwrap_model(trainer.adapter).prepare_grad_etc(trainer._text_encoder, trainer.unet)

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
