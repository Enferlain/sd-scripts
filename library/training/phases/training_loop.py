"""
Training Loop Phase - Main training loop execution.

This module contains the core training loop that's called by Trainer.
All state is accessed via the trainer instance.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from library.data import CaptionConfig, prepare_epoch, create_training_dataloader
from library.logging.resource_monitor import NoOpResourceMonitor
from library.logging.step_logging import generate_step_logs, step_logging
from library.logging.training_plots import save_timestep_distribution_plot
from library.losses.edm2_loss_utils import plot_edm2_loss_weighting_check, plot_edm2_loss_weighting
from library.training.checkpointing import (
    get_step_ckpt_name,
    save_and_remove_state_stepwise,
    get_remove_step_no,
    get_epoch_ckpt_name,
    get_remove_epoch_no,
    save_and_remove_state_on_epoch_end,
)
from library.training.sample_generation import sample_images_check
from library.training.trainer_utils import determine_grad_sync_context
from library.training.phases.triggers import (
    StepTriggerContext,
    compute_step_actions,
    EpochEndTriggerContext,
    compute_epoch_end_actions,
)


if TYPE_CHECKING:
    from library.training.runners.trainer import Trainer


logger = logging.getLogger(__name__)
_NOOP_RESOURCE_MONITOR = NoOpResourceMonitor()


def _resource_monitor(trainer: Trainer):
    monitor = getattr(trainer, "_resource_monitor", None)
    return monitor if monitor is not None else _NOOP_RESOURCE_MONITOR


def _save_step_checkpoint_artifacts(trainer: Trainer) -> None:
    """Save/remove step checkpoint artifacts for the current global step."""
    cfg = trainer.cfg
    accelerator = trainer.accelerator

    ckpt_name = get_step_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as, trainer.global_step)
    trainer.save_checkpoint(
        ckpt_name,
        accelerator.unwrap_model(trainer.trainable_model),
        trainer.global_step,
        trainer._current_epoch_state.value,
    )

    if cfg.loss.edm2.edm2_loss_weighting:
        loss_weights_ckpt_name = get_step_ckpt_name(
            cfg.output.saving,
            "." + cfg.output.saving.save_model_as,
            trainer.global_step,
            "_edm2_loss_weights",
        )
        trainer.save_checkpoint(
            loss_weights_ckpt_name,
            accelerator.unwrap_model(trainer._edm2_model),
            trainer.global_step,
            trainer._current_epoch_state.value,
            dtype_override=torch.float32,
        )

    if cfg.output.saving.save_state:
        save_and_remove_state_stepwise(cfg.output.saving, accelerator, trainer.global_step)

    remove_step_no = get_remove_step_no(cfg.output.saving, trainer.global_step)
    if remove_step_no is None:
        return

    remove_ckpt_name = get_step_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as, remove_step_no)
    trainer.remove_checkpoint(remove_ckpt_name)

    if cfg.loss.edm2.edm2_loss_weighting:
        remove_loss_weights_ckpt_name = get_step_ckpt_name(
            cfg.output.saving,
            "." + cfg.output.saving.save_model_as,
            remove_step_no,
            "_edm2_loss_weights",
        )
        trainer.remove_checkpoint(remove_loss_weights_ckpt_name)


def _run_step_side_effects(
    trainer: Trainer,
    *,
    step: int,
    batch: dict[str, torch.Tensor],
    num_steps_in_epoch: int,
) -> None:
    """Run validation/sampling/checkpoint side-effects for an optimization step."""
    cfg = trainer.cfg
    accelerator = trainer.accelerator
    strategies = trainer.strategies

    step_actions = compute_step_actions(
        validation_scheduler=trainer._validation_scheduler,
        sampling_config=cfg.output.sampling,
        saving_config=cfg.output.saving,
        ctx=StepTriggerContext(
            global_step=trainer.global_step,
            epoch_step=step,
            current_epoch=trainer._current_epoch_state.value,
            num_steps_in_epoch=num_steps_in_epoch,
            max_train_steps=cfg.training.max_train_steps,
            has_validation_data=trainer._val_dataloader is not None,
        ),
        # Keep explicit patch target in training_loop tests.
        sample_images_check_fn=sample_images_check,
    )
    if not step_actions.should_enter_eval_mode:
        return

    trainer.mode.set_eval(trainer)
    trainer.optimizer_eval_fn()

    if step_actions.should_sample:
        strategies.sample_images(
            accelerator,
            cfg,
            None,
            trainer.global_step,
            accelerator.device,
            trainer.vae,
            trainer.tokenizers,
            trainer._text_encoder,
            trainer.unet,
        )

    if step_actions.should_validate:
        trainer._current_val_loss, trainer._average_val_loss = strategies.calculate_val_loss(
            trainer.global_step,
            step,
            trainer.num_batches_per_epoch,
            trainer._val_loss_recorder,
            trainer._val_dataloader,
            trainer._cyclic_val_dataloader,
            trainer.trainable_model,
            trainer._tokenize_strategy,
            trainer.text_encoders,
            trainer._text_encoding_strategy,
            trainer.unet,
            trainer.vae,
            trainer.noise_scheduler,
            trainer.vae_dtype,
            trainer.weight_dtype,
            accelerator,
            cfg,
            batch,
            trainer._current_epoch_state.value,
            trainer._train_text_encoder,
        )
        accelerator.print(f"  val_loss: {trainer._current_val_loss:.4f}  (avg: {trainer._average_val_loss:.4f})")
    else:
        trainer._current_val_loss, trainer._average_val_loss = None, None

    if step_actions.should_save_step:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _save_step_checkpoint_artifacts(trainer)

    if plot_edm2_loss_weighting_check(cfg.loss.edm2, cfg.training, trainer.global_step):
        plot_edm2_loss_weighting(
            cfg.loss.edm2,
            cfg.output.saving.output_name,
            trainer.global_step,
            trainer._edm2_model,
            1000,
            accelerator.device,
        )
    trainer.optimizer_train_fn()
    trainer.mode.set_train(trainer)


def _update_live_timestep_outputs(trainer: Trainer, *, timesteps: torch.Tensor) -> None:
    """Update live plotter stream and static timestep plots on the main process."""
    if not trainer.is_main_process:
        return

    cfg = trainer.cfg
    strategies = trainer.strategies
    timesteps_np = timesteps.cpu().numpy()

    if strategies.live_plotter_process and strategies.live_plotter_process.poll() is None:
        try:
            strategies.live_plotter_process.stdin.write(f"{','.join(map(str, timesteps_np))}\n".encode())
            strategies.live_plotter_process.stdin.flush()
        except (BrokenPipeError, OSError):
            logger.error("Live plotter connection lost.")
            strategies.live_plotter_process = None

    if trainer._timestep_counts is None:
        return

    unique, counts = np.unique(timesteps_np, return_counts=True)
    trainer._timestep_counts[unique] += counts

    if (
        cfg.output.logging.log_timestep_distribution_every_n_steps
        and trainer.global_step % cfg.output.logging.log_timestep_distribution_every_n_steps == 0
    ):
        save_timestep_distribution_plot(cfg, trainer.global_step, trainer._timestep_counts, trainer._plotter_settings)


def _emit_step_tracking_logs(
    trainer: Trainer,
    *,
    timesteps: torch.Tensor,
    keys_scaled: float | None,
    mean_norm: float | None,
    maximum_norm: float | None,
) -> None:
    """Emit tracker metrics for the current optimization step when enabled."""
    cfg = trainer.cfg
    accelerator = trainer.accelerator
    strategies = trainer.strategies

    current_global_step_loss = trainer._current_global_step_loss / trainer._accumulation_counter
    if cfg.loss.edm2.edm2_loss_weighting:
        assert trainer._current_global_step_loss_scaled is not None and trainer._loss_scaled_recorder is not None
        current_global_step_loss_scaled = trainer._current_global_step_loss_scaled / trainer._accumulation_counter
        average_loss_scaled: float | None = trainer._loss_scaled_recorder.average
    else:
        current_global_step_loss_scaled = None
        average_loss_scaled = None

    logs = generate_step_logs(
        cfg,
        current_global_step_loss,
        trainer._loss_recorder.average,
        trainer.lr_scheduler,
        trainer.lr_descriptions,
        la_sampler=strategies.la_sampler,
        optimizer=trainer.optimizer,
        keys_scaled=keys_scaled,
        mean_norm=mean_norm,
        maximum_norm=maximum_norm,
        mean_grad_norm=None,
        mean_combined_norm=None,
        edm2_lr_scheduler=trainer._edm2_lr_scheduler,
        current_loss_scaled=current_global_step_loss_scaled,
        average_loss_scaled=average_loss_scaled,
        current_val_loss=trainer._current_val_loss,
        average_val_loss=trainer._average_val_loss,
        timesteps=timesteps,
    )
    log_every = cfg.output.logging.log_every_n_steps
    if trainer.global_step % log_every == 0:
        step_logging(accelerator, logs, trainer.global_step, trainer._current_epoch_state.value)


def _save_epoch_checkpoint_artifacts(trainer: Trainer) -> None:
    """Save/remove epoch checkpoint artifacts for the current epoch."""
    cfg = trainer.cfg
    accelerator = trainer.accelerator

    ckpt_name = get_epoch_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as, trainer._current_epoch_state.value)
    trainer.save_checkpoint(
        ckpt_name,
        accelerator.unwrap_model(trainer.trainable_model),
        trainer.global_step,
        trainer._current_epoch_state.value,
    )

    if cfg.loss.edm2.edm2_loss_weighting:
        loss_weights_ckpt_name = get_epoch_ckpt_name(
            cfg.output.saving,
            "." + cfg.output.saving.save_model_as,
            trainer._current_epoch_state.value,
            "_edm2_loss_weights",
        )
        trainer.save_checkpoint(
            loss_weights_ckpt_name,
            accelerator.unwrap_model(trainer._edm2_model),
            trainer.global_step,
            trainer._current_epoch_state.value,
            dtype_override=torch.float32,
        )

    remove_epoch_no = get_remove_epoch_no(cfg.output.saving, trainer._current_epoch_state.value)
    if remove_epoch_no is not None:
        remove_ckpt_name = get_epoch_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as, remove_epoch_no)
        trainer.remove_checkpoint(remove_ckpt_name)

        if cfg.loss.edm2.edm2_loss_weighting:
            remove_loss_weights_ckpt_name = get_epoch_ckpt_name(
                cfg.output.saving, "." + cfg.output.saving.save_model_as, remove_epoch_no, "_edm2_loss_weights"
            )
            trainer.remove_checkpoint(remove_loss_weights_ckpt_name)

    if cfg.output.saving.save_state:
        save_and_remove_state_on_epoch_end(cfg.output.saving, accelerator, trainer._current_epoch_state.value)


def _finalize_epoch(
    trainer: Trainer,
    *,
    tokens_path: Path | None,
    phase_name: str,
) -> None:
    """Run end-of-epoch cleanup, tracking, and epoch-end side-effects."""
    cfg = trainer.cfg
    accelerator = trainer.accelerator
    strategies = trainer.strategies
    resource_monitor = _resource_monitor(trainer)

    if tokens_path and tokens_path.exists():
        tokens_path.unlink()
        logger.debug(f"Cleaned up epoch token file: {tokens_path}")

    if trainer._is_tracking:
        logs = {"loss/epoch_average": trainer._loss_recorder.average}
        accelerator.log(logs, step=trainer.global_step)

    accelerator.wait_for_everyone()

    epoch_end_actions = compute_epoch_end_actions(
        sampling_config=cfg.output.sampling,
        saving_config=cfg.output.saving,
        ctx=EpochEndTriggerContext(
            current_epoch=trainer._current_epoch_state.value,
            num_train_epochs=trainer.num_train_epochs,
            global_step=trainer.global_step,
        ),
        # Keep explicit patch target in training_loop tests.
        sample_images_check_fn=sample_images_check,
    )
    try:
        if not epoch_end_actions.should_enter_eval_mode:
            return

        trainer.optimizer_eval_fn()
        trainer.mode.set_eval(trainer)
        if trainer.is_main_process and epoch_end_actions.should_save_epoch:
            _save_epoch_checkpoint_artifacts(trainer)

        # Preserve existing behavior: when entering epoch-end eval mode, sampling runs.
        strategies.sample_images(
            accelerator,
            cfg,
            trainer._current_epoch_state.value,
            trainer.global_step,
            accelerator.device,
            trainer.vae,
            trainer.tokenizers,
            trainer._text_encoder,
            trainer.unet,
        )
        trainer._progress_bar.unpause()
        trainer.optimizer_train_fn()
        trainer.mode.set_train(trainer)
    finally:
        resource_monitor.phase_end(phase_name)


def run_training_loop(trainer: Trainer) -> None:
    """Execute the main training loop.

    Iterates over epochs and steps, handling:
    - Per-epoch dataloader creation
    - Gradient accumulation
    - Checkpointing
    - Sample image generation
    - Validation loss calculation

    Updates trainer.global_step as training progresses.

    Args:
        trainer: Trainer instance containing all training state
    """
    # Unpack frequently used attributes for readability
    cfg = trainer.cfg
    accelerator = trainer.accelerator
    strategies = trainer.strategies

    # Mode must set _grad_sync_handle during prepare_with_accelerator;
    # if missing, grad-sync context will silently fail.
    assert trainer._grad_sync_handle is not None, (
        "trainer._grad_sync_handle must be set by mode.prepare_with_accelerator() before the training loop starts"
    )
    assert trainer.trainable_model is not None, (
        "trainer.trainable_model must be set by mode.prepare_with_accelerator() before the training loop starts"
    )
    assert trainer._validation_scheduler is not None, "trainer._validation_scheduler must be initialized before the training loop starts"

    for epoch in range(trainer.epoch_to_start, trainer.num_train_epochs):
        if trainer.global_step >= cfg.training.max_train_steps:
            break

        trainer._current_epoch_state.value = epoch + 1
        accelerator.print(f"Epoch {trainer._current_epoch_state.value}/{trainer.num_train_epochs}")

        trainer._metadata["ss_epoch"] = str(trainer._current_epoch_state.value)

        trainer.mode.on_epoch_start(trainer)
        epoch_phase_name = f"training_epoch_{trainer._current_epoch_state.value}"
        resource_monitor = _resource_monitor(trainer)
        resource_monitor.phase_start(epoch_phase_name)

        # Phase G: Create per-epoch DataLoader with fresh epoch manifest
        caption_config = CaptionConfig(
            shuffle_caption=cfg.data.caption.shuffle_caption,
            keep_tokens=cfg.data.caption.keep_tokens,
            caption_dropout_rate=cfg.data.caption.caption_dropout_rate,
            caption_tag_dropout_rate=cfg.data.caption.caption_tag_dropout_rate,
            enable_wildcard=cfg.data.caption.enable_wildcard,
            caption_separator=cfg.data.caption.caption_separator,
            secondary_separator=cfg.data.caption.secondary_separator,
            keep_tokens_separator=cfg.data.caption.keep_tokens_separator,
            token_warmup_min=cfg.data.caption.token_warmup_min,
            token_warmup_step=cfg.data.caption.token_warmup_step,
        )
        epoch_manifest = prepare_epoch(
            manifest=trainer.train_manifest,
            epoch=epoch,
            seed=cfg.training.seed,
            batch_size=cfg.training.train_batch_size,
            caption_config=caption_config,
        )

        # Phase G.1: Optional epoch tokenization (when TE caching is disabled)
        tokens_path = None
        if cfg.data.caching.cache_tokens_per_epoch and not cfg.data.caching.cache_text_encoder_outputs:
            from library.data import tokenize_epoch_manifest

            tokens_path = Path(trainer._cache_dir) / f"epoch_{epoch}_tokens.safetensors"

            def tokenize_fn(captions: list[str]) -> list[torch.Tensor]:
                return strategies.tokenize_captions(trainer.tokenizers, captions, cfg.training.max_token_length)

            if accelerator.is_main_process:
                tokenize_epoch_manifest(
                    epoch_manifest,
                    tokenize_fn,
                    tokens_path,
                    encoder_names=["clip_l", "clip_g"],
                    max_token_length=cfg.training.max_token_length,
                )
            accelerator.wait_for_everyone()

        train_dataloader = create_training_dataloader(
            dataset_manifest=trainer.train_manifest,
            epoch_manifest=epoch_manifest,
            latent_strategy=trainer.latent_strategy,
            te_strategy=trainer.te_strategy,
            flip_aug=cfg.data.preprocessing.flip_aug,
            prior_loss_weight=cfg.loss.prior_loss_weight,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            num_workers=trainer._n_workers,
            prefetch_factor=cfg.data.loader.prefetch_factor,
            pin_memory=cfg.data.loader.pin_memory,
            persistent_workers=cfg.data.loader.persistent_workers,
            tokens_path=str(tokens_path) if tokens_path else None,
        )
        num_steps_in_epoch = len(train_dataloader)

        # TRAINING
        dataloader_iter = iter(train_dataloader)
        if trainer._initial_step > 0:
            dataloader_iter = itertools.islice(dataloader_iter, trainer._initial_step - 1, None)
            trainer._initial_step = 1

        for step, batch in enumerate(dataloader_iter):
            trainer._current_step_state.value = trainer.global_step

            # Dynamic timestep schedule
            if (
                trainer._dynamic_timestep_schedule
                and len(trainer._dynamic_timestep_schedule) > 0
                and trainer.global_step >= trainer._dynamic_timestep_schedule[0][0]
            ):
                _, new_min, new_max = trainer._dynamic_timestep_schedule.pop(0)
                trainer._current_min_timestep = new_min
                trainer._current_max_timestep = new_max
                accelerator.print(
                    f"\nStep {trainer.global_step}: Timestep range dynamically changed to "
                    f"[{trainer._current_min_timestep}, {trainer._current_max_timestep})"
                )

            if trainer._initial_step > 0:
                trainer._initial_step -= 1
                continue

            with determine_grad_sync_context(cfg.performance.precision, accelerator, None, trainer._grad_sync_handle, trainer._edm2_model):
                trainer.mode.on_step_start(trainer)

                trainer._accumulation_counter += 1

                # preprocess batch for each model
                strategies.on_step_start(
                    cfg,
                    accelerator,
                    trainer.trainable_model,
                    trainer.text_encoders,
                    trainer.unet,
                    batch,
                    trainer.weight_dtype,
                    is_train=True,
                )

                loss, pre_scaling_loss, loss_scaled, timesteps = strategies.process_batch(
                    batch,
                    trainer.text_encoders,
                    trainer.unet,
                    trainer.trainable_model,
                    trainer.vae,
                    trainer.noise_scheduler,
                    trainer.vae_dtype,
                    trainer.weight_dtype,
                    accelerator,
                    cfg,
                    trainer._text_encoding_strategy,
                    trainer._tokenize_strategy,
                    is_train=True,
                    train_text_encoder=trainer._train_text_encoder,
                    train_unet=trainer._train_unet,
                    edm2_model=trainer._edm2_model,
                    min_timestep_override=trainer._current_min_timestep,
                    max_timestep_override=trainer._current_max_timestep,
                    global_step=trainer.global_step,
                )

                accelerator.backward(loss)

                loss = pre_scaling_loss

                if accelerator.sync_gradients:
                    strategies.all_reduce_trainable(accelerator, trainer.trainable_model)
                    if cfg.optimizer.max_grad_norm != 0.0:
                        params_to_clip = trainer.mode.get_trainable_params(trainer)
                        accelerator.clip_grad_norm_(params_to_clip, cfg.optimizer.max_grad_norm)

                trainer.optimizer.step()
                trainer.lr_scheduler.step()
                trainer.optimizer.zero_grad(set_to_none=True)

                if cfg.loss.edm2.edm2_loss_weighting:
                    trainer._edm2_optimizer.step()
                    trainer._edm2_lr_scheduler.step()
                    trainer._edm2_optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                max_mean_logs = trainer.mode.on_step_end(trainer)
                keys_scaled = max_mean_logs.get("Keys Scaled")
                mean_norm = max_mean_logs.get("Average key norm")
                maximum_norm = None
            else:
                keys_scaled, mean_norm, maximum_norm = None, None, None
                max_mean_logs = {}

            # Checks if the accelerator has performed an optimization step
            if accelerator.sync_gradients:
                trainer._progress_bar.update(1)
                trainer.global_step += 1
                _run_step_side_effects(
                    trainer,
                    step=step,
                    batch=batch,
                    num_steps_in_epoch=num_steps_in_epoch,
                )
                resource_monitor.step_end(trainer.global_step, trainer._current_epoch_state.value)

            trainer._current_global_step_loss += loss.detach().item()
            if cfg.loss.edm2.edm2_loss_weighting:
                assert loss_scaled is not None and trainer._current_global_step_loss_scaled is not None
                trainer._current_global_step_loss_scaled += loss_scaled.detach().item()
            else:
                trainer._current_global_step_loss_scaled = None

            if accelerator.sync_gradients:
                trainer._loss_recorder.add(trainer._current_global_step_loss / trainer._accumulation_counter)
                if cfg.loss.edm2.edm2_loss_weighting:
                    assert trainer._loss_scaled_recorder is not None and trainer._current_global_step_loss_scaled is not None
                    trainer._loss_scaled_recorder.add(trainer._current_global_step_loss_scaled / trainer._accumulation_counter)
                avr_loss: float = trainer._loss_recorder.average
                logs = {"avr_loss": avr_loss}
                trainer._progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if trainer._is_tracking:
                    _emit_step_tracking_logs(
                        trainer,
                        timesteps=timesteps,
                        keys_scaled=keys_scaled,
                        mean_norm=mean_norm,
                        maximum_norm=maximum_norm,
                    )

                trainer._current_global_step_loss = 0.0
                if cfg.loss.edm2.edm2_loss_weighting:
                    trainer._current_global_step_loss_scaled = 0.0
                trainer._accumulation_counter = 0

                _update_live_timestep_outputs(trainer, timesteps=timesteps)

            if trainer.global_step >= cfg.training.max_train_steps:
                break

        _finalize_epoch(
            trainer,
            tokens_path=tokens_path,
            phase_name=epoch_phase_name,
        )

        # end of epoch
