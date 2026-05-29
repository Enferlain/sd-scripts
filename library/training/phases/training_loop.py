"""
Training Loop Phase - Main training loop execution.

This module contains the core training loop that's called by Trainer.
All state is accessed via the trainer instance.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

from library.data import CaptionConfig, prepare_epoch, create_training_dataloader
from library.logging.phase_tags import (
    EVENT_TRAINING_FIRST_STEP_STARTED,
    EVENT_TRAINING_FIRST_STEP_SYNCED,
    EVENT_TRAINING_PROGRESS_BAR_STARTED,
    training_epoch_phase,
)
from library.logging.metrics import generate_step_logs
from library.logging.training_plots import save_timestep_distribution_plot
from library.optimization.optimizer_utils import apply_optimizer_runtime_mode
from library.training.checkpointing import (
    get_step_ckpt_name,
    save_and_remove_state_stepwise,
    get_remove_step_no,
    get_epoch_ckpt_name,
    get_remove_epoch_no,
    save_and_remove_state_on_epoch_end,
)
from library.training.sample_generation import sample_images_check
from library.training.phases.orchestration_helpers import (
    get_resource_monitor,
    monitored_phase,
    run_sampling_and_validation,
    temporarily_in_eval_mode,
)
from library.training.trainer_utils import determine_grad_sync_context, all_reduce_trainable
from library.training.phases.triggers import (
    StepTriggerContext,
    compute_step_actions,
    EpochEndTriggerContext,
    compute_epoch_end_actions,
)


if TYPE_CHECKING:
    from library.training.runners.trainer import Trainer


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochContext:
    """Prepared epoch-local runtime inputs for step execution."""

    train_dataloader: object
    num_steps_in_epoch: int
    remaining_batches_in_epoch: int
    tokens_path: Path | None


@dataclass(frozen=True)
class EpochRunResult:
    """Outcome of executing one training epoch."""

    completed: bool
    stop_training: bool
    interrupted_by_step_cap: bool
    batches_seen: int
    expected_batches: int
    tokens_path: Path | None


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

    if trainer.loss_modifier.is_enabled and trainer.loss_modifier.sidecar_suffix:
        loss_weights_ckpt_name = get_step_ckpt_name(
            cfg.output.saving,
            "." + cfg.output.saving.save_model_as,
            trainer.global_step,
            trainer.loss_modifier.sidecar_suffix,
        )
        sidecar_path = str(Path(cfg.output.saving.output_dir) / loss_weights_ckpt_name)
        trainer.loss_modifier.save_sidecar(sidecar_path, trainer._build_checkpoint_metadata())

    if cfg.output.saving.save_state:
        save_and_remove_state_stepwise(cfg.output.saving, accelerator, trainer.global_step)

    remove_step_no = get_remove_step_no(cfg.output.saving, trainer.global_step)
    if remove_step_no is None:
        return

    remove_ckpt_name = get_step_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as, remove_step_no)
    trainer.remove_checkpoint(remove_ckpt_name)

    if trainer.loss_modifier.is_enabled and trainer.loss_modifier.sidecar_suffix:
        remove_loss_weights_ckpt_name = get_step_ckpt_name(
            cfg.output.saving,
            "." + cfg.output.saving.save_model_as,
            remove_step_no,
            trainer.loss_modifier.sidecar_suffix,
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

    with temporarily_in_eval_mode(trainer):
        run_sampling_and_validation(
            trainer,
            should_sample=step_actions.should_sample,
            should_validate=step_actions.should_validate,
            sample_epoch=None,
            validation_step=step,
            validation_epoch=trainer._current_epoch_state.value,
            validation_batch=batch,
        )

        if step_actions.should_save_step:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                _save_step_checkpoint_artifacts(trainer)

        if trainer.loss_modifier.should_plot(trainer.global_step):
            trainer.loss_modifier.plot(
                cfg.output.saving.output_name,
                trainer.global_step,
                accelerator.device,
            )


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
    """Emit the current optimization step's scalar metrics through the training observer."""
    cfg = trainer.cfg
    observer = getattr(trainer, "_observer", None)
    if observer is None:
        raise RuntimeError("observer must be initialized before step metrics are emitted")

    current_global_step_loss = trainer._current_global_step_loss / trainer._accumulation_counter
    current_loss_scaled_total = trainer._current_loss_modifier_metrics.get("loss/current_scaled")
    current_global_step_loss_scaled = (
        current_loss_scaled_total / trainer._accumulation_counter if current_loss_scaled_total is not None else None
    )
    scaled_loss_recorder = trainer._loss_modifier_metric_recorders.get("loss/current_scaled")
    average_loss_scaled: float | None = scaled_loss_recorder.average if scaled_loss_recorder is not None else None
    modifier_lrs = None
    modifier_lr = trainer.loss_modifier.get_lr()
    if modifier_lr is not None:
        modifier_lrs = {trainer.loss_modifier.name: modifier_lr}

    logs = generate_step_logs(
        cfg,
        current_global_step_loss,
        trainer._loss_recorder.average,
        trainer.lr_scheduler,
        lr_descriptions=None if trainer.optimization_plan is not None else trainer.lr_descriptions,
        optimization_plan=trainer.optimization_plan,
        timestep_runtime=trainer.objective_runtime.timestep_runtime if trainer.objective_runtime is not None else None,
        optimizer=trainer.optimizer,
        keys_scaled=keys_scaled,
        mean_norm=mean_norm,
        maximum_norm=maximum_norm,
        mean_grad_norm=None,
        mean_combined_norm=None,
        modifier_lrs=modifier_lrs,
        current_loss_scaled=current_global_step_loss_scaled,
        average_loss_scaled=average_loss_scaled,
        current_val_loss=trainer._current_val_loss,
        average_val_loss=trainer._average_val_loss,
        timesteps=timesteps,
    )
    log_every = cfg.output.logging.log_every_n_steps
    if trainer.global_step % log_every == 0:
        observer.log_metrics(trainer.global_step, logs, epoch=trainer._current_epoch_state.value)


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

    if trainer.loss_modifier.is_enabled and trainer.loss_modifier.sidecar_suffix:
        loss_weights_ckpt_name = get_epoch_ckpt_name(
            cfg.output.saving,
            "." + cfg.output.saving.save_model_as,
            trainer._current_epoch_state.value,
            trainer.loss_modifier.sidecar_suffix,
        )
        sidecar_path = str(Path(cfg.output.saving.output_dir) / loss_weights_ckpt_name)
        trainer.loss_modifier.save_sidecar(sidecar_path, trainer._build_checkpoint_metadata())

    remove_epoch_no = get_remove_epoch_no(cfg.output.saving, trainer._current_epoch_state.value)
    if remove_epoch_no is not None:
        remove_ckpt_name = get_epoch_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as, remove_epoch_no)
        trainer.remove_checkpoint(remove_ckpt_name)

        if trainer.loss_modifier.is_enabled and trainer.loss_modifier.sidecar_suffix:
            remove_loss_weights_ckpt_name = get_epoch_ckpt_name(
                cfg.output.saving, "." + cfg.output.saving.save_model_as, remove_epoch_no, trainer.loss_modifier.sidecar_suffix
            )
            trainer.remove_checkpoint(remove_loss_weights_ckpt_name)

    if cfg.output.saving.save_state:
        save_and_remove_state_on_epoch_end(cfg.output.saving, accelerator, trainer._current_epoch_state.value)


def _maybe_cache_epoch_tokens(trainer: Trainer, *, epoch: int, epoch_manifest) -> Path | None:
    """Persist per-epoch token caches when enabled for the current strategy."""
    cfg = trainer.cfg
    accelerator = trainer.accelerator
    strategies = trainer.strategies

    if not cfg.data.caching.cache_tokens_per_epoch or cfg.data.caching.cache_text_encoder_outputs:
        return None

    from library.data.epoch_preparation import tokenize_epoch_manifest

    tokens_path = Path(trainer._cache_dir) / f"epoch_{epoch}_tokens.safetensors"

    def tokenize_fn(captions: list[str]) -> list[torch.Tensor]:
        return strategies.tokenize_captions(trainer.tokenizers, captions, cfg.training.max_token_length)

    if accelerator.is_main_process:
        tokenize_epoch_manifest(
            epoch_manifest,
            tokenize_fn,
            tokens_path,
            encoder_names=strategies.get_token_cache_encoder_names(),
            max_token_length=cfg.training.max_token_length,
        )
    accelerator.wait_for_everyone()
    return tokens_path


def _finalize_epoch(
    trainer: Trainer,
    *,
    tokens_path: Path | None,
) -> None:
    """Run end-of-epoch cleanup, tracking, and epoch-end side-effects."""
    cfg = trainer.cfg
    accelerator = trainer.accelerator
    strategies = trainer.strategies

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
    if not epoch_end_actions.should_enter_eval_mode:
        return

    apply_optimizer_runtime_mode(trainer.optimizer, trainer.optimization_plan, training=False)
    trainer.mode.set_eval(trainer)
    if trainer.is_main_process and epoch_end_actions.should_save_epoch:
        _save_epoch_checkpoint_artifacts(trainer)

    if epoch_end_actions.should_sample:
        strategies.sample_images(
            accelerator,
            cfg,
            trainer._current_epoch_state.value,
            trainer.global_step,
            accelerator.device,
            trainer.vae,
            trainer.tokenizers,
            trainer._text_encoder,
            trainer.denoiser,
        )
    trainer._progress_bar.unpause()
    apply_optimizer_runtime_mode(trainer.optimizer, trainer.optimization_plan, training=True)
    trainer.mode.set_train(trainer)


def _build_caption_config(cfg) -> CaptionConfig:
    """Build the per-epoch caption mutation config."""
    return CaptionConfig(
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


def _prepare_epoch_context(trainer: Trainer, *, epoch: int) -> EpochContext:
    """Prepare epoch-local manifest, token cache, and dataloader state."""
    cfg = trainer.cfg
    accelerator = trainer.accelerator

    epoch_manifest = prepare_epoch(
        manifest=trainer.train_manifest,
        epoch=epoch,
        seed=cfg.training.seed,
        batch_size=cfg.training.train_batch_size,
        caption_config=_build_caption_config(cfg),
    )
    epoch_message = f"prepared epoch {epoch}: {epoch_manifest.num_batches} batches, {epoch_manifest.num_images} images"
    trainer.log_progress_message(epoch_message, tag="epoch", stacklevel=3)

    tokens_path = _maybe_cache_epoch_tokens(
        trainer,
        epoch=epoch,
        epoch_manifest=epoch_manifest,
    )

    train_dataloader = create_training_dataloader(
        dataset_manifest=trainer.train_manifest,
        epoch_manifest=epoch_manifest,
        latent_cache_backend=trainer.latent_cache_backend,
        te_cache_backend=trainer.te_cache_backend,
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
    initial_batches_to_skip = max(trainer._initial_step, 0)
    remaining_batches_in_epoch = max(num_steps_in_epoch - initial_batches_to_skip, 0)

    return EpochContext(
        train_dataloader=train_dataloader,
        num_steps_in_epoch=num_steps_in_epoch,
        remaining_batches_in_epoch=remaining_batches_in_epoch,
        tokens_path=tokens_path,
    )


def _run_epoch_steps(trainer: Trainer, *, epoch_ctx: EpochContext) -> EpochRunResult:
    """Execute one epoch's training steps and return an explicit outcome."""
    cfg = trainer.cfg
    accelerator = trainer.accelerator
    strategies = trainer.strategies
    assert trainer.objective_runtime is not None, "objective_runtime must be initialized before running the training loop"

    batches_seen_in_epoch = 0
    dataloader_iter = iter(epoch_ctx.train_dataloader)
    if trainer._initial_step > 0:
        dataloader_iter = itertools.islice(dataloader_iter, trainer._initial_step - 1, None)
        trainer._initial_step = 1

    for step, batch in enumerate(dataloader_iter):
        batches_seen_in_epoch += 1
        trainer._current_step_state.value = trainer.global_step

        updated_range = trainer.objective_runtime.advance_to_step(trainer.global_step)

        if updated_range is not None:
            new_min, new_max = updated_range
            accelerator.print(
                f"\nStep {trainer.global_step}: Timestep range dynamically changed to "
                f"[{new_min}, {new_max})"
            )

        if trainer._initial_step > 0:
            trainer._initial_step -= 1
            continue

        if not trainer._runtime_trace_first_step_started:
            trainer.runtime_trace.event(EVENT_TRAINING_FIRST_STEP_STARTED)
            trainer._runtime_trace_first_step_started = True

        with determine_grad_sync_context(
            cfg.performance.precision,
            accelerator,
            None,
            trainer._grad_sync_handle,
            trainer.loss_modifier.accumulation_model,
        ):
            trainer.mode.on_step_start(trainer)

            trainer._accumulation_counter += 1

            batch_loss = strategies.process_batch(
                batch,
                trainer.text_encoders,
                trainer.denoiser,
                trainer.trainable_model,
                trainer.vae,
                trainer.objective_runtime,
                trainer.vae_dtype,
                trainer.weight_dtype,
                accelerator,
                cfg,
                is_train=True,
                train_text_encoder=trainer._train_text_encoder,
                train_denoiser=trainer._train_denoiser,
                global_step=trainer.global_step,
            )

            sampling_loss = batch_loss.sampling_loss if batch_loss.sampling_loss is not None else batch_loss.per_sample_loss
            trainer.objective_runtime.update_from_batch(batch_loss.timesteps, sampling_loss)

            modifier_output = trainer.loss_modifier.apply(
                per_sample_loss=batch_loss.per_sample_loss,
                timesteps=batch_loss.timesteps,
                batch=batch,
                global_step=trainer.global_step,
            )
            accelerator.backward(modifier_output.loss)

            if accelerator.sync_gradients:
                all_reduce_trainable(accelerator, trainer.trainable_model)
                if cfg.optimizer.max_grad_norm != 0.0:
                    params_to_clip = trainer.mode.get_trainable_params(trainer)
                    accelerator.clip_grad_norm_(params_to_clip, cfg.optimizer.max_grad_norm)

            trainer.optimizer.step()
            trainer.lr_scheduler.step()
            trainer.optimizer.zero_grad(set_to_none=True)

            trainer.loss_modifier.optimizer_step()
            trainer.loss_modifier.zero_grad()

        if accelerator.sync_gradients:
            max_mean_logs = trainer.mode.on_step_end(trainer)
            keys_scaled = max_mean_logs.get("Keys Scaled")
            mean_norm = max_mean_logs.get("Average key norm")
            maximum_norm = None
        else:
            keys_scaled, mean_norm, maximum_norm = None, None, None
            max_mean_logs = {}

        if accelerator.sync_gradients:
            trainer._progress_bar.update(1)
            trainer.global_step += 1
            if not trainer._runtime_trace_first_step_synced:
                trainer.runtime_trace.event(EVENT_TRAINING_FIRST_STEP_SYNCED)
                trainer._runtime_trace_first_step_synced = True
            _run_step_side_effects(
                trainer,
                step=step,
                batch=batch,
                num_steps_in_epoch=epoch_ctx.num_steps_in_epoch,
            )
            get_resource_monitor(trainer).step_end(trainer.global_step, trainer._current_epoch_state.value)

        trainer._current_global_step_loss += batch_loss.loss.detach().item()
        for metric_name, metric_value in modifier_output.metrics.items():
            trainer._current_loss_modifier_metrics[metric_name] = trainer._current_loss_modifier_metrics.get(metric_name, 0.0) + float(metric_value)

        if accelerator.sync_gradients:
            trainer._loss_recorder.add(trainer._current_global_step_loss / trainer._accumulation_counter)
            for metric_name, metric_total in trainer._current_loss_modifier_metrics.items():
                if metric_name not in trainer._loss_modifier_metric_recorders:
                    from library.losses.loss import EMARecorder

                    trainer._loss_modifier_metric_recorders[metric_name] = EMARecorder()
                trainer._loss_modifier_metric_recorders[metric_name].add(metric_total / trainer._accumulation_counter)
            avr_loss: float = trainer._loss_recorder.average
            logs = {"avr_loss": avr_loss}
            trainer._progress_bar.set_postfix(**{**max_mean_logs, **logs})

            if trainer._is_tracking:
                _emit_step_tracking_logs(
                    trainer,
                    timesteps=batch_loss.timesteps,
                    keys_scaled=keys_scaled,
                    mean_norm=mean_norm,
                    maximum_norm=maximum_norm,
                )

            trainer._current_global_step_loss = 0.0
            trainer._current_loss_modifier_metrics = {}
            trainer._accumulation_counter = 0

            _update_live_timestep_outputs(trainer, timesteps=batch_loss.timesteps)

        if trainer.global_step >= cfg.training.max_train_steps:
            return EpochRunResult(
                completed=False,
                stop_training=True,
                interrupted_by_step_cap=True,
                batches_seen=batches_seen_in_epoch,
                expected_batches=epoch_ctx.remaining_batches_in_epoch,
                tokens_path=epoch_ctx.tokens_path,
            )

    return EpochRunResult(
        completed=batches_seen_in_epoch >= epoch_ctx.remaining_batches_in_epoch,
        stop_training=False,
        interrupted_by_step_cap=False,
        batches_seen=batches_seen_in_epoch,
        expected_batches=epoch_ctx.remaining_batches_in_epoch,
        tokens_path=epoch_ctx.tokens_path,
    )


def _finalize_epoch_if_completed(trainer: Trainer, *, epoch_result: EpochRunResult) -> None:
    """Finalize epoch-owned side effects only when the epoch fully completed."""
    if not epoch_result.completed:
        return

    _finalize_epoch(
        trainer,
        tokens_path=epoch_result.tokens_path,
    )


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

    # Mode must set _grad_sync_handle during prepare_with_accelerator;
    # if missing, grad-sync context will silently fail.
    assert trainer._grad_sync_handle is not None, (
        "trainer._grad_sync_handle must be set by mode.prepare_with_accelerator() before the training loop starts"
    )
    assert trainer.trainable_model is not None, (
        "trainer.trainable_model must be set by mode.prepare_with_accelerator() before the training loop starts"
    )
    assert trainer._validation_scheduler is not None, "trainer._validation_scheduler must be initialized before the training loop starts"

    if trainer._progress_bar is None:
        logger.info("[%s] starting progress bar", EVENT_TRAINING_PROGRESS_BAR_STARTED)
        trainer._progress_bar = tqdm(
            range(trainer.max_train_steps - trainer._initial_step),
            smoothing=0,
            disable=not trainer.accelerator.is_local_main_process,
            desc="steps",
        )
        trainer.runtime_trace.event(EVENT_TRAINING_PROGRESS_BAR_STARTED)

    for epoch_index in range(trainer.epoch_to_start, trainer.num_train_epochs):
        if trainer.global_step >= cfg.training.max_train_steps:
            break

        # Keep the user-facing progress banner one-based while runtime phase
        # identifiers continue to use the true zero-based epoch index.
        display_epoch = epoch_index + 1
        trainer._current_epoch_state.value = display_epoch
        trainer.print_progress_message(f"Epoch {display_epoch}/{trainer.num_train_epochs}")

        trainer._set_training_metadata_fact("ss_epoch", trainer._current_epoch_state.value)

        trainer.mode.on_epoch_start(trainer)
        epoch_phase_name = training_epoch_phase(epoch_index)
        with monitored_phase(trainer, epoch_phase_name):
            epoch_ctx = _prepare_epoch_context(trainer, epoch=epoch_index)
            epoch_result = _run_epoch_steps(trainer, epoch_ctx=epoch_ctx)

        _finalize_epoch_if_completed(trainer, epoch_result=epoch_result)

        if epoch_result.stop_training:
            break

        # end of epoch
