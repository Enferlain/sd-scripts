"""
Training Loop Phase - Main training loop execution.

This module contains the core training loop that's called by PeftTrainer.
All state is accessed via the trainer instance.
"""

from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from library.data import CaptionConfig, prepare_epoch, create_training_dataloader
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
from library.training.trainer_utils import determine_grad_sync_context, calculate_val_loss_check
from library.utils.common_utils import setup_logging

if TYPE_CHECKING:
    from library.training.runners.peft_trainer import PeftTrainer

setup_logging()
logger = logging.getLogger(__name__)


def run_training_loop(trainer: PeftTrainer) -> None:
    """Execute the main training loop.

    Iterates over epochs and steps, handling:
    - Per-epoch dataloader creation
    - Gradient accumulation
    - Checkpointing
    - Sample image generation
    - Validation loss calculation

    Updates trainer.global_step as training progresses.

    Args:
        trainer: PeftTrainer instance containing all training state
    """
    # Unpack frequently used attributes for readability
    cfg = trainer.cfg
    accelerator = trainer.accelerator
    strategies = trainer.strategies

    for epoch in range(trainer.epoch_to_start, trainer.num_train_epochs):
        if trainer.global_step >= cfg.training.max_train_steps:
            break

        trainer._current_epoch_state.value = epoch + 1
        accelerator.print(f"Epoch {trainer._current_epoch_state.value}/{trainer.num_train_epochs}")

        trainer._metadata["ss_epoch"] = str(trainer._current_epoch_state.value)

        trainer.mode.on_epoch_start(trainer)

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

        # TRAINING
        dataloader_iter = iter(train_dataloader)
        if trainer._initial_step > 0:
            dataloader_iter = itertools.islice(dataloader_iter, trainer._initial_step - 1, None)
            trainer._initial_step = 1

        # RESOURCE TRACKER START
        training_resource_tracker = None
        if os.environ.get("BENCHMARK_RESOURCES", "").lower() in ("1", "true", "yes"):
            from library.utils.resource_tracker import ResourceTracker

            training_resource_tracker = ResourceTracker("Training")
            training_resource_tracker.start()
        # RESOURCE TRACKER END

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

            with determine_grad_sync_context(cfg.performance.precision, accelerator, None, trainer._training_model, trainer._edm2_model):
                trainer._on_step_start_for_adapter(trainer._text_encoder, trainer.unet)

                trainer._accumulation_counter += 1

                # preprocess batch for each model
                strategies.on_step_start(
                    cfg, accelerator, trainer.adapter, trainer.text_encoders, trainer.unet, batch, trainer.weight_dtype, is_train=True
                )

                loss, pre_scaling_loss, loss_scaled, timesteps = strategies.process_batch(
                    batch,
                    trainer.text_encoders,
                    trainer.unet,
                    trainer.adapter,
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
                    strategies.all_reduce_adapter(accelerator, trainer.adapter)
                    if cfg.optimizer.max_grad_norm != 0.0:
                        params_to_clip = accelerator.unwrap_model(trainer.adapter).get_trainable_params()
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

                if (
                    sample_images_check(cfg.output.sampling, None, trainer.global_step)
                    or calculate_val_loss_check(
                        cfg.validation, cfg.training, trainer.global_step, step, trainer._val_dataloader, train_dataloader
                    )
                    or cfg.output.saving.save_every_n_steps is not None
                    and trainer.global_step % cfg.output.saving.save_every_n_steps == 0
                ):
                    accelerator.unwrap_model(trainer.adapter).eval()
                    trainer.optimizer_eval_fn()
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

                    if calculate_val_loss_check(
                        cfg.validation, cfg.training, trainer.global_step, step, trainer._val_dataloader, trainer.num_batches_per_epoch
                    ):
                        trainer._current_val_loss, trainer._average_val_loss = strategies.calculate_val_loss(
                            trainer.global_step,
                            step,
                            trainer.num_batches_per_epoch,
                            trainer._val_loss_recorder,
                            trainer._val_dataloader,
                            trainer._cyclic_val_dataloader,
                            trainer.adapter,
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
                    else:
                        trainer._current_val_loss, trainer._average_val_loss = None, None

                    # Save model at specified steps
                    if cfg.output.saving.save_every_n_steps is not None and trainer.global_step % cfg.output.saving.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = get_step_ckpt_name(cfg.output.saving, "." + cfg.output.saving.save_model_as, trainer.global_step)
                            trainer.save_checkpoint(ckpt_name, accelerator.unwrap_model(trainer.adapter), trainer.global_step, epoch)

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
                                    epoch,
                                    dtype_override=torch.float32,
                                )

                            if cfg.output.saving.save_state:
                                save_and_remove_state_stepwise(cfg.output.saving, accelerator, trainer.global_step)

                            remove_step_no = get_remove_step_no(cfg.output.saving, trainer.global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = get_step_ckpt_name(
                                    cfg.output.saving, "." + cfg.output.saving.save_model_as, remove_step_no
                                )
                                trainer.remove_checkpoint(remove_ckpt_name)

                                if cfg.loss.edm2.edm2_loss_weighting:
                                    remove_loss_weights_ckpt_name = get_step_ckpt_name(
                                        cfg.output.saving,
                                        "." + cfg.output.saving.save_model_as,
                                        remove_step_no,
                                        "_edm2_loss_weights",
                                    )
                                    trainer.remove_checkpoint(remove_loss_weights_ckpt_name)

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
                    accelerator.unwrap_model(trainer.adapter).train()

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
                    current_global_step_loss = trainer._current_global_step_loss / trainer._accumulation_counter
                    if cfg.loss.edm2.edm2_loss_weighting:
                        assert trainer._current_global_step_loss_scaled is not None and trainer._loss_scaled_recorder is not None
                        current_global_step_loss_scaled = trainer._current_global_step_loss_scaled / trainer._accumulation_counter
                        average_loss_scaled: float = trainer._loss_scaled_recorder.average
                    else:
                        current_global_step_loss_scaled = None
                        average_loss_scaled = None

                    logs = generate_step_logs(
                        cfg,
                        current_global_step_loss,
                        avr_loss,
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
                    step_logging(accelerator, logs, trainer.global_step, epoch + 1)

                trainer._current_global_step_loss = 0.0
                if cfg.loss.edm2.edm2_loss_weighting:
                    trainer._current_global_step_loss_scaled = 0.0
                trainer._accumulation_counter = 0

                # --- LIVE PLOTTER & STATIC PLOT UPDATE ---
                if trainer.is_main_process:
                    timesteps_np = timesteps.cpu().numpy()

                    # Send data to the live plotter
                    if strategies.live_plotter_process and strategies.live_plotter_process.poll() is None:
                        try:
                            strategies.live_plotter_process.stdin.write(f"{','.join(map(str, timesteps_np))}\n".encode())
                            strategies.live_plotter_process.stdin.flush()
                        except (BrokenPipeError, OSError):
                            logger.error("Live plotter connection lost.")
                            strategies.live_plotter_process = None

                    # Update counts and save static plot if needed
                    if trainer._timestep_counts is not None:
                        unique, counts = np.unique(timesteps_np, return_counts=True)
                        trainer._timestep_counts[unique] += counts

                        if (
                            cfg.output.logging.log_timestep_distribution_every_n_steps
                            and trainer.global_step % cfg.output.logging.log_timestep_distribution_every_n_steps == 0
                        ):
                            save_timestep_distribution_plot(cfg, trainer.global_step, trainer._timestep_counts, trainer._plotter_settings)

            if trainer.global_step >= cfg.training.max_train_steps:
                break

        # END OF EPOCH
        if training_resource_tracker:
            stats = training_resource_tracker.stop()
            logger.info(f"\n{stats.summary()}")
            training_resource_tracker = None

        # Cleanup epoch token file if it was created
        if tokens_path and tokens_path.exists():
            tokens_path.unlink()
            logger.debug(f"Cleaned up epoch token file: {tokens_path}")

        if trainer._is_tracking:
            logs = {"loss/epoch_average": trainer._loss_recorder.average}
            accelerator.log(logs, step=trainer.global_step)

        accelerator.wait_for_everyone()

        if (
            sample_images_check(cfg.output.sampling, trainer._current_epoch_state.value, trainer.global_step)
            or cfg.output.saving.save_every_n_epochs is not None
        ):
            trainer.optimizer_eval_fn()
            accelerator.unwrap_model(trainer.adapter).eval()
            if cfg.output.saving.save_every_n_epochs is not None and cfg.output.saving.save_every_n_epochs > 0:
                saving = (
                    trainer._current_epoch_state.value % cfg.output.saving.save_every_n_epochs == 0
                    and trainer._current_epoch_state.value < trainer.num_train_epochs
                )
                if trainer.is_main_process and saving:
                    ckpt_name = get_epoch_ckpt_name(
                        cfg.output.saving, "." + cfg.output.saving.save_model_as, trainer._current_epoch_state.value
                    )
                    trainer.save_checkpoint(
                        ckpt_name, accelerator.unwrap_model(trainer.adapter), trainer.global_step, trainer._current_epoch_state.value
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
            accelerator.unwrap_model(trainer.adapter).train()

        # end of epoch
