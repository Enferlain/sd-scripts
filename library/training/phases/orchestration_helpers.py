"""Shared orchestration helpers for generic training phases."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from library.logging.resource_monitor import NoOpResourceMonitor

if TYPE_CHECKING:
    from collections.abc import Iterator
    import torch


_NOOP_RESOURCE_MONITOR = NoOpResourceMonitor()


def get_resource_monitor(trainer):
    """Return the trainer resource monitor or a no-op fallback."""
    monitor = getattr(trainer, "_resource_monitor", None)
    return monitor if monitor is not None else _NOOP_RESOURCE_MONITOR


@contextmanager
def monitored_phase(trainer, phase_name: str) -> Iterator[object]:
    """Wrap a shared phase in paired resource-monitor lifecycle hooks."""
    monitor = get_resource_monitor(trainer)
    monitor.phase_start(phase_name)
    try:
        yield monitor
    finally:
        monitor.phase_end(phase_name)


@contextmanager
def temporarily_in_eval_mode(trainer) -> Iterator[None]:
    """Temporarily switch trainer-owned runtime state into eval mode."""
    trainer.mode.set_eval(trainer)
    trainer.optimizer_eval_fn()
    try:
        yield
    finally:
        trainer.optimizer_train_fn()
        trainer.mode.set_train(trainer)


def run_sampling_and_validation(
    trainer,
    *,
    should_sample: bool,
    should_validate: bool,
    sample_epoch: int | None,
    validation_step: int,
    validation_epoch: int,
    validation_batch: dict[str, torch.Tensor] | None,
) -> None:
    """Run the shared sample/validate eval actions and update trainer val state."""
    cfg = trainer.cfg
    accelerator = trainer.accelerator
    strategies = trainer.strategies

    if should_sample:
        strategies.sample_images(
            accelerator,
            cfg,
            sample_epoch,
            trainer.global_step,
            accelerator.device,
            trainer.vae,
            trainer.tokenizers,
            trainer._text_encoder,
            trainer.denoiser,
        )

    if should_validate:
        assert trainer.vae_dtype is not None, "vae_dtype must be set"
        assert trainer.weight_dtype is not None, "weight_dtype must be set"
        trainer._current_val_loss, trainer._average_val_loss = strategies.calculate_val_loss(
            trainer.global_step,
            validation_step,
            trainer.num_batches_per_epoch,
            trainer._val_loss_recorder,
            trainer._val_dataloader,
            trainer._cyclic_val_dataloader,
            trainer.trainable_model,
            trainer.text_encoders,
            trainer.denoiser,
            trainer.vae,
            trainer.noise_scheduler,
            trainer.vae_dtype,
            trainer.weight_dtype,
            accelerator,
            cfg,
            validation_epoch,
            validation_batch,
            trainer._train_text_encoder,
        )
        accelerator.print(f"  val_loss: {trainer._current_val_loss:.4f}  (avg: {trainer._average_val_loss:.4f})")
    else:
        trainer._current_val_loss, trainer._average_val_loss = None, None
