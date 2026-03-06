"""
Trigger policy helpers for training loop side effects.

This module centralizes step/epoch trigger decisions (validation, sampling,
checkpoint saves) so the training loop can focus on orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING

from library.training.phases.validation import ValidationStepContext
from library.training.sample_generation import sample_images_check

if TYPE_CHECKING:
    from library.config.dataclasses.output import SamplingConfig, SavingConfig
    from library.training.phases.validation import ValidationScheduler


@dataclass(frozen=True)
class StepTriggerContext:
    """Runtime state for step-level trigger decisions."""

    global_step: int
    epoch_step: int
    current_epoch: int
    num_steps_in_epoch: int
    max_train_steps: int
    has_validation_data: bool


@dataclass(frozen=True)
class StepActions:
    """Decided step actions for the current optimization step."""

    should_validate: bool
    should_sample: bool
    should_save_step: bool

    @property
    def should_enter_eval_mode(self) -> bool:
        """Whether eval mode is needed for any side action on this step."""
        return self.should_sample or self.should_validate or self.should_save_step


@dataclass(frozen=True)
class EpochEndTriggerContext:
    """Runtime state for epoch-end trigger decisions."""

    current_epoch: int
    num_train_epochs: int
    global_step: int


@dataclass(frozen=True)
class EpochEndActions:
    """Decided actions for epoch-end branch."""

    should_sample: bool
    should_save_epoch: bool
    should_enter_eval_mode: bool


def _should_save_step(saving_config: SavingConfig, global_step: int) -> bool:
    if saving_config.save_every_n_steps is None:
        return False
    return global_step % saving_config.save_every_n_steps == 0


def compute_step_actions(
    *,
    validation_scheduler: ValidationScheduler,
    sampling_config: SamplingConfig,
    saving_config: SavingConfig,
    ctx: StepTriggerContext,
    sample_images_check_fn: Callable[[SamplingConfig, int | None, int], bool] = sample_images_check,
) -> StepActions:
    """Compute step-level side-effect actions from typed context."""
    val_ctx = ValidationStepContext(
        global_step=ctx.global_step,
        epoch_step=ctx.epoch_step,
        current_epoch=ctx.current_epoch,
        is_last_step_in_epoch=ctx.epoch_step == ctx.num_steps_in_epoch - 1,
        is_training_start=False,
        is_training_end=ctx.global_step >= ctx.max_train_steps,
        has_validation_data=ctx.has_validation_data,
    )
    should_validate = validation_scheduler.should_run(val_ctx)
    should_sample = sample_images_check_fn(sampling_config, None, ctx.global_step)
    should_save_step = _should_save_step(saving_config, ctx.global_step)
    return StepActions(
        should_validate=should_validate,
        should_sample=should_sample,
        should_save_step=should_save_step,
    )


def compute_epoch_end_actions(
    *,
    sampling_config: SamplingConfig,
    saving_config: SavingConfig,
    ctx: EpochEndTriggerContext,
    sample_images_check_fn: Callable[[SamplingConfig, int | None, int], bool] = sample_images_check,
) -> EpochEndActions:
    """Compute epoch-end side-effect actions from typed context."""
    should_sample = sample_images_check_fn(sampling_config, ctx.current_epoch, ctx.global_step)

    has_epoch_save_config = saving_config.save_every_n_epochs is not None
    should_save_epoch = False
    if has_epoch_save_config and saving_config.save_every_n_epochs > 0:
        should_save_epoch = (
            ctx.current_epoch % saving_config.save_every_n_epochs == 0 and ctx.current_epoch < ctx.num_train_epochs
        )

    # Enter eval mode only when an epoch-end side effect is actually scheduled.
    should_enter_eval_mode = should_sample or should_save_epoch

    return EpochEndActions(
        should_sample=should_sample,
        should_save_epoch=should_save_epoch,
        should_enter_eval_mode=should_enter_eval_mode,
    )
