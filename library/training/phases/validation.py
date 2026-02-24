"""
Validation Phase - Scheduling and orchestration for validation during training.

This module owns the validation trigger logic. Strategy files own model-specific
validation math (process_val_batch), but this module decides WHEN to validate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.config.dataclasses.validation import ValidationConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationStepContext:
    """Typed context for validation scheduling decisions.

    Passed to ValidationScheduler.should_run() instead of raw
    dataloader/int arguments. No dataloader objects here — only
    scalars that describe where we are in the training run.
    """

    global_step: int
    epoch_step: int
    current_epoch: int
    is_last_step_in_epoch: bool
    is_training_start: bool
    is_training_end: bool
    has_validation_data: bool


class ValidationScheduler:
    """Single source of truth for validation trigger decisions.

    Replaces the old calculate_val_loss_check() with typed, testable
    scheduling logic that supports step-based, epoch-based, start,
    and end triggers.

    Args:
        config: ValidationConfig with scheduling fields.
    """

    def __init__(self, config: ValidationConfig) -> None:
        self._run_at_start = config.run_at_start
        self._run_at_end = config.run_at_end
        self._every_n_steps = config.validate_every_n_steps
        self._every_n_epochs = config.validate_every_n_epochs

    def should_run(self, ctx: ValidationStepContext) -> bool:
        """Determine if validation should run at this point in training.

        Rules:
        - If no validation data exists, always False.
        - ``run_at_start=True`` triggers at ``is_training_start``.
        - ``run_at_end=True`` triggers at ``is_training_end``.
        - ``every_n_steps`` triggers when ``global_step % n == 0`` (excludes step 0
          unless ``run_at_start`` handles it separately).
        - ``every_n_epochs`` triggers when ``is_last_step_in_epoch`` and
          ``current_epoch % n == 0``.
        - If neither cadence field is set, defaults to epoch-end validation.
        - Step and epoch cadences use OR semantics when both are set.

        Args:
            ctx: Validation step context with current training state.

        Returns:
            True if validation should run, False otherwise.
        """
        if not ctx.has_validation_data:
            return False

        # Start-of-training trigger
        if ctx.is_training_start and self._run_at_start:
            return True

        # End-of-training trigger
        if ctx.is_training_end and self._run_at_end:
            return True

        # Step-based cadence
        if self._every_n_steps is not None and ctx.global_step > 0 and ctx.global_step % self._every_n_steps == 0:
            return True

        # Epoch-based cadence
        if self._every_n_epochs is not None and ctx.is_last_step_in_epoch and ctx.current_epoch % self._every_n_epochs == 0:
            return True

        # Default: epoch-end validation when no cadence is explicitly set
        return self._every_n_steps is None and self._every_n_epochs is None and ctx.is_last_step_in_epoch
