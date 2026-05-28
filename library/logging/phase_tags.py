"""Structured runtime trace vocabulary for training startup and execution.

This module intentionally separates:
- phase tags: duration-bearing spans that can be started/ended and timed
- event tags: point-in-time milestones that should not be treated as spans
"""

from __future__ import annotations

PHASE_STARTUP_ACCELERATOR = "startup.accelerator"
PHASE_STARTUP_DATASET_MANIFEST = "startup.dataset_manifest"
PHASE_STARTUP_SUMMARY = "startup.summary"
PHASE_STARTUP_METADATA = "startup.metadata"
PHASE_STARTUP_RUNTIME = "startup.runtime"

PHASE_CACHE_LATENTS = "cache.latents"
PHASE_CACHE_TEXT_ENCODER = "cache.text_encoder"

PHASE_TRAINING_PREP_DEFERRED_DENOISER = "training.prep.deferred_denoiser"
PHASE_TRAINING_PREP_TRAINABLES = "training.prep.trainables"
PHASE_TRAINING_PREP_SHARED_PRECISION = "training.prep.shared_precision"
PHASE_TRAINING_PREP_TRAINABLE_PRECISION = "training.prep.trainable_precision"
PHASE_TRAINING_PREP_OPTIMIZER_GROUPS = "training.prep.optimizer_groups"
PHASE_TRAINING_PREP_VALIDATION_DATALOADER = "training.prep.validation_dataloader"
PHASE_TRAINING_PREP_LR_SCHEDULER = "training.prep.lr_scheduler"
PHASE_TRAINING_PREP_ACCELERATOR = "training.prep.accelerator"
PHASE_TRAINING_PREP_GRADIENT_CHECKPOINTING = "training.prep.gradient_checkpointing"

PHASE_CHECKPOINT_SAVE = "checkpoint.save"

EVENT_TRAINING_PROGRESS_BAR_STARTED = "training.progress_bar.started"
EVENT_TRAINING_FIRST_STEP_STARTED = "training.first_step.started"
EVENT_TRAINING_FIRST_STEP_SYNCED = "training.first_step.synced"
EVENT_CHECKPOINT_SAVED = "checkpoint.saved"

_TRAINING_EPOCH_PHASE_PREFIX = "training.epoch."


def training_epoch_phase(epoch_index: int) -> str:
    """Return the canonical phase tag for one training epoch.

    Epoch phase tags use the true zero-based runtime epoch index. User-facing
    console banners may still render one-based epoch counts separately.
    """
    return f"training.epoch.{epoch_index}"


def is_training_epoch_phase(phase_name: str | None) -> bool:
    """Return whether a phase tag identifies a training epoch span."""
    if not phase_name or not phase_name.startswith(_TRAINING_EPOCH_PHASE_PREFIX):
        return False

    epoch_suffix = phase_name[len(_TRAINING_EPOCH_PHASE_PREFIX) :]
    return epoch_suffix.isdigit()


__all__ = [
    "EVENT_CHECKPOINT_SAVED",
    "EVENT_TRAINING_FIRST_STEP_STARTED",
    "EVENT_TRAINING_FIRST_STEP_SYNCED",
    "EVENT_TRAINING_PROGRESS_BAR_STARTED",
    "PHASE_CACHE_LATENTS",
    "PHASE_CACHE_TEXT_ENCODER",
    "PHASE_CHECKPOINT_SAVE",
    "PHASE_STARTUP_ACCELERATOR",
    "PHASE_STARTUP_DATASET_MANIFEST",
    "PHASE_STARTUP_METADATA",
    "PHASE_STARTUP_RUNTIME",
    "PHASE_STARTUP_SUMMARY",
    "PHASE_TRAINING_PREP_ACCELERATOR",
    "PHASE_TRAINING_PREP_DEFERRED_DENOISER",
    "PHASE_TRAINING_PREP_GRADIENT_CHECKPOINTING",
    "PHASE_TRAINING_PREP_LR_SCHEDULER",
    "PHASE_TRAINING_PREP_OPTIMIZER_GROUPS",
    "PHASE_TRAINING_PREP_SHARED_PRECISION",
    "PHASE_TRAINING_PREP_TRAINABLES",
    "PHASE_TRAINING_PREP_TRAINABLE_PRECISION",
    "PHASE_TRAINING_PREP_VALIDATION_DATALOADER",
    "is_training_epoch_phase",
    "training_epoch_phase",
]
