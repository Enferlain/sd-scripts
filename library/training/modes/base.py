"""
TrainingMode Protocol — pluggable interface for training-mode differences.

Each mode (PEFT, future fine-tune, etc.) implements this protocol to define
how its trainable model is created, prepared, saved, and updated per-step.
The runner and phases provide the *shared* default pipeline (temporal policy:
*when* things happen), while modes own the *divergent* concerns (*how* each
step is done for a specific training type).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from collections.abc import Callable

import torch
from torch import nn

if TYPE_CHECKING:
    from library.training.runners.peft_trainer import PeftTrainer

    # Use a type alias for the trainer to keep the protocol signatures clean.
    # This will eventually be a generic Trainer base, but today it's PeftTrainer.
    Trainer = PeftTrainer


@runtime_checkable
class TrainingMode(Protocol):
    """Plugin interface for training-mode differences (PEFT vs fine-tune).

    All hooks receive ``trainer`` as the sole "context" argument so
    signatures stay stable as the trainer grows.  Mode implementations
    read whatever they need from ``trainer`` directly.
    """

    # --- Model creation & precision ---

    def prepare_trainables(self, trainer: Trainer) -> None:
        """Create/configure the trainable model target.

        For PEFT: import the adapter module, create an adapter, apply it to
        UNet/text-encoders, load weights, merge base weights.
        For fine-tune (future): unfreeze UNet layers.

        After this call, ``trainer.adapter`` (or equivalent) must be set.
        """
        ...

    def configure_trainable_precision(self, trainer: Trainer) -> None:
        """Mode-specific precision casting & freeze/unfreeze logic.

        For PEFT: ``adapter.to(weight_dtype)``, freeze base model
        (``unet.requires_grad_(False)``), freeze text-encoders.
        Shared dtype setup (FP8, TE dtype) is handled by the phase caller.
        """
        ...

    # --- Optimizer & accelerator ---

    def build_optimizer_params(self, trainer: Trainer) -> tuple[str, dict, Any, Any, Any, list[str]]:
        """Build optimizer parameter groups and create the optimizer.

        Returns:
            (optimizer_name, optimizer_args, optimizer,
             train_fn, eval_fn, lr_descriptions)
        """
        ...

    def prepare_with_accelerator(self, trainer: Trainer) -> None:
        """Wrap trainable models with ``accelerator.prepare()``.

        For PEFT: wrap adapter + optimizer + dataloader + lr_scheduler.
        For DeepSpeed: wrap via ``prepare_deepspeed_model``.
        Sets ``trainer._training_model`` to the wrapped trainable.
        """
        ...

    def setup_gradient_training(self, trainer: Trainer) -> None:
        """Mode-specific gradient checkpointing & grad preparation.

        For PEFT: ``adapter.enable_gradient_checkpointing()``,
        ``adapter.prepare_grad_etc(text_encoder, unet)``.
        Shared UNet/TE gradient-checkpointing is handled by the phase caller.
        """
        ...

    def register_state_hooks(self, trainer: Trainer) -> Callable[[], int | None]:
        """Register save/load hooks for checkpointing.

        Returns a callable that retrieves the ``steps_from_state`` after
        a checkpoint is loaded (or ``None`` if training from scratch).
        """
        ...

    # --- Per-epoch / per-step callbacks ---

    def on_epoch_start(self, trainer: Trainer) -> None:
        """Mode-specific epoch start callback.

        For PEFT: ``adapter.on_epoch_start(text_encoder, unet)``
        (which calls ``adapter.train()`` internally).
        """
        ...

    def on_step_end(self, trainer: Trainer) -> dict[str, Any]:
        """Post-step operations, e.g. weight-norm regularization.

        Returns a dict of extra log entries (can be empty).
        """
        ...

    # --- Checkpoint saving ---

    def save_checkpoint(
        self,
        trainer: Trainer,
        ckpt_name: str,
        step: int,
        epoch: int,
        metadata: dict[str, str],
        force_sync_upload: bool = False,
        dtype_override: torch.dtype | None = None,
        target_model: nn.Module | None = None,
    ) -> None:
        """Save a mode-specific checkpoint.

        Receives *logical intent* (name, step, epoch, metadata) so the
        mode decides the physical format:
        - PEFT: single ``.safetensors`` file via ``adapter.save_weights``
        - Fine-tune (future): directory with multiple components

        Args:
            target_model: Model to save. When ``None`` the mode uses its
                default trainable (adapter for PEFT). Callers pass an
                explicit model for alternate targets such as EDM2 loss
                weights.
        """
        ...
