"""
TrainingMode Protocol — pluggable interface for training-mode differences.

Each mode (adapters, fine-tune, etc.) implements this protocol to define
how its trainable model is created, prepared, saved, and updated per-step.
The runner and phases provide the *shared* default pipeline (temporal policy:
*when* things happen), while modes own the *divergent* concerns (*how* each
step is done for a specific training type).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from torch import nn

from library.optimization.types import OptimizerBuildResult

if TYPE_CHECKING:
    from library.training.checkpointing import ResumeState
    from library.training.runners.trainer import Trainer


@runtime_checkable
class TrainingMode(Protocol):
    """Plugin interface for training-mode differences (adapters vs fine-tune).

    All hooks receive ``trainer`` as the sole "context" argument so
    signatures stay stable as the trainer grows.  Mode implementations
    read whatever they need from ``trainer`` directly.
    """

    checkpoint_artifact_role: str

    # --- Model creation & precision ---

    def prepare_trainables(self, trainer: Trainer) -> None:
        """Create/configure the trainable model target.

        For adapters: import the adapter module, create an adapter, apply it to
        denoiser/text-encoders, load weights, merge base weights.
        For fine-tune (future): unfreeze denoiser layers.

        After this call the mode's trainable target must be ready
        (e.g. ``trainer.adapter`` for adapters, denoiser for fine-tune).
        """
        ...

    def configure_trainable_precision(self, trainer: Trainer) -> None:
        """Mode-specific precision casting & freeze/unfreeze logic.

        For adapters: ``adapter.to(weight_dtype)``, freeze base model
        (``denoiser.requires_grad_(False)``), freeze text-encoders.
        Shared dtype setup (FP8, TE dtype) is handled by the phase caller.
        """
        ...

    # --- Optimizer & accelerator ---

    def build_optimizer_params(self, trainer: Trainer) -> OptimizerBuildResult | tuple[str, dict, Any, Any, Any, list[str]]:
        """Build optimizer parameter groups and create the optimizer.

        Returns:
            Either a normalized ``OptimizerBuildResult`` for plan-aware paths,
            or the legacy compatibility tuple:
            (optimizer_name, optimizer_args, optimizer,
             train_fn, eval_fn, lr_descriptions)
        """
        ...

    def prepare_with_accelerator(self, trainer: Trainer) -> None:
        """Wrap trainable models with ``accelerator.prepare()``.

        For adapters: wrap adapter + optimizer + dataloader + lr_scheduler.
        For DeepSpeed: wrap via ``prepare_deepspeed_model``.

        **Required contracts** (asserted in ``run_training_loop``):

        - ``trainer._grad_sync_handle``: the object passed to
          ``accelerator.accumulate()`` for gradient synchronization.
          DeepSpeed: the composite model. Otherwise: the prepared trainable.
        - ``trainer._primary_trainable``: the semantic trainable module.
          Exposed via ``trainer.trainable_model`` and passed to strategies.
        """
        ...

    def setup_gradient_training(self, trainer: Trainer) -> None:
        """Mode-specific gradient checkpointing & grad preparation.

        For adapters: ``adapter.enable_gradient_checkpointing()``,
        ``adapter.prepare_grad_etc(text_encoder, denoiser)``.
        Shared denoiser/TE gradient-checkpointing is handled by the phase caller.
        """
        ...

    def register_state_hooks(self, trainer: Trainer) -> ResumeState:
        """Register save/load hooks for checkpointing.

        Returns the mutable resume state populated by checkpoint load hooks
        (or left empty when training from scratch).
        """
        ...

    # --- Per-epoch / per-step callbacks ---

    def on_epoch_start(self, trainer: Trainer) -> None:
        """Mode-specific epoch start callback.

        For adapters: ``adapter.on_epoch_start(text_encoder, denoiser)``
        (which calls ``adapter.train()`` internally).
        """
        ...

    def on_step_start(self, trainer: Trainer) -> None:
        """Mode-specific callback before each training step.

        For adapters: ``adapter.on_step_start(text_encoder, denoiser)`` if the
        adapter defines it.
        For fine-tune: typically a no-op.
        """
        ...

    def on_step_end(self, trainer: Trainer) -> dict[str, Any]:
        """Post-step operations, e.g. weight-norm regularization.

        Returns a dict of extra log entries (can be empty).
        """
        ...

    # --- Eval / train transitions ---

    def get_trainable_params(self, trainer: Trainer) -> list:
        """Return parameters for gradient clipping.

        For adapters: ``adapter.get_trainable_params()``
        For fine-tune: denoiser (+ optional TE) parameters.
        """
        ...

    def set_eval(self, trainer: Trainer) -> None:
        """Switch primary trainable module(s) to eval mode.

        For adapters: ``adapter.eval()``
        For fine-tune: ``denoiser.eval()`` + optional TE.
        """
        ...

    def set_train(self, trainer: Trainer) -> None:
        """Switch primary trainable module(s) to train mode.

        For adapters: ``adapter.train()``
        For fine-tune: ``denoiser.train()`` + optional TE.
        """
        ...

    # --- Checkpoint saving ---

    def resolve_checkpoint_artifact_format(self, trainer: Trainer) -> str:
        """Return the physical format used for this mode's checkpoint artifact."""
        ...

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
        - Adapters: single ``.safetensors`` file via ``adapter.save_weights``
        - Fine-tune (future): directory with multiple components

        Args:
            target_model: Model to save. When ``None`` the mode uses its
                default trainable (adapter for adapters). Callers pass an
                explicit model for alternate targets such as EDM2 loss
                weights.
        """
        ...

    # --- Diagnostics ---

    def get_diagnostics_components(self, trainer: Trainer) -> tuple[list[tuple[str, nn.Module]], list[tuple[str, str]] | None]:
        """Return components relevant for training diagnostics.

        Each mode decides what to show:
        - Adapters: adapter only (frozen backbone is noise).
        - Fine-tune: declared loaded backbone components in family order.

        Returns:
            (components, aliases) where components is a list of
            (component_key, module) tuples and aliases is an optional list of
            (alias_name, target_name) for display.
        """
        ...
