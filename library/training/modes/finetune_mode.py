"""
FineTuneMode – Full-model fine-tuning.

Implements the TrainingMode protocol for training denoiser (and optionally
text encoders) directly, as opposed to training a PEFT adapter on top.

All model-family specifics (serialization format, TE freeze behavior)
are delegated to the strategy; this mode only handles generic lifecycle.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from library.optimization.arguments import parse_key_value_args
from library.optimization.optimizer_factory import get_optimizer
from library.optimization.types import build_module_parameter_group
from library.optimization.optimizer_utils import (
    get_optimizer_train_eval_fn,
    get_text_encoders_train_flags,
    should_train_denoiser,
    should_train_text_encoder,
)
from library.performance import deepspeed_utils
from library.training.checkpointing import ResumeState, load_train_state_metadata, save_train_state_metadata


if TYPE_CHECKING:
    from library.training.runners.trainer import Trainer


class FineTuneMode:
    """Full-model fine-tuning mode.

    Unfreezes denoiser and (optionally) text encoders, builds standard
    per-module optimizer param groups, and delegates checkpoint
    serialization to the strategy.
    """

    # ------------------------------------------------------------------
    # Model preparation
    # ------------------------------------------------------------------

    _te_train_flags: list[bool]

    def prepare_trainables(self, trainer: Trainer) -> None:
        """Unfreeze denoiser and optionally text encoders for training.

        Sets ``trainer._train_denoiser``, ``trainer._train_text_encoder``,
        ``trainer._primary_trainable``, and per-TE training flags on
        the mode instance.

        Uses shared trainability helpers for generic LR-driven train flags and
        the strategy for model-specific post-processing (e.g. SDXL TE1
        last-layer freezing).
        """
        cfg = trainer.cfg
        strategies = trainer.strategies

        # Determine what to train from generic LR policy
        trainer._train_denoiser = should_train_denoiser(cfg.optimizer.learning_rates)
        trainer._train_text_encoder = should_train_text_encoder(cfg.optimizer.learning_rates)

        # Unfreeze denoiser
        assert trainer.denoiser is not None, "denoiser must be loaded before prepare_trainables"
        if trainer._train_denoiser:
            trainer.denoiser.requires_grad_(True)
            trainer.denoiser.train()
        else:
            trainer.denoiser.requires_grad_(False)
            trainer.denoiser.eval()

        self._te_train_flags = get_text_encoders_train_flags(cfg.optimizer.learning_rates, trainer.text_encoders)
        trainer._train_text_encoder = any(self._te_train_flags)

        # Freeze / unfreeze each TE
        for t_enc, flag in zip(trainer.text_encoders, self._te_train_flags):
            if flag:
                t_enc.requires_grad_(True)
                t_enc.train()
            else:
                t_enc.requires_grad_(False)
                t_enc.eval()

        # Delegate model-specific post-processing to strategy
        # (e.g. SDXL freezes TE1's last encoder layer + final_layer_norm)
        strategies.post_process_trainable(cfg, trainer.accelerator, trainer.denoiser, trainer.text_encoders, trainer.denoiser)

        # Primary trainable = denoiser in fine-tune mode
        trainer._primary_trainable = trainer.denoiser

    def configure_trainable_precision(self, trainer: Trainer) -> None:
        """Cast trainable models to weight_dtype for full fp16/bf16 training."""
        cfg = trainer.cfg
        weight_dtype = trainer.weight_dtype
        assert trainer.denoiser is not None, "denoiser must be loaded before configure_trainable_precision"

        if cfg.performance.precision.full_fp16:
            trainer.accelerator.print("enable full fp16 training.")
            if trainer._train_denoiser:
                trainer.denoiser.to(weight_dtype)
            for t_enc, flag in zip(trainer.text_encoders, self._te_train_flags):
                if flag:
                    t_enc.to(weight_dtype)
        elif cfg.performance.precision.full_bf16:
            trainer.accelerator.print("enable full bf16 training.")
            if trainer._train_denoiser:
                trainer.denoiser.to(weight_dtype)
            for t_enc, flag in zip(trainer.text_encoders, self._te_train_flags):
                if flag:
                    t_enc.to(weight_dtype)

        # Cast non-trained TEs to weight_dtype so they run efficiently
        for t_enc, flag in zip(trainer.text_encoders, self._te_train_flags):
            if not flag:
                t_enc.to(weight_dtype)

    # ------------------------------------------------------------------
    # Optimizer & accelerator
    # ------------------------------------------------------------------

    def build_optimizer_params(self, trainer: Trainer) -> tuple[str, dict, Any, Any, Any, list[str]]:
        """Build optimizer with standard denoiser + TE param groups.

        Block-level LR grouping and pattern-based grouping are deferred
        to a later mode-agnostic phase and are not supported in 2B.
        """
        cfg = trainer.cfg
        lr = cfg.optimizer.learning_rates

        # --- Fail-fast for deferred features ---
        # Block LR is configured via optimizer_args or dedicated config
        if cfg.optimizer.optimizer_args:
            for arg in cfg.optimizer.optimizer_args:
                if "block_lr" in arg.lower():
                    raise NotImplementedError(
                        "Block-level learning rates are deferred to a later mode-agnostic "
                        "optimizer-group phase. Remove block_lr from optimizer_args for 2B."
                    )

        # --- Build param groups ---
        trainable_params = []
        lr_descriptions = []

        if trainer._train_denoiser:
            assert trainer.denoiser is not None, "denoiser must be loaded before build_optimizer_params"
            denoiser_lr = lr.denoiser if lr.denoiser is not None else lr.base
            trainable_params.append(build_module_parameter_group(trainer.denoiser, lr=denoiser_lr, label="denoiser"))
            lr_descriptions.append(f"denoiser lr: {denoiser_lr}")

        te_lr_raw = lr.text_encoders
        for i, (t_enc, flag) in enumerate(zip(trainer.text_encoders, self._te_train_flags)):
            if flag:
                if te_lr_raw is None:
                    te_lr = lr.base
                elif isinstance(te_lr_raw, (int, float)):
                    te_lr = te_lr_raw
                else:
                    te_lr = te_lr_raw[i] if i < len(te_lr_raw) else lr.base
                trainable_params.append(build_module_parameter_group(t_enc, lr=te_lr, label=f"text_encoder{i + 1}"))
                lr_descriptions.append(f"text_encoder{i + 1} lr: {te_lr}")

        # --- Parse optimizer kwargs ---
        optimizer_kwargs = parse_key_value_args(cfg.optimizer.optimizer_args)

        # --- Create optimizer ---
        optimizer_name, optimizer_args, optimizer = get_optimizer(
            cfg.optimizer, lr, cfg.optimizer.scheduler, trainable_params, optimizer_kwargs
        )
        optimizer_train_fn, optimizer_eval_fn = get_optimizer_train_eval_fn(optimizer, cfg.optimizer)  # type: ignore[arg-type]

        return optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn, lr_descriptions  # type: ignore[return-value]

    def prepare_with_accelerator(self, trainer: Trainer) -> None:
        """Wrap denoiser/TEs with ``accelerator.prepare()``."""
        cfg = trainer.cfg

        if cfg.performance.deepspeed.deepspeed:
            # Build dynamic kwargs from flag list — no fixed TE count assumption
            te_flags = self._te_train_flags
            ds_kwargs: dict[str, Any] = {}
            if trainer._train_denoiser:
                ds_kwargs["denoiser"] = trainer.denoiser
            for i, (t_enc, flag) in enumerate(zip(trainer.text_encoders, te_flags)):
                if flag:
                    ds_kwargs[f"text_encoder{i + 1}"] = t_enc
            # No adapter for fine-tune

            ds_model = deepspeed_utils.prepare_deepspeed_model(cfg.performance.precision, **ds_kwargs)
            ds_model, trainer.optimizer, trainer.lr_scheduler = trainer.accelerator.prepare(
                ds_model, trainer.optimizer, trainer.lr_scheduler
            )
            trainer._grad_sync_handle = ds_model
            trainer._primary_trainable = trainer.denoiser
        else:
            # Prepare denoiser
            if trainer._train_denoiser:
                trainer.denoiser = trainer.accelerator.prepare(trainer.denoiser)

            # Prepare trained TEs; move non-trained TEs to device
            for i, (t_enc, flag) in enumerate(zip(trainer.text_encoders, self._te_train_flags)):
                if flag:
                    trainer.text_encoders[i] = trainer.accelerator.prepare(t_enc)
                # Non-trained TEs are already on device from caching phase

            trainer._text_encoder = trainer.text_encoders if len(trainer.text_encoders) > 1 else trainer.text_encoders[0]

            # Prepare optimizer + scheduler
            trainer.optimizer, trainer.lr_scheduler = trainer.accelerator.prepare(trainer.optimizer, trainer.lr_scheduler)

            # Grad sync handle = denoiser (the largest trainable component)
            trainer._grad_sync_handle = trainer.denoiser
            trainer._primary_trainable = trainer.denoiser

    def setup_gradient_training(self, trainer: Trainer) -> None:
        """No-op for fine-tune.

        Shared ``_setup_gradient_checkpointing`` in ``optimizer.py``
        already handles denoiser/TE gradient checkpointing and train mode.
        """
        pass

    def register_state_hooks(self, trainer: Trainer) -> ResumeState:
        """Register save/load hooks for epoch/step metadata.

        For full fine-tune, accelerator handles model state natively.
        Hooks only manage the epoch/step metadata file.
        """
        accelerator = trainer.accelerator
        cfg = trainer.cfg
        current_epoch = trainer._current_epoch_state
        current_step = trainer._current_step_state
        resume_state = ResumeState()

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process or cfg.performance.deepspeed.deepspeed:
                save_train_state_metadata(output_dir, current_epoch, current_step)

        def load_model_hook(models, input_dir):
            load_train_state_metadata(input_dir, current_epoch, current_step, resume_state)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        return resume_state

    # ------------------------------------------------------------------
    # Per-epoch / per-step callbacks
    # ------------------------------------------------------------------

    def on_epoch_start(self, trainer: Trainer) -> None:
        """Set all training models to train mode."""
        if trainer._train_denoiser and trainer.denoiser is not None:
            trainer.denoiser.train()
        for t_enc, flag in zip(trainer.text_encoders, self._te_train_flags):
            if flag:
                t_enc.train()

    def on_step_start(self, trainer: Trainer) -> None:
        """No-op for fine-tune."""
        pass

    def on_step_end(self, trainer: Trainer) -> dict[str, Any]:
        """No weight-norm regularization in fine-tune mode."""
        return {}

    # ------------------------------------------------------------------
    # Eval / train transitions
    # ------------------------------------------------------------------

    def get_trainable_params(self, trainer: Trainer) -> list:
        """Return all trainable parameters for gradient clipping."""
        params: list[nn.Parameter] = []
        if trainer._train_denoiser and trainer.denoiser is not None:
            params.extend(list(trainer.denoiser.parameters()))
        for t_enc, flag in zip(trainer.text_encoders, self._te_train_flags):
            if flag:
                params.extend(list(t_enc.parameters()))
        return params

    def set_eval(self, trainer: Trainer) -> None:
        """Switch denoiser + trained TEs to eval mode."""
        if trainer._train_denoiser and trainer.denoiser is not None:
            trainer.denoiser.eval()
        for t_enc, flag in zip(trainer.text_encoders, self._te_train_flags):
            if flag:
                t_enc.eval()

    def set_train(self, trainer: Trainer) -> None:
        """Switch denoiser + trained TEs to train mode."""
        if trainer._train_denoiser and trainer.denoiser is not None:
            trainer.denoiser.train()
        for t_enc, flag in zip(trainer.text_encoders, self._te_train_flags):
            if flag:
                t_enc.train()

    # ------------------------------------------------------------------
    # Checkpoint saving
    # ------------------------------------------------------------------

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
        """Serialize checkpoint, delegating to strategy for full-model saves.

        Trainer owns save timing and retention policy; this method only
        handles serialization routing.

        For EDM2 side-saves (``_edm2_loss_weights`` suffix in
        ``ckpt_name``), the target_model (EDM2 loss module) is saved
        via ``.save_weights()`` — this is mode-owned since it's not a
        full model checkpoint.

        For normal checkpoints, the full model is delegated to
        ``strategies.save_model_checkpoint(...)`` which handles
        architecture-specific serialization.
        """
        # EDM2 side artifact — delegate to simple .save_weights()
        if "_edm2_loss_weights" in ckpt_name and target_model is not None:
            os.makedirs(trainer.cfg.output.saving.output_dir, exist_ok=True)
            ckpt_file = os.path.join(trainer.cfg.output.saving.output_dir, ckpt_name)
            save_dtype = dtype_override or trainer.save_dtype
            target_model.save_weights(ckpt_file, save_dtype, metadata)  # type: ignore[union-attr]
            return

        # Full-model checkpoint — delegate to strategy
        save_dtype = dtype_override or trainer.save_dtype
        assert save_dtype is not None, "save_dtype must be set (via dtype_override or trainer.save_dtype)"
        trainer.strategies.save_model_checkpoint(
            trainer=trainer,
            ckpt_name=ckpt_name,
            step=step,
            epoch=epoch,
            metadata=metadata,
            save_dtype=save_dtype,
            force_sync_upload=force_sync_upload,
        )

    def get_diagnostics_components(self, trainer: Trainer) -> tuple[list[tuple[str, nn.Module]], list[tuple[str, str]] | None]:
        """Return all backbone components — they're all relevant in fine-tune."""
        components: list[tuple[str, nn.Module]] = []
        if trainer.denoiser is not None:
            components.append(("denoiser", trainer.denoiser))
        for i, te in enumerate(trainer.text_encoders):
            components.append((f"text_encoder{i + 1}", te))
        if trainer.vae is not None:
            components.append(("vae", trainer.vae))
        return components, None
