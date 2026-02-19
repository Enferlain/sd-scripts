"""
PeftMode — TrainingMode implementation for PEFT (LoRA/LyCORIS) training.

This is a direct extraction of adapter-specific logic that was previously
inline in the phase files and trainer.  Every method mirrors the behavior
of the code it replaces — **zero behavior change** in Phase 1.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

import torch
from torch import nn

from library.adapters.lora_utils import resolve_adapter_kwargs
from library.optimizers.optimizer_utils import prepare_optimizer as _prepare_optimizer_util
from library.performance import deepspeed_utils
from library.training.checkpointing import register_adapter_state_hooks

if TYPE_CHECKING:
    from library.training.runners.trainer import Trainer

from library.utils.common_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class PeftMode:
    """PEFT (LoRA/LyCORIS) training mode.

    Implements the ``TrainingMode`` protocol for adapter-based training.
    Each method is extracted from inline phase/trainer code with no
    behavior change.
    """

    # ------------------------------------------------------------------
    # Model creation & precision
    # ------------------------------------------------------------------

    def prepare_trainables(self, trainer: Trainer) -> None:
        """Create and configure the adapter for training.

        Extracted from ``model_prep.create_adapter()``.
        """
        cfg = trainer.cfg
        accelerator = trainer.accelerator
        vae = trainer.vae
        unet = trainer.unet
        text_encoder = trainer._text_encoder
        text_encoders = trainer.text_encoders
        weight_dtype = trainer.weight_dtype

        # Import adapter module dynamically
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import peft module:", cfg.peft.adapter_module)
        adapter_module = importlib.import_module(cfg.peft.adapter_module)

        # Merge base weights if specified
        if cfg.peft.base_weights is not None:
            for i, weight_path in enumerate(cfg.peft.base_weights):
                if cfg.peft.base_weights_multiplier is None or len(cfg.peft.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = cfg.peft.base_weights_multiplier[i]

                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

                module, weights_sd = adapter_module.create_adapter_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                module.merge_to(
                    text_encoder,
                    unet,
                    weights_sd,
                    weight_dtype,
                    accelerator.device if cfg.performance.memory.lowram else "cpu",
                )

            accelerator.print(f"all weights merged: {', '.join(cfg.peft.base_weights)}")

        # Prepare adapter kwargs
        net_kwargs: dict[str, Any] = {}
        if cfg.peft.adapter_args is not None:
            for net_arg in cfg.peft.adapter_args:
                key, value = net_arg.split("=", 1)
                net_kwargs[key] = value

        resolve_adapter_kwargs(cfg.peft, net_kwargs)

        # Create adapter
        if cfg.peft.adapter_rank_from_weights:
            adapter, _ = adapter_module.create_adapter_from_weights(1, cfg.peft.adapter_weights, vae, text_encoder, unet, **net_kwargs)
        else:
            if "dropout" not in net_kwargs:
                net_kwargs["dropout"] = cfg.peft.neuron_dropout

            adapter = adapter_module.create_adapter(
                1.0,
                cfg.peft.adapter_rank,
                cfg.peft.adapter_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=cfg.peft.neuron_dropout,
                **net_kwargs,
            )

        if adapter is None:
            raise RuntimeError("Adapter creation returned None - check adapter module configuration")

        if hasattr(adapter, "prepare_adapter"):
            adapter.prepare_adapter(cfg)

        if cfg.peft.scale_weight_norms and not hasattr(adapter, "apply_max_norm_regularization"):
            logger.warning("warning: scale_weight_norms is specified but the peft does not support it")
            cfg.peft.scale_weight_norms = False

        trainer.strategies.post_process_trainable(cfg, accelerator, adapter, text_encoders, unet)

        # Apply adapter to unet and text_encoder
        trainer._train_unet = trainer.strategies.is_train_unet(cfg)
        trainer._train_text_encoder = trainer.strategies.is_train_text_encoder(cfg)
        adapter.apply_to(text_encoder, unet, trainer._train_text_encoder, trainer._train_unet)

        # Load weights if specified
        if cfg.peft.adapter_weights is not None:
            info = adapter.load_weights(cfg.peft.adapter_weights)
            accelerator.print(f"load peft weights from {cfg.peft.adapter_weights}: {info}")

        trainer.adapter = adapter
        trainer.net_kwargs = net_kwargs

    def configure_trainable_precision(self, trainer: Trainer) -> None:
        """PEFT-specific precision: cast adapter, freeze base model.

        Extracted from ``model_prep.configure_precision()`` —
        the adapter cast + ``requires_grad_(False)`` portions.
        """
        cfg = trainer.cfg
        adapter = trainer.adapter
        weight_dtype = trainer.weight_dtype
        accelerator = trainer.accelerator

        # Full fp16/bf16 training — cast entire adapter
        if cfg.performance.precision.full_fp16:
            accelerator.print("enable full fp16 training.")
            adapter.to(weight_dtype)
        elif cfg.performance.precision.full_bf16:
            accelerator.print("enable full bf16 training.")
            adapter.to(weight_dtype)

        # Freeze base model
        trainer.unet.requires_grad_(False)
        for t_enc in trainer.text_encoders:
            t_enc.requires_grad_(False)

    # ------------------------------------------------------------------
    # Optimizer & accelerator
    # ------------------------------------------------------------------

    def build_optimizer_params(self, trainer: Trainer) -> tuple[str, dict, Any, Any, Any, list[str]]:
        """Build optimizer for adapter params.

        Extracted from ``optimizer.prepare_optimizer()`` L52-57.
        """
        cfg = trainer.cfg
        return _prepare_optimizer_util(
            cfg.optimizer,
            cfg.optimizer.learning_rates,
            cfg.peft,
            trainer.adapter,
        )

    def prepare_with_accelerator(self, trainer: Trainer) -> None:
        """Wrap adapter/models with ``accelerator.prepare()``.

        Extracted from ``optimizer._prepare_with_accelerator()``.
        """
        cfg = trainer.cfg

        if cfg.performance.deepspeed:
            flags = trainer.strategies.get_text_encoders_train_flags(cfg, trainer.text_encoders)
            ds_model = deepspeed_utils.prepare_deepspeed_model(
                cfg.performance.precision,
                unet=trainer.unet if trainer._train_unet else None,
                text_encoder1=trainer.text_encoders[0] if flags[0] else None,
                text_encoder2=(trainer.text_encoders[1] if flags[1] else None) if len(trainer.text_encoders) > 1 else None,
                adapter=trainer.adapter,
            )
            ds_model, trainer.optimizer, trainer.lr_scheduler = trainer.accelerator.prepare(
                ds_model, trainer.optimizer, trainer.lr_scheduler
            )
            trainer._grad_sync_handle = ds_model
            trainer._primary_trainable = trainer.adapter
        else:
            if trainer._train_unet:
                trainer.unet = trainer.strategies.prepare_unet_with_accelerator(cfg, trainer.accelerator, trainer.unet)
            else:
                trainer.unet.to(
                    trainer.accelerator.device,
                    dtype=trainer.unet_weight_dtype if trainer.strategies.cast_unet(cfg) else None,
                )

            if trainer._train_text_encoder:
                trainer.text_encoders = [
                    (trainer.accelerator.prepare(t_enc) if flag else t_enc)
                    for t_enc, flag in zip(
                        trainer.text_encoders,
                        trainer.strategies.get_text_encoders_train_flags(cfg, trainer.text_encoders),
                    )
                ]
                trainer._text_encoder = trainer.text_encoders if len(trainer.text_encoders) > 1 else trainer.text_encoders[0]

            trainer.adapter, trainer.optimizer, trainer.lr_scheduler = trainer.accelerator.prepare(
                trainer.adapter, trainer.optimizer, trainer.lr_scheduler
            )
            trainer._grad_sync_handle = trainer.adapter
            trainer._primary_trainable = trainer.adapter

    def setup_gradient_training(self, trainer: Trainer) -> None:
        """Adapter-specific gradient checkpointing & ``prepare_grad_etc``.

        Extracted from ``optimizer._setup_gradient_checkpointing()`` —
        the adapter-specific portions (line 198 + line 214).
        """
        cfg = trainer.cfg

        if cfg.performance.memory.gradient_checkpointing:
            trainer.adapter.enable_gradient_checkpointing()

        # prepare_grad_etc is always called (regardless of gradient_checkpointing flag)
        trainer.accelerator.unwrap_model(trainer.adapter).prepare_grad_etc(trainer._text_encoder, trainer.unet)

    def register_state_hooks(self, trainer: Trainer) -> Callable[[], int | None]:
        """Register adapter save/load hooks for checkpointing.

        Extracted from ``optimizer.prepare_optimizer()`` L131-134.
        """
        return register_adapter_state_hooks(
            trainer.accelerator,
            trainer.adapter,
            trainer.cfg,
            trainer._current_epoch_state,
            trainer._current_step_state,
        )

    # ------------------------------------------------------------------
    # Per-epoch / per-step callbacks
    # ------------------------------------------------------------------

    def on_epoch_start(self, trainer: Trainer) -> None:
        """Call adapter's ``on_epoch_start`` (which calls ``.train()``).

        Extracted from ``training_loop.py`` L71.
        """
        trainer.accelerator.unwrap_model(trainer.adapter).on_epoch_start(trainer._text_encoder, trainer.unet)

    def on_step_start(self, trainer: Trainer) -> None:
        """Call adapter's ``on_step_start`` if it defines one.

        Absorbed from ``trainer._on_step_start_for_adapter`` callback.
        """
        unwrapped = trainer.accelerator.unwrap_model(trainer.adapter)
        if hasattr(unwrapped, "on_step_start"):
            unwrapped.on_step_start(trainer._text_encoder, trainer.unet)

    def on_step_end(self, trainer: Trainer) -> dict[str, Any]:
        """Apply weight-norm regularization if configured.

        Extracted from ``training_loop.py`` L217-224.
        """
        cfg = trainer.cfg
        accelerator = trainer.accelerator

        if cfg.peft.scale_weight_norms and accelerator.sync_gradients:
            keys_scaled, mean_norm, _maximum_norm = accelerator.unwrap_model(trainer.adapter).apply_max_norm_regularization(
                cfg.peft.scale_weight_norms, accelerator.device
            )
            return {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}

        return {}

    # ------------------------------------------------------------------
    # Eval / train transitions
    # ------------------------------------------------------------------

    def get_trainable_params(self, trainer: Trainer) -> list:
        """Return adapter parameters for gradient clipping."""
        return trainer.accelerator.unwrap_model(trainer.adapter).get_trainable_params()

    def set_eval(self, trainer: Trainer) -> None:
        """Switch adapter to eval mode."""
        trainer.accelerator.unwrap_model(trainer.adapter).eval()

    def set_train(self, trainer: Trainer) -> None:
        """Switch adapter to train mode."""
        trainer.accelerator.unwrap_model(trainer.adapter).train()

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
        """Save model weights as a single ``.safetensors`` file.

        When *target_model* is ``None`` (the default), the adapter is
        unwrapped and saved.  Callers pass an explicit *target_model*
        for alternate save targets such as EDM2 loss weights.

        Extracted from ``trainer.save_checkpoint()`` L384-390.
        """
        os.makedirs(trainer.cfg.output.saving.output_dir, exist_ok=True)
        ckpt_file = os.path.join(trainer.cfg.output.saving.output_dir, ckpt_name)

        trainer.accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        metadata["ss_training_finished_at"] = str(time.time())
        metadata["ss_steps"] = str(step)
        metadata["ss_epoch"] = str(epoch)

        modelspec_metadata = trainer.strategies.get_model_metadata(trainer.cfg)
        metadata.update(modelspec_metadata)

        save_dtype = dtype_override or trainer.save_dtype
        model_to_save = target_model if target_model is not None else trainer.accelerator.unwrap_model(trainer.adapter)
        model_to_save.save_weights(ckpt_file, save_dtype, metadata)

        if trainer.cfg.output.huggingface is not None and trainer.cfg.output.huggingface.huggingface_repo_id is not None:
            from library.utils import huggingface_util

            huggingface_util.upload(
                trainer.cfg.output.huggingface,
                ckpt_file,
                "/" + ckpt_name,
                force_sync_upload=force_sync_upload,
            )
