"""
PeftMode — TrainingMode implementation for PEFT/adapter training.

This began as a direct extraction of adapter-specific logic that was previously
inline in the phase files and trainer. The class now also owns the repo-owned
adapter runtime/orchestration seam that replaced the older compatibility-era
optimizer boundary.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from library.adapters import (
    AdapterBuildContext,
    AdapterBuildRequest,
    AdapterExportLoadRequest,
    AdapterExportSaveRequest,
    AdapterModelContext,
    AdapterRuntimeSpec,
    build_adapter_for_legacy_module,
    build_adapter_from_weights_for_legacy_module,
    get_adapter_method_for_legacy_module,
    load_adapter_export,
    register_adapter_checkpoint_state_hooks,
    save_adapter_export,
)
from library.adapters.runtime import AdapterMergeRequest
from library.adapters.lora_utils import resolve_adapter_kwargs
from library.optimization.arguments import parse_key_value_args
from library.optimization.grouping import build_adapter_grouping, resolve_adapter_target_selection, resolve_learning_rate_groups
from library.optimization.optimizer_factory import get_optimizer
from library.optimization.optimizer_utils import get_text_encoders_train_flags
from library.optimization.types import OptimizationPlan, OptimizerBuildResult
from library.performance import deepspeed_utils
from library.training.checkpointing import ResumeState

if TYPE_CHECKING:
    from library.training.runners.trainer import Trainer


logger = logging.getLogger(__name__)


_UNSUPPORTED_ADAPTER_OPTIMIZER_POLICY_KEYS = (
    "down_lr_weight",
    "mid_lr_weight",
    "up_lr_weight",
    "block_lr_zero_threshold",
)


def _ensure_supported_adapter_optimizer_policy(cfg, net_kwargs: dict[str, Any] | None = None) -> None:
    legacy_optimizer_policy_fields = {
        "peft.lora.down_lr_weight": cfg.peft.lora.down_lr_weight,
        "peft.lora.mid_lr_weight": cfg.peft.lora.mid_lr_weight,
        "peft.lora.up_lr_weight": cfg.peft.lora.up_lr_weight,
        "peft.lora.block_lr_zero_threshold": cfg.peft.lora.block_lr_zero_threshold,
        "peft.lora.loraplus_lr_ratio": cfg.peft.lora.loraplus_lr_ratio,
        "peft.lora.loraplus_unet_lr_ratio": cfg.peft.lora.loraplus_unet_lr_ratio,
        "peft.lora.loraplus_text_encoder_lr_ratio": cfg.peft.lora.loraplus_text_encoder_lr_ratio,
    }
    active_fields = [name for name, value in legacy_optimizer_policy_fields.items() if value is not None]
    if active_fields:
        raise NotImplementedError(
            "Legacy built-in adapter optimizer policy is not supported by the repo-owned trainable-ref handoff: "
            + ", ".join(active_fields)
        )

    adapter_kwargs = net_kwargs if isinstance(net_kwargs, dict) else {}
    active_kwargs = [key for key in _UNSUPPORTED_ADAPTER_OPTIMIZER_POLICY_KEYS if adapter_kwargs.get(key) is not None]
    if active_kwargs:
        raise NotImplementedError(
            "Legacy built-in adapter optimizer policy in adapter args is not supported by the repo-owned trainable-ref handoff: "
            + ", ".join(active_kwargs)
        )


def _build_adapter_settings(peft_config, net_kwargs: dict[str, Any]) -> dict[str, Any]:
    lora_config = peft_config.lora
    settings = dict(net_kwargs)
    settings["adapter_rank"] = lora_config.rank
    settings["adapter_alpha"] = lora_config.alpha
    settings["neuron_dropout"] = lora_config.dropout
    return settings


def _build_adapter_request(
    *,
    vae,
    text_encoders: list[nn.Module],
    denoiser,
    adapter_type: str,
    settings: dict[str, Any],
    resolved_targets,
    for_inference: bool = False,
) -> AdapterBuildRequest:
    return AdapterBuildRequest(
        adapter=AdapterRuntimeSpec(adapter_type=adapter_type, settings=settings),
        context=AdapterBuildContext(
            model=AdapterModelContext(vae=vae, text_encoder=text_encoders, denoiser=denoiser),
            multiplier=1.0,
            for_inference=for_inference,
        ),
        resolved_targets=resolved_targets,
    )


class PeftMode:
    """PEFT (LoRA/LyCORIS) training mode.

    Implements the ``TrainingMode`` protocol for adapter-based training while
    keeping training-side orchestration in the mode layer and leaving
    optimizer grouping ownership in the optimization layer.
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
        denoiser = trainer.denoiser
        text_encoder = trainer._text_encoder
        text_encoders = trainer.text_encoders
        weight_dtype = trainer.weight_dtype
        adapter_registration = get_adapter_method_for_legacy_module(cfg.peft.adapter_module)
        adapter_type = adapter_registration.name

        accelerator.print("import peft module:", cfg.peft.adapter_module)

        target_selection = resolve_adapter_target_selection(
            model_type=cfg.model.model_type,
            denoiser=denoiser,
            text_encoders=text_encoders,
            learning_rates=cfg.optimizer.learning_rates,
        )
        trainer._train_denoiser = target_selection.train_denoiser
        trainer._train_text_encoder = target_selection.train_any_text_encoder

        # Merge base weights if specified
        if cfg.peft.base_weights is not None:
            for i, weight_path in enumerate(cfg.peft.base_weights):
                if cfg.peft.base_weights_multiplier is None or len(cfg.peft.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = cfg.peft.base_weights_multiplier[i]

                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

                merge_request = AdapterBuildRequest(
                    adapter=AdapterRuntimeSpec(adapter_type=adapter_type, settings={}),
                    context=AdapterBuildContext(
                        model=AdapterModelContext(vae=vae, text_encoder=text_encoders, denoiser=denoiser),
                        multiplier=multiplier,
                        for_inference=True,
                    ),
                    resolved_targets=target_selection.resolved_targets,
                )
                loaded_runtime = build_adapter_from_weights_for_legacy_module(cfg.peft.adapter_module, merge_request, weight_path)
                loaded_runtime.merge_into(
                    AdapterMergeRequest(
                        model=AdapterModelContext(vae=vae, text_encoder=text_encoders, denoiser=denoiser),
                        resolved_targets=target_selection.resolved_targets,
                        dtype=weight_dtype,
                        device=accelerator.device if cfg.performance.memory.lowram else "cpu",
                    ),
                )

            accelerator.print(f"all weights merged: {', '.join(cfg.peft.base_weights)}")

        # Prepare adapter kwargs
        net_kwargs: dict[str, Any] = parse_key_value_args(cfg.peft.adapter_args)

        resolve_adapter_kwargs(cfg.peft, net_kwargs)
        _ensure_supported_adapter_optimizer_policy(cfg, net_kwargs)
        adapter_settings = _build_adapter_settings(cfg.peft, net_kwargs)

        # Create adapter
        build_request = _build_adapter_request(
            vae=vae,
            text_encoders=text_encoders,
            denoiser=denoiser,
            adapter_type=adapter_type,
            settings=adapter_settings,
            resolved_targets=target_selection.resolved_targets,
        )
        if cfg.peft.adapter_rank_from_weights:
            loaded_runtime = build_adapter_from_weights_for_legacy_module(
                cfg.peft.adapter_module,
                build_request,
                cfg.peft.adapter_weights,
            )
            adapter = loaded_runtime.adapter
        else:
            if "dropout" not in net_kwargs:
                net_kwargs["dropout"] = cfg.peft.lora.dropout
            build_request.adapter.settings["dropout"] = net_kwargs["dropout"]
            adapter = build_adapter_for_legacy_module(cfg.peft.adapter_module, build_request)

        if adapter is None:
            raise RuntimeError("Adapter creation returned None - check adapter module configuration")

        if hasattr(adapter, "prepare_adapter"):
            adapter.prepare_adapter(cfg)

        if cfg.peft.scale_weight_norms and not hasattr(adapter, "apply_max_norm_regularization"):
            logger.warning("warning: scale_weight_norms is specified but the peft does not support it")
            cfg.peft.scale_weight_norms = False

        trainer.strategies.post_process_trainable(cfg, accelerator, adapter, text_encoders, denoiser)

        # Apply adapter to denoiser and text_encoder
        adapter.apply_to(text_encoder, denoiser, trainer._train_text_encoder, trainer._train_denoiser)

        # Load weights if specified
        if cfg.peft.adapter_weights is not None:
            info = load_adapter_export(adapter, AdapterExportLoadRequest(file=cfg.peft.adapter_weights))
            accelerator.print(f"load peft weights from {cfg.peft.adapter_weights}: {info}")

        adapter.requires_grad_(True)
        trainer.adapter = adapter
        trainer.adapter_resolved_targets = build_request.resolved_targets
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
        trainer.denoiser.requires_grad_(False)
        for t_enc in trainer.text_encoders:
            t_enc.requires_grad_(False)

    # ------------------------------------------------------------------
    # Optimizer & accelerator
    # ------------------------------------------------------------------

    def build_optimizer_params(self, trainer: Trainer) -> OptimizerBuildResult:
        """Build optimization-owned adapter groups and create the optimizer."""
        cfg = trainer.cfg
        if resolve_learning_rate_groups(cfg.optimizer.learning_rates):
            raise NotImplementedError("optimizer.learning_rates.groups are currently supported only for fine-tune mode")
        _ensure_supported_adapter_optimizer_policy(cfg, getattr(trainer, "net_kwargs", None))

        grouping = build_adapter_grouping(
            adapter=trainer.adapter,
            learning_rates=cfg.optimizer.learning_rates,
        )
        optimization_plan = OptimizationPlan(
            logical_groups=grouping.logical_groups,
            execution_groups=grouping.execution_groups,
        )
        optimizer_kwargs = parse_key_value_args(cfg.optimizer.optimizer_args)
        optimizer_name, optimizer_args, optimizer = get_optimizer(
            cfg.optimizer,
            cfg.optimizer.learning_rates,
            cfg.optimizer.scheduler,
            optimization_plan.execution_groups,
            optimizer_kwargs,
        )

        return OptimizerBuildResult(
            optimizer_name=optimizer_name,
            optimizer_args=optimizer_args,
            optimizer=optimizer,
            optimization_plan=optimization_plan,
        )

    def prepare_with_accelerator(self, trainer: Trainer) -> None:
        """Wrap adapter/models with ``accelerator.prepare()``.

        Extracted from ``optimizer._prepare_with_accelerator()``.
        """
        cfg = trainer.cfg

        if cfg.performance.deepspeed.deepspeed:
            flags = get_text_encoders_train_flags(cfg.optimizer.learning_rates, trainer.text_encoders)
            # Build dynamic kwargs — no fixed TE count assumption
            ds_kwargs: dict[str, Any] = {}
            if trainer._train_denoiser:
                ds_kwargs["denoiser"] = trainer.denoiser
            for i, (t_enc, flag) in enumerate(zip(trainer.text_encoders, flags)):
                if flag:
                    ds_kwargs[f"text_encoder{i + 1}"] = t_enc
            ds_kwargs["adapter"] = trainer.adapter

            ds_model = deepspeed_utils.prepare_deepspeed_model(cfg.performance.precision, **ds_kwargs)
            ds_model, trainer.optimizer, trainer.lr_scheduler = trainer.accelerator.prepare(
                ds_model, trainer.optimizer, trainer.lr_scheduler
            )
            trainer._grad_sync_handle = ds_model
            trainer._primary_trainable = trainer.adapter
        else:
            if trainer._train_denoiser:
                trainer.denoiser = trainer.accelerator.prepare(trainer.denoiser)
            else:
                trainer.denoiser.to(
                    trainer.accelerator.device,
                    dtype=trainer.denoiser_weight_dtype if trainer.strategies.cast_denoiser(cfg) else None,
                )

            if trainer._train_text_encoder:
                trainer.text_encoders = [
                    (trainer.accelerator.prepare(t_enc) if flag else t_enc)
                    for t_enc, flag in zip(
                        trainer.text_encoders,
                        get_text_encoders_train_flags(cfg.optimizer.learning_rates, trainer.text_encoders),
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
        trainer.accelerator.unwrap_model(trainer.adapter).prepare_grad_etc(trainer._text_encoder, trainer.denoiser)

    def register_state_hooks(self, trainer: Trainer) -> ResumeState:
        """Register adapter-only training checkpoint hooks."""
        return register_adapter_checkpoint_state_hooks(
            trainer.accelerator,
            trainer.adapter,
            save_for_deepspeed=trainer.cfg.performance.deepspeed.deepspeed,
            current_epoch=trainer._current_epoch_state,
            current_step=trainer._current_step_state,
        )

    # ------------------------------------------------------------------
    # Per-epoch / per-step callbacks
    # ------------------------------------------------------------------

    def on_epoch_start(self, trainer: Trainer) -> None:
        """Call adapter's ``on_epoch_start`` (which calls ``.train()``).

        Extracted from ``training_loop.py`` L71.
        """
        trainer.accelerator.unwrap_model(trainer.adapter).on_epoch_start(trainer._text_encoder, trainer.denoiser)

    def on_step_start(self, trainer: Trainer) -> None:
        """Call adapter's ``on_step_start`` if it defines one.

        Absorbed from ``trainer._on_step_start_for_adapter`` callback.
        """
        unwrapped = trainer.accelerator.unwrap_model(trainer.adapter)
        if hasattr(unwrapped, "on_step_start"):
            unwrapped.on_step_start(trainer._text_encoder, trainer.denoiser)

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
        save_adapter_export(
            model_to_save,
            AdapterExportSaveRequest(file=ckpt_file, dtype=save_dtype, metadata=metadata),
        )

        if trainer.cfg.output.huggingface is not None and trainer.cfg.output.huggingface.huggingface_repo_id is not None:
            from library.utils import huggingface_util

            huggingface_util.upload(
                trainer.cfg.output.huggingface,
                ckpt_file,
                "/" + ckpt_name,
                force_sync_upload=force_sync_upload,
            )

    def get_diagnostics_components(self, trainer: Trainer) -> tuple[list[tuple[str, nn.Module]], list[tuple[str, str]] | None]:
        """Return adapter diagnostics components.

        If the adapter implements ``get_diagnostics_components()``, use its
        per-component breakdown (e.g. denoiser modules vs TE modules).
        Otherwise fall back to showing the adapter as a single component.
        """
        adapter = trainer.adapter
        if adapter is not None and hasattr(adapter, "get_diagnostics_components"):
            return adapter.get_diagnostics_components()

        # Fallback: show adapter as a single component
        components: list[tuple[str, nn.Module]] = []
        if adapter is not None:
            components.append(("adapter", adapter))
        aliases = [("trainable_model", "adapter")]
        return components, aliases
