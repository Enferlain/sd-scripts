"""
AdapterMode — TrainingMode implementation for PEFT/adapter training.

This began as a direct extraction of adapter-specific logic that was previously
inline in the phase files and trainer. The class now also owns the repo-owned
adapter runtime/orchestration seam that replaced the older compatibility-era
optimizer boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    build_adapter,
    build_adapter_from_weights,
    load_adapter_export,
    register_adapter_checkpoint_state_hooks,
    save_adapter_export,
)
from library.adapters.methods.peft.config_resolution import (
    build_adapter_runtime_spec,
    get_adapter_peft_config,
    resolve_adapter_method_registration,
)
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


@dataclass(slots=True)
class PeftContinuationPlan:
    """AdapterMode-owned continuation plan derived from intent-shaped config."""

    continue_from: str | None
    continue_mode: str


def _build_continuation_plan(peft_config) -> PeftContinuationPlan:
    continue_from = peft_config.continue_from
    continue_mode = peft_config.continue_mode

    if continue_from is not None and continue_mode is None:
        raise ValueError(
            "adapter.peft.continue_mode must be explicitly set when adapter.peft.continue_from is provided. "
            "Choose 'strict' or 'initialize_from_artifact'."
        )
    if continue_mode is None:
        continue_mode = "strict"

    return PeftContinuationPlan(continue_from=continue_from, continue_mode=continue_mode)


def _build_adapter_request(
    *,
    vae,
    text_encoders: list[nn.Module],
    denoiser,
    runtime_spec: AdapterRuntimeSpec,
    resolved_targets,
    for_inference: bool = False,
) -> AdapterBuildRequest:
    return AdapterBuildRequest(
        adapter=runtime_spec,
        context=AdapterBuildContext(
            model=AdapterModelContext(vae=vae, text_encoder=text_encoders, denoiser=denoiser),
            multiplier=1.0,
            for_inference=for_inference,
        ),
        resolved_targets=resolved_targets,
    )


class AdapterMode:
    """Adapter (peft, other) training mode.

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
        peft_config = get_adapter_peft_config(cfg)
        if peft_config is None:
            raise ValueError("mode=adapter requires adapter.peft config.")
        adapter_registration = resolve_adapter_method_registration(peft_config)
        runtime_spec = build_adapter_runtime_spec(peft_config)
        continuation_plan = _build_continuation_plan(peft_config)
        trainer.adapter_method_name = adapter_registration.name

        target_selection = resolve_adapter_target_selection(
            model_type=cfg.model.model_type,
            denoiser=denoiser,
            text_encoders=text_encoders,
            learning_rates=cfg.optimizer.learning_rates,
        )
        trainer._train_denoiser = target_selection.train_denoiser
        trainer._train_text_encoder = target_selection.train_any_text_encoder

        # Create adapter
        build_request = _build_adapter_request(
            vae=vae,
            text_encoders=text_encoders,
            denoiser=denoiser,
            runtime_spec=runtime_spec,
            resolved_targets=target_selection.resolved_targets,
        )

        if continuation_plan.continue_from is not None and continuation_plan.continue_mode == "strict":
            loaded_runtime = build_adapter_from_weights(build_request, continuation_plan.continue_from)
            adapter = loaded_runtime.adapter
        else:
            if build_request.adapter.adapter_type == "lora" and "dropout" not in build_request.adapter.settings:
                build_request.adapter.settings["dropout"] = getattr(peft_config.lora, "dropout", None)
            adapter = build_adapter(build_request)

        if adapter is None:
            raise RuntimeError("Adapter creation returned None - check adapter module configuration")

        if hasattr(adapter, "prepare_adapter"):
            adapter.prepare_adapter(cfg)

        if peft_config.scale_weight_norms and not hasattr(adapter, "apply_max_norm_regularization"):
            logger.warning("warning: scale_weight_norms is specified but the peft does not support it")
            peft_config.scale_weight_norms = False

        trainer.strategies.post_process_trainable(cfg, accelerator, adapter, text_encoders, denoiser)

        # Apply adapter to denoiser and text_encoder
        adapter.apply_to(text_encoder, denoiser, trainer._train_text_encoder, trainer._train_denoiser)

        if continuation_plan.continue_from is not None and continuation_plan.continue_mode == "initialize_from_artifact":
            info = load_adapter_export(adapter, AdapterExportLoadRequest(file=continuation_plan.continue_from))
            accelerator.print(f"initialized adapter from {continuation_plan.continue_from}: {info}")

        adapter.requires_grad_(True)
        trainer.adapter = adapter
        trainer.adapter_resolved_targets = build_request.resolved_targets
        trainer.net_kwargs = {}

    def configure_trainable_precision(self, trainer: Trainer) -> None:
        """Adapter-specific precision: cast adapter, freeze base model.

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
        peft_config = get_adapter_peft_config(cfg)
        if peft_config is None:
            raise ValueError("mode=adapter requires adapter.peft config.")

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
        peft_config = get_adapter_peft_config(cfg)

        if peft_config is not None and peft_config.scale_weight_norms and accelerator.sync_gradients:
            keys_scaled, mean_norm, _maximum_norm = accelerator.unwrap_model(trainer.adapter).apply_max_norm_regularization(
                peft_config.scale_weight_norms, accelerator.device
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

        logger.info("[checkpoint] saving checkpoint: %s", ckpt_file)
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
