from typing import Any, cast

import torch

from library.losses.huber import get_huber_threshold_if_needed
from library.losses.loss import conditional_loss
from library.losses.loss_modifiers import BatchLossOutput
from library.losses.masking import apply_masked_loss
from library.objectives.base import ObjectiveRuntime
from library.objectives.ddpm import (
    DDPMObjectiveRuntime,
    build_ddpm_training_target,
    post_process_ddpm_loss,
    prepare_ddpm_training_inputs,
)
from library.strategies.base.context import StrategyPhase
from library.strategies.base.contracts import DiffusionTrainingStrategy
from library.training.diffusion import prepare_latents


class SdDiffusionTrainingStrategy(DiffusionTrainingStrategy):
    """Diffusion-training facet for SD 1.5/2.0 training strategies."""

    def get_noise_pred_and_target(
        self,
        cfg: Any,
        accelerator: Any,
        objective_runtime: ObjectiveRuntime,
        latents: torch.Tensor,
        batch: Any,
        text_encoder_conds: list[Any],
        denoiser: Any,
        trainable_model: Any,
        weight_dtype: torch.dtype,
        train_denoiser: bool,
        fixed_timesteps: torch.Tensor | None = None,
        is_train: bool = True,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sample noise, call the denoiser, and build the training target."""
        phase = StrategyPhase.TRAIN if is_train else StrategyPhase.VALIDATION
        ddpm_runtime = cast(DDPMObjectiveRuntime, objective_runtime)
        noise_scheduler = ddpm_runtime.noise_scheduler
        noise, noisy_latents, timesteps = prepare_ddpm_training_inputs(
            cfg.loss.regularization,
            cfg.timestep,
            cfg.training,
            noise_scheduler,
            latents,
            timestep_runtime=ddpm_runtime.timestep_runtime,
            global_step=global_step,
            fixed_timesteps=fixed_timesteps,
            is_train=is_train,
            output_dtype=weight_dtype,
        )

        if is_train and cfg.performance.memory.gradient_checkpointing:
            for x in noisy_latents:
                x.requires_grad_(True)
            for t in text_encoder_conds:
                t.requires_grad_(True)

        noise_pred = self.call_denoiser(
            cfg,
            accelerator,
            denoiser,
            noisy_latents,
            timesteps,
            text_encoder_conds,
            batch,
            weight_dtype,
            phase=phase,
            global_step=global_step,
            is_train=is_train,
            train_denoiser=train_denoiser,
        )

        target = build_ddpm_training_target(noise_scheduler, latents, noise, timesteps, cfg.objective.prediction)

        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0 and hasattr(trainable_model, "set_multiplier"):
                trainable_model.set_multiplier(0.0)
                noise_pred_prior = self.call_denoiser(
                    cfg,
                    accelerator,
                    denoiser,
                    noisy_latents,
                    timesteps,
                    text_encoder_conds,
                    batch,
                    weight_dtype,
                    phase=phase,
                    global_step=global_step,
                    is_train=is_train,
                    train_denoiser=False,
                    sample_indices=tuple(diff_output_pr_indices),
                    enable_grad=False,
                )
                trainable_model.set_multiplier(1.0)
                target[diff_output_pr_indices] = noise_pred_prior.to(target.dtype)

        return noise_pred, target, timesteps, None

    def process_batch(
        self,
        batch: Any,
        text_encoders: list[Any],
        denoiser: Any,
        trainable_model: Any,
        vae: Any,
        objective_runtime: ObjectiveRuntime,
        vae_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        accelerator: Any,
        cfg: Any,
        is_train: bool = True,
        train_text_encoder: bool = True,
        train_denoiser: bool = True,
        global_step: int = 0,
    ) -> BatchLossOutput:
        """Process a training or validation batch for SD diffusion training."""
        ddpm_runtime = cast(DDPMObjectiveRuntime, objective_runtime)

        with torch.no_grad():
            latents = prepare_latents(
                batch,
                cfg.data.caching,
                accelerator.device,
                vae,
                vae_dtype,
                self.vae_latent_scale,
                log_fn=accelerator.print,
            )

        text_encoder_conds = self.resolve_conditioning(
            batch=batch,
            text_encoders=text_encoders,
            accelerator=accelerator,
            cfg=cfg,
            train_text_encoder=train_text_encoder,
            is_train=is_train,
            weight_dtype=weight_dtype,
        )

        noise_pred, target, timesteps, weighting = self.get_noise_pred_and_target(
            cfg,
            accelerator,
            objective_runtime,
            latents,
            batch,
            text_encoder_conds,
            denoiser,
            trainable_model,
            weight_dtype,
            train_denoiser,
            is_train=is_train,
            global_step=global_step,
        )

        if is_train:
            huber_c = get_huber_threshold_if_needed(
                cfg.loss,
                cfg.loss.huber,
                timesteps,
                num_train_timesteps=ddpm_runtime.num_train_timesteps,
                alphas_cumprod=ddpm_runtime.alphas_cumprod,
            )
            loss = conditional_loss(
                noise_pred.float(), target.float(), cfg.loss.loss_type, "none", huber_c, scale=float(cfg.loss.loss_scale)
            )
            if weighting is not None:
                loss = loss * weighting
            if cfg.loss.masked.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                loss = apply_masked_loss(loss, batch)
        else:
            loss = conditional_loss(noise_pred.float(), target.float(), "l2", "none", None)

        per_sample_loss = loss.mean([1, 2, 3])

        loss = per_sample_loss
        if is_train:
            loss = loss * batch["loss_weights"]
            loss = post_process_ddpm_loss(loss, cfg, timesteps, ddpm_runtime.noise_scheduler)

        if is_train and cfg.loss.loss_multiplier:
            loss.mul_(float(cfg.loss.loss_multiplier) if cfg.loss.loss_multiplier is not None else 1.0)

        return BatchLossOutput(
            loss=loss.mean(),
            per_sample_loss=loss,
            timesteps=timesteps,
            sampling_loss=per_sample_loss,
        )
