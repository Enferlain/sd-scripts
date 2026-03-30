from typing import Any

import torch

from library.losses.loss import conditional_loss, get_huber_threshold_if_needed
from library.losses.loss_modifiers import BatchLossOutput
from library.losses.loss_weighting import apply_masked_loss, post_process_loss
from library.objectives.ddpm import prepare_ddpm_training_inputs
from library.strategies.base.contracts import DiffusionTrainingStrategy
from library.training.diffusion import prepare_latents


class SdxlDiffusionTrainingStrategy(DiffusionTrainingStrategy):
    """Diffusion-training facet for SDXL training strategies."""

    def get_noise_pred_and_target(
        self,
        cfg: Any,
        accelerator: Any,
        noise_scheduler: Any,
        latents: torch.Tensor,
        batch: Any,
        text_encoder_conds: Any,
        unet: Any,
        trainable_model: Any,
        weight_dtype: torch.dtype,
        train_denoiser: bool,
        fixed_timesteps: torch.Tensor | None = None,
        is_train: bool = True,
        timestep_runtime: Any | None = None,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Sample noise, call the SDXL denoiser, and build the training target.

        Returns:
            Tuple of (noise_pred, target, timesteps, weighting).
        """
        noise, noisy_latents, timesteps = prepare_ddpm_training_inputs(
            cfg.loss.regularization,
            cfg.timestep,
            cfg.training,
            noise_scheduler,
            latents,
            timestep_runtime=timestep_runtime,
            global_step=global_step,
            fixed_timesteps=fixed_timesteps,
            is_train=is_train,
            output_dtype=weight_dtype,
        )

        if is_train and cfg.performance.memory.gradient_checkpointing:
            for x in noisy_latents:
                x.requires_grad_(True)
            if isinstance(text_encoder_conds, (list, tuple)):
                for t in text_encoder_conds:
                    if t is not None and hasattr(t, "requires_grad_"):
                        t.requires_grad_(True)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            noise_pred = self.call_denoiser(
                cfg, accelerator, unet, noisy_latents.requires_grad_(train_denoiser), timesteps, text_encoder_conds, batch, weight_dtype
            )

        if cfg.loss.v_parameterization:
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0 and hasattr(trainable_model, "set_multiplier"):
                trainable_model.set_multiplier(0.0)
                with torch.no_grad(), accelerator.autocast():
                    noise_pred_prior = self.call_denoiser(
                        cfg,
                        accelerator,
                        unet,
                        noisy_latents,
                        timesteps,
                        text_encoder_conds,
                        batch,
                        weight_dtype,
                        indices=diff_output_pr_indices,
                    )
                trainable_model.set_multiplier(1.0)
                target[diff_output_pr_indices] = noise_pred_prior.to(target.dtype)

        return noise_pred, target, timesteps, None

    def process_batch(
        self,
        batch: Any,
        text_encoders: list[Any],
        unet: Any,
        trainable_model: Any,
        vae: Any,
        noise_scheduler: Any,
        vae_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        accelerator: Any,
        cfg: Any,
        is_train: bool = True,
        train_text_encoder: bool = True,
        train_denoiser: bool = True,
        timestep_runtime: Any | None = None,
        global_step: int = 0,
    ) -> BatchLossOutput:
        """Process a training or validation batch for SDXL diffusion training."""
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
            cfg,
            accelerator,
            batch,
            text_encoders,
            weight_dtype,
            train_text_encoder=train_text_encoder,
            is_train=is_train,
        )

        noise_pred, target, timesteps, weighting = self.get_noise_pred_and_target(
            cfg,
            accelerator,
            noise_scheduler,
            latents,
            batch,
            text_encoder_conds,
            unet,
            trainable_model,
            weight_dtype,
            train_denoiser,
            is_train=is_train,
            timestep_runtime=timestep_runtime,
            global_step=global_step,
        )

        if is_train:
            huber_c = get_huber_threshold_if_needed(cfg.loss, cfg.loss.huber, timesteps, noise_scheduler)
            loss = conditional_loss(
                noise_pred.float(), target.float(), cfg.loss.loss_type, "none", huber_c, scale=float(cfg.loss.loss_scale)
            )
            if weighting is not None:
                loss = loss * weighting
            if cfg.loss.masked.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                if cfg.loss.masked.masked_loss:
                    has_cond = "conditioning_images" in batch
                    has_alpha = "alpha_masks" in batch and batch["alpha_masks"] is not None
                    if not has_cond and not has_alpha:
                        raise ValueError(
                            "cfg.loss.masked.masked_loss=True but no masks found in batch. "
                            "Ensure your dataset has alpha channels or conditioning images. "
                            "Set cfg.loss.masked.masked_loss=False if masking is not intended."
                        )
                loss = apply_masked_loss(loss, batch)
        else:
            loss = conditional_loss(noise_pred.float(), target.float(), "l2", "none", None)

        per_sample_loss = loss.mean([1, 2, 3])

        loss = per_sample_loss
        if is_train:
            loss = loss * batch["loss_weights"].to(loss.device)
            loss = post_process_loss(loss, cfg, timesteps, noise_scheduler)

        if is_train and cfg.loss.loss_multiplier:
            loss.mul_(float(cfg.loss.loss_multiplier) if cfg.loss.loss_multiplier is not None else 1.0)

        return BatchLossOutput(
            loss=loss.mean(),
            per_sample_loss=loss,
            timesteps=timesteps,
            sampling_loss=per_sample_loss,
        )
