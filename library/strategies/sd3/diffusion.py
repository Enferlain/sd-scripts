from typing import Any, cast

import torch

from library.losses.huber import get_huber_threshold_if_needed
from library.losses.loss import conditional_loss
from library.losses.loss_modifiers import BatchLossOutput
from library.losses.masking import apply_masked_loss
from library.models.sd3.vae import SDVAE
from library.objectives.base import ObjectiveRuntime
from library.objectives.rectified_flow import (
    RectifiedFlowObjectiveRuntime,
    resolve_rectified_flow_prediction_type,
)
from library.strategies.base.context import StrategyPhase
from library.strategies.base.contracts import DiffusionTrainingStrategy
from library.strategies.sd3.encoding import Sd3TextConditioning
from library.training.diffusion import prepare_latents


def encode_sd3_images_to_latents(vae: SDVAE, images: torch.Tensor) -> torch.Tensor:
    """Encode SD3 images to latents using the family-specific VAE output."""
    return vae.encode(images)


def shift_scale_sd3_latents(latents: torch.Tensor, _vae_latent_scale: float) -> torch.Tensor:
    """Apply the SD3 VAE input transform used by the MMDiT training path."""
    return SDVAE.process_in(latents)


def build_sd3_flow_target(latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Build the paper-style SD3 rectified-flow velocity target."""
    return noise - latents


class Sd3DiffusionTrainingStrategy(DiffusionTrainingStrategy):
    """Diffusion-training facet for SD3 flow-matching."""

    def get_noise_pred_and_target(
        self,
        cfg: Any,
        accelerator: Any,
        objective_runtime: ObjectiveRuntime,
        latents: torch.Tensor,
        batch: Any,
        text_encoder_conds: Sd3TextConditioning,
        denoiser: Any,
        trainable_model: Any,
        weight_dtype: torch.dtype,
        train_denoiser: bool,
        fixed_timesteps: torch.Tensor | None = None,
        is_train: bool = True,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample SD3 flow noise, run MMDiT, and build the flow-matching target."""
        rf_runtime = cast(RectifiedFlowObjectiveRuntime, objective_runtime)
        phase = StrategyPhase.TRAIN if is_train else StrategyPhase.VALIDATION
        resolve_rectified_flow_prediction_type(cfg.objective.prediction)

        batch_state = rf_runtime.build_training_batch_state(
            latents,
            device=accelerator.device,
            dtype=weight_dtype,
            fixed_timesteps=fixed_timesteps,
        )
        noisy_model_input = batch_state.noisy_model_input
        timesteps = batch_state.timesteps

        if is_train and cfg.performance.memory.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for text_cond in text_encoder_conds.to_tensor_list():
                if text_cond is not None and text_cond.dtype.is_floating_point:
                    text_cond.requires_grad_(True)

        model_pred = self.call_denoiser(
            cfg,
            accelerator,
            denoiser,
            noisy_model_input,
            timesteps,
            text_encoder_conds,
            batch,
            weight_dtype,
            phase=phase,
            global_step=global_step,
            is_train=is_train,
            train_denoiser=train_denoiser,
        )

        weighting = batch_state.loss_weighting
        target = build_sd3_flow_target(latents, batch_state.noise)

        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for index, custom_attributes in enumerate(batch["custom_attributes"]):
                if custom_attributes.get("diff_output_preservation"):
                    diff_output_pr_indices.append(index)

            if diff_output_pr_indices and hasattr(trainable_model, "set_multiplier"):
                trainable_model.set_multiplier(0.0)
                model_pred_prior = self.call_denoiser(
                    cfg,
                    accelerator,
                    denoiser,
                    noisy_model_input,
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
                target[diff_output_pr_indices] = model_pred_prior.to(target.dtype)

        return model_pred, target, timesteps, weighting

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
        """Process a training or validation batch for SD3 flow matching."""
        with torch.no_grad():
            latents = prepare_latents(
                batch,
                cfg.data.caching,
                accelerator.device,
                vae,
                vae_dtype,
                1.0,
                log_fn=accelerator.print,
                encode_images_to_latents_fn=encode_sd3_images_to_latents,
                shift_scale_latents_fn=shift_scale_sd3_latents,
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
                num_train_timesteps=objective_runtime.num_train_timesteps,
            )
            loss = conditional_loss(
                noise_pred.float(),
                target.float(),
                cfg.loss.loss_type,
                "none",
                huber_c,
                scale=float(cfg.loss.loss_scale),
            )
            loss = loss * weighting
            if cfg.loss.masked.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                loss = apply_masked_loss(loss, batch)
        else:
            loss = conditional_loss(noise_pred.float(), target.float(), "l2", "none", None)

        per_sample_loss = loss.mean([1, 2, 3])

        loss = per_sample_loss
        if is_train:
            loss = loss * batch["loss_weights"].to(loss.device)

        if is_train and cfg.loss.loss_multiplier:
            loss.mul_(float(cfg.loss.loss_multiplier) if cfg.loss.loss_multiplier is not None else 1.0)

        return BatchLossOutput(
            loss=loss.mean(),
            per_sample_loss=loss,
            timesteps=timesteps,
            sampling_loss=per_sample_loss,
        )


__all__ = [
    "Sd3DiffusionTrainingStrategy",
    "build_sd3_flow_target",
    "encode_sd3_images_to_latents",
    "shift_scale_sd3_latents",
]
