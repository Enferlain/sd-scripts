from typing import Any

import torch

from library.losses.loss import conditional_loss, get_huber_threshold_if_needed
from library.losses.loss_weighting import apply_masked_loss, post_process_loss
from library.strategies.base.training import DiffusionTrainingStrategy
from library.training.diffusion import get_noise_noisy_latents_and_timesteps, prepare_latents


class SdDiffusionTrainingStrategy(DiffusionTrainingStrategy):
    """Diffusion-training facet for SD 1.5/2.0 training strategies."""

    def _get_text_conds(
        self,
        batch: Any,
        text_encoders: list[Any],
        accelerator: Any,
        cfg: Any,
        train_text_encoder: bool,
        is_train: bool,
        weight_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Get SD text conditioning from cached outputs, cached tokens, or live captions."""
        te_outputs = batch.get("text_encoder_outputs")
        text_encoder_conds = [te_outputs["hidden_state"].to(accelerator.device, dtype=weight_dtype)] if te_outputs is not None else []

        if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
            with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
                if cfg.data.caption.weighted_captions:
                    input_ids_list, weights_list = self.tokenize_with_weights(batch["captions"])
                    encoded_text_encoder_conds = self.encode_tokens_with_weights(
                        self.get_models_for_text_encoding(cfg, accelerator, text_encoders),
                        input_ids_list,
                        weights_list,
                    )
                else:
                    input_ids_dict = batch.get("input_ids")
                    if input_ids_dict is not None:
                        input_ids = [input_ids_dict["clip"].to(accelerator.device)]
                    else:
                        input_ids = [ids.to(accelerator.device) for ids in self.tokenize(batch["captions"])]
                    encoded_text_encoder_conds = self.encode_tokens(
                        self.get_models_for_text_encoding(cfg, accelerator, text_encoders),
                        input_ids,
                    )

                if cfg.performance.precision.full_fp16:
                    encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                for i, cond in enumerate(encoded_text_encoder_conds):
                    if cond is not None:
                        text_encoder_conds[i] = cond

        return text_encoder_conds

    def get_noise_pred_and_target(
        self,
        cfg: Any,
        accelerator: Any,
        noise_scheduler: Any,
        latents: torch.Tensor,
        batch: Any,
        text_encoder_conds: list[Any],
        denoiser: Any,
        trainable_model: Any,
        weight_dtype: torch.dtype,
        train_denoiser: bool,
        fixed_timesteps: torch.Tensor | None = None,
        is_train: bool = True,
        min_timestep_override: int | None = None,
        max_timestep_override: int | None = None,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sample noise, call the denoiser, and build the training target."""
        noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(
            cfg.loss.regularization,
            cfg.timestep,
            cfg.training,
            noise_scheduler,
            latents,
            la_sampler=self.la_sampler,
            global_step=global_step,
            fixed_timesteps=fixed_timesteps,
            is_train=is_train,
            min_timestep_override=min_timestep_override,
            max_timestep_override=max_timestep_override,
            output_dtype=weight_dtype,
        )

        if is_train and cfg.performance.memory.gradient_checkpointing:
            for x in noisy_latents:
                x.requires_grad_(True)
            for t in text_encoder_conds:
                t.requires_grad_(True)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            noise_pred = self.call_denoiser(
                cfg,
                accelerator,
                denoiser,
                noisy_latents.requires_grad_(train_denoiser),
                timesteps,
                text_encoder_conds,
                batch,
                weight_dtype,
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
                        denoiser,
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
        denoiser: Any,
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
        edm2_model: Any | None = None,
        min_timestep_override: int | None = None,
        max_timestep_override: int | None = None,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Process a training or validation batch for SD diffusion training."""
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

        text_encoder_conds = self._get_text_conds(
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
            noise_scheduler,
            latents,
            batch,
            text_encoder_conds,
            denoiser,
            trainable_model,
            weight_dtype,
            train_denoiser,
            is_train=is_train,
            min_timestep_override=min_timestep_override,
            max_timestep_override=max_timestep_override,
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
                loss = apply_masked_loss(loss, batch)
        else:
            loss = conditional_loss(noise_pred.float(), target.float(), "l2", "none", None)

        per_sample_loss = loss.mean([1, 2, 3])

        if is_train and self.la_sampler is not None and hasattr(self.la_sampler, "update"):
            self.la_sampler.update(timesteps.detach(), per_sample_loss.detach())

        loss = per_sample_loss
        if is_train:
            loss = loss * batch["loss_weights"]
            loss = post_process_loss(loss, cfg, timesteps, noise_scheduler)

        if is_train and cfg.loss.loss_multiplier:
            loss.mul_(float(cfg.loss.loss_multiplier) if cfg.loss.loss_multiplier is not None else 1.0)

        pre_scaling_loss = loss.mean()

        if is_train and cfg.loss.edm2.edm2_loss_weighting and edm2_model is not None:
            loss, loss_scaled = edm2_model(loss, timesteps)
            loss_scaled = loss_scaled.mean()
        else:
            loss_scaled = None

        return loss.mean(), pre_scaling_loss, loss_scaled, timesteps
