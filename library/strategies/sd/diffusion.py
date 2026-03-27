from typing import Any

import torch

from library.losses.loss import conditional_loss, get_huber_threshold_if_needed
from library.losses.loss_modifiers import BatchLossOutput
from library.losses.loss_weighting import apply_masked_loss, post_process_loss
from library.strategies.base.contracts import DiffusionTrainingStrategy
from library.training.diffusion import get_noise_noisy_latents_and_timesteps, prepare_latents


class SdDiffusionTrainingStrategy(DiffusionTrainingStrategy):
    """Diffusion-training facet for SD 1.5/2.0 training strategies."""

    def _get_cached_text_conds(
        self,
        batch: Any,
        accelerator: Any,
        weight_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Load cached SD text-encoder outputs already attached to the batch."""
        te_outputs = batch.get("text_encoder_outputs")
        if te_outputs is None:
            return []
        return [te_outputs["hidden_state"].to(accelerator.device, dtype=weight_dtype)]

    def _encode_live_text_conds(
        self,
        batch: Any,
        text_encoders: list[Any],
        accelerator: Any,
        cfg: Any,
        weight_dtype: torch.dtype,
        train_text_encoder: bool,
        is_train: bool,
    ) -> list[torch.Tensor]:
        """Encode SD text conditioning from captions or cached input IDs."""
        models = self.get_models_for_text_encoding(cfg, accelerator, text_encoders)

        with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
            if cfg.data.caption.weighted_captions:
                input_ids_list, weights_list = self.tokenize_with_weights(batch["captions"])
                encoded_text_conds = self.encode_tokens_with_weights(models, input_ids_list, weights_list)
            else:
                input_ids_dict = batch.get("input_ids")
                if input_ids_dict is not None:
                    input_ids_list = [input_ids_dict["clip"].to(accelerator.device)]
                else:
                    input_ids_list = [ids.to(accelerator.device) for ids in self.tokenize(batch["captions"])]
                encoded_text_conds = self.encode_tokens(models, input_ids_list)

            if cfg.performance.precision.full_fp16:
                encoded_text_conds = [cond.to(weight_dtype) for cond in encoded_text_conds]

        return encoded_text_conds

    def _merge_text_conds(
        self,
        cached_text_conds: list[torch.Tensor],
        encoded_text_conds: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Prefer live-encoded outputs when they are recomputed for the current batch."""
        if not cached_text_conds:
            return encoded_text_conds

        merged_text_conds = list(cached_text_conds)
        for i, cond in enumerate(encoded_text_conds):
            if cond is not None:
                merged_text_conds[i] = cond
        return merged_text_conds

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
        """
        Get SD text conditioning for this batch.

        Diffusion owns choosing whether the batch should use cached TE outputs
        or run a live encoding pass; encoding still owns how tokens become text
        encoder outputs.
        """
        cached_text_conds = self._get_cached_text_conds(batch, accelerator, weight_dtype)
        needs_live_encoding = not cached_text_conds or cached_text_conds[0] is None or train_text_encoder

        if not needs_live_encoding:
            return cached_text_conds

        encoded_text_conds = self._encode_live_text_conds(
            batch=batch,
            text_encoders=text_encoders,
            accelerator=accelerator,
            cfg=cfg,
            weight_dtype=weight_dtype,
            train_text_encoder=train_text_encoder,
            is_train=is_train,
        )
        return self._merge_text_conds(cached_text_conds, encoded_text_conds)

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
        timestep_runtime: Any | None = None,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sample noise, call the denoiser, and build the training target."""
        noise, noisy_latents, timesteps = get_noise_noisy_latents_and_timesteps(
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
        timestep_runtime: Any | None = None,
        global_step: int = 0,
    ) -> BatchLossOutput:
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
                loss = apply_masked_loss(loss, batch)
        else:
            loss = conditional_loss(noise_pred.float(), target.float(), "l2", "none", None)

        per_sample_loss = loss.mean([1, 2, 3])

        loss = per_sample_loss
        if is_train:
            loss = loss * batch["loss_weights"]
            loss = post_process_loss(loss, cfg, timesteps, noise_scheduler)

        if is_train and cfg.loss.loss_multiplier:
            loss.mul_(float(cfg.loss.loss_multiplier) if cfg.loss.loss_multiplier is not None else 1.0)

        return BatchLossOutput(
            loss=loss.mean(),
            per_sample_loss=loss,
            timesteps=timesteps,
            sampling_loss=per_sample_loss,
        )
