from typing import Any

import torch

from library.losses.loss import conditional_loss, get_huber_threshold_if_needed
from library.losses.loss_modifiers import BatchLossOutput
from library.losses.loss_weighting import apply_masked_loss, post_process_loss
from library.strategies.base.contracts import DiffusionTrainingStrategy
from library.training.diffusion import get_noise_noisy_latents_and_timesteps, prepare_latents


class SdxlDiffusionTrainingStrategy(DiffusionTrainingStrategy):
    """Diffusion-training facet for SDXL training strategies."""

    def _get_cached_text_conds(
        self,
        batch: Any,
        accelerator: Any,
        weight_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Load cached SDXL text-encoder outputs already attached to the batch."""
        te_outputs = batch.get("text_encoder_outputs")
        if te_outputs is None:
            return []
        return [
            te_outputs["hidden_state1"].to(accelerator.device, dtype=weight_dtype),
            te_outputs["hidden_state2"].to(accelerator.device, dtype=weight_dtype),
            te_outputs["pool2"].to(accelerator.device, dtype=weight_dtype),
        ]

    def _encode_live_text_conds(
        self,
        cfg: Any,
        accelerator: Any,
        batch: Any,
        text_encoders: list[Any],
        weight_dtype: torch.dtype,
        train_text_encoder: bool,
        is_train: bool,
    ) -> list[torch.Tensor]:
        """Encode SDXL text conditioning from captions or cached input IDs."""
        te_device = text_encoders[0].device
        models = self.get_models_for_text_encoding(cfg, accelerator, text_encoders)

        with torch.set_grad_enabled(is_train and train_text_encoder):
            if cfg.data.caption.weighted_captions:
                captions = batch.get("captions", [])
                if not captions:
                    raise ValueError("Weighted captions require raw captions in the batch - cannot encode text from cached inputs alone")

                input_ids_list, weights_list = self.tokenize_with_weights(captions)
                input_ids_list = [input_ids.to(te_device) for input_ids in input_ids_list]
                encoded_text_conds = self.encode_tokens_with_weights(models, input_ids_list, weights_list)
            else:
                input_ids = batch.get("input_ids")

                if input_ids is None:
                    captions = batch.get("captions", [])
                    if not captions:
                        raise ValueError("Batch has neither 'input_ids' nor 'captions' - cannot encode text")

                    input_ids_list = self.tokenize(captions)
                    input_ids_list = [input_ids.to(te_device) for input_ids in input_ids_list]
                else:
                    input_ids_list = [
                        input_ids["clip_l"].to(te_device),
                        input_ids["clip_g"].to(te_device),
                    ]

                encoded_text_conds = self.encode_tokens(models, input_ids_list)

            return [cond.to(accelerator.device, dtype=weight_dtype) for cond in encoded_text_conds]

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
        cfg: Any,
        accelerator: Any,
        batch: Any,
        text_encoders: list[Any],
        weight_dtype: torch.dtype,
        train_text_encoder: bool,
        is_train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get SDXL text conditioning from cached outputs, cached tokens, or live captions.

        Returns:
            Tuple of (encoder_hidden_states1, encoder_hidden_states2, pool2).
        """
        cached_text_conds = self._get_cached_text_conds(batch, accelerator, weight_dtype)
        needs_live_encoding = not cached_text_conds or any(cond is None for cond in cached_text_conds) or train_text_encoder

        if not needs_live_encoding:
            return tuple(cached_text_conds)

        encoded_text_conds = self._encode_live_text_conds(
            cfg=cfg,
            accelerator=accelerator,
            batch=batch,
            text_encoders=text_encoders,
            weight_dtype=weight_dtype,
            train_text_encoder=train_text_encoder,
            is_train=is_train,
        )
        return tuple(self._merge_text_conds(cached_text_conds, encoded_text_conds))

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

        text_encoder_conds = self._get_text_conds(
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
