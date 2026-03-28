import math
from typing import Any

import torch

from library.losses.loss import conditional_loss, get_huber_threshold_if_needed
from library.losses.loss_modifiers import BatchLossOutput
from library.losses.loss_weighting import apply_masked_loss
from library.models.sd3.vae import SDVAE
from library.strategies.base.contracts import DiffusionTrainingStrategy
from library.strategies.sd3.encoding import build_sd3_attention_masks
from library.training.diffusion import prepare_latents


def resolve_sd3_weighting_scheme(cfg: Any) -> str:
    """Resolve the active SD3 weighting scheme from temporary model config fields."""
    return getattr(cfg.model, "weighting_scheme", "uniform") or "uniform"


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    *,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
) -> torch.Tensor:
    """Compute the SD3 timestep density used for flow-matching training."""
    if weighting_scheme == "logit_normal":
        samples = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        return torch.nn.functional.sigmoid(samples)
    if weighting_scheme == "mode":
        samples = torch.rand(size=(batch_size,), device="cpu")
        return 1 - samples - mode_scale * (torch.cos(math.pi * samples / 2) ** 2 - 1 + samples)
    return torch.rand(size=(batch_size,), device="cpu")


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas: torch.Tensor) -> torch.Tensor:
    """Compute SD3 post-loss weighting from the sampled flow sigmas."""
    if weighting_scheme == "sigma_sqrt":
        return (sigmas**-2.0).float()
    if weighting_scheme == "cosmap":
        denominator = 1 - 2 * sigmas + 2 * sigmas**2
        return 2 / (math.pi * denominator)
    return torch.ones_like(sigmas)


def get_noisy_model_input_and_timesteps(
    cfg: Any,
    latents: torch.Tensor,
    noise: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    fixed_timesteps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build SD3 flow-matching inputs from clean latents and sampled noise."""
    timestep_cfg = cfg.timestep
    batch_size = latents.shape[0]

    if fixed_timesteps is None:
        weighting_scheme = resolve_sd3_weighting_scheme(cfg)
        timestep_samples = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=batch_size,
            logit_mean=float(getattr(cfg.model, "logit_mean", 0.0)),
            logit_std=float(getattr(cfg.model, "logit_std", 1.0)),
            mode_scale=float(getattr(cfg.model, "mode_scale", 1.29)),
        )

        t_min = timestep_cfg.min_timestep if timestep_cfg.min_timestep is not None else 0
        t_max = timestep_cfg.max_timestep if timestep_cfg.max_timestep is not None else 1000
        shift = float(timestep_cfg.discrete_flow_shift)
        timestep_samples = (timestep_samples * shift) / (1 + (shift - 1) * timestep_samples)
        timestep_indices = (timestep_samples * (t_max - t_min) + t_min).long()
        timesteps = timestep_indices.to(device=device, dtype=torch.long)
    else:
        timesteps = fixed_timesteps.to(device=device, dtype=torch.long)

    sigmas = (timesteps.to(dtype) / 1000).view(-1, 1, 1, 1)
    noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents
    return noisy_model_input.to(dtype), timesteps, sigmas.to(dtype)


def encode_sd3_images_to_latents(vae: SDVAE, images: torch.Tensor) -> torch.Tensor:
    """Encode SD3 images to latents using the family-specific VAE output."""
    return vae.encode(images)


def shift_scale_sd3_latents(latents: torch.Tensor, _vae_latent_scale: float) -> torch.Tensor:
    """Apply the SD3 VAE input transform used by the MMDiT training path."""
    return SDVAE.process_in(latents)


class Sd3DiffusionTrainingStrategy(DiffusionTrainingStrategy):
    """Diffusion-training facet for SD3 flow-matching."""

    def _get_cached_text_conds(
        self,
        batch: Any,
        accelerator: Any,
        weight_dtype: torch.dtype,
    ) -> list[torch.Tensor] | list[Any]:
        """Load cached SD3 text-encoder outputs already attached to the batch."""
        te_outputs = batch.get("text_encoder_outputs")
        if te_outputs is None:
            return []

        lg_out = te_outputs["lg_out"].to(accelerator.device, dtype=weight_dtype).clone()
        lg_pooled = te_outputs["lg_pooled"].to(accelerator.device, dtype=weight_dtype).clone()
        t5_out = te_outputs.get("t5_out")
        if t5_out is not None:
            t5_out = t5_out.to(accelerator.device, dtype=weight_dtype).clone()

        clip_l_attn_mask = te_outputs.get("clip_l_attn_mask")
        if clip_l_attn_mask is not None:
            clip_l_attn_mask = clip_l_attn_mask.to(accelerator.device).clone()
        clip_g_attn_mask = te_outputs.get("clip_g_attn_mask")
        if clip_g_attn_mask is not None:
            clip_g_attn_mask = clip_g_attn_mask.to(accelerator.device).clone()
        t5_attn_mask = te_outputs.get("t5_attn_mask")
        if t5_attn_mask is not None:
            t5_attn_mask = t5_attn_mask.to(accelerator.device).clone()

        return self.drop_cached_text_encoder_outputs(
            lg_out,
            t5_out,
            lg_pooled,
            clip_l_attn_mask,
            clip_g_attn_mask,
            t5_attn_mask,
        )

    def _encode_live_text_conds(
        self,
        batch: Any,
        text_encoders: list[Any],
        accelerator: Any,
        cfg: Any,
        weight_dtype: torch.dtype,
        train_text_encoder: bool,
        is_train: bool,
    ) -> list[torch.Tensor | None]:
        """Encode SD3 text conditioning from token-cache tensors or live captions."""
        models = self.get_models_for_text_encoding(cfg, accelerator, text_encoders)

        if cfg.data.caption.weighted_captions:
            raise NotImplementedError("SD3 weighted captions are not implemented in the current strategy port")

        input_ids_dict = batch.get("input_ids")
        if input_ids_dict is not None:
            input_ids_list = [
                input_ids_dict["clip_l"].to(accelerator.device),
                input_ids_dict["clip_g"].to(accelerator.device),
                input_ids_dict["t5"].to(accelerator.device),
            ]
            input_ids_list.extend(build_sd3_attention_masks(self.tokenizers, input_ids_list))
        else:
            captions = batch.get("captions", [])
            if not captions:
                raise ValueError("Batch has neither 'input_ids' nor 'captions' - cannot encode SD3 text")
            input_ids_list = [tensor.to(accelerator.device) for tensor in self.tokenize(captions)]

        with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
            encoded_text_conds = self.encode_tokens(models, input_ids_list)

        for index in range(3):
            if encoded_text_conds[index] is not None:
                encoded_text_conds[index] = encoded_text_conds[index].to(accelerator.device, dtype=weight_dtype)
        for index in range(3, 6):
            if encoded_text_conds[index] is not None:
                encoded_text_conds[index] = encoded_text_conds[index].to(accelerator.device)

        return encoded_text_conds

    def _merge_text_conds(
        self,
        cached_text_conds: list[torch.Tensor | None],
        encoded_text_conds: list[torch.Tensor | None],
    ) -> list[torch.Tensor | None]:
        """Prefer live-encoded outputs when they are recomputed for the current batch."""
        if not cached_text_conds:
            return encoded_text_conds

        merged_text_conds = list(cached_text_conds)
        for index, cond in enumerate(encoded_text_conds):
            if cond is not None:
                merged_text_conds[index] = cond
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
    ) -> list[torch.Tensor | None]:
        """Resolve SD3 text conditioning from cache or a live encoding pass."""
        cached_text_conds = self._get_cached_text_conds(batch, accelerator, weight_dtype)
        needs_live_encoding = not cached_text_conds or any(cond is None for cond in cached_text_conds[:3]) or train_text_encoder

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
        text_encoder_conds: list[torch.Tensor | None],
        denoiser: Any,
        trainable_model: Any,
        weight_dtype: torch.dtype,
        train_denoiser: bool,
        fixed_timesteps: torch.Tensor | None = None,
        is_train: bool = True,
        timestep_runtime: Any | None = None,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample SD3 flow noise, run MMDiT, and build the flow-matching target."""
        del noise_scheduler, timestep_runtime, global_step

        noise = torch.randn_like(latents)
        noisy_model_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(
            cfg,
            latents,
            noise,
            device=accelerator.device,
            dtype=weight_dtype,
            fixed_timesteps=fixed_timesteps,
        )

        if is_train and cfg.performance.memory.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for text_cond in text_encoder_conds:
                if text_cond is not None and text_cond.dtype.is_floating_point:
                    text_cond.requires_grad_(True)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = self.call_denoiser(
                cfg,
                accelerator,
                denoiser,
                noisy_model_input.requires_grad_(train_denoiser),
                timesteps,
                text_encoder_conds,
                batch,
                weight_dtype,
            )

        model_pred = model_pred * (-sigmas) + noisy_model_input
        weighting = compute_loss_weighting_for_sd3(resolve_sd3_weighting_scheme(cfg), sigmas=sigmas)
        target = latents

        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for index, custom_attributes in enumerate(batch["custom_attributes"]):
                if custom_attributes.get("diff_output_preservation"):
                    diff_output_pr_indices.append(index)

            if diff_output_pr_indices and hasattr(trainable_model, "set_multiplier"):
                trainable_model.set_multiplier(0.0)
                with torch.no_grad(), accelerator.autocast():
                    model_pred_prior = self.call_denoiser(
                        cfg,
                        accelerator,
                        denoiser,
                        noisy_model_input,
                        timesteps,
                        text_encoder_conds,
                        batch,
                        weight_dtype,
                        indices=diff_output_pr_indices,
                    )
                trainable_model.set_multiplier(1.0)
                model_pred_prior = model_pred_prior * (-sigmas[diff_output_pr_indices]) + noisy_model_input[diff_output_pr_indices]
                target[diff_output_pr_indices] = model_pred_prior.to(target.dtype)

        return model_pred, target, timesteps, weighting

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
    "compute_density_for_timestep_sampling",
    "compute_loss_weighting_for_sd3",
    "encode_sd3_images_to_latents",
    "get_noisy_model_input_and_timesteps",
    "resolve_sd3_weighting_scheme",
    "shift_scale_sd3_latents",
]
