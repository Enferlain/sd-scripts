from __future__ import annotations

import logging
from typing import Any

import torch
from diffusers import DDPMScheduler
from torch.types import Number

from library.config.dataclasses.loss import RegularizationConfig
from library.config.dataclasses.timestep import TimestepConfig
from library.config.dataclasses.training import TrainingConfig
from library.losses.loss_modifiers import build_loss_modifier
from library.objectives.base import ObjectiveDefinition, ObjectiveRuntime
from library.timesteps.runtime import build_timestep_runtime
from library.training.noise_utils import apply_noise_offset, pyramid_noise_like


logger = logging.getLogger(__name__)

DDPM_PREDICTION_TYPE_EPSILON = "epsilon"
DDPM_PREDICTION_TYPE_V = "v_prediction"


def resolve_ddpm_prediction_type(prediction: str) -> str:
    """Validate and return the active DDPM prediction-target convention."""
    if prediction not in {DDPM_PREDICTION_TYPE_EPSILON, DDPM_PREDICTION_TYPE_V}:
        raise ValueError(f"Unsupported DDPM prediction type: {prediction!r}")
    return prediction


def build_ddpm_training_target(
    noise_scheduler: Any,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    """Build the DDPM training target tensor for the chosen prediction convention."""
    resolved_prediction_type = resolve_ddpm_prediction_type(prediction_type)
    if resolved_prediction_type == DDPM_PREDICTION_TYPE_V:
        return noise_scheduler.get_velocity(latents, noise, timesteps)
    return noise


def build_ddpm_noise_scheduler(cfg: Any, device: torch.device) -> DDPMScheduler:
    """Create the DDPM-style scheduler used by the current diffusion objective path."""
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )

    if cfg.loss.regularization.zero_terminal_snr:
        fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    prepare_scheduler_for_custom_training(noise_scheduler, device)
    return noise_scheduler


def prepare_scheduler_for_custom_training(noise_scheduler: Any, device: torch.device) -> None:
    """Attach derived scheduler state used by the active DDPM-style runtime path."""
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to(device)


def fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler: Any) -> None:
    """Adjust scheduler betas to enforce zero terminal SNR."""
    logger.info("fix noise scheduler betas: https://arxiv.org/abs/2305.08891")

    def enforce_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_t = alphas_bar_sqrt[-1].clone()
        alphas_bar_sqrt -= alphas_bar_sqrt_t
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_t)

        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        return 1 - alphas

    betas = enforce_zero_terminal_snr(noise_scheduler.betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    noise_scheduler.betas = betas
    noise_scheduler.alphas = alphas
    noise_scheduler.alphas_cumprod = alphas_cumprod


def apply_snr_weight(
    loss: torch.Tensor, timesteps: torch.Tensor, noise_scheduler: DDPMScheduler, gamma: Number, v_prediction: bool = False
) -> torch.Tensor:
    """Apply Min-SNR weighting to per-sample DDPM diffusion loss."""
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
    return loss * snr_weight


def get_snr_scale(timesteps: torch.Tensor, noise_scheduler: DDPMScheduler) -> torch.Tensor:
    """Compute the DDPM SNR scaling factor used by v-pred post-processing."""
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)
    return snr_t / (snr_t + 1)


def scale_v_prediction_loss_like_noise_prediction(
    loss: torch.Tensor, timesteps: torch.Tensor, noise_scheduler: DDPMScheduler
) -> torch.Tensor:
    """Scale v-prediction loss to match the DDPM epsilon-prediction loss shape."""
    return loss * get_snr_scale(timesteps, noise_scheduler)


def add_v_prediction_like_loss(
    loss: torch.Tensor, timesteps: torch.Tensor, noise_scheduler: DDPMScheduler, v_pred_like_loss: torch.Tensor
) -> torch.Tensor:
    """Add the configured DDPM v-pred-like auxiliary loss term."""
    scale = get_snr_scale(timesteps, noise_scheduler)
    return loss + loss / scale * v_pred_like_loss


def apply_debiased_estimation(
    loss: torch.Tensor, timesteps: torch.Tensor, noise_scheduler: DDPMScheduler, v_prediction: bool = False
) -> torch.Tensor:
    """Apply DDPM debiased-estimation weighting to per-sample loss."""
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)
    if v_prediction:
        weight = 1 / (snr_t + 1)
    else:
        weight = 1 / torch.sqrt(snr_t)
    return weight * loss


def post_process_ddpm_loss(loss: torch.Tensor, cfg: Any, timesteps: torch.Tensor, noise_scheduler: DDPMScheduler) -> torch.Tensor:
    """Apply configured DDPM-only post-loss weighting in the shared order."""
    is_v_prediction = resolve_ddpm_prediction_type(cfg.objective.prediction) == DDPM_PREDICTION_TYPE_V
    if cfg.loss.snr.min_snr_gamma:
        loss = apply_snr_weight(loss, timesteps, noise_scheduler, cfg.loss.snr.min_snr_gamma, is_v_prediction)
    if cfg.loss.snr.scale_v_pred_loss_like_noise_pred:
        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
    if cfg.loss.snr.v_pred_like_loss:
        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, cfg.loss.snr.v_pred_like_loss)
    if cfg.loss.snr.debiased_estimation_loss:
        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, is_v_prediction)
    return loss


def prepare_ddpm_training_inputs(
    regularization_config: RegularizationConfig,
    timestep_config: TimestepConfig,
    training_config: TrainingConfig,
    noise_scheduler: Any,
    latents: torch.Tensor,
    timestep_runtime=None,
    la_sampler=None,
    global_step: int = 0,
    fixed_timesteps=None,
    is_train: bool = True,
    min_timestep_override=None,
    max_timestep_override=None,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate DDPM-style training noise, noisy latents, and timesteps."""
    noise = torch.randn_like(latents, device=latents.device)
    if regularization_config.noise_offset and is_train:
        noise_offset = (
            torch.rand(1, device=latents.device) * regularization_config.noise_offset
            if regularization_config.noise_offset_random_strength
            else regularization_config.noise_offset
        )
        noise = apply_noise_offset(latents, noise, noise_offset, regularization_config.adaptive_noise_scale)

    b_size = latents.shape[0]

    runtime = timestep_runtime
    if runtime is None:
        runtime = build_timestep_runtime(
            timestep_config,
            noise_scheduler,
            sampler_override=la_sampler,
            min_timestep_override=min_timestep_override,
            max_timestep_override=max_timestep_override,
            global_step=global_step,
        )

    timesteps = runtime.sample_timesteps(
        timestep_config=timestep_config,
        training_config=training_config,
        noise_scheduler=noise_scheduler,
        batch_size=b_size,
        device=latents.device,
        global_step=global_step,
        fixed_timesteps=fixed_timesteps,
        is_train=is_train,
    )

    if regularization_config.multires_noise_iterations and is_train:
        noise = pyramid_noise_like(
            noise, latents.device, regularization_config.multires_noise_iterations, regularization_config.multires_noise_discount
        )

    if regularization_config.ip_noise_gamma and is_train:
        strength = (
            torch.rand(1, device=latents.device) * regularization_config.ip_noise_gamma
            if regularization_config.ip_noise_gamma_random_strength
            else regularization_config.ip_noise_gamma
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise + strength * torch.randn_like(latents), timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    if output_dtype is not None:
        noisy_latents = noisy_latents.to(output_dtype)

    return noise, noisy_latents, timesteps


class DDPMObjective(ObjectiveDefinition):
    """Default diffusion-style objective/runtime owner."""

    name = "ddpm"

    def build_runtime(self, cfg: Any, accelerator: Any) -> ObjectiveRuntime:
        noise_scheduler = build_ddpm_noise_scheduler(cfg, accelerator.device)
        timestep_runtime = build_timestep_runtime(cfg.timestep, noise_scheduler, accelerator)
        loss_modifier = build_loss_modifier(cfg.loss, cfg.training, noise_scheduler, accelerator)
        return ObjectiveRuntime(
            noise_scheduler=noise_scheduler,
            timestep_runtime=timestep_runtime,
            loss_modifier=loss_modifier,
        )
