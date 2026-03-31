from __future__ import annotations

import math

import torch

from library.config.dataclasses.timestep import TimestepConfig
from library.objectives.ddpm import DDPMObjective
from library.timesteps.continuous_sampling import apply_training_shift, sample_continuous_timesteps


def compute_flow_matching_timestep_density(
    timestep_sampling: str,
    batch_size: int,
    *,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    cosine_shape_scale: float = 1.29,
) -> torch.Tensor:
    """Compute pre-index timestep values for flow-matching training."""
    return sample_continuous_timesteps(
        timestep_sampling=timestep_sampling,
        batch_size=batch_size,
        logit_mean=logit_mean,
        logit_std=logit_std,
        cosine_shape_scale=cosine_shape_scale,
    )


def compute_flow_matching_loss_weighting(loss_weighting_scheme: str, sigmas: torch.Tensor) -> torch.Tensor:
    """Compute post-loss weighting from sampled flow sigmas."""
    if loss_weighting_scheme == "sigma_sqrt":
        return (sigmas**-2.0).float()
    if loss_weighting_scheme == "cosmap":
        denominator = 1 - 2 * sigmas + 2 * sigmas**2
        return 2 / (math.pi * denominator)
    return torch.ones_like(sigmas)


def build_flow_matching_model_input_and_timesteps(
    timestep_config: TimestepConfig,
    latents: torch.Tensor,
    noise: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    fixed_timesteps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build flow-matching model input, timesteps, and sigmas from clean latents and noise."""
    batch_size = latents.shape[0]

    if fixed_timesteps is None:
        u = compute_flow_matching_timestep_density(
            timestep_sampling=timestep_config.timestep_sampling,
            batch_size=batch_size,
            logit_mean=float(timestep_config.logit_mean),
            logit_std=float(timestep_config.logit_std),
            cosine_shape_scale=float(timestep_config.cosine_shape_scale),
        )

        t_min = timestep_config.min_timestep if timestep_config.min_timestep is not None else 0
        t_max = timestep_config.max_timestep if timestep_config.max_timestep is not None else 1000
        u = apply_training_shift(u, float(timestep_config.training_shift))
        timestep_indices = (u * (t_max - t_min) + t_min).long()
        timesteps = timestep_indices.to(device=device, dtype=torch.long)
    else:
        timesteps = fixed_timesteps.to(device=device, dtype=torch.long)

    sigmas = (timesteps.to(dtype) / 1000).view(-1, 1, 1, 1)
    noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents
    return noisy_model_input.to(dtype), timesteps, sigmas.to(dtype)


class RectifiedFlowObjective(DDPMObjective):
    """Rectified-flow objective/runtime owner for the current SD3 path."""

    name = "rectified_flow"
