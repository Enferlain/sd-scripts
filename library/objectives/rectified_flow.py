from __future__ import annotations

import math

import torch

from library.config.dataclasses.timestep import TimestepConfig
from library.objectives.ddpm import DDPMObjective


def compute_flow_matching_timestep_density(
    weighting_scheme: str,
    batch_size: int,
    *,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
) -> torch.Tensor:
    """Compute timestep-density samples for flow-matching training."""
    if weighting_scheme == "logit_normal":
        samples = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        return torch.nn.functional.sigmoid(samples)
    if weighting_scheme == "mode":
        samples = torch.rand(size=(batch_size,), device="cpu")
        return 1 - samples - mode_scale * (torch.cos(math.pi * samples / 2) ** 2 - 1 + samples)
    return torch.rand(size=(batch_size,), device="cpu")


def compute_flow_matching_loss_weighting(weighting_scheme: str, sigmas: torch.Tensor) -> torch.Tensor:
    """Compute post-loss weighting from sampled flow sigmas."""
    if weighting_scheme == "sigma_sqrt":
        return (sigmas**-2.0).float()
    if weighting_scheme == "cosmap":
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
        timestep_samples = compute_flow_matching_timestep_density(
            weighting_scheme=timestep_config.weighting_scheme,
            batch_size=batch_size,
            logit_mean=float(timestep_config.logit_mean),
            logit_std=float(timestep_config.logit_std),
            mode_scale=float(timestep_config.mode_scale),
        )

        t_min = timestep_config.min_timestep if timestep_config.min_timestep is not None else 0
        t_max = timestep_config.max_timestep if timestep_config.max_timestep is not None else 1000
        shift = float(timestep_config.discrete_flow_shift)
        timestep_samples = (timestep_samples * shift) / (1 + (shift - 1) * timestep_samples)
        timestep_indices = (timestep_samples * (t_max - t_min) + t_min).long()
        timesteps = timestep_indices.to(device=device, dtype=torch.long)
    else:
        timesteps = fixed_timesteps.to(device=device, dtype=torch.long)

    sigmas = (timesteps.to(dtype) / 1000).view(-1, 1, 1, 1)
    noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents
    return noisy_model_input.to(dtype), timesteps, sigmas.to(dtype)


class RectifiedFlowObjective(DDPMObjective):
    """Rectified-flow objective/runtime owner for the current SD3 path."""

    name = "rectified_flow"
