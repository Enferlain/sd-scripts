"""The DDPM formulation: noising, the v-prediction target and the signal-to-noise ratio.

Each piece reads a noise schedule value the strategy hands it and no
configuration. The arithmetic follows Diffusers' ``DDPMScheduler``
(``add_noise``, ``get_velocity``) and the SNR table today's trainer attaches
to it, in the same order and dtypes, so the same inputs give the same bits.
Diffusers serves the tests as the reference; the runtime never imports it.
"""

import torch

from library.objectives.schedules import NoiseSchedule


def _per_sample(values: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """One value per sample, shaped to multiply ``like`` sample by sample."""
    values = values.flatten()
    while values.dim() < like.dim():
        values = values.unsqueeze(-1)
    return values


def _coefficients(schedule: NoiseSchedule, timesteps: torch.Tensor, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """√ᾱ and √(1 - ᾱ) at each sample's timestep, in ``like``'s device and dtype."""
    alphas_cumprod = schedule.on("alphas_cumprod", like.device, like.dtype)
    timesteps = timesteps.to(like.device)
    signal = _per_sample(alphas_cumprod[timesteps] ** 0.5, like)
    noise_level = _per_sample((1 - alphas_cumprod[timesteps]) ** 0.5, like)
    return signal, noise_level


def ddpm_noise(schedule: NoiseSchedule, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """The latents noised to each sample's timestep: √ᾱ·x + √(1 - ᾱ)·ε, in the latents' dtype."""
    signal, noise_level = _coefficients(schedule, timesteps, latents)
    return signal * latents + noise_level * noise


def ddpm_v_target(schedule: NoiseSchedule, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """The v-prediction target at each sample's timestep: √ᾱ·ε - √(1 - ᾱ)·x (Salimans and Ho 2022)."""
    signal, noise_level = _coefficients(schedule, timesteps, latents)
    return signal * noise - noise_level * latents


def ddpm_snr(schedule: NoiseSchedule, timesteps: torch.Tensor) -> torch.Tensor:
    """Each sample's signal-to-noise ratio ᾱ / (1 - ᾱ), in float32 on the timesteps' device."""
    return schedule.on("snr", timesteps.device, torch.float32)[timesteps]


__all__ = ["ddpm_noise", "ddpm_snr", "ddpm_v_target"]
