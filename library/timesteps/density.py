from __future__ import annotations

import math

import torch


def sample_timestep_density(
    density_scheme: str,
    batch_size: int,
    *,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    cosine_shape_scale: float = 1.29,
) -> torch.Tensor:
    """Sample normalized training-time density values in ``[0, 1]``.

    Historical note:
    The old RF-specific ``shift`` sampler used ``sigmoid(N(0, s^2))``.
    That is just the zero-mean slice of this same logit-normal family, so the
    active surface keeps ``logit_normal`` as the canonical sampler instead of
    carrying two names for the same distribution family.
    """
    if density_scheme == "logit_normal":
        samples = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        return torch.nn.functional.sigmoid(samples)
    if density_scheme == "cosine_shaped":
        samples = torch.rand(size=(batch_size,), device="cpu")
        return 1 - samples - cosine_shape_scale * (torch.cos(math.pi * samples / 2) ** 2 - 1 + samples)
    return torch.rand(size=(batch_size,), device="cpu")


def apply_training_shift(timestep_density: torch.Tensor, training_shift: float) -> torch.Tensor:
    """Warp sampled training-time density toward earlier or later timesteps."""
    if training_shift == 1.0:
        return timestep_density
    return (timestep_density * training_shift) / (1 + (training_shift - 1) * timestep_density)
