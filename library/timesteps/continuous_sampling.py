from __future__ import annotations

import math

import torch


def sample_continuous_timesteps(
    timestep_sampling: str,
    batch_size: int,
    *,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    cosine_shape_scale: float = 1.29,
) -> torch.Tensor:
    """Sample continuous timesteps ``u`` in ``[0, 1]`` before index conversion.

    Historical note:
    The old RF-specific ``shift`` sampler used ``sigmoid(N(0, s^2))``.
    That is just the zero-mean slice of this same logit-normal family, so the
    active surface keeps ``logit_normal`` as the canonical sampler instead of
    carrying two names for the same distribution family.

    The ``timestep_sampling`` value comes from the active timestep config
    (`cfg.timestep.timestep_sampling`) or, when sampling through the shared
    runtime, from the resolved runtime mode.

    In the RF path this is the old "sample ``u``, then map it into timestep
    indices" step extracted into one shared helper.
    """
    if timestep_sampling == "logit_normal":
        samples = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        return torch.nn.functional.sigmoid(samples)
    if timestep_sampling == "cosine_shaped":
        samples = torch.rand(size=(batch_size,), device="cpu")
        return 1 - samples - cosine_shape_scale * (torch.cos(math.pi * samples / 2) ** 2 - 1 + samples)
    return torch.rand(size=(batch_size,), device="cpu")


def apply_training_shift(u: torch.Tensor, training_shift: float) -> torch.Tensor:
    """Warp sampled continuous timesteps toward earlier or later values."""
    if training_shift == 1.0:
        return u
    return (u * training_shift) / (1 + (training_shift - 1) * u)
