"""Unit tests for shared flow-matching training helpers."""

import pytest
import torch

from library.config.dataclasses.timestep import TimestepConfig
from library.training.flow import (
    build_flow_matching_model_input_and_timesteps,
    compute_flow_matching_loss_weighting,
    compute_flow_matching_timestep_density,
)


@pytest.mark.training
@pytest.mark.unit
def test_build_flow_matching_model_input_and_timesteps_uses_timestep_config() -> None:
    timestep_config = TimestepConfig(
        weighting_scheme="mode",
        logit_mean=0.5,
        logit_std=1.0,
        mode_scale=1.5,
        min_timestep=0,
        max_timestep=1000,
        discrete_flow_shift=1.0,
    )
    latents = torch.zeros(2, 4, 8, 8)
    noise = torch.ones_like(latents)

    noisy_model_input, timesteps, sigmas = build_flow_matching_model_input_and_timesteps(
        timestep_config,
        latents,
        noise,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert noisy_model_input.shape == latents.shape
    assert timesteps.shape == (2,)
    assert sigmas.shape == (2, 1, 1, 1)


@pytest.mark.training
@pytest.mark.unit
def test_compute_flow_matching_loss_weighting_sigma_sqrt() -> None:
    sigmas = torch.tensor([[[[0.5]]], [[[0.25]]]], dtype=torch.float32)

    weighting = compute_flow_matching_loss_weighting("sigma_sqrt", sigmas)

    assert torch.allclose(weighting, torch.tensor([[[[4.0]]], [[[16.0]]]], dtype=torch.float32))


@pytest.mark.training
@pytest.mark.unit
def test_compute_flow_matching_timestep_density_returns_expected_shape() -> None:
    density = compute_flow_matching_timestep_density("uniform", batch_size=4)

    assert density.shape == (4,)
