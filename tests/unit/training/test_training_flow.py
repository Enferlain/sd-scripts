"""Unit tests for shared flow-matching training helpers."""

import pytest
import torch

from library.config.dataclasses.timestep import TimestepConfig
from library.losses.loss_modifiers import NoOpLossModifier
from library.objectives.rectified_flow import (
    RectifiedFlowObjectiveRuntime,
    build_flow_matching_model_input_and_timesteps,
    compute_flow_matching_loss_weighting,
    compute_flow_matching_timestep_density,
)


@pytest.mark.training
@pytest.mark.unit
def test_build_flow_matching_model_input_and_timesteps_uses_timestep_config() -> None:
    timestep_config = TimestepConfig(
        timestep_sampling="cosine_shaped",
        logit_mean=0.5,
        logit_std=1.0,
        cosine_shape_scale=1.5,
        min_timestep=0,
        max_timestep=1000,
        training_shift=1.0,
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
def test_rectified_flow_runtime_builds_training_batch_state() -> None:
    timestep_config = TimestepConfig(
        timestep_sampling="logit_normal",
        logit_mean=0.0,
        logit_std=1.0,
        min_timestep=0,
        max_timestep=1000,
        training_shift=1.0,
        rf_loss_weighting_scheme="sigma_sqrt",
    )
    runtime = RectifiedFlowObjectiveRuntime(
        name="rectified_flow",
        num_train_timesteps=1000,
        timestep_runtime=None,
        loss_modifier=NoOpLossModifier(),
        timestep_config=timestep_config,
        loss_weighting_scheme=timestep_config.rf_loss_weighting_scheme,
    )
    latents = torch.zeros(2, 4, 8, 8)

    batch_state = runtime.build_training_batch_state(
        latents,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert batch_state.noisy_model_input.shape == latents.shape
    assert batch_state.timesteps.shape == (2,)
    assert batch_state.sigmas.shape == (2, 1, 1, 1)
    assert batch_state.loss_weighting.shape == (2, 1, 1, 1)
    assert batch_state.noise.shape == latents.shape


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


@pytest.mark.training
@pytest.mark.unit
def test_compute_flow_matching_timestep_density_logit_normal_returns_expected_shape() -> None:
    density = compute_flow_matching_timestep_density("logit_normal", batch_size=4, logit_mean=0.0, logit_std=1.0)

    assert density.shape == (4,)
