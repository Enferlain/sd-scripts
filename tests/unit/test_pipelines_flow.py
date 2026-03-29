"""Unit tests for shared discrete-flow sampling helpers."""

import pytest
import torch

from library.pipelines.flow import (
    DiscreteFlowModelSampling,
    get_discrete_flow_sigmas,
    starts_at_max_denoise,
)


@pytest.mark.unit
def test_get_discrete_flow_sigmas_returns_terminal_zero() -> None:
    sampling = DiscreteFlowModelSampling(shift=3.0)

    sigmas = get_discrete_flow_sigmas(sampling, steps=4)

    assert sigmas.shape == (5,)
    assert sigmas[-1].item() == 0.0


@pytest.mark.unit
def test_starts_at_max_denoise_detects_schedule_start() -> None:
    sampling = DiscreteFlowModelSampling(shift=1.0)
    sigmas = get_discrete_flow_sigmas(sampling, steps=4)

    assert starts_at_max_denoise(sampling, sigmas) is True


@pytest.mark.unit
def test_discrete_flow_model_sampling_noise_scaling_preserves_shape() -> None:
    sampling = DiscreteFlowModelSampling(shift=1.0)
    sigma = torch.tensor(0.5)
    noise = torch.randn(1, 4, 8, 8)
    latent = torch.zeros_like(noise)

    scaled = sampling.noise_scaling(sigma, noise, latent)

    assert scaled.shape == noise.shape
