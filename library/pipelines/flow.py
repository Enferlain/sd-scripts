"""Shared sampling-side helpers for discrete-flow inference runtime math."""

import torch


class DiscreteFlowModelSampling:
    """Discrete-flow sigma/timestep helper used by flow-style sampling loops."""

    def __init__(self, shift: float = 1.0) -> None:
        self.shift = shift
        timesteps = 1000
        self.sigmas = self.sigma(torch.arange(1, timesteps + 1, 1))

    @property
    def sigma_min(self) -> torch.Tensor:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> torch.Tensor:
        return self.sigmas[-1]

    def timestep(self, sigma: torch.Tensor | float) -> torch.Tensor | float:
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma: torch.Tensor, model_output: torch.Tensor, model_input: torch.Tensor) -> torch.Tensor:
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(
        self,
        sigma: torch.Tensor,
        noise: torch.Tensor,
        latent_image: torch.Tensor,
        max_denoise: bool = False,
    ) -> torch.Tensor:
        del max_denoise
        return sigma * noise + (1.0 - sigma) * latent_image


def get_discrete_flow_sigmas(sampling: DiscreteFlowModelSampling, steps: int) -> torch.Tensor:
    """Build the discrete-flow sigma schedule for the requested number of inference steps."""
    start = sampling.timestep(sampling.sigma_max)
    end = sampling.timestep(sampling.sigma_min)
    timesteps = torch.linspace(start, end, steps)
    sigmas = [sampling.sigma(timestep) for timestep in timesteps]
    sigmas.append(torch.tensor(0.0))
    return torch.stack(sigmas).float()


def starts_at_max_denoise(model_sampling: DiscreteFlowModelSampling, sigmas: torch.Tensor) -> bool:
    """Return whether the denoising schedule starts at or above the model's max sigma."""
    max_sigma = float(model_sampling.sigma_max)
    sigma = float(sigmas[0])
    return torch.isclose(torch.tensor(max_sigma), torch.tensor(sigma), rtol=1e-5).item() or sigma > max_sigma


__all__ = [
    "DiscreteFlowModelSampling",
    "get_discrete_flow_sigmas",
    "starts_at_max_denoise",
]
