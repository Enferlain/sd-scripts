"""Noise schedules as values.

A diffusion formulation's noise schedule is a value the strategy hands it. An
assembly records the schedule its model was trained with, a technique that
changes it, such as zero-terminal SNR, is a transform that returns a new
schedule, and noising, the target and the signal-to-noise ratio all read the
one result (:mod:`library.objectives.formulations.ddpm`). How training draws
its timesteps and which steps a sampler visits are separate choices.

The arithmetic follows Diffusers' ``DDPMScheduler`` step for step, in the
same dtypes, so a schedule built here gives the same bits; Diffusers serves
the tests as the reference and the runtime never imports it.
"""

import functools
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, eq=False)
class NoiseSchedule:
    """A discrete schedule, given by its betas: the share of noise each step adds, one float32 per step on the host.

    Everything else derives from the betas: ``alphas_cumprod`` (ᾱ, the
    cumulative product of ``1 - beta``) and each step's signal-to-noise
    ratio. Copies on a device are made once per device and dtype and kept,
    as scratch that never changes a result.
    """

    betas: torch.Tensor
    _copies: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        betas = self.betas
        if betas.dim() != 1 or betas.device.type != "cpu" or betas.dtype != torch.float32:
            raise ValueError(
                f"a noise schedule's betas are one float32 per step on the host, not {tuple(betas.shape)} {betas.dtype} on {betas.device}"
            )

    @functools.cached_property
    def alphas_cumprod(self) -> torch.Tensor:
        """ᾱ for each step: the cumulative product of ``1 - beta``."""
        return torch.cumprod(1.0 - self.betas, dim=0)

    @functools.cached_property
    def snr(self) -> torch.Tensor:
        """Each step's signal-to-noise ratio, ᾱ / (1 - ᾱ), computed as (√ᾱ / √(1 - ᾱ))²; zero where ᾱ is."""
        alphas_cumprod = self.alphas_cumprod
        return (torch.sqrt(alphas_cumprod) / torch.sqrt(1.0 - alphas_cumprod)) ** 2

    def on(self, quantity: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """``alphas_cumprod`` or ``snr`` on a device in a dtype, copied there once and kept.

        A copy on a device is complete before it is kept: the copy waits for
        the device once, so work on any stream may read it afterwards.
        """
        key = (quantity, device, dtype)
        found = self._copies.get(key)
        if found is None:
            copy = getattr(self, quantity).to(device=device).to(dtype=dtype)
            if copy.device.type != "cpu":
                torch.accelerator.synchronize(copy.device)
            found = self._copies.setdefault(key, copy)
        return found


def scaled_linear(beta_start: float, beta_end: float, steps: int) -> NoiseSchedule:
    """Latent diffusion's scaled-linear schedule: betas whose square roots are evenly spaced from ``beta_start`` to ``beta_end``."""
    return NoiseSchedule(torch.linspace(beta_start**0.5, beta_end**0.5, steps, dtype=torch.float32) ** 2)


def zero_terminal_snr(schedule: NoiseSchedule) -> NoiseSchedule:
    """The schedule rescaled so its last step has a signal-to-noise ratio of exactly zero (Lin et al. 2023, arXiv 2305.08891).

    √ᾱ is shifted so its last value is zero and scaled so its first stays
    the same, and the betas are recovered from the result.
    """
    alphas = 1 - schedule.betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()
    first = alphas_bar_sqrt[0].clone()
    last = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= last
    alphas_bar_sqrt *= first / (first - last)
    alphas_bar = alphas_bar_sqrt**2
    alphas = torch.cat([alphas_bar[0:1], alphas_bar[1:] / alphas_bar[:-1]])
    return NoiseSchedule(1 - alphas)


__all__ = ["NoiseSchedule", "scaled_linear", "zero_terminal_snr"]
