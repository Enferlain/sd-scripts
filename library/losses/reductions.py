"""Reductions that turn elementwise errors into a loss with its mass.

``per_sample_mean`` reduces each sample's errors, ``weigh`` multiplies in one
weight per sample, and ``sample_mean`` averages over the samples and reports
the loss with its mass, the number of samples it divided by, so a window can
divide once by its total mass. Weights are multiplied in one at a time, in
the order the strategy calls ``weigh``, since the order decides the result's
last bits.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class Loss:
    """A loss with its mass: what its final reduction divides by, which a window sums and divides by once."""

    value: torch.Tensor
    mass: float | torch.Tensor


def per_sample_mean(errors: torch.Tensor) -> torch.Tensor:
    """Each sample's mean error: the mean over every dimension but the first; errors with one value per sample are already that."""
    if errors.dim() == 1:
        return errors
    return errors.mean(dim=tuple(range(1, errors.dim())))


def weigh(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Each sample's loss times its weight."""
    return losses * weights.to(losses.device)


def sample_mean(losses: torch.Tensor) -> Loss:
    """The mean over the samples, with the number of samples as its mass."""
    return Loss(losses.mean(), float(losses.shape[0]))


__all__ = ["Loss", "per_sample_mean", "sample_mean", "weigh"]
