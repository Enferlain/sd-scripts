"""Loss weightings over a formulation's signal-to-noise ratio.

A formulation defines each sample's signal-to-noise ratio
(:mod:`library.objectives.formulations.ddpm`), and a weighting turns it into
one weight per sample, which ``weigh`` multiplies in
(:mod:`library.losses.reductions`). A weighting that differs by prediction
type is one function per type.
"""

import torch


def min_snr_v_weight(snr: torch.Tensor, gamma: float) -> torch.Tensor:
    """Min-SNR weighting for a v-prediction loss, min(snr, gamma) / (snr + 1), in float32 (Hang et al. 2023, arXiv 2303.09556)."""
    return torch.div(torch.minimum(snr, torch.full_like(snr, gamma)), snr + 1).float()


__all__ = ["min_snr_v_weight"]
