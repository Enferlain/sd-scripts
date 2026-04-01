from __future__ import annotations

import math

import torch

from library.config.dataclasses.loss import HuberConfig, LossConfig


def get_huber_threshold_if_needed(
    loss_config: LossConfig,
    huber_config: HuberConfig,
    timesteps: torch.Tensor,
    *,
    num_train_timesteps: int | None = None,
    alphas_cumprod: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Calculate the configured Huber-like threshold schedule for the current timesteps."""
    if loss_config.loss_type not in {
        "huber",
        "smooth_l1",
        "standard_pseudo_huber",
        "standard_huber",
        "standard_smooth_l1",
        "soft_welsch",
        "scaled_quadratic",
        "smooth_l2_log",
    }:
        return None

    if huber_config.huber_schedule == "constant":
        result = torch.tensor(huber_config.huber_c * float(huber_config.huber_scale), device=timesteps.device)
    elif huber_config.huber_schedule == "exponential":
        if num_train_timesteps is None:
            raise NotImplementedError("Huber schedule 'exponential' requires objective runtime timestep metadata.")
        alpha = -math.log(huber_config.huber_c) / num_train_timesteps
        result = torch.exp(-alpha * timesteps) * float(huber_config.huber_scale)
    elif huber_config.huber_schedule == "snr":
        if alphas_cumprod is None:
            raise NotImplementedError("Huber schedule 'snr' is not supported with the current model.")
        indexed_alphas_cumprod = torch.index_select(alphas_cumprod, 0, timesteps)
        sigmas = ((1.0 - indexed_alphas_cumprod) / indexed_alphas_cumprod) ** 0.5
        result = (1 - huber_config.huber_c) / (1 + sigmas) ** 2 + huber_config.huber_c
        result = result.to(timesteps.device)
    else:
        raise NotImplementedError(f"Unknown Huber loss schedule {huber_config.huber_schedule}!")

    return result
