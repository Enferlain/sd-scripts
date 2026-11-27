# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2022 kohya-ss
# SPDX-FileCopyrightText: 2023 Stability AI
# Follows kohya-ss/sd-scripts library/sdxl_model_util.py and SDXL's configuration in
# Stability-AI/generative-models. Modified. See THIRD_PARTY_NOTICES.md.

"""SDXL as released: its UNet's loading, its conditioning layout and its denoise call.

Plain code: a strategy, or whatever stands in for one, decides how these are
used. The numbers follow today's loader and denoiser call
(``library/models/sdxl/conversion.py``, ``library/strategies/sdxl/denoiser.py``)
in the same order and dtypes, so the same inputs give the same bits, except
the size embedding: today's denoiser call makes the sizes in the weights'
dtype before embedding them, which in bfloat16 rounds an original size of
1417 to 1416, where upstream kohya-ss and the reference embed the exact
numbers, as this does.

SDXL's released noise schedule is scaled-linear over ``TRAIN_STEPS`` steps
from ``BETA_START`` to ``BETA_END``; a formulation builds it from these
values, since models sit below objectives.
"""

import math
from collections.abc import Sequence
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors import safe_open

from library.models.denoisers.sd_unet import SDUNet, SDUNetConfig

BETA_START = 0.00085
BETA_END = 0.012
TRAIN_STEPS = 1000

UNET = SDUNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=320,
    channel_mult=(1, 2, 4),
    num_res_blocks=2,
    transformer_depth=(0, 2, 10),
    middle_transformer_depth=10,
    context_dim=2048,
    adm_in_channels=2816,
    head_dim=64,
    linear_projection=True,
)
"""SDXL's UNet values, as the released base model was trained with."""

UNET_PREFIX = "model.diffusion_model."  # where a released single-file checkpoint keeps the UNet's tensors
SIZE_EMBEDDING = 256  # each size number's sinusoidal embedding width


def load_unet(path: str | Path, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> SDUNet:
    """The UNet of a released single-file SDXL checkpoint, built without weights and filled on ``device`` in ``dtype``.

    Every key of the checkpoint's UNet must match the model's, or loading
    fails naming the differences.
    """
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(path), "pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118 (a safetensors file, not a dict)
            if key.startswith(UNET_PREFIX):
                state[key.removeprefix(UNET_PREFIX)] = f.get_tensor(key)
    with init_empty_weights():
        unet = SDUNet(UNET)
    expected = set(unet.state_dict())
    missing, unexpected = sorted(expected - state.keys()), sorted(state.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            f"{path} is not an SDXL UNet: missing {missing[:5]} ({len(missing)}), unexpected {unexpected[:5]} ({len(unexpected)})"
        )
    for key in list(state):
        set_module_tensor_to_device(unet, key, device, value=state.pop(key), dtype=dtype)
    return unet


def _sinusoidal(values: torch.Tensor, width: int, max_period: int = 10000) -> torch.Tensor:
    """Each value's sinusoidal embedding, cosines then sines, computed in float32."""
    half = width // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=values.device)
    angles = values[:, None].float() * freqs[None]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


def size_embedding(
    original_hw: Sequence[Sequence[int]],
    crop_top_left: Sequence[Sequence[int]],
    target_hw: Sequence[Sequence[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """SDXL's size conditioning, one row per sample: the original size, the crop's top left and the target size, each number embedded.

    The numbers are embedded exactly, in float32, as upstream kohya-ss and the
    reference embed them; only the embedding is made in ``dtype``.
    """
    parts = []
    for sizes in (original_hw, crop_top_left, target_hw):
        values = torch.tensor([list(s) for s in sizes], device=device, dtype=torch.float32)  # exact for any image size
        rows, numbers = values.shape
        parts.append(_sinusoidal(values.flatten(), SIZE_EMBEDDING).reshape(rows, numbers * SIZE_EMBEDDING))
    return torch.cat(parts, dim=1).to(device).to(dtype)


def conditioning(
    hidden_states1: torch.Tensor, hidden_states2: torch.Tensor, pooled2: torch.Tensor, sizes: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """The UNet's context and vector from the two text encoders' states, the second's pooled output and the size embedding.

    The context joins the two encoders' hidden states along the feature
    axis; the vector joins the pooled output and the sizes.
    """
    device = sizes.device
    hidden_states1 = hidden_states1.to(device, dtype=dtype)
    hidden_states2 = hidden_states2.to(device, dtype=dtype)
    pooled2 = pooled2.to(device, dtype=dtype)
    rows = {
        "first encoder's states": hidden_states1.shape[0],
        "second encoder's states": hidden_states2.shape[0],
        "pooled outputs": pooled2.shape[0],
    }
    if any(count != sizes.shape[0] for count in rows.values()):
        raise ValueError(f"{sizes.shape[0]} size rows, and " + ", ".join(f"{count} {name}" for name, count in rows.items()))
    vector = torch.cat([pooled2, sizes], dim=1).to(dtype)
    context = torch.cat([hidden_states1, hidden_states2], dim=2).to(dtype)
    return context, vector


def denoise(unet: SDUNet, noisy: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """The UNet's prediction for noisy latents at their timesteps under the given context and vector."""
    return unet(noisy, timesteps, context, vector)


__all__ = [
    "BETA_END",
    "BETA_START",
    "TRAIN_STEPS",
    "UNET",
    "conditioning",
    "denoise",
    "load_unet",
    "size_embedding",
]
