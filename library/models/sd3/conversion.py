"""SD3 checkpoint serialization helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import CLIPTextModelWithProjection, T5EncoderModel

from library.models.sd3.mmdit import MMDiT
from library.models.sd3.vae import SDVAE


def _to_save_tensor(tensor: torch.Tensor, save_dtype: torch.dtype | None) -> torch.Tensor:
    """Detach, move, and optionally cast a tensor for checkpoint saving."""
    tensor = tensor.detach().clone().to("cpu")
    if save_dtype is not None:
        tensor = tensor.to(save_dtype)
    return tensor


def _prefixed_state_dict(
    prefix: str,
    state_dict: dict[str, torch.Tensor],
    save_dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    """Prefix and normalize a module state dict for SD3 unified checkpoint saving."""
    return {prefix + key: _to_save_tensor(value, save_dtype) for key, value in state_dict.items()}


def _save_sidecar(
    path: Path,
    state_dict: dict[str, torch.Tensor],
    save_dtype: torch.dtype | None,
) -> None:
    """Persist a sidecar safetensors checkpoint."""
    save_file({key: _to_save_tensor(value, save_dtype) for key, value in state_dict.items()}, str(path))


def save_models(
    ckpt_path: str,
    mmdit: MMDiT,
    vae: SDVAE,
    clip_l: CLIPTextModelWithProjection | None,
    clip_g: CLIPTextModelWithProjection | None,
    t5xxl: T5EncoderModel | None,
    metadata: dict[str, str] | None,
    save_dtype: torch.dtype | None = None,
) -> list[str]:
    """Save an SD3 checkpoint plus any text-encoder sidecars."""
    ckpt_file = Path(ckpt_path)

    state_dict: dict[str, torch.Tensor] = {}
    state_dict.update(_prefixed_state_dict("model.diffusion_model.", mmdit.state_dict(), save_dtype))
    state_dict.update(_prefixed_state_dict("first_stage_model.", vae.state_dict(), save_dtype))
    save_file(state_dict, str(ckpt_file), metadata=metadata)

    saved_paths = [str(ckpt_file)]
    if clip_l is not None:
        clip_l_path = ckpt_file.with_name(ckpt_file.stem + "_clip_l.safetensors")
        _save_sidecar(clip_l_path, clip_l.state_dict(), save_dtype)
        saved_paths.append(str(clip_l_path))

    if clip_g is not None:
        clip_g_path = ckpt_file.with_name(ckpt_file.stem + "_clip_g.safetensors")
        _save_sidecar(clip_g_path, clip_g.state_dict(), save_dtype)
        saved_paths.append(str(clip_g_path))

    if t5xxl is not None:
        t5xxl_path = ckpt_file.with_name(ckpt_file.stem + "_t5xxl.safetensors")
        t5xxl_state_dict = dict(t5xxl.state_dict())
        if "shared.weight" in t5xxl_state_dict:
            t5xxl_state_dict["shared.weight"] = t5xxl_state_dict["shared.weight"].detach().clone()
        _save_sidecar(t5xxl_path, t5xxl_state_dict, save_dtype)
        saved_paths.append(str(t5xxl_path))

    return saved_paths


__all__ = ["save_models"]
