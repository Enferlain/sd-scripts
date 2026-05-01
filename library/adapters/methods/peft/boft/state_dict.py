from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .module import BoftModule


def _is_safetensors_path(path: str) -> bool:
    return Path(path).suffix == ".safetensors"


def load_boft_state_dict(weights_path: str) -> dict[str, torch.Tensor]:
    """Load a BOFT checkpoint payload from either PyTorch or safetensors format."""

    if _is_safetensors_path(weights_path):
        return load_file(weights_path)
    return torch.load(weights_path, map_location="cpu")


def save_boft_state_dict(
    modules: list[BoftModule],
    file: str,
    *,
    dtype,
    metadata: dict[str, str] | None,
) -> None:
    """Save a repo-owned BOFT checkpoint payload for the provided modules."""

    state_dict: dict[str, torch.Tensor] = {}
    for module in modules:
        for key, value in module.export_state_dict().items():
            export_value = value.detach().clone().to("cpu")
            if dtype is not None:
                export_value = export_value.to(dtype)
            state_dict[f"{module.lora_name}.{key}"] = export_value

    if _is_safetensors_path(file):
        save_file(state_dict, file, metadata or {})
        return
    torch.save(state_dict, file)
