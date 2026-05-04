from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .module import VeraModule, VeraSharedProjectionBank


_VERA_SHARED_PREFIX = "vera_shared"
_VERA_SAVE_PROJECTION_METADATA_KEY = "sd_scripts_vera.save_projection"
_VERA_PROJECTION_PRNG_KEY_METADATA_KEY = "sd_scripts_vera.projection_prng_key"


def _is_safetensors_path(path: str) -> bool:
    return Path(path).suffix == ".safetensors"


def load_vera_state_dict(weights_path: str) -> dict[str, torch.Tensor]:
    """Load a VeRA checkpoint payload from either PyTorch or safetensors format."""

    if _is_safetensors_path(weights_path):
        return load_file(weights_path)
    return torch.load(weights_path, map_location="cpu")


def load_vera_metadata(weights_path: str) -> dict[str, str]:
    """Load VeRA-specific adapter metadata when the artifact format supports it."""

    if not _is_safetensors_path(weights_path):
        return {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def resolve_projection_metadata(metadata: Mapping[str, str] | None) -> tuple[bool | None, int | None]:
    """Parse VeRA projection settings from exported adapter metadata when present."""

    if not metadata:
        return None, None
    save_projection_raw = metadata.get(_VERA_SAVE_PROJECTION_METADATA_KEY)
    projection_key_raw = metadata.get(_VERA_PROJECTION_PRNG_KEY_METADATA_KEY)

    save_projection = None
    if save_projection_raw is not None:
        normalized = save_projection_raw.strip().lower()
        if normalized in {"true", "1"}:
            save_projection = True
        elif normalized in {"false", "0"}:
            save_projection = False
        else:
            raise ValueError(
                "VeRA checkpoint metadata stores an invalid save_projection flag: "
                f"{save_projection_raw!r}."
            )

    projection_prng_key = None
    if projection_key_raw is not None:
        try:
            projection_prng_key = int(projection_key_raw)
        except ValueError as exc:
            raise ValueError(
                "VeRA checkpoint metadata stores an invalid projection_prng_key value: "
                f"{projection_key_raw!r}."
            ) from exc
    return save_projection, projection_prng_key


def extract_vera_shared_weights(state_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return the shared VeRA projection tensors from an exported checkpoint payload."""

    return state_dict.get(f"{_VERA_SHARED_PREFIX}.vera_A"), state_dict.get(f"{_VERA_SHARED_PREFIX}.vera_B")


def save_vera_state_dict(
    shared_bank: VeraSharedProjectionBank,
    modules: list[VeraModule],
    file: str,
    *,
    dtype,
    metadata: dict[str, str] | None,
) -> None:
    """Save a repo-owned VeRA checkpoint payload for the provided shared bank and modules."""

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in shared_bank.export_state_dict().items():
        export_value = value.detach().clone().to("cpu")
        if dtype is not None and torch.is_floating_point(export_value):
            export_value = export_value.to(dtype)
        state_dict[f"{_VERA_SHARED_PREFIX}.{key}"] = export_value

    for module in modules:
        for key, value in module.export_state_dict().items():
            export_value = value.detach().clone().to("cpu")
            if dtype is not None and torch.is_floating_point(export_value):
                export_value = export_value.to(dtype)
            state_dict[f"{module.lora_name}.{key}"] = export_value

    export_metadata = dict(metadata or {})
    export_metadata[_VERA_SAVE_PROJECTION_METADATA_KEY] = "true" if shared_bank.save_projection else "false"
    export_metadata[_VERA_PROJECTION_PRNG_KEY_METADATA_KEY] = str(shared_bank.projection_prng_key)

    if _is_safetensors_path(file):
        save_file(state_dict, file, export_metadata)
        return
    torch.save(state_dict, file)
