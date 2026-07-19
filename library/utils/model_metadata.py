"""Narrow helpers used while assembling and reading model artifact metadata."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import subprocess

import safetensors


logger = logging.getLogger(__name__)


def get_implementation_version() -> str:
    """Return the current implementation version as ``sd-scripts/<commit>``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
            timeout=5,
        )
        if result.returncode == 0:
            return f"sd-scripts/{result.stdout.strip()}"

        logger.warning("Failed to get git commit hash, using fallback")
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.warning("Could not determine git commit: %s", exc)
    return "sd-scripts/unknown"


def file_to_data_url(file_path: str) -> str:
    """Convert a file into a data URL suitable for artifact metadata."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(file_path, "rb") as file:
        encoded_data = base64.b64encode(file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_data}"


def load_metadata_from_safetensors(model: str) -> dict[str, str]:
    """Read the string metadata map from a safetensors artifact."""
    if not model.endswith(".safetensors"):
        return {}

    with safetensors.safe_open(model, framework="pt") as file:
        return file.metadata() or {}


__all__ = [
    "file_to_data_url",
    "get_implementation_version",
    "load_metadata_from_safetensors",
]
