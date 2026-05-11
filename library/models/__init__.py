"""Shared model-package metadata helpers and component naming seams."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True, slots=True)
class NamedParameterComponentNames:
    """Family-owned public names for top-level loaded model components."""

    text_encoder_names: tuple[str, ...] = ()
    vae_name: str = "vae"
    denoiser_name: str = "denoiser"


_MODEL_PACKAGE_ALIASES = {"sd1": "sd", "sd15": "sd", "sd2": "sd"}


def resolve_component_names(model_type: str) -> NamedParameterComponentNames | None:
    """Resolve family-owned component labels from model-package metadata."""
    package_name = model_type
    direct_spec = importlib.util.find_spec(f"library.models.{package_name}")
    if direct_spec is None:
        package_name = _MODEL_PACKAGE_ALIASES.get(model_type)
        if package_name is None:
            return None

    model_package = importlib.import_module(f"library.models.{package_name}")
    component_names = getattr(model_package, "NAMED_PARAMETER_COMPONENT_NAMES", None)
    if isinstance(component_names, NamedParameterComponentNames):
        return component_names
    return None


def build_selector_name(component_name: str, local_name: str) -> str:
    """Build the canonical public selector path for a component-local name."""
    return component_name if not local_name else f"{component_name}.{local_name}"


def _normalize_text_encoders(text_encoders: nn.Module | Sequence[nn.Module | None] | None) -> list[nn.Module | None]:
    if text_encoders is None:
        return []
    if isinstance(text_encoders, nn.Module):
        return [text_encoders]
    return list(text_encoders)


def build_named_components(
    *,
    component_names: NamedParameterComponentNames | None,
    text_encoders: nn.Module | Sequence[nn.Module | None] | None,
    vae: nn.Module | None,
    denoiser: nn.Module | None,
) -> list[tuple[str, nn.Module]]:
    """Group loaded modules under stable top-level public component names."""
    encoder_modules = _normalize_text_encoders(text_encoders)
    components: list[tuple[str, nn.Module]] = []

    if component_names is not None:
        for label, module in zip(component_names.text_encoder_names, encoder_modules):
            if module is not None:
                components.append((label, module))
        if vae is not None:
            components.append((component_names.vae_name, vae))
        if denoiser is not None:
            components.append((component_names.denoiser_name, denoiser))
        return components

    for index, module in enumerate(encoder_modules):
        if module is not None:
            components.append((f"text_encoder{index + 1}", module))
    if vae is not None:
        components.append(("vae", vae))
    if denoiser is not None:
        components.append(("denoiser", denoiser))
    return components


__all__ = [
    "NamedParameterComponentNames",
    "build_named_components",
    "build_selector_name",
    "resolve_component_names",
]
