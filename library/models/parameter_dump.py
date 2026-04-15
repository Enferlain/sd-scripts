"""Utilities for dumping live model named parameters in a compact YAML shape."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True, slots=True)
class NamedParameterComponentNames:
    """Family-owned top-level component names for parameter dump tooling."""

    text_encoder_names: tuple[str, ...] = ()
    vae_name: str = "vae"
    denoiser_name: str = "denoiser"


def derive_parameter_dump_identifier(model_type: str, model_version: str | None) -> str:
    """Build a readable identifier for a parameter dump."""
    if model_version is None or model_version == model_type:
        return model_type
    return f"{model_type}-{model_version}"


def _normalize_text_encoders(text_encoders: nn.Module | Sequence[nn.Module | None] | None) -> list[nn.Module | None]:
    if text_encoders is None:
        return []
    if isinstance(text_encoders, nn.Module):
        return [text_encoders]
    return list(text_encoders)


_MODEL_TYPE_TO_PACKAGE = {
    "sd1": "sd",
    "sd15": "sd",
    "sd2": "sd",
    "sdxl": "sdxl",
    "sd3": "sd3",
}


def _resolve_component_names(model_type: str) -> NamedParameterComponentNames | None:
    package_name = _MODEL_TYPE_TO_PACKAGE.get(model_type)
    if package_name is None:
        return None

    model_package = importlib.import_module(f"library.models.{package_name}")
    component_names = getattr(model_package, "NAMED_PARAMETER_COMPONENT_NAMES", None)
    if isinstance(component_names, NamedParameterComponentNames):
        return component_names
    return None


def resolve_named_parameter_components(
    *,
    model_type: str,
    text_encoders: nn.Module | Sequence[nn.Module | None] | None,
    vae: nn.Module | None,
    denoiser: nn.Module | None,
) -> list[tuple[str, nn.Module]]:
    """Resolve component names through family-owned name metadata when available."""
    encoder_modules = _normalize_text_encoders(text_encoders)
    component_names = _resolve_component_names(model_type)
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


def _iter_component_parameter_lines(
    component_name: str,
    module: nn.Module,
    *,
    trainable_only: bool = False,
) -> Iterable[str]:
    del component_name
    for name, param in sorted(module.named_parameters(), key=lambda item: item[0]):
        if trainable_only and not param.requires_grad:
            continue
        shape = list(param.shape)
        dtype = str(param.dtype).removeprefix("torch.")
        requires_grad = str(bool(param.requires_grad)).lower()
        yield f"    {name}: {{shape: {shape}, dtype: {dtype}, requires_grad: {requires_grad}}}"


def format_named_parameter_dump(
    *,
    identifier: str,
    components: Sequence[tuple[str, nn.Module]],
    trainable_only: bool = False,
) -> str:
    """Render a compact YAML dump grouped by component with one line per parameter."""
    lines = [f"identifier: {identifier}", "components:"]

    for component_name, module in components:
        parameter_lines = list(_iter_component_parameter_lines(component_name, module, trainable_only=trainable_only))
        if trainable_only and not parameter_lines:
            continue

        lines.append(f"  {component_name}:")
        lines.extend(parameter_lines)

    return "\n".join(lines) + "\n"
