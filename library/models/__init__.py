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


@dataclass(frozen=True, slots=True)
class LoadedModelComponentSpec:
    """Family-declared top-level component identity and semantics."""

    key: str
    public_name: str
    roles: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()

    def has_role(self, role: str) -> bool:
        """Return whether the component declares the requested generic role."""
        return role in self.roles

    def has_capability(self, capability: str) -> bool:
        """Return whether the component declares the requested capability."""
        return capability in self.capabilities


_MODEL_PACKAGE_ALIASES = {"sd1": "sd", "sd15": "sd", "sd2": "sd"}


def _resolve_model_package(model_type: str):
    package_name = model_type
    direct_spec = importlib.util.find_spec(f"library.models.{package_name}")
    if direct_spec is None:
        package_name = _MODEL_PACKAGE_ALIASES.get(model_type)
        if package_name is None:
            return None
    return importlib.import_module(f"library.models.{package_name}")


def resolve_component_specs(model_type: str) -> tuple[LoadedModelComponentSpec, ...] | None:
    """Resolve family-declared loaded component definitions from model-package metadata."""
    model_package = _resolve_model_package(model_type)
    if model_package is None:
        return None

    component_specs = getattr(model_package, "LOADED_MODEL_COMPONENT_SPECS", None)
    if not isinstance(component_specs, tuple):
        return None
    if all(isinstance(spec, LoadedModelComponentSpec) for spec in component_specs):
        return component_specs
    return None


def _build_legacy_component_names(
    component_specs: Sequence[LoadedModelComponentSpec] | None,
) -> NamedParameterComponentNames | None:
    if component_specs is None:
        return None

    text_encoder_names = tuple(spec.public_name for spec in component_specs if spec.has_role("text_encoder"))
    vae_spec = next((spec for spec in component_specs if spec.has_role("vae")), None)
    denoiser_spec = next((spec for spec in component_specs if spec.has_role("denoiser")), None)

    if not text_encoder_names and vae_spec is None and denoiser_spec is None:
        return None

    return NamedParameterComponentNames(
        text_encoder_names=text_encoder_names,
        vae_name=vae_spec.public_name if vae_spec is not None else "vae",
        denoiser_name=denoiser_spec.public_name if denoiser_spec is not None else "denoiser",
    )


def resolve_component_names(model_type: str) -> NamedParameterComponentNames | None:
    """Resolve legacy component labels derived from the declared component surface."""
    return _build_legacy_component_names(resolve_component_specs(model_type))


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
    "LoadedModelComponentSpec",
    "NamedParameterComponentNames",
    "build_named_components",
    "build_selector_name",
    "resolve_component_specs",
    "resolve_component_names",
]
