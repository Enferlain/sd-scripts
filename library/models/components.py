"""Shared model-component metadata helpers and loaded-component runtime seams."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

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


@dataclass(frozen=True, slots=True)
class LoadedModelComponent:
    """Live loaded top-level model component bound to a family declaration."""

    key: str
    public_name: str
    module: Any
    roles: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()

    @classmethod
    def from_spec(cls, spec: LoadedModelComponentSpec, module: Any) -> LoadedModelComponent:
        """Bind a live module object to its family-declared component spec."""
        return cls(
            key=spec.key,
            public_name=spec.public_name,
            module=module,
            roles=spec.roles,
            capabilities=spec.capabilities,
        )

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


def build_loaded_components(
    model_type: str,
    modules_by_key: Mapping[str, Any],
) -> tuple[LoadedModelComponent, ...]:
    """Bind live modules to the declared component order for a model family."""
    component_specs = resolve_component_specs(model_type)
    if component_specs is None:
        raise ValueError(f"No component declarations were found for model type {model_type!r}.")

    missing_keys = [spec.key for spec in component_specs if spec.key not in modules_by_key]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Missing loaded component modules for {model_type!r}: {missing}")

    return tuple(LoadedModelComponent.from_spec(spec, modules_by_key[spec.key]) for spec in component_specs)


def find_loaded_components(
    loaded_components: Sequence[LoadedModelComponent],
    *,
    role: str | None = None,
    capability: str | None = None,
    include_unloaded: bool = False,
) -> list[LoadedModelComponent]:
    """Filter loaded components by declared role/capability while preserving order."""
    components: list[LoadedModelComponent] = []
    for component in loaded_components:
        if not include_unloaded and component.module is None:
            continue
        if role is not None and not component.has_role(role):
            continue
        if capability is not None and not component.has_capability(capability):
            continue
        components.append(component)
    return components


def get_loaded_component_modules(
    loaded_components: Sequence[LoadedModelComponent],
    *,
    role: str | None = None,
    capability: str | None = None,
    include_unloaded: bool = False,
) -> list[Any]:
    """Return live component modules filtered by declared role/capability."""
    return [
        component.module
        for component in find_loaded_components(
            loaded_components,
            role=role,
            capability=capability,
            include_unloaded=include_unloaded,
        )
    ]


def get_loaded_component_module(
    loaded_components: Sequence[LoadedModelComponent],
    *,
    role: str | None = None,
    capability: str | None = None,
) -> Any | None:
    """Return the first live component module matching the requested semantics."""
    modules = get_loaded_component_modules(
        loaded_components,
        role=role,
        capability=capability,
        include_unloaded=False,
    )
    return modules[0] if modules else None


def _default_component_key(role: str, index: int) -> str:
    if role == "text_encoder":
        return f"text_encoder{index + 1}"
    if index == 0:
        return role
    return f"{role}{index + 1}"


def _synthesized_component(role: str, index: int, module: Any) -> LoadedModelComponent:
    key = _default_component_key(role, index)
    return LoadedModelComponent(key=key, public_name=key, module=module, roles=(role,))


def update_loaded_component_modules_by_role(
    loaded_components: Sequence[LoadedModelComponent],
    *,
    role: str,
    modules: Sequence[Any],
) -> tuple[LoadedModelComponent, ...]:
    """Replace or synthesize component modules for a given generic role."""
    updated_components = list(loaded_components)
    role_indices = [index for index, component in enumerate(updated_components) if component.has_role(role)]

    for index, module in enumerate(modules):
        if index < len(role_indices):
            component_index = role_indices[index]
            updated_components[component_index] = replace(updated_components[component_index], module=module)
        else:
            updated_components.append(_synthesized_component(role, index, module))

    for component_index in role_indices[len(modules) :]:
        updated_components[component_index] = replace(updated_components[component_index], module=None)

    return tuple(updated_components)


def update_loaded_component_module(
    loaded_components: Sequence[LoadedModelComponent],
    *,
    role: str,
    module: Any,
) -> tuple[LoadedModelComponent, ...]:
    """Replace or synthesize the first component module for a generic role."""
    return update_loaded_component_modules_by_role(loaded_components, role=role, modules=[module])


def build_component_module_pairs(loaded_components: Sequence[LoadedModelComponent]) -> list[tuple[str, nn.Module]]:
    """Project loaded components into ordered public-name/module pairs for diagnostics or tools."""
    component_pairs: list[tuple[str, nn.Module]] = []
    for component in loaded_components:
        if isinstance(component.module, nn.Module):
            component_pairs.append((component.public_name, component.module))
    return component_pairs


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
