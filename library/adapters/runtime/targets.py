from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from torch import nn

from library.models import LoadedModelComponent
from library.optimization.targets import (
    OptimizationTargetRef,
    build_component_target_ref,
    build_module_target_ref,
)

try:
    from transformers.pytorch_utils import Conv1D as TransformersConv1D
except (ImportError, AttributeError):  # pragma: no cover - optional dependency
    TransformersConv1D = None


# The first module-targeting slice intentionally stays on common weight-bearing
# module types that existing adapter families commonly realize against.
# Broader target classes such as embeddings or finer parameter-granular binding
# can be added later once a concrete absorbed-method consumer proves the need.
_ADAPTER_TARGET_MODULE_TYPES_LIST: list[type[nn.Module]] = [
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.LayerNorm,
    nn.GroupNorm,
]
if TransformersConv1D is not None:
    _ADAPTER_TARGET_MODULE_TYPES_LIST.append(TransformersConv1D)
_ADAPTER_TARGET_MODULE_TYPES = tuple(_ADAPTER_TARGET_MODULE_TYPES_LIST)


@dataclass(slots=True)
class AdapterResolvedTarget:
    """Adapter-facing view of one resolved original-model target.

    `path` intentionally remains the public component-qualified selector for
    compatibility with existing adapter callers. Use `local_path` when code
    needs the component-local module path carried by the shared target ref.
    """

    target_ref: OptimizationTargetRef
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def component(self) -> str:
        return self.target_ref.component

    @property
    def component_key(self) -> str:
        return self.target_ref.component_key

    @property
    def path(self) -> str:
        """Compatibility selector path such as ``unet.to_q``."""
        return self.target_ref.selector

    @property
    def selector(self) -> str:
        return self.target_ref.selector

    @property
    def local_path(self) -> str:
        """Component-local module path such as ``to_q``."""
        return self.target_ref.path

    @property
    def module(self) -> Any:
        return self.target_ref.obj

    @property
    def module_type(self) -> str | None:
        return self.target_ref.module_type


@dataclass(slots=True)
class AdapterResolvedTargets:
    """Adapter-facing bundle of resolved targets for one training run."""

    family: str
    targets: list[AdapterResolvedTarget] = field(default_factory=list)


def _build_resolved_target(
    *,
    target_ref: OptimizationTargetRef,
    tags: frozenset[str],
) -> AdapterResolvedTarget:
    return AdapterResolvedTarget(
        target_ref=target_ref,
        tags=tags,
        metadata={"component_key": target_ref.component_key},
    )


def _iter_component_module_targets(
    *,
    component: str,
    component_key: str,
    root_module: Any,
    tags: frozenset[str],
) -> list[AdapterResolvedTarget]:
    if root_module is None:
        return []

    if not isinstance(root_module, nn.Module):
        return [
            _build_resolved_target(
                target_ref=build_module_target_ref(
                    component=component,
                    component_key=component_key,
                    path="",
                    module=root_module,
                    tags=tags | frozenset({"component_root"}),
                ),
                tags=tags | frozenset({"component_root"}),
            )
        ]

    targets: list[AdapterResolvedTarget] = []
    for local_name, module in root_module.named_modules():
        if not isinstance(module, _ADAPTER_TARGET_MODULE_TYPES):
            continue

        module_tags = tags | frozenset({"module_target"})
        # A selected component can itself be a targetable module (for example a
        # standalone Linear used in focused tests). Mark that edge case
        # explicitly while keeping child-module targets on the same shared path.
        if not local_name:
            module_tags = module_tags | frozenset({"component_root"})
        targets.append(
            _build_resolved_target(
                target_ref=build_module_target_ref(
                    component=component,
                    component_key=component_key,
                    path=local_name,
                    module=module,
                    tags=module_tags,
                ),
                tags=module_tags,
            )
        )

    return targets


def _iter_selected_loaded_components(
    loaded_components: Sequence[LoadedModelComponent],
    *,
    include_text_encoders: list[bool] | None,
    include_vae: bool,
    include_denoiser: bool,
) -> list[tuple[LoadedModelComponent, frozenset[str]]]:
    encoder_flags = list(include_text_encoders or [])
    text_encoder_index = 0
    selected_components: list[tuple[LoadedModelComponent, frozenset[str]]] = []

    for component in loaded_components:
        if component.has_role("text_encoder"):
            include_target = encoder_flags[text_encoder_index] if text_encoder_index < len(encoder_flags) else False
            text_encoder_index += 1
            if include_target and component.module is not None:
                selected_components.append((component, frozenset({"text_encoder"})))
            continue

        if component.has_role("vae"):
            if include_vae and component.module is not None:
                selected_components.append((component, frozenset({"vae"})))
            continue

        if component.has_role("denoiser") and include_denoiser and component.module is not None:
            selected_components.append((component, frozenset({"denoiser"})))

    return selected_components


def build_component_root_targets(
    *,
    model_type: str,
    loaded_components: Sequence[LoadedModelComponent],
    include_text_encoders: list[bool] | None = None,
    include_vae: bool = False,
    include_denoiser: bool = False,
) -> AdapterResolvedTargets:
    """Build component-root adapter targets from declared loaded components."""

    targets: list[AdapterResolvedTarget] = []
    for component, role_tags in _iter_selected_loaded_components(
        loaded_components,
        include_text_encoders=include_text_encoders,
        include_vae=include_vae,
        include_denoiser=include_denoiser,
    ):
        component_tags = role_tags | frozenset({"component_root"})
        targets.append(
            _build_resolved_target(
                target_ref=build_component_target_ref(
                    component=component.public_name,
                    component_key=component.key,
                    obj=component.module,
                    tags=component_tags,
                ),
                tags=component_tags,
            )
        )

    return AdapterResolvedTargets(family=model_type, targets=targets)


def build_component_module_targets(
    *,
    model_type: str,
    loaded_components: Sequence[LoadedModelComponent],
    include_text_encoders: list[bool] | None = None,
    include_vae: bool = False,
    include_denoiser: bool = False,
) -> AdapterResolvedTargets:
    """Build module-resolved adapter targets from selected declared components.

    Optimization still decides which model components are in scope. This helper
    expands those selected components into concrete target modules with stable
    component/path provenance for downstream adapter runtimes.
    """

    targets: list[AdapterResolvedTarget] = []
    for component, role_tags in _iter_selected_loaded_components(
        loaded_components,
        include_text_encoders=include_text_encoders,
        include_vae=include_vae,
        include_denoiser=include_denoiser,
    ):
        targets.extend(
            _iter_component_module_targets(
                component=component.public_name,
                component_key=component.key,
                root_module=component.module,
                tags=role_tags,
            )
        )

    return AdapterResolvedTargets(family=model_type, targets=targets)
