from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import nn

from library.models.parameter_dump import resolve_component_names
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


def build_component_root_targets(
    *,
    model_type: str,
    text_encoders: list[Any],
    vae: Any,
    denoiser: Any,
    include_text_encoders: list[bool] | None = None,
    include_vae: bool = False,
    include_denoiser: bool = False,
) -> AdapterResolvedTargets:
    """Build component-root adapter targets for compatibility-oriented callers."""

    component_names = resolve_component_names(model_type)
    encoder_flags = list(include_text_encoders or [])
    while len(encoder_flags) < len(text_encoders):
        encoder_flags.append(False)

    targets: list[AdapterResolvedTarget] = []
    for index, (text_encoder, include_target) in enumerate(zip(text_encoders, encoder_flags, strict=False)):
        if not include_target or text_encoder is None:
            continue
        public_label = (
            component_names.text_encoder_names[index]
            if component_names is not None and index < len(component_names.text_encoder_names)
            else f"text_encoder{index + 1}"
        )
        targets.append(
            _build_resolved_target(
                target_ref=build_component_target_ref(
                    component=public_label,
                    component_key=f"text_encoder{index + 1}",
                    obj=text_encoder,
                    tags=frozenset({"component_root", "text_encoder"}),
                ),
                tags=frozenset({"component_root", "text_encoder"}),
            )
        )

    if include_vae and vae is not None:
        public_label = component_names.vae_name if component_names is not None else "vae"
        targets.append(
            _build_resolved_target(
                target_ref=build_component_target_ref(
                    component=public_label,
                    component_key="vae",
                    obj=vae,
                    tags=frozenset({"component_root", "vae"}),
                ),
                tags=frozenset({"component_root", "vae"}),
            )
        )

    if include_denoiser and denoiser is not None:
        public_label = component_names.denoiser_name if component_names is not None else "denoiser"
        targets.append(
            _build_resolved_target(
                target_ref=build_component_target_ref(
                    component=public_label,
                    component_key="denoiser",
                    obj=denoiser,
                    tags=frozenset({"component_root", "denoiser"}),
                ),
                tags=frozenset({"component_root", "denoiser"}),
            )
        )

    return AdapterResolvedTargets(family=model_type, targets=targets)


def build_component_module_targets(
    *,
    model_type: str,
    text_encoders: list[Any],
    vae: Any,
    denoiser: Any,
    include_text_encoders: list[bool] | None = None,
    include_vae: bool = False,
    include_denoiser: bool = False,
) -> AdapterResolvedTargets:
    """Build module-resolved adapter targets from selected top-level components.

    Optimization still decides which model components are in scope. This helper
    expands those selected components into concrete target modules with stable
    component/path provenance for downstream adapter runtimes.
    """

    component_names = resolve_component_names(model_type)
    encoder_flags = list(include_text_encoders or [])
    while len(encoder_flags) < len(text_encoders):
        encoder_flags.append(False)

    targets: list[AdapterResolvedTarget] = []
    for index, (text_encoder, include_target) in enumerate(zip(text_encoders, encoder_flags, strict=False)):
        if not include_target or text_encoder is None:
            continue
        public_label = (
            component_names.text_encoder_names[index]
            if component_names is not None and index < len(component_names.text_encoder_names)
            else f"text_encoder{index + 1}"
        )
        targets.extend(
            _iter_component_module_targets(
                component=public_label,
                component_key=f"text_encoder{index + 1}",
                root_module=text_encoder,
                tags=frozenset({"text_encoder"}),
            )
        )

    if include_vae and vae is not None:
        public_label = component_names.vae_name if component_names is not None else "vae"
        targets.extend(
            _iter_component_module_targets(
                component=public_label,
                component_key="vae",
                root_module=vae,
                tags=frozenset({"vae"}),
            )
        )

    if include_denoiser and denoiser is not None:
        public_label = component_names.denoiser_name if component_names is not None else "denoiser"
        targets.extend(
            _iter_component_module_targets(
                component=public_label,
                component_key="denoiser",
                root_module=denoiser,
                tags=frozenset({"denoiser"}),
            )
        )

    return AdapterResolvedTargets(family=model_type, targets=targets)
