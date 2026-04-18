from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from library.models.parameter_dump import resolve_component_names

@dataclass(slots=True)
class AdapterResolvedTarget:
    """Adapter-facing view of one resolved original-model target."""

    component: str
    path: str
    module: Any
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AdapterResolvedTargets:
    """Adapter-facing bundle of resolved targets for one training run."""

    family: str
    targets: list[AdapterResolvedTarget] = field(default_factory=list)


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
    """Build the first adapter target bundle from selected top-level components.

    This intentionally keeps the first migration slice narrow: optimization
    resolves which model components are in scope, and the adapter runtime sees
    those selected component roots as its input bundle.
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
        targets.append(
            AdapterResolvedTarget(
                component=public_label,
                path=public_label,
                module=text_encoder,
                tags=frozenset({"component_root", "text_encoder"}),
                metadata={"component_key": f"text_encoder{index + 1}"},
            )
        )

    if include_vae and vae is not None:
        public_label = component_names.vae_name if component_names is not None else "vae"
        targets.append(
            AdapterResolvedTarget(
                component=public_label,
                path=public_label,
                module=vae,
                tags=frozenset({"component_root", "vae"}),
                metadata={"component_key": "vae"},
            )
        )

    if include_denoiser and denoiser is not None:
        public_label = component_names.denoiser_name if component_names is not None else "denoiser"
        targets.append(
            AdapterResolvedTarget(
                component=public_label,
                path=public_label,
                module=denoiser,
                tags=frozenset({"component_root", "denoiser"}),
                metadata={"component_key": "denoiser"},
            )
        )

    return AdapterResolvedTargets(family=model_type, targets=targets)
