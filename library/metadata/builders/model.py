"""Central builders for model-realization metadata facts."""

from __future__ import annotations

import base64
import logging
import mimetypes

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from library.config.dataclasses.output import MetadataConfig
from library.metadata.dataclasses.model import (
    ModelArtifactPresentation,
    ModelArtifactResolutionContext,
    ModelRealizationFacts,
    RealizedModelComponentFacts,
)
from library.metadata.validation import validate_metadata_items


logger = logging.getLogger(__name__)


class _LoadedModelComponentView(Protocol):
    """Narrow loaded-component input required to describe a realization."""

    @property
    def key(self) -> str: ...

    @property
    def public_name(self) -> str: ...

    @property
    def module(self) -> Any: ...

    @property
    def roles(self) -> tuple[str, ...]: ...

    @property
    def capabilities(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class ModelRealizationState:
    """Validated realization facts retained for later identity references."""

    realization: ModelRealizationFacts
    components: tuple[RealizedModelComponentFacts, ...]

    @property
    def realization_identifier(self) -> str:
        """Return the qualified model-realization identity."""
        return self.realization.realization_identifier

    def component_identifier(self, component_key: str) -> str | None:
        """Return the qualified identity for one family-local component key."""
        component = next((item for item in self.components if item.component_key == component_key), None)
        return None if component is None else component.component_identifier

    def as_metadata_items(self) -> tuple[ModelRealizationFacts | RealizedModelComponentFacts, ...]:
        """Return realization then components in family-declared filing order."""
        return (self.realization, *self.components)


def build_model_realization_state(
    *,
    run_identifier: str,
    realization_key: str,
    family_identifier: str,
    model_version: str | None,
    loaded_components: Sequence[_LoadedModelComponentView],
) -> ModelRealizationState:
    """Build validated, module-free facts from an explicit loaded-component view."""
    realization = ModelRealizationFacts.for_run(
        run_identifier=run_identifier,
        realization_key=realization_key,
        family_identifier=family_identifier,
        model_version=model_version,
    )
    components = tuple(
        RealizedModelComponentFacts.for_realization(
            run_identifier=run_identifier,
            realization_identifier=realization.realization_identifier,
            component_key=component.key,
            public_name=component.public_name,
            declaration_order=declaration_order,
            roles=component.roles,
            capabilities=component.capabilities,
            present=component.module is not None,
        )
        for declaration_order, component in enumerate(loaded_components)
    )
    state = ModelRealizationState(realization=realization, components=components)
    validate_metadata_items(state.as_metadata_items())
    return state


def build_model_artifact_resolution_context(
    *,
    metadata_config: MetadataConfig,
    family_identifier: str,
    model_version: str,
    artifact_identifier: str,
    artifact_role: str,
    serialization_format: str,
    resolution: str | int | Sequence[int] | None,
    created_at: float,
    realization_identifier: str | None = None,
    implementation_version: str | None = None,
    prediction_type: str | None = None,
    min_timestep: int | None = None,
    max_timestep: int | None = None,
    encoder_layer: int | None = None,
    extension_fields: Mapping[str, str] | None = None,
) -> ModelArtifactResolutionContext:
    """Build narrow family-resolution inputs from explicit artifact facts."""
    return ModelArtifactResolutionContext(
        family_identifier=family_identifier,
        model_version=model_version,
        artifact_identifier=artifact_identifier,
        artifact_role=artifact_role,
        serialization_format=serialization_format,
        resolution=_normalize_resolution(resolution),
        created_at=created_at,
        presentation=ModelArtifactPresentation(
            title=metadata_config.metadata_title,
            description=metadata_config.metadata_description,
            author=metadata_config.metadata_author,
            license=metadata_config.metadata_license,
            tags=metadata_config.metadata_tags,
            usage_hint=metadata_config.metadata_usage_hint,
            thumbnail=_resolve_thumbnail(metadata_config.metadata_thumbnail),
            merged_from=metadata_config.metadata_merged_from,
            trigger_phrase=metadata_config.metadata_trigger_phrase,
            preprocessor=metadata_config.metadata_preprocessor,
            is_negative_embedding=metadata_config.metadata_is_negative_embedding,
            extension_fields={} if extension_fields is None else dict(extension_fields),
        ),
        realization_identifier=realization_identifier,
        implementation_version=implementation_version,
        prediction_type=prediction_type,
        timestep_range=_normalize_timestep_range(min_timestep, max_timestep),
        encoder_layer=encoder_layer,
    )


def _normalize_resolution(resolution: str | int | Sequence[int] | None) -> tuple[int, int]:
    if resolution is None:
        raise ValueError("Model artifact metadata requires an explicit resolution.")
    if isinstance(resolution, int):
        values = (resolution, resolution)
    elif isinstance(resolution, str):
        parts = tuple(part.strip() for part in resolution.lower().replace("x", ",").split(",") if part.strip())
        if not parts:
            raise ValueError("Model artifact resolution must contain at least one integer.")
        parsed = tuple(int(part) for part in parts)
        values = (parsed[0], parsed[1] if len(parsed) > 1 else parsed[0])
    else:
        parsed = tuple(int(value) for value in resolution)
        if not parsed:
            raise ValueError("Model artifact resolution must contain at least one integer.")
        values = (parsed[0], parsed[1] if len(parsed) > 1 else parsed[0])
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError("Model artifact resolution dimensions must be positive integers.")
    return values


def _normalize_timestep_range(
    min_timestep: int | None,
    max_timestep: int | None,
) -> tuple[int, int] | None:
    if min_timestep is None and max_timestep is None:
        return None
    return (
        0 if min_timestep is None else min_timestep,
        1000 if max_timestep is None else max_timestep,
    )


def _resolve_thumbnail(thumbnail: str | None) -> str | None:
    if thumbnail is None or thumbnail.startswith("data:"):
        return thumbnail

    try:
        return _file_to_data_url(thumbnail)
    except FileNotFoundError as exc:
        logger.warning("Thumbnail file not found, skipping: %s", exc)
    except Exception as exc:  # pragma: no cover - format/IO-specific failure
        logger.warning("Failed to convert thumbnail file %s to a data URL: %s", thumbnail, exc)
    return None


def _file_to_data_url(file_path: str) -> str:
    """Convert a configured thumbnail file into artifact presentation data."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    mime_type, _ = mimetypes.guess_type(path)
    encoded_data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type or 'application/octet-stream'};base64,{encoded_data}"


__all__ = [
    "ModelRealizationState",
    "build_model_artifact_resolution_context",
    "build_model_realization_state",
]
