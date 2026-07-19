"""Shared model-family, realization, component, and artifact fact shapes."""

from __future__ import annotations

import datetime

from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import quote


def build_model_realization_identifier(run_identifier: str, realization_key: str) -> str:
    """Build a run-qualified durable identifier for one realized model."""
    return f"run/{_identity_segment(run_identifier)}/model/{_identity_segment(realization_key)}"


def build_model_component_identifier(realization_identifier: str, component_key: str) -> str:
    """Build a realization-qualified durable identifier for one component."""
    return f"{realization_identifier}/component/{_identity_segment(component_key)}"


def build_model_family_contribution_identifier(
    realization_identifier: str,
    contribution_namespace: str,
    contribution_version: str,
) -> str:
    """Build a qualified identity for a versioned family-local contribution."""
    return f"{realization_identifier}/family/{_identity_segment(contribution_namespace)}/{_identity_segment(contribution_version)}"


def _identity_segment(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Model metadata identity segments must be non-empty strings.")
    return quote(value, safe="")


@dataclass(frozen=True, slots=True)
class ModelFamilyDeclarationReference:
    """Reference to a mandatory version of the family-owned declaration surface."""

    family_identifier: str
    declaration_version: str = "1"


@dataclass(frozen=True, slots=True)
class ModelRealizationFacts:
    """Run-scoped facts for one loaded model realization."""

    run_identifier: str
    realization_key: str
    realization_identifier: str
    family: ModelFamilyDeclarationReference
    model_version: str | None = None

    @classmethod
    def for_run(
        cls,
        *,
        run_identifier: str,
        realization_key: str,
        family_identifier: str,
        model_version: str | None = None,
        declaration_version: str = "1",
    ) -> ModelRealizationFacts:
        """Build realization facts with the shared qualified identity."""
        return cls(
            run_identifier=run_identifier,
            realization_key=realization_key,
            realization_identifier=build_model_realization_identifier(run_identifier, realization_key),
            family=ModelFamilyDeclarationReference(
                family_identifier=family_identifier,
                declaration_version=declaration_version,
            ),
            model_version=model_version,
        )


@dataclass(frozen=True, slots=True)
class RealizedModelComponentFacts:
    """Persistable declaration and presence facts for one loaded component."""

    run_identifier: str
    realization_identifier: str
    component_identifier: str
    component_key: str
    public_name: str
    declaration_order: int
    roles: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    present: bool = True

    @classmethod
    def for_realization(
        cls,
        *,
        run_identifier: str,
        realization_identifier: str,
        component_key: str,
        public_name: str,
        declaration_order: int,
        roles: tuple[str, ...] = (),
        capabilities: tuple[str, ...] = (),
        present: bool = True,
    ) -> RealizedModelComponentFacts:
        """Build component facts with the shared qualified identity."""
        return cls(
            run_identifier=run_identifier,
            realization_identifier=realization_identifier,
            component_identifier=build_model_component_identifier(realization_identifier, component_key),
            component_key=component_key,
            public_name=public_name,
            declaration_order=declaration_order,
            roles=roles,
            capabilities=capabilities,
            present=present,
        )


@dataclass(frozen=True, slots=True)
class ModelFamilyMetadataField:
    """One canonical field in an explicit versioned family contribution."""

    name: str
    value: str | int | float | bool


@dataclass(frozen=True, slots=True)
class ModelFamilyMetadataContribution:
    """Namespaced family-local facts that do not belong in the shared schema."""

    run_identifier: str
    realization_identifier: str
    contribution_identifier: str
    contribution_namespace: str
    contribution_version: str
    fields: tuple[ModelFamilyMetadataField, ...]

    @classmethod
    def for_realization(
        cls,
        *,
        run_identifier: str,
        realization_identifier: str,
        contribution_namespace: str,
        contribution_version: str,
        fields: tuple[ModelFamilyMetadataField, ...],
    ) -> ModelFamilyMetadataContribution:
        """Build family-local facts with a shared qualified identity."""
        return cls(
            run_identifier=run_identifier,
            realization_identifier=realization_identifier,
            contribution_identifier=build_model_family_contribution_identifier(
                realization_identifier,
                contribution_namespace,
                contribution_version,
            ),
            contribution_namespace=contribution_namespace,
            contribution_version=contribution_version,
            fields=fields,
        )


@dataclass(frozen=True, slots=True)
class ModelArtifactPresentation:
    """User-authored presentation facts supplied to artifact resolution.

    Extension fields remain string-valued because they feed compatibility
    projections. Typed family-local scalar facts belong in an explicit
    ``ModelFamilyMetadataContribution`` instead.
    """

    title: str | None = None
    description: str | None = None
    author: str | None = None
    license: str | None = None
    tags: str | None = None
    usage_hint: str | None = None
    thumbnail: str | None = None
    merged_from: str | None = None
    trigger_phrase: str | None = None
    preprocessor: str | None = None
    is_negative_embedding: str | None = None
    extension_fields: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelArtifactResolutionContext:
    """Narrow typed inputs a family uses to resolve artifact semantics."""

    family_identifier: str
    model_version: str
    artifact_identifier: str
    artifact_role: str
    serialization_format: str
    resolution: tuple[int, int]
    created_at: float
    presentation: ModelArtifactPresentation = field(default_factory=ModelArtifactPresentation)
    realization_identifier: str | None = None
    implementation_version: str | None = None
    prediction_type: str | None = None
    timestep_range: tuple[int, int] | None = None
    encoder_layer: int | None = None


@dataclass(frozen=True, slots=True)
class ModelArtifactFacts:
    """Canonical model facts for one output artifact boundary.

    ``extension_fields`` is the string-only compatibility-extension seam;
    typed family-local values use ``ModelFamilyMetadataContribution``.
    """

    artifact_identifier: str
    family_identifier: str
    artifact_role: str
    artifact_format: str
    architecture: str
    implementation: str
    title: str
    resolution: str
    realization_identifier: str | None = None
    description: str | None = None
    author: str | None = None
    date: str | None = None
    hash_sha256: str | None = None
    implementation_version: str | None = None
    license: str | None = None
    usage_hint: str | None = None
    thumbnail: str | None = None
    tags: str | None = None
    merged_from: str | None = None
    trigger_phrase: str | None = None
    prediction_type: str | None = None
    timestep_range: str | None = None
    encoder_layer: str | None = None
    preprocessor: str | None = None
    is_negative_embedding: str | None = None
    unet_dtype: str | None = None
    vae_dtype: str | None = None
    extension_fields: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_resolution_context(
        cls,
        context: ModelArtifactResolutionContext,
        *,
        architecture: str,
        implementation: str,
        default_title: str,
    ) -> ModelArtifactFacts:
        """Build canonical artifact facts from family-resolved semantics."""
        presentation = context.presentation
        timestep_range = None if context.timestep_range is None else f"{context.timestep_range[0]},{context.timestep_range[1]}"
        return cls(
            artifact_identifier=context.artifact_identifier,
            family_identifier=context.family_identifier,
            artifact_role=context.artifact_role,
            artifact_format=context.serialization_format,
            architecture=architecture,
            implementation=implementation,
            title=presentation.title or default_title,
            resolution=f"{context.resolution[0]}x{context.resolution[1]}",
            realization_identifier=context.realization_identifier,
            description=presentation.description,
            author=presentation.author,
            date=datetime.datetime.fromtimestamp(int(context.created_at)).isoformat(),
            implementation_version=context.implementation_version,
            license=presentation.license,
            usage_hint=presentation.usage_hint,
            thumbnail=presentation.thumbnail,
            tags=presentation.tags,
            merged_from=presentation.merged_from,
            trigger_phrase=presentation.trigger_phrase,
            prediction_type=context.prediction_type,
            timestep_range=timestep_range,
            encoder_layer=None if context.encoder_layer is None else str(context.encoder_layer),
            preprocessor=presentation.preprocessor,
            is_negative_embedding=presentation.is_negative_embedding,
            extension_fields=dict(presentation.extension_fields),
        )
