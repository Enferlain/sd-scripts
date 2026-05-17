"""Repo-owned metadata backbone contracts and first in-memory backend."""

from library.metadata.backends import InMemoryMetadataBackend, MetadataBackend, MetadataSnapshot
from library.metadata.projections import (
    KuroMetadataProjection,
    MetadataProjection,
    ModelSpecCompatibilityProjection,
    ProjectionResult,
    SafetensorsMetadataProjection,
    SsCompatibilityProjection,
)
from library.metadata.providers import MetadataProvider, MetadataProviderResult, MetadataRequiredFact
from library.metadata.records import (
    AdapterMetadataRecord,
    ArtifactMetadataRecord,
    MetadataEdge,
    MetadataEvent,
    MetadataIdentity,
    MetadataRecord,
    ModelComponentMetadataRecord,
    RunMetadataRecord,
)
from library.metadata.storage import (
    InMemoryMetadataStore,
    MetadataSchemaVersionError,
    MetadataStore,
    SQLiteMetadataStore,
    SCHEMA_VERSION,
)
from library.metadata.validation import MetadataValidationError, MissingMetadataFact, validate_required_facts

__all__ = [
    "AdapterMetadataRecord",
    "ArtifactMetadataRecord",
    "InMemoryMetadataBackend",
    "InMemoryMetadataStore",
    "KuroMetadataProjection",
    "MetadataBackend",
    "MetadataEdge",
    "MetadataEvent",
    "MetadataIdentity",
    "MetadataProjection",
    "MetadataProvider",
    "MetadataProviderResult",
    "MetadataRecord",
    "MetadataRequiredFact",
    "MetadataSnapshot",
    "MetadataStore",
    "MetadataSchemaVersionError",
    "MetadataValidationError",
    "MissingMetadataFact",
    "ModelSpecCompatibilityProjection",
    "ModelComponentMetadataRecord",
    "ProjectionResult",
    "RunMetadataRecord",
    "SafetensorsMetadataProjection",
    "SCHEMA_VERSION",
    "SsCompatibilityProjection",
    "SQLiteMetadataStore",
    "validate_required_facts",
]
