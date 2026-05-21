"""Shared model metadata fact shapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpecFacts:
    """Model facts that feed the metadata emitters."""

    compatibility_metadata: Mapping[str, str]
    model_identifier: str = "active-model"
    architecture: str | None = None
    implementation: str | None = None
    prediction_type: str | None = None

    @classmethod
    def from_modelspec_metadata(
        cls,
        metadata: Mapping[str, str],
        *,
        model_identifier: str = "active-model",
    ) -> ModelSpecFacts:
        """Build typed model facts from a `modelspec.*` compatibility dict."""
        return cls(
            compatibility_metadata=dict(metadata),
            model_identifier=model_identifier,
            architecture=metadata.get("modelspec.architecture"),
            implementation=metadata.get("modelspec.implementation"),
            prediction_type=metadata.get("modelspec.prediction_type"),
        )
