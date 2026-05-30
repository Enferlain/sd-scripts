"""Shared training-run metadata fact shapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from library.metadata.records import MetadataValue


@dataclass(frozen=True)
class RunMetadataFacts:
    """Training-run facts used by checkpoint/export projections.

    Run metadata stays string-shaped because the active checkpoint export path
    ultimately targets string-only artifact metadata.
    """

    run_identifier: str
    metadata: Mapping[str, str]

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, str],
        *,
        run_identifier: str,
    ) -> RunMetadataFacts:
        """Build run facts from repo-owned string metadata."""
        return cls(run_identifier=run_identifier, metadata=dict(metadata))

    def with_metadata(self, metadata: Mapping[str, MetadataValue]) -> RunMetadataFacts:
        """Return a copy with additional stringified training-run facts."""
        return replace(
            self,
            metadata={
                **dict(self.metadata),
                **{key: str(value) for key, value in metadata.items()},
            },
        )
