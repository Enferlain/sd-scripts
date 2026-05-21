"""Shared training-run metadata fact shapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from library.metadata.records import MetadataValue


@dataclass(frozen=True)
class RunMetadataFacts:
    """Training-run facts used by checkpoint/export projections.

    Compatibility metadata stays string-shaped because the active checkpoint
    export path ultimately targets string-only artifact metadata.
    """

    run_identifier: str
    compatibility_metadata: Mapping[str, str]

    @classmethod
    def from_ss_metadata(
        cls,
        metadata: Mapping[str, str],
        *,
        run_identifier: str,
    ) -> RunMetadataFacts:
        """Build run facts from legacy Kohya-compatible metadata."""
        return cls(run_identifier=run_identifier, compatibility_metadata=dict(metadata))

    def with_compatibility_metadata(self, metadata: Mapping[str, MetadataValue]) -> RunMetadataFacts:
        """Return a copy with additional stringified compatibility facts."""
        return replace(
            self,
            compatibility_metadata={
                **dict(self.compatibility_metadata),
                **{key: str(value) for key, value in metadata.items()},
            },
        )
