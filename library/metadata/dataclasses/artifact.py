"""Shared artifact metadata fact shapes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointArtifactFacts:
    """Facts for a checkpoint artifact export boundary."""

    artifact_identifier: str
    metadata_policy: str
    kind: str = "checkpoint"
    artifact_format: str = "safetensors"
    step: int | None = None
    epoch: int | None = None

    @classmethod
    def for_checkpoint(
        cls,
        *,
        artifact_identifier: str,
        no_metadata: bool,
        step: int | None = None,
        epoch: int | None = None,
    ) -> CheckpointArtifactFacts:
        """Build checkpoint facts from the current save request."""
        return cls(
            artifact_identifier=artifact_identifier,
            metadata_policy="none" if no_metadata else "full",
            step=step,
            epoch=epoch,
        )
