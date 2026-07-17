"""Temporary shared normalization for metadata values at string-only boundaries."""

from __future__ import annotations

import json

from collections.abc import Mapping

from library.metadata.records import MetadataValue


def stringify_metadata_value(value: MetadataValue) -> str:
    """Convert one metadata value into a stable string representation."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list | tuple | dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def stringify_metadata_mapping(values: Mapping[str, MetadataValue]) -> dict[str, str]:
    """Convert a metadata mapping into stable string values."""
    return {key: stringify_metadata_value(value) for key, value in values.items()}


__all__ = ["stringify_metadata_mapping", "stringify_metadata_value"]
