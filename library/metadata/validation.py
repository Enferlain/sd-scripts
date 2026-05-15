"""Metadata validation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from library.metadata.providers import MetadataRequiredFact
from library.metadata.records import MetadataRecord


@dataclass(frozen=True)
class MissingMetadataFact:
    """A required fact that was not present in collected metadata."""

    fact_key: str
    owner: str
    entity_type: str | None = None
    identifier: str | None = None
    namespace: str | None = None
    description: str | None = None


class MetadataValidationError(ValueError):
    """Raised when required metadata facts are missing or invalid."""

    def __init__(self, missing_facts: Iterable[MissingMetadataFact]) -> None:
        self.missing_facts = tuple(missing_facts)
        details = ", ".join(_format_missing_fact(fact) for fact in self.missing_facts)
        super().__init__(f"Missing required metadata facts: {details}")


def validate_required_facts(records: Iterable[MetadataRecord], required_facts: Iterable[MetadataRequiredFact]) -> None:
    """Fail when any declared required fact is absent from matching records."""
    record_tuple = tuple(records)
    missing: list[MissingMetadataFact] = []
    for required_fact in required_facts:
        matching_records = _matching_records(record_tuple, required_fact)
        if not matching_records or not any(record.has_fact(required_fact.fact_key) for record in matching_records):
            missing.append(
                MissingMetadataFact(
                    fact_key=required_fact.fact_key,
                    owner=required_fact.owner,
                    entity_type=required_fact.entity_type,
                    identifier=required_fact.identifier,
                    namespace=required_fact.namespace,
                    description=required_fact.description,
                )
            )
    if missing:
        raise MetadataValidationError(missing)


def _matching_records(records: tuple[MetadataRecord, ...], required_fact) -> tuple[MetadataRecord, ...]:
    return tuple(
        record
        for record in records
        if (required_fact.entity_type is None or record.identity.entity_type == required_fact.entity_type)
        and (required_fact.identifier is None or record.identity.identifier == required_fact.identifier)
        and (required_fact.namespace is None or record.identity.namespace == required_fact.namespace)
    )


def _format_missing_fact(fact: MissingMetadataFact) -> str:
    scope_parts = [
        part
        for part in (
            fact.namespace,
            fact.entity_type,
            fact.identifier,
        )
        if part is not None
    ]
    scope = ":".join(scope_parts) if scope_parts else "any"
    return f"{fact.fact_key} (owner={fact.owner}, scope={scope})"
