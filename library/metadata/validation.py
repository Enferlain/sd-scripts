"""Metadata validation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping as AbcMapping
from dataclasses import dataclass, fields, is_dataclass
from types import UnionType
from typing import Any, cast, TypeGuard, Union, get_args, get_origin, get_type_hints

from library.metadata.providers import MetadataRequiredFact
from library.metadata.records import MetadataEdge, MetadataIdentity, MetadataRecord
from library.metadata.registry import METADATA_ITEM_TYPES, supported_metadata_item_names

from library.metadata.dataclasses.resource import (
    ResourceAccountingFacts,
    ResourceAccountingGapFacts,
    ResourceFactReference,
    ResourceObservationFrameFacts,
    ResourceProfileFacts,
)
from library.metadata.dataclasses.model import (
    build_model_component_identifier,
    build_model_family_contribution_identifier,
    build_model_realization_identifier,
    ModelArtifactFacts,
    ModelFamilyMetadataContribution,
    ModelRealizationFacts,
    RealizedModelComponentFacts,
)


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


class MetadataItemValidationError(TypeError):
    """Raised when a filed metadata item does not match the accepted shape."""


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


def validate_resource_source_evidence(
    records: Iterable[MetadataRecord],
    edges: Iterable[MetadataEdge],
) -> None:
    """Fail when derived resource records reference missing evidence or edges."""
    record_tuple = tuple(records)
    records_by_key = {record.identity.key: record for record in record_tuple}
    edge_keys = {(edge.source.key, edge.target.key, edge.relationship) for edge in edges}
    missing: list[MissingMetadataFact] = []

    for record in record_tuple:
        if not _is_derived_resource_record(record):
            continue
        if not record.has_fact("derivation_version"):
            missing.append(_missing_resource_evidence(record, "derivation_version"))
        references = _record_source_fact_references(record)
        if not references:
            missing.append(_missing_resource_evidence(record, "source_fact_references"))
            continue
        for index, reference in enumerate(references):
            reference_identity = _resource_reference_identity(reference, default_namespace=record.identity.namespace)
            relationship = reference.get("relationship")
            if reference_identity is None or not _is_non_empty_string(relationship):
                missing.append(_missing_resource_evidence(record, f"source_fact_references[{index}]"))
                continue
            if reference_identity.key not in records_by_key:
                missing.append(_missing_resource_evidence(record, f"source_fact_references[{index}].record"))
            if (record.identity.key, reference_identity.key, relationship) not in edge_keys:
                missing.append(_missing_resource_evidence(record, f"source_fact_references[{index}].relationship"))

    if missing:
        raise MetadataValidationError(missing)


def validate_metadata_item(item: object) -> None:
    """Validate that one filed metadata item matches the accepted shared shape."""
    if not is_dataclass(item):
        raise MetadataItemValidationError("Metadata items must be dataclass instances.")
    if not isinstance(item, METADATA_ITEM_TYPES):
        raise MetadataItemValidationError(
            f"Unsupported metadata item type {type(item).__name__}. Supported types: {supported_metadata_item_names()}."
        )
    _validate_dataclass_fields(item, error_type=MetadataItemValidationError)
    _validate_resource_observation_frame(item)
    _validate_resource_item_relationships(item)
    _validate_model_item(item)


def validate_metadata_items(items: Iterable[object]) -> None:
    """Validate accepted items plus collection-level identity constraints."""
    item_tuple = tuple(items)
    for item in item_tuple:
        validate_metadata_item(item)

    components_by_realization: dict[str, list[RealizedModelComponentFacts]] = {}
    for item in item_tuple:
        if isinstance(item, RealizedModelComponentFacts):
            components_by_realization.setdefault(item.realization_identifier, []).append(item)

    for realization_identifier, components in components_by_realization.items():
        _validate_realized_component_collection(realization_identifier, components)


def _validate_model_item(item: object) -> None:
    if isinstance(item, ModelRealizationFacts):
        _validate_non_empty_fields(
            item,
            "run_identifier",
            "realization_key",
            "realization_identifier",
        )
        _validate_non_empty_fields(item.family, "family_identifier", "declaration_version")
        expected_identifier = build_model_realization_identifier(item.run_identifier, item.realization_key)
        if item.realization_identifier != expected_identifier:
            raise MetadataItemValidationError(
                "ModelRealizationFacts.realization_identifier must use the shared run-qualified identity constructor."
            )
        if item.model_version is not None and not _is_non_empty_string(item.model_version):
            raise MetadataItemValidationError("ModelRealizationFacts.model_version must be None or a non-empty string.")
        return

    if isinstance(item, RealizedModelComponentFacts):
        _validate_non_empty_fields(
            item,
            "run_identifier",
            "realization_identifier",
            "component_identifier",
            "component_key",
            "public_name",
        )
        if item.declaration_order < 0:
            raise MetadataItemValidationError("RealizedModelComponentFacts.declaration_order must be non-negative.")
        expected_identifier = build_model_component_identifier(item.realization_identifier, item.component_key)
        if item.component_identifier != expected_identifier:
            raise MetadataItemValidationError(
                "RealizedModelComponentFacts.component_identifier must use the shared realization-qualified identity constructor."
            )
        _validate_string_terms(item.roles, field_name="roles")
        _validate_string_terms(item.capabilities, field_name="capabilities")
        return

    if isinstance(item, ModelArtifactFacts):
        _validate_non_empty_fields(
            item,
            "artifact_identifier",
            "family_identifier",
            "artifact_role",
            "artifact_format",
            "architecture",
            "implementation",
            "title",
            "resolution",
        )
        if item.realization_identifier is not None and (
            not _is_non_empty_string(item.realization_identifier) or not item.realization_identifier.startswith("run/")
        ):
            raise MetadataItemValidationError(
                "ModelArtifactFacts.realization_identifier must be None or a shared run-qualified model identity."
            )
        for key, value in item.extension_fields.items():
            if not _is_non_empty_string(key):
                raise MetadataItemValidationError("ModelArtifactFacts extension field names must be non-empty strings.")
            if key.startswith(("modelspec.", "ss_")):
                raise MetadataItemValidationError(
                    "ModelArtifactFacts extension fields must use canonical names, not rendered compatibility prefixes."
                )
            if not isinstance(value, str):
                raise MetadataItemValidationError("ModelArtifactFacts extension field values must be strings.")
        return

    if isinstance(item, ModelFamilyMetadataContribution):
        _validate_non_empty_fields(
            item,
            "run_identifier",
            "realization_identifier",
            "contribution_identifier",
            "contribution_namespace",
            "contribution_version",
        )
        expected_identifier = build_model_family_contribution_identifier(
            item.realization_identifier,
            item.contribution_namespace,
            item.contribution_version,
        )
        if item.contribution_identifier != expected_identifier:
            raise MetadataItemValidationError(
                "ModelFamilyMetadataContribution.contribution_identifier must use the shared qualified identity constructor."
            )
        if not item.fields:
            raise MetadataItemValidationError("ModelFamilyMetadataContribution must include at least one field.")
        field_names: list[str] = []
        for field in item.fields:
            if not _is_non_empty_string(field.name):
                raise MetadataItemValidationError("Model family contribution field names must be non-empty strings.")
            if field.name.startswith(("modelspec.", "ss_")):
                raise MetadataItemValidationError(
                    "Model family contribution fields must use canonical names, not rendered compatibility prefixes."
                )
            if not isinstance(field.value, str | int | float | bool):
                raise MetadataItemValidationError("Model family contribution values must be scalar metadata values.")
            field_names.append(field.name)
        if len(field_names) != len(set(field_names)):
            raise MetadataItemValidationError("Model family contribution field names must be unique.")


def _validate_non_empty_fields(item: object, *field_names: str) -> None:
    for field_name in field_names:
        if not _is_non_empty_string(getattr(item, field_name)):
            raise MetadataItemValidationError(
                f"Metadata item {type(item).__name__}.{field_name} must be a non-empty string."
            )


def _validate_string_terms(values: tuple[str, ...], *, field_name: str) -> None:
    if any(not _is_non_empty_string(value) for value in values):
        raise MetadataItemValidationError(f"Model component {field_name} must contain only non-empty strings.")
    if len(values) != len(set(values)):
        raise MetadataItemValidationError(f"Model component {field_name} must not contain duplicates.")


def _validate_realized_component_collection(
    realization_identifier: str,
    components: list[RealizedModelComponentFacts],
) -> None:
    component_keys = [component.component_key for component in components]
    component_identifiers = [component.component_identifier for component in components]
    declaration_orders = [component.declaration_order for component in components]
    for field_name, values in (
        ("component_key", component_keys),
        ("component_identifier", component_identifiers),
        ("declaration_order", declaration_orders),
    ):
        if len(values) != len(set(values)):
            raise MetadataItemValidationError(
                f"Model components for {realization_identifier!r} must have unique {field_name} values."
            )


def _validate_resource_observation_frame(item: object) -> None:
    if not isinstance(item, ResourceObservationFrameFacts):
        return
    if not item.measurements:
        raise MetadataItemValidationError("ResourceObservationFrameFacts must include at least one measurement.")
    measurement_identifiers = [measurement.measurement_identifier for measurement in item.measurements]
    if len(measurement_identifiers) != len(set(measurement_identifiers)):
        raise MetadataItemValidationError("ResourceObservationFrameFacts measurement identifiers must be unique.")
    for measurement in item.measurements:
        _validate_dataclass_fields(measurement, error_type=MetadataItemValidationError)


def _validate_resource_item_relationships(item: object) -> None:
    if not _requires_resource_source_references(item):
        return
    derivation_version = getattr(item, "derivation_version", None)
    if not _is_non_empty_string(derivation_version):
        raise MetadataItemValidationError(f"Metadata item {type(item).__name__} must include a non-empty derivation_version.")
    if not item.source_fact_references:
        raise MetadataItemValidationError(f"Metadata item {type(item).__name__} must include at least one source_fact_reference.")
    for index, reference in enumerate(item.source_fact_references):
        _validate_resource_source_reference(reference, item_type=type(item).__name__, index=index)


def _requires_resource_source_references(
    item: object,
) -> TypeGuard[ResourceProfileFacts | ResourceAccountingFacts | ResourceAccountingGapFacts]:
    return isinstance(item, (ResourceProfileFacts, ResourceAccountingFacts, ResourceAccountingGapFacts))


def _is_derived_resource_record(record: MetadataRecord) -> bool:
    return record.identity.entity_type in {
        "resource_profile",
        "resource_accounting",
        "resource_accounting_gap",
    }


def _record_source_fact_references(record: MetadataRecord) -> tuple[dict[str, object], ...]:
    references = record.facts.get("source_fact_references")
    if not isinstance(references, list | tuple):
        return ()
    return tuple(cast(dict[str, object], reference) for reference in references if isinstance(reference, dict))


def _resource_reference_identity(
    reference: AbcMapping[str, object],
    *,
    default_namespace: str,
) -> MetadataIdentity | None:
    entity_type = reference.get("entity_type")
    identifier = reference.get("identifier")
    namespace = reference.get("namespace", default_namespace)
    if not _is_non_empty_string(entity_type) or not _is_non_empty_string(identifier):
        return None
    if namespace is not None and not _is_non_empty_string(namespace):
        return None
    return MetadataIdentity(
        entity_type=entity_type,
        identifier=identifier,
        namespace=default_namespace if namespace is None else namespace,
    )


def _missing_resource_evidence(record: MetadataRecord, fact_key: str) -> MissingMetadataFact:
    return MissingMetadataFact(
        fact_key=fact_key,
        owner=record.producer,
        entity_type=record.identity.entity_type,
        identifier=record.identity.identifier,
        namespace=record.identity.namespace,
        description="derived resource facts require existing source evidence and declared source relationships",
    )


def _validate_resource_source_reference(
    reference: ResourceFactReference,
    *,
    item_type: str,
    index: int,
) -> None:
    field_prefix = f"{item_type}.source_fact_references[{index}]"
    for field_name in ("entity_type", "identifier", "relationship"):
        value = getattr(reference, field_name)
        if not _is_non_empty_string(value):
            raise MetadataItemValidationError(f"{field_prefix}.{field_name} must be a non-empty string.")
    if reference.namespace is not None and not _is_non_empty_string(reference.namespace):
        raise MetadataItemValidationError(f"{field_prefix}.namespace must be None or a non-empty string.")


def _is_non_empty_string(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip())


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


def _validate_dataclass_fields(item: object, *, error_type: type[Exception]) -> None:
    type_hints = get_type_hints(type(item))
    for field in fields(cast(Any, item)):
        expected_type = type_hints.get(field.name)
        if expected_type is None:
            continue
        _validate_value(
            getattr(item, field.name),
            expected_type,
            field_name=field.name,
            error_type=error_type,
        )


def _validate_value(value: object, expected_type: Any, *, field_name: str, error_type: type[Exception]) -> None:
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if expected_type is Any:
        return

    if origin in (UnionType, Union):
        if any(_matches_type(value, option) for option in args):
            return
        expected_names = ", ".join(_type_name(option) for option in args)
        raise error_type(f"Metadata field {field_name!r} must match one of ({expected_names}), got {type(value).__name__}.")

    if origin is tuple:
        if not isinstance(value, tuple):
            raise error_type(f"Metadata field {field_name!r} must be a tuple, got {type(value).__name__}.")
        if len(args) == 2 and args[1] is Ellipsis:
            item_type = args[0]
            for index, item in enumerate(value):
                try:
                    _validate_value(
                        item,
                        item_type,
                        field_name=f"{field_name}[{index}]",
                        error_type=error_type,
                    )
                except error_type as exc:
                    raise error_type(str(exc)) from exc
            return
        if len(args) != len(value):
            raise error_type(f"Metadata field {field_name!r} must contain {len(args)} items, got {len(value)}.")
        for index, (item, item_type) in enumerate(zip(value, args, strict=True)):
            _validate_value(
                item,
                item_type,
                field_name=f"{field_name}[{index}]",
                error_type=error_type,
            )
        return

    if origin in (dict,):
        if not isinstance(value, dict):
            raise error_type(f"Metadata field {field_name!r} must be a dict, got {type(value).__name__}.")
        key_type, value_type = args
        for key, item in value.items():
            _validate_value(key, key_type, field_name=f"{field_name}.key", error_type=error_type)
            _validate_value(item, value_type, field_name=f"{field_name}[{key!r}]", error_type=error_type)
        return

    if origin is AbcMapping:
        if not isinstance(value, AbcMapping):
            raise error_type(f"Metadata field {field_name!r} must be a mapping, got {type(value).__name__}.")
        key_type, value_type = args
        for key, item in value.items():
            _validate_value(key, key_type, field_name=f"{field_name}.key", error_type=error_type)
            _validate_value(item, value_type, field_name=f"{field_name}[{key!r}]", error_type=error_type)
        return

    if not _matches_type(value, expected_type):
        raise error_type(f"Metadata field {field_name!r} must be {_type_name(expected_type)}, got {type(value).__name__}.")


def _matches_type(value: object, expected_type: Any) -> bool:
    if expected_type is Any:
        return True

    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin in (UnionType, Union):
        return any(_matches_type(value, option) for option in args)

    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_matches_type(item, args[0]) for item in value)
        return len(args) == len(value) and all(_matches_type(item, item_type) for item, item_type in zip(value, args))

    if origin in (dict,):
        if not isinstance(value, dict):
            return False
        key_type, value_type = args
        return all(_matches_type(key, key_type) and _matches_type(item, value_type) for key, item in value.items())

    if origin is AbcMapping:
        if not isinstance(value, AbcMapping):
            return False
        key_type, value_type = args
        return all(_matches_type(key, key_type) and _matches_type(item, value_type) for key, item in value.items())

    if isinstance(expected_type, type):
        return isinstance(value, expected_type)
    return True


def _type_name(expected_type: Any) -> str:
    if isinstance(expected_type, type):
        return expected_type.__name__
    return str(expected_type)
