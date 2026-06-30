"""Resource-domain derivation for evidence-constrained accounting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from library.metadata.dataclasses.resource import ResourceAccountingFacts, ResourceAccountingGapFacts, ResourceFactReference
from library.metadata.graph import MetadataRelationship
from library.metadata.records import MetadataRecord, MetadataValue
from library.metadata.views import ResourceRunView

from .profiles import RUN_RESOURCE_SUMMARY_DERIVATION_VERSION, RUN_RESOURCE_SUMMARY_PROFILE_KIND


STRUCTURAL_RESOURCE_ACCOUNTING_DERIVATION_VERSION = "structural_resource_accounting_v1"
ACCOUNTING_GAP_DERIVATION_VERSION = "resource_accounting_gap_v1"

_ELIGIBLE_STRUCTURAL_SOURCES_BY_KIND = {
    "gradient_memory": frozenset({"startup_training_state_estimator"}),
    "optimizer_state_memory": frozenset({"startup_training_state_estimator"}),
    "parameter_memory": frozenset({"startup_component_memory"}),
    "trainable_parameter_memory": frozenset({"startup_component_memory"}),
}

_PROFILE_GAP_VALUE_MAP = {
    "gpu_allocated_peak_mib": ("gpu_memory", "MiB", "run:gpu_allocated_peak"),
    "gpu_reserved_peak_mib": ("gpu_memory", "MiB", "run:gpu_reserved_peak"),
    "gpu_used_peak_mib": ("gpu_memory", "MiB", "run:gpu_used_peak"),
    "gradient_memory_estimate_mib": ("gradient_memory", "MiB", "run:startup_gradient_estimate"),
    "loaded_parameter_memory_mib": ("parameter_memory", "MiB", "run:loaded_parameter_memory"),
    "optimizer_state_memory_estimate_mib": ("optimizer_state_memory", "MiB", "run:startup_optimizer_state_estimate"),
    "trainable_parameter_memory_mib": ("trainable_parameter_memory", "MiB", "run:trainable_parameter_memory"),
}


def build_structural_resource_accounting(view: ResourceRunView) -> tuple[ResourceAccountingFacts, ...]:
    """Derive owner-bearing accounting statements from accepted structural facts."""
    statements: list[ResourceAccountingFacts] = []
    for structural_record in view.structural_facts():
        quantity = _numeric_fact(structural_record, "quantity")
        if quantity is None or not _is_eligible_structural_record(structural_record):
            continue
        statements.append(
            ResourceAccountingFacts(
                accounting_identifier=_accounting_identifier(view.run_identifier, structural_record),
                run_identifier=view.run_identifier,
                resource_kind=str(structural_record.facts["resource_kind"]),
                quantity=quantity,
                unit=str(structural_record.facts["unit"]),
                owner_type=str(structural_record.facts["owner_type"]),
                owner_identifier=str(structural_record.facts["owner_identifier"]),
                basis=_accounting_basis(structural_record),
                derivation_method="accepted_structural_resource_fact",
                derivation_version=STRUCTURAL_RESOURCE_ACCOUNTING_DERIVATION_VERSION,
                source_fact_references=(
                    ResourceFactReference(
                        entity_type=structural_record.identity.entity_type,
                        identifier=structural_record.identity.identifier,
                        relationship=MetadataRelationship.DERIVED_FROM,
                        namespace=structural_record.identity.namespace,
                    ),
                ),
                validity_scope=_validity_scope(structural_record),
                metadata=_accounting_metadata(structural_record),
            )
        )
    return tuple(statements)


def build_operation_window_resource_accounting(view: ResourceRunView) -> tuple[ResourceAccountingFacts, ...]:
    """Return operation/window accounting statements supported by current owner-scope evidence.

    Current phase, step, and diagnostic-window observations preserve useful
    runtime context, but they do not declare a resource owner. Until a runtime
    or owning domain files accepted owner-scope evidence, operation/window
    accounting must not create owner-bearing statements.
    """
    _ = view
    return ()


def build_resource_accounting_gaps(view: ResourceRunView) -> tuple[ResourceAccountingGapFacts, ...]:
    """Derive explicit gaps for accepted profile quantities not explained by accounting."""
    profile = view.latest_profile(
        profile_kind=RUN_RESOURCE_SUMMARY_PROFILE_KIND,
        derivation_version=RUN_RESOURCE_SUMMARY_DERIVATION_VERSION,
    )
    if profile is None:
        return ()

    values = profile.facts.get("values")
    if not isinstance(values, dict):
        return ()

    statements_by_resource = _accounted_quantities(view.accounting_statements())
    gaps: list[ResourceAccountingGapFacts] = []
    for value_key, (resource_kind, unit, scope) in _PROFILE_GAP_VALUE_MAP.items():
        profile_quantity = _numeric_mapping_value(values, value_key)
        if profile_quantity is None:
            continue
        accounted_quantity, source_statements = statements_by_resource.get((resource_kind, unit), (0.0, ()))
        gap_quantity = max(0.0, profile_quantity - accounted_quantity)
        if gap_quantity <= 0.0:
            continue
        gaps.append(
            ResourceAccountingGapFacts(
                gap_identifier=f"{view.run_identifier}:accounting_gap:{value_key}:{ACCOUNTING_GAP_DERIVATION_VERSION}",
                run_identifier=view.run_identifier,
                resource_kind=resource_kind,
                quantity=gap_quantity,
                unit=unit,
                basis="profile_value_minus_accounted",
                derivation_method="run_resource_summary_profile_gap",
                derivation_version=ACCOUNTING_GAP_DERIVATION_VERSION,
                source_fact_references=(
                    _source_reference(profile),
                    *(_source_reference(statement) for statement in source_statements),
                ),
                scope=scope,
                reason="no_accepted_accounting_for_profile_value" if accounted_quantity == 0.0 else "accepted_accounting_incomplete",
                metadata={
                    "accounted_quantity": accounted_quantity,
                    "profile_identifier": profile.identity.identifier,
                    "profile_value_key": value_key,
                },
            )
        )
    return tuple(gaps)


def _is_eligible_structural_record(record: MetadataRecord) -> bool:
    resource_kind = record.facts.get("resource_kind")
    source = record.facts.get("source")
    if not isinstance(resource_kind, str) or not isinstance(source, str):
        return False
    if source not in _ELIGIBLE_STRUCTURAL_SOURCES_BY_KIND.get(resource_kind, frozenset()):
        return False
    return (
        _has_string_fact(record, "owner_type")
        and _has_string_fact(record, "owner_identifier")
        and _has_string_fact(record, "unit")
        and _has_string_fact(record, "basis")
        and (quantity := _numeric_fact(record, "quantity")) is not None
        and quantity >= 0.0
    )


def _accounting_identifier(run_identifier: str, structural_record: MetadataRecord) -> str:
    return f"{run_identifier}:accounting:structural:{structural_record.identity.identifier}"


def _accounted_quantities(
    statements: tuple[MetadataRecord, ...],
) -> dict[tuple[str, str], tuple[float, tuple[MetadataRecord, ...]]]:
    accounted: dict[tuple[str, str], tuple[float, tuple[MetadataRecord, ...]]] = {}
    for statement in statements:
        resource_kind = statement.facts.get("resource_kind")
        unit = statement.facts.get("unit")
        quantity = _numeric_fact(statement, "quantity")
        if not isinstance(resource_kind, str) or not isinstance(unit, str) or quantity is None:
            continue
        key = (resource_kind, unit)
        current_quantity, current_statements = accounted.get(key, (0.0, ()))
        accounted[key] = (current_quantity + quantity, (*current_statements, statement))
    return accounted


def _accounting_basis(record: MetadataRecord) -> str:
    basis = record.facts.get("basis")
    source = record.facts.get("source")
    if source == "startup_training_state_estimator" or isinstance(basis, str) and ("estimate" in basis or "heuristic" in basis):
        return "estimated_structural"
    return "structural"


def _validity_scope(record: MetadataRecord) -> str:
    validity_scope = record.facts.get("validity_scope")
    if isinstance(validity_scope, str) and validity_scope:
        return validity_scope
    return "run_startup_structure"


def _accounting_metadata(record: MetadataRecord) -> dict[str, MetadataValue]:
    metadata: dict[str, MetadataValue] = {
        "accounting_scope": "known_structural_state",
        "derivation_owner": "library.logging.resource_monitor",
        "structural_basis": str(record.facts["basis"]),
        "structural_source": str(record.facts["source"]),
    }
    _copy_optional_facts(
        record,
        metadata,
        (
            "component_key",
            "group_identifier",
            "validity_scope",
        ),
    )
    source_metadata = record.facts.get("metadata")
    if isinstance(source_metadata, dict):
        metadata["source_metadata"] = dict(source_metadata)
    return metadata


def _copy_optional_facts(
    record: MetadataRecord,
    metadata: dict[str, MetadataValue],
    keys: Iterable[str],
) -> None:
    for key in keys:
        value = record.facts.get(key)
        if isinstance(value, str) and value:
            metadata[key] = value


def _has_string_fact(record: MetadataRecord, key: str) -> bool:
    value = record.facts.get(key)
    return isinstance(value, str) and bool(value)


def _numeric_fact(record: MetadataRecord, key: str) -> float | None:
    value = record.facts.get(key)
    return _numeric_value(value)


def _numeric_mapping_value(values: Mapping[str, object], key: str) -> float | None:
    return _numeric_value(values.get(key))


def _numeric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _source_reference(record: MetadataRecord) -> ResourceFactReference:
    return ResourceFactReference(
        entity_type=record.identity.entity_type,
        identifier=record.identity.identifier,
        relationship=MetadataRelationship.DERIVED_FROM,
        namespace=record.identity.namespace,
    )


__all__ = [
    "ACCOUNTING_GAP_DERIVATION_VERSION",
    "build_operation_window_resource_accounting",
    "build_resource_accounting_gaps",
    "build_structural_resource_accounting",
    "STRUCTURAL_RESOURCE_ACCOUNTING_DERIVATION_VERSION",
]
