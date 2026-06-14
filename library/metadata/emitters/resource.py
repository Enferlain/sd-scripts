"""Central emitters for resource-intelligence metadata facts."""

from __future__ import annotations

from library.metadata.dataclasses.resource import (
    ResourceAccountingFacts,
    ResourceAccountingGapFacts,
    ResourceFactReference,
    ResourceObservationFrameFacts,
    ResourceObservationFacts,
    ResourceObservationMeasurementFacts,
    ResourceProfileFacts,
    StructuralResourceFacts,
)
from library.metadata.graph import MetadataEntityType, MetadataRelationship, metadata_edge, metadata_identity
from library.metadata.providers import MetadataProviderResult
from library.metadata.records import MetadataEdge, MetadataIdentity, MetadataRecord, MetadataValue
from library.metadata.versions import METADATA_PAYLOAD_VERSION


def build_resource_observation_metadata(
    facts: ResourceObservationFacts,
    *,
    provider_id: str = "resource_intelligence.observation",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build metadata for one measured resource observation."""
    identity = _resource_identity(
        entity_type="resource_observation",
        identifier=facts.observation_identifier,
        label=f"{facts.resource_kind}:{facts.measurement_kind}",
    )
    record = MetadataRecord(
        identity=identity,
        producer=provider_id,
        facts={
            "semantic_class": "observation",
            "run_identifier": facts.run_identifier,
            "resource_kind": facts.resource_kind,
            "measurement_kind": facts.measurement_kind,
            "value": facts.value,
            "unit": facts.unit,
            "source": facts.source,
            **_optional_metadata(
                ts=facts.ts,
                collector_id=facts.collector_id,
                scope_type=facts.scope_type,
                phase=facts.phase,
                global_step=facts.global_step,
                epoch=facts.epoch,
                rank=facts.rank,
                world_size=facts.world_size,
                host_identifier=facts.host_identifier,
                process_identifier=facts.process_identifier,
                device_identifier=facts.device_identifier,
                quality=facts.quality,
                metadata=dict(facts.metadata) if facts.metadata else None,
            ),
        },
        schema_version=schema_version,
    )
    edges = (
        _run_edge(identity, facts.run_identifier, producer=provider_id, relationship=MetadataRelationship.OBSERVED_DURING),
        *_observation_scope_edges(identity, facts, producer=provider_id),
    )
    return _result_from_record(
        provider_id=provider_id,
        schema_version=schema_version,
        record=record,
        edges=edges,
    )


def build_resource_observation_frame_metadata(
    facts: ResourceObservationFrameFacts,
    *,
    provider_id: str = "resource_intelligence.observation_frame",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build one shared-context frame plus individually addressable observations."""
    frame_identity = _resource_identity(
        entity_type="resource_observation_frame",
        identifier=facts.frame_identifier,
        label=facts.event_name,
    )
    frame_record = MetadataRecord(
        identity=frame_identity,
        producer=provider_id,
        facts={
            "semantic_class": "observation",
            "container_kind": "observation_frame",
            "run_identifier": facts.run_identifier,
            "event_name": facts.event_name,
            "measurement_identifiers": [measurement.measurement_identifier for measurement in facts.measurements],
            **_optional_metadata(
                ts=facts.ts,
                collector_id=facts.collector_id,
                collection_policy=facts.collection_policy,
                phase=facts.phase,
                global_step=facts.global_step,
                epoch=facts.epoch,
                rank=facts.rank,
                world_size=facts.world_size,
                host_identifier=facts.host_identifier,
                process_identifier=facts.process_identifier,
                quality=facts.quality,
                metadata=dict(facts.metadata) if facts.metadata else None,
            ),
        },
        schema_version=schema_version,
    )
    measurement_records = tuple(
        _build_frame_measurement_record(measurement, frame_identifier=facts.frame_identifier, producer=provider_id)
        for measurement in facts.measurements
    )
    edges = (
        _run_edge(
            frame_identity,
            facts.run_identifier,
            producer=provider_id,
            relationship=MetadataRelationship.OBSERVED_DURING,
        ),
        *_frame_scope_edges(frame_identity, facts, producer=provider_id),
        *(
            edge
            for measurement in facts.measurements
            for edge in _frame_measurement_edges(measurement, frame_identity=frame_identity, producer=provider_id)
        ),
    )
    return MetadataProviderResult.from_sequences(
        provider_id=provider_id,
        schema_version=schema_version,
        records=(frame_record, *measurement_records),
        edges=edges,
    )


def build_structural_resource_metadata(
    facts: StructuralResourceFacts,
    *,
    provider_id: str = "resource_intelligence.structural",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build metadata for one known structural resource fact."""
    identity = _resource_identity(
        entity_type="resource_structural_fact",
        identifier=facts.structural_identifier,
        label=f"{facts.owner_type}:{facts.owner_identifier}",
    )
    record = MetadataRecord(
        identity=identity,
        producer=provider_id,
        facts={
            "semantic_class": "structural",
            "run_identifier": facts.run_identifier,
            "owner_type": facts.owner_type,
            "owner_identifier": facts.owner_identifier,
            "resource_kind": facts.resource_kind,
            "quantity": facts.quantity,
            "unit": facts.unit,
            "basis": facts.basis,
            "source": facts.source,
            **_optional_metadata(
                component_key=facts.component_key,
                group_identifier=facts.group_identifier,
                validity_scope=facts.validity_scope,
                metadata=dict(facts.metadata) if facts.metadata else None,
            ),
        },
        schema_version=schema_version,
    )
    return _result_from_record(
        provider_id=provider_id,
        schema_version=schema_version,
        record=record,
        edges=(
            _run_edge(
                identity,
                facts.run_identifier,
                producer=provider_id,
                relationship=MetadataRelationship.DESCRIBES,
            ),
            *_structural_scope_edges(identity, facts, producer=provider_id),
        ),
    )


def build_resource_profile_metadata(
    facts: ResourceProfileFacts,
    *,
    provider_id: str = "resource_intelligence.profile",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build metadata for one versioned resource profile."""
    identity = _resource_identity(
        entity_type="resource_profile",
        identifier=facts.profile_identifier,
        label=facts.profile_kind,
    )
    record = MetadataRecord(
        identity=identity,
        producer=provider_id,
        facts={
            "semantic_class": "profile",
            "run_identifier": facts.run_identifier,
            "profile_kind": facts.profile_kind,
            "derivation_version": facts.derivation_version,
            "values": dict(facts.values),
            "source_fact_references": _reference_payloads(facts.source_fact_references),
            **_optional_metadata(
                generated_at=facts.generated_at,
                metadata=dict(facts.metadata) if facts.metadata else None,
            ),
        },
        schema_version=schema_version,
    )
    edges = (
        _run_edge(identity, facts.run_identifier, producer=provider_id, relationship=MetadataRelationship.PROFILES),
        *_reference_edges(identity, facts.source_fact_references, producer=provider_id),
    )
    return _result_from_record(provider_id=provider_id, schema_version=schema_version, record=record, edges=edges)


def build_resource_accounting_metadata(
    facts: ResourceAccountingFacts,
    *,
    provider_id: str = "resource_intelligence.accounting",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build metadata for one evidence-constrained accounting statement."""
    identity = _resource_identity(
        entity_type="resource_accounting",
        identifier=facts.accounting_identifier,
        label=f"{facts.owner_type}:{facts.owner_identifier}",
    )
    record = MetadataRecord(
        identity=identity,
        producer=provider_id,
        facts={
            "semantic_class": "accounting",
            "run_identifier": facts.run_identifier,
            "resource_kind": facts.resource_kind,
            "quantity": facts.quantity,
            "unit": facts.unit,
            "owner_type": facts.owner_type,
            "owner_identifier": facts.owner_identifier,
            "basis": facts.basis,
            "derivation_method": facts.derivation_method,
            "derivation_version": facts.derivation_version,
            "source_fact_references": _reference_payloads(facts.source_fact_references),
            **_optional_metadata(
                validity_scope=facts.validity_scope,
                window_start=facts.window_start,
                window_end=facts.window_end,
                metadata=dict(facts.metadata) if facts.metadata else None,
            ),
        },
        schema_version=schema_version,
    )
    edges = (
        _run_edge(
            identity,
            facts.run_identifier,
            producer=provider_id,
            relationship=MetadataRelationship.ACCOUNTS_FOR,
        ),
        *_reference_edges(identity, facts.source_fact_references, producer=provider_id),
    )
    return _result_from_record(provider_id=provider_id, schema_version=schema_version, record=record, edges=edges)


def build_resource_accounting_gap_metadata(
    facts: ResourceAccountingGapFacts,
    *,
    provider_id: str = "resource_intelligence.accounting_gap",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build metadata for one unresolved resource-accounting gap."""
    identity = _resource_identity(
        entity_type="resource_accounting_gap",
        identifier=facts.gap_identifier,
        label=f"{facts.resource_kind}:gap",
    )
    record = MetadataRecord(
        identity=identity,
        producer=provider_id,
        facts={
            "semantic_class": "accounting_gap",
            "run_identifier": facts.run_identifier,
            "resource_kind": facts.resource_kind,
            "quantity": facts.quantity,
            "unit": facts.unit,
            "basis": facts.basis,
            "derivation_method": facts.derivation_method,
            "derivation_version": facts.derivation_version,
            "source_fact_references": _reference_payloads(facts.source_fact_references),
            **_optional_metadata(
                scope=facts.scope,
                reason=facts.reason,
                metadata=dict(facts.metadata) if facts.metadata else None,
            ),
        },
        schema_version=schema_version,
    )
    edges = (
        _run_edge(
            identity,
            facts.run_identifier,
            producer=provider_id,
            relationship=MetadataRelationship.GAP_FOR,
        ),
        *_reference_edges(identity, facts.source_fact_references, producer=provider_id),
    )
    return _result_from_record(provider_id=provider_id, schema_version=schema_version, record=record, edges=edges)


def _result_from_record(
    *,
    provider_id: str,
    schema_version: str,
    record: MetadataRecord,
    edges: tuple[MetadataEdge, ...],
) -> MetadataProviderResult:
    return MetadataProviderResult.from_sequences(
        provider_id=provider_id,
        schema_version=schema_version,
        records=(record,),
        edges=edges,
    )


def _resource_identity(*, entity_type: str, identifier: str, label: str | None) -> MetadataIdentity:
    return metadata_identity(
        entity_type=entity_type,
        identifier=identifier,
        label=label,
    )


def _run_identity(run_identifier: str) -> MetadataIdentity:
    return metadata_identity(
        entity_type=MetadataEntityType.RUN,
        identifier=run_identifier,
    )


def _run_edge(source: MetadataIdentity, run_identifier: str, *, producer: str, relationship: str) -> MetadataEdge:
    return metadata_edge(
        source=source,
        target=_run_identity(run_identifier),
        relationship=relationship,
        producer=producer,
    )


def _scope_edge(
    source: MetadataIdentity,
    *,
    entity_type: str,
    identifier: str,
    relationship: str,
    producer: str,
    label: str | None = None,
    facts: dict[str, MetadataValue] | None = None,
) -> MetadataEdge:
    return metadata_edge(
        source=source,
        target=metadata_identity(
            entity_type=entity_type,
            identifier=identifier,
            label=label,
        ),
        relationship=relationship,
        producer=producer,
        facts={} if facts is None else facts,
    )


def _observation_scope_edges(
    source: MetadataIdentity,
    facts: ResourceObservationFacts,
    *,
    producer: str,
) -> tuple[MetadataEdge, ...]:
    edges = list(
        _runtime_scope_edges(
            source,
            run_identifier=facts.run_identifier,
            host_identifier=facts.host_identifier,
            process_identifier=facts.process_identifier,
            rank=facts.rank,
            world_size=facts.world_size,
            phase=facts.phase,
            global_step=facts.global_step,
            epoch=facts.epoch,
            collector_id=facts.collector_id,
            producer=producer,
        )
    )
    if facts.device_identifier is not None:
        edges.append(
            _scope_edge(
                source,
                entity_type=MetadataEntityType.DEVICE,
                identifier=facts.device_identifier,
                relationship=MetadataRelationship.OBSERVED_ON,
                producer=producer,
            )
        )
    return tuple(edges)


def _runtime_scope_edges(
    source: MetadataIdentity,
    *,
    run_identifier: str,
    host_identifier: str | None,
    process_identifier: str | None,
    rank: int | None,
    world_size: int | None,
    phase: str | None,
    global_step: int | None,
    epoch: int | None,
    collector_id: str | None,
    producer: str,
) -> tuple[MetadataEdge, ...]:
    edges: list[MetadataEdge] = []
    if host_identifier is not None:
        edges.append(
            _scope_edge(
                source,
                entity_type=MetadataEntityType.HOST,
                identifier=host_identifier,
                relationship=MetadataRelationship.OBSERVED_ON,
                producer=producer,
            )
        )
    if process_identifier is not None:
        edges.append(
            _scope_edge(
                source,
                entity_type=MetadataEntityType.PROCESS,
                identifier=process_identifier,
                relationship=MetadataRelationship.OBSERVED_IN,
                producer=producer,
            )
        )
    if rank is not None:
        edges.append(
            _scope_edge(
                source,
                entity_type=MetadataEntityType.RANK,
                identifier=_rank_identifier(run_identifier=run_identifier, rank=rank),
                relationship=MetadataRelationship.OBSERVED_IN,
                producer=producer,
                label=f"rank {rank}",
                facts=_optional_metadata(rank=rank, world_size=world_size),
            )
        )
    if phase is not None:
        edges.append(
            _scope_edge(
                source,
                entity_type=MetadataEntityType.PHASE,
                identifier=_phase_identifier(run_identifier=run_identifier, phase=phase),
                relationship=MetadataRelationship.OBSERVED_DURING,
                producer=producer,
                label=phase,
                facts={"phase": phase},
            )
        )
    if global_step is not None:
        edges.append(
            _scope_edge(
                source,
                entity_type=MetadataEntityType.STEP,
                identifier=_step_identifier(run_identifier=run_identifier, global_step=global_step),
                relationship=MetadataRelationship.OBSERVED_DURING,
                producer=producer,
                label=f"step {global_step}",
                facts=_optional_metadata(global_step=global_step, epoch=epoch),
            )
        )
    if collector_id is not None:
        edges.append(
            _scope_edge(
                source,
                entity_type=MetadataEntityType.COLLECTOR,
                identifier=collector_id,
                relationship=MetadataRelationship.PRODUCED_BY,
                producer=producer,
                label=collector_id,
            )
        )
    return tuple(edges)


def _build_frame_measurement_record(
    measurement: ResourceObservationMeasurementFacts,
    *,
    frame_identifier: str,
    producer: str,
) -> MetadataRecord:
    return MetadataRecord(
        identity=_resource_identity(
            entity_type="resource_observation",
            identifier=measurement.measurement_identifier,
            label=f"{measurement.resource_kind}:{measurement.measurement_kind}",
        ),
        producer=producer,
        facts={
            "semantic_class": "observation",
            "frame_identifier": frame_identifier,
            "resource_kind": measurement.resource_kind,
            "measurement_kind": measurement.measurement_kind,
            "value": measurement.value,
            "unit": measurement.unit,
            "source": measurement.source,
            **_optional_metadata(
                scope_type=measurement.scope_type,
                device_identifier=measurement.device_identifier,
                quality=measurement.quality,
                metadata=dict(measurement.metadata) if measurement.metadata else None,
            ),
        },
    )


def _frame_scope_edges(
    source: MetadataIdentity,
    facts: ResourceObservationFrameFacts,
    *,
    producer: str,
) -> tuple[MetadataEdge, ...]:
    edges = list(
        _runtime_scope_edges(
            source,
            run_identifier=facts.run_identifier,
            host_identifier=facts.host_identifier,
            process_identifier=facts.process_identifier,
            rank=facts.rank,
            world_size=facts.world_size,
            phase=facts.phase,
            global_step=facts.global_step,
            epoch=facts.epoch,
            collector_id=facts.collector_id,
            producer=producer,
        )
    )
    edges.append(
        _scope_edge(
            source,
            entity_type=MetadataEntityType.EVENT,
            identifier=_event_identifier(run_identifier=facts.run_identifier, event_name=facts.event_name),
            relationship=MetadataRelationship.OBSERVED_DURING,
            producer=producer,
            label=facts.event_name,
            facts={"event_name": facts.event_name},
        )
    )
    return tuple(edges)


def _structural_scope_edges(
    source: MetadataIdentity,
    facts: StructuralResourceFacts,
    *,
    producer: str,
) -> tuple[MetadataEdge, ...]:
    edges: list[MetadataEdge] = [
        _scope_edge(
            source,
            entity_type=facts.owner_type,
            identifier=facts.owner_identifier,
            relationship=MetadataRelationship.OWNED_BY,
            producer=producer,
            label=facts.owner_identifier,
            facts={"owner_type": facts.owner_type},
        )
    ]
    if facts.component_key is not None:
        edges.append(
            _scope_edge(
                source,
                entity_type=MetadataEntityType.COMPONENT,
                identifier=facts.component_key,
                relationship=MetadataRelationship.DESCRIBES,
                producer=producer,
                label=facts.component_key,
                facts={"component_key": facts.component_key},
            )
        )
    if facts.group_identifier is not None:
        edges.append(
            _scope_edge(
                source,
                entity_type=MetadataEntityType.GROUP,
                identifier=facts.group_identifier,
                relationship=MetadataRelationship.DESCRIBES,
                producer=producer,
                label=facts.group_identifier,
                facts={"group_identifier": facts.group_identifier},
            )
        )
    return tuple(edges)


def _frame_measurement_edges(
    measurement: ResourceObservationMeasurementFacts,
    *,
    frame_identity: MetadataIdentity,
    producer: str,
) -> tuple[MetadataEdge, ...]:
    measurement_identity = _resource_identity(
        entity_type="resource_observation",
        identifier=measurement.measurement_identifier,
        label=f"{measurement.resource_kind}:{measurement.measurement_kind}",
    )
    edges = [
        metadata_edge(
            source=measurement_identity,
            target=frame_identity,
            relationship=MetadataRelationship.CONTAINED_IN,
            producer=producer,
        )
    ]
    if measurement.device_identifier is not None:
        edges.append(
            _scope_edge(
                measurement_identity,
                entity_type=MetadataEntityType.DEVICE,
                identifier=measurement.device_identifier,
                relationship=MetadataRelationship.OBSERVED_ON,
                producer=producer,
            )
        )
    return tuple(edges)


def _reference_payloads(references: tuple[ResourceFactReference, ...]) -> list[object]:
    return [
        {
            "entity_type": reference.entity_type,
            "identifier": reference.identifier,
            "relationship": reference.relationship,
            **_optional_metadata(namespace=reference.namespace),
        }
        for reference in references
    ]


def _reference_edges(
    source: MetadataIdentity,
    references: tuple[ResourceFactReference, ...],
    *,
    producer: str,
) -> tuple[MetadataEdge, ...]:
    """Build evidence edges.

    A reference without an explicit namespace inherits the source namespace.
    Cross-namespace references should set ``ResourceFactReference.namespace``.
    """
    return tuple(
        MetadataEdge(
            source=source,
            target=metadata_identity(
                entity_type=reference.entity_type,
                identifier=reference.identifier,
                namespace=source.namespace if reference.namespace is None else reference.namespace,
            ),
            relationship=reference.relationship,
            producer=producer,
        )
        for reference in references
    )


def _optional_metadata(**values: MetadataValue | None) -> dict[str, MetadataValue]:
    return {key: value for key, value in values.items() if value is not None}


def _rank_identifier(*, run_identifier: str, rank: int) -> str:
    return f"{run_identifier}:rank:{rank}"


def _phase_identifier(*, run_identifier: str, phase: str) -> str:
    return f"{run_identifier}:phase:{phase}"


def _step_identifier(*, run_identifier: str, global_step: int) -> str:
    return f"{run_identifier}:step:{global_step}"


def _event_identifier(*, run_identifier: str, event_name: str) -> str:
    return f"{run_identifier}:event:{event_name}"
