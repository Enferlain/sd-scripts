"""Central emitters for observability and report metadata assembly."""

from __future__ import annotations

from library.metadata.dataclasses.observability import (
    AnalyticsSnapshotFacts,
    LoggedArtifactFacts,
    ResourceMonitorFacts,
    RunLifecycleFacts,
    RunReportFacts,
)
from library.metadata.providers import MetadataProviderResult
from library.metadata.records import ArtifactMetadataRecord, MetadataEvent, MetadataIdentity, MetadataValue
from library.metadata.versions import METADATA_PAYLOAD_VERSION


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def build_logged_artifact_metadata(
    facts: LoggedArtifactFacts,
    *,
    provider_id: str = "observability.artifact_registration",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build collected metadata for one logging artifact-registration boundary."""
    return _result_from_event(
        provider_id=provider_id,
        schema_version=schema_version,
        event=_build_logged_artifact_event(facts, producer=provider_id),
    )


def build_run_lifecycle_metadata(
    facts: RunLifecycleFacts,
    *,
    provider_id: str = "observability.run_lifecycle",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build collected metadata for a run lifecycle event."""
    return _result_from_event(
        provider_id=provider_id,
        schema_version=schema_version,
        event=_build_run_lifecycle_event(facts, producer=provider_id),
    )


def build_resource_monitor_metadata(
    facts: ResourceMonitorFacts,
    *,
    provider_id: str = "observability.resource_monitor",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build collected metadata for a resource monitor event."""
    return _result_from_event(
        provider_id=provider_id,
        schema_version=schema_version,
        event=_build_resource_monitor_event(facts, producer=provider_id),
    )


def build_run_report_metadata(
    facts: RunReportFacts,
    *,
    provider_id: str = "observability.run_report",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build collected metadata for a benchmark/report boundary."""
    return _result_from_record(
        provider_id=provider_id,
        schema_version=schema_version,
        record=_build_run_report_record(facts, producer=provider_id),
    )


def build_analytics_snapshot_metadata(
    facts: AnalyticsSnapshotFacts,
    *,
    provider_id: str = "observability.analytics_snapshot",
    schema_version: str = METADATA_PAYLOAD_VERSION,
) -> MetadataProviderResult:
    """Build collected metadata for a backend-neutral analytics snapshot."""
    return _result_from_record(
        provider_id=provider_id,
        schema_version=schema_version,
        record=_build_analytics_snapshot_record(facts, producer=provider_id),
    )


# ---------------------------------------------------------------------------
# Result wrappers
# ---------------------------------------------------------------------------


def _result_from_event(*, provider_id: str, schema_version: str, event: MetadataEvent) -> MetadataProviderResult:
    return MetadataProviderResult.from_sequences(
        provider_id=provider_id,
        schema_version=schema_version,
        events=(event,),
    )


def _result_from_record(
    *,
    provider_id: str,
    schema_version: str,
    record: ArtifactMetadataRecord,
) -> MetadataProviderResult:
    return MetadataProviderResult.from_sequences(
        provider_id=provider_id,
        schema_version=schema_version,
        records=(record,),
    )


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------


def _build_logged_artifact_event(facts: LoggedArtifactFacts, *, producer: str) -> MetadataEvent:
    metadata: dict[str, MetadataValue] = {
        "path": facts.path,
        "kind": facts.kind,
    }
    if facts.metadata:
        metadata["metadata"] = dict(facts.metadata)
    return MetadataEvent(
        event_type="artifact_registered",
        identity=_artifact_identity(identifier=facts.path, label=facts.kind),
        producer=producer,
        facts=metadata,
        schema_version=METADATA_PAYLOAD_VERSION,
    )


def _build_run_lifecycle_event(facts: RunLifecycleFacts, *, producer: str) -> MetadataEvent:
    metadata = _optional_metadata(
        run_name=facts.run_name,
        status=facts.status,
        mode_name=facts.mode_name,
        strategy_name=facts.strategy_name,
        optimizer_name=facts.optimizer_name,
        config_name=facts.config_name,
        global_step=facts.global_step,
        epoch=facts.epoch,
        duration_ms=facts.duration_ms,
        error_message=facts.error_message,
    )
    return MetadataEvent(
        event_type=facts.event_type,
        identity=_run_identity(identifier=facts.run_identifier, label=facts.run_name),
        producer=producer,
        facts=metadata,
        schema_version=METADATA_PAYLOAD_VERSION,
    )


def _build_resource_monitor_event(facts: ResourceMonitorFacts, *, producer: str) -> MetadataEvent:
    metadata = _optional_metadata(
        ts=facts.ts,
        rank=facts.rank,
        world_size=facts.world_size,
        mode=facts.mode,
        device_scope=facts.device_scope,
        config_name=facts.config_name,
        git_sha=facts.git_sha,
        git_dirty=facts.git_dirty,
        global_step=facts.global_step,
        epoch=facts.epoch,
        phase=facts.phase,
        duration_ms=facts.duration_ms,
        gpu_allocated_mb=facts.gpu_allocated_mb,
        gpu_reserved_mb=facts.gpu_reserved_mb,
        gpu_peak_allocated_mb=facts.gpu_peak_allocated_mb,
        gpu_used_mb=facts.gpu_used_mb,
        cpu_rss_mb=facts.cpu_rss_mb,
        steps_per_sec=facts.steps_per_sec,
        samples_per_sec=facts.samples_per_sec,
        dropped_samples=facts.dropped_samples,
        collection_ms=facts.collection_ms,
        deep_alloc_retries=facts.deep_alloc_retries,
        deep_ooms=facts.deep_ooms,
        deep_active_mb=facts.deep_active_mb,
        deep_reserved_mb=facts.deep_reserved_mb,
        deep_inactive_split_mb=facts.deep_inactive_split_mb,
        deep_window_active=facts.deep_window_active,
    )
    return MetadataEvent(
        event_type=facts.event_name,
        identity=_run_identity(identifier=facts.run_identifier, label=facts.phase or facts.event_name),
        producer=producer,
        facts=metadata,
        schema_version=METADATA_PAYLOAD_VERSION,
    )


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------


def _build_run_report_record(facts: RunReportFacts, *, producer: str) -> ArtifactMetadataRecord:
    metadata: dict[str, MetadataValue] = {
        "kind": "benchmark_report",
        "status": facts.status,
        "generated_at": facts.generated_at,
        "run_identifier": facts.run_identifier,
    }
    if facts.output_name is not None:
        metadata["output_name"] = facts.output_name
    if facts.output_dir is not None:
        metadata["output_dir"] = facts.output_dir
    if facts.mode_name is not None:
        metadata["mode_name"] = facts.mode_name
    if facts.strategy_name is not None:
        metadata["strategy_name"] = facts.strategy_name
    if facts.optimizer_name is not None:
        metadata["optimizer_name"] = facts.optimizer_name
    if facts.global_step is not None:
        metadata["global_step"] = facts.global_step
    if facts.num_train_epochs is not None:
        metadata["num_train_epochs"] = facts.num_train_epochs
    if facts.resource_event_count is not None:
        metadata["resource_event_count"] = facts.resource_event_count
    if facts.phase_count is not None:
        metadata["phase_count"] = facts.phase_count
    if facts.include_full_config is not None:
        metadata["include_full_config"] = facts.include_full_config
    if facts.resource_jsonl_path is not None:
        metadata["resource_jsonl_path"] = facts.resource_jsonl_path
    return ArtifactMetadataRecord(
        identity=_artifact_identity(identifier=facts.report_identifier, label=facts.output_name),
        producer=producer,
        facts=metadata,
        schema_version=METADATA_PAYLOAD_VERSION,
    )


def _build_analytics_snapshot_record(
    facts: AnalyticsSnapshotFacts,
    *,
    producer: str,
) -> ArtifactMetadataRecord:
    metadata: dict[str, MetadataValue] = {
        "kind": facts.snapshot_kind,
        "source": facts.source,
        "payload": dict(facts.payload),
    }
    if facts.generated_at is not None:
        metadata["generated_at"] = facts.generated_at
    if facts.run_identifier is not None:
        metadata["run_identifier"] = facts.run_identifier
    return ArtifactMetadataRecord(
        identity=_artifact_identity(identifier=facts.snapshot_identifier, label=facts.snapshot_kind),
        producer=producer,
        facts=metadata,
        schema_version=METADATA_PAYLOAD_VERSION,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_identity(*, identifier: str, label: str | None) -> MetadataIdentity:
    return MetadataIdentity(
        entity_type="run",
        identifier=identifier,
        label=label,
        schema_version=METADATA_PAYLOAD_VERSION,
    )


def _artifact_identity(*, identifier: str, label: str | None) -> MetadataIdentity:
    return MetadataIdentity(
        entity_type="artifact",
        identifier=identifier,
        label=label,
        schema_version=METADATA_PAYLOAD_VERSION,
    )


def _optional_metadata(**values: MetadataValue | None) -> dict[str, MetadataValue]:
    return {key: value for key, value in values.items() if value is not None}
