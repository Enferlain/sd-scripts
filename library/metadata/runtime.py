"""Runtime-facing metadata filing API."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal

from library.metadata.providers import MetadataProviderResult
from library.metadata.records import MetadataEvent, MetadataIdentity, MetadataValue
from library.metadata.versions import METADATA_PAYLOAD_VERSION

from library.metadata.backends import (
    InMemoryMetadataBackend,
    MetadataBackend,
    MetadataSnapshot,
)

from library.metadata.emitters.observability import (
    build_analytics_snapshot_metadata,
    build_logged_artifact_metadata,
    build_resource_monitor_metadata,
    build_run_lifecycle_metadata,
    build_run_report_metadata,
)

from library.metadata.emitters.resource import (
    build_resource_accounting_gap_metadata,
    build_resource_accounting_metadata,
    build_resource_collector_status_metadata,
    build_resource_observation_frame_metadata,
    build_resource_observation_metadata,
    build_resource_profile_metadata,
    build_structural_resource_metadata,
)
from library.metadata.emitters.model import (
    build_model_artifact_metadata,
    build_model_family_contribution_metadata,
    build_model_realization_metadata,
    build_realized_model_component_metadata,
)

from library.metadata.registry import (
    metadata_item_route,
    METADATA_ITEM_ROUTES,
    MetadataRuntimeItem,
    supported_metadata_item_names,
)

from library.metadata.validation import (
    MetadataItemValidationError,
    validate_metadata_item,
    validate_metadata_items,
)


_METADATA_EMITTERS_BY_ROUTE: dict[str, Callable[[Any], MetadataProviderResult]] = {
    "observability.logged_artifact": build_logged_artifact_metadata,
    "observability.run_lifecycle": build_run_lifecycle_metadata,
    "observability.resource_monitor_compatibility": build_resource_monitor_metadata,
    "observability.run_report": build_run_report_metadata,
    "observability.analytics_snapshot": build_analytics_snapshot_metadata,
    "resource.observation_frame": build_resource_observation_frame_metadata,
    "resource.observation": build_resource_observation_metadata,
    "resource.structural": build_structural_resource_metadata,
    "resource.profile": build_resource_profile_metadata,
    "resource.accounting": build_resource_accounting_metadata,
    "resource.accounting_gap": build_resource_accounting_gap_metadata,
    "resource.collector_status": build_resource_collector_status_metadata,
    "model.realization": build_model_realization_metadata,
    "model.realized_component": build_realized_model_component_metadata,
    "model.artifact": build_model_artifact_metadata,
    "model.family_contribution": build_model_family_contribution_metadata,
}

_UNRESOLVED_METADATA_ROUTES = set(METADATA_ITEM_ROUTES.values()) - set(_METADATA_EMITTERS_BY_ROUTE)
if _UNRESOLVED_METADATA_ROUTES:
    unresolved = ", ".join(sorted(_UNRESOLVED_METADATA_ROUTES))
    raise RuntimeError(f"Metadata item registry contains unresolved emitter routes: {unresolved}.")


def build_metadata_result(item: MetadataRuntimeItem) -> MetadataProviderResult:
    """Build one provider result from one accepted typed metadata item."""
    validate_metadata_item(item)
    route = metadata_item_route(item)
    emitter = None if route is None else _METADATA_EMITTERS_BY_ROUTE.get(route)
    if emitter is not None:
        return emitter(item)
    if route is not None:
        raise MetadataItemValidationError(f"Metadata item type {type(item).__name__} uses unresolved emitter route {route!r}.")
    raise MetadataItemValidationError(
        f"Unsupported metadata item type {type(item).__name__}. Supported types: {supported_metadata_item_names()}."
    )


@dataclass(frozen=True, slots=True)
class MetadataBufferPolicy:
    """Bounded retention policy for high-frequency metadata items."""

    capacity: int = 1024
    drop_policy: Literal["drop_oldest", "drop_newest"] = "drop_oldest"
    flush_max_items: int = 256
    buffer_identifier: str = "resource_telemetry"

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("Metadata buffer capacity must be greater than zero.")
        if self.drop_policy not in {"drop_oldest", "drop_newest"}:
            raise ValueError(f"Unsupported metadata buffer drop policy: {self.drop_policy}.")
        if self.flush_max_items <= 0:
            raise ValueError("Metadata buffer flush_max_items must be greater than zero.")


@dataclass(frozen=True, slots=True)
class MetadataBufferReport:
    """Observable state returned by telemetry buffer operations."""

    accepted_items: int
    flushed_items: int
    dropped_items: int
    capacity_dropped_items: int
    ingestion_failed_items: int
    pending_items: int
    degraded: bool
    error_message: str | None = None


@dataclass(slots=True)
class MetadataRuntime:
    """Runtime-facing API for direct and bounded buffered metadata filing."""

    backend: MetadataBackend = field(default_factory=InMemoryMetadataBackend)
    telemetry_buffer_policy: MetadataBufferPolicy = field(default_factory=MetadataBufferPolicy)
    _telemetry_buffer: deque[MetadataRuntimeItem] = field(default_factory=deque, init=False, repr=False)
    _telemetry_buffer_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _telemetry_flush_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _buffer_accepted_items: int = field(default=0, init=False, repr=False)
    _buffer_flushed_items: int = field(default=0, init=False, repr=False)
    _buffer_capacity_dropped_items: int = field(default=0, init=False, repr=False)
    _buffer_ingestion_failed_items: int = field(default=0, init=False, repr=False)
    _buffer_capacity_dropped_since_status: int = field(default=0, init=False, repr=False)
    _buffer_ingestion_failed_since_status: int = field(default=0, init=False, repr=False)
    _buffer_error_message: str | None = field(default=None, init=False, repr=False)

    def file(self, item: MetadataRuntimeItem) -> MetadataProviderResult:
        """File one accepted typed metadata item into the metadata backend."""
        result = build_metadata_result(item)
        self.backend.ingest(result)
        return result

    def file_many(self, items: Sequence[MetadataRuntimeItem]) -> tuple[MetadataProviderResult, ...]:
        """File accepted typed metadata items in one backend-owned batch."""
        item_tuple = tuple(items)
        validate_metadata_items(item_tuple)
        # ``build_metadata_result`` intentionally repeats item-level validation
        # so that its standalone public contract remains identical to ``file``.
        # Runtime batches are low-volume, making the small redundancy preferable
        # to an unvalidated private construction path.
        results = tuple(build_metadata_result(item) for item in item_tuple)
        self.backend.ingest_many(results)
        return results

    def buffer(self, item: MetadataRuntimeItem) -> bool:
        """Queue one telemetry item without performing backend or storage work."""
        validate_metadata_item(item)
        with self._telemetry_buffer_lock:
            if len(self._telemetry_buffer) >= self.telemetry_buffer_policy.capacity:
                self._buffer_capacity_dropped_items += 1
                self._buffer_capacity_dropped_since_status += 1
                if self.telemetry_buffer_policy.drop_policy == "drop_newest":
                    return False
                self._telemetry_buffer.popleft()
            self._telemetry_buffer.append(item)
            self._buffer_accepted_items += 1
            return True

    def flush_buffer(self, *, max_items: int | None = None) -> MetadataBufferReport:
        """Flush a bounded telemetry batch and preserve degradation counters."""
        batch_limit = self.telemetry_buffer_policy.flush_max_items if max_items is None else max_items
        if batch_limit <= 0:
            raise ValueError("Metadata buffer max_items must be greater than zero.")

        with self._telemetry_flush_lock:
            return self._flush_buffer_batch(batch_limit)

    def _flush_buffer_batch(self, batch_limit: int) -> MetadataBufferReport:
        """Flush one batch while serialized against other telemetry flushes."""
        with self._telemetry_buffer_lock:
            items = tuple(self._telemetry_buffer.popleft() for _ in range(min(batch_limit, len(self._telemetry_buffer))))
            capacity_dropped_since_status = self._buffer_capacity_dropped_since_status
            ingestion_failed_since_status = self._buffer_ingestion_failed_since_status
            prior_error_message = self._buffer_error_message

        try:
            results = tuple(build_metadata_result(item) for item in items)
            if capacity_dropped_since_status or ingestion_failed_since_status:
                results = (
                    _build_ingestion_degradation_result(
                        policy=self.telemetry_buffer_policy,
                        capacity_dropped_items=capacity_dropped_since_status,
                        ingestion_failed_items=ingestion_failed_since_status,
                        error_message=prior_error_message,
                    ),
                    *results,
                )
            if results:
                self.backend.ingest_many(results)
        except Exception as exc:  # pragma: no cover - backend-specific failure
            with self._telemetry_buffer_lock:
                self._buffer_ingestion_failed_items += len(items)
                self._buffer_ingestion_failed_since_status += len(items)
                self._buffer_error_message = str(exc)
            return self.buffer_report()

        with self._telemetry_buffer_lock:
            self._buffer_flushed_items += len(items)
            if capacity_dropped_since_status:
                self._buffer_capacity_dropped_since_status = max(
                    self._buffer_capacity_dropped_since_status - capacity_dropped_since_status,
                    0,
                )
            if ingestion_failed_since_status:
                self._buffer_ingestion_failed_since_status = max(
                    self._buffer_ingestion_failed_since_status - ingestion_failed_since_status,
                    0,
                )
            self._buffer_error_message = None
        return self.buffer_report()

    def buffer_report(self) -> MetadataBufferReport:
        """Return current bounded-ingestion counters without flushing."""
        with self._telemetry_buffer_lock:
            dropped_items = self._buffer_capacity_dropped_items + self._buffer_ingestion_failed_items
            return MetadataBufferReport(
                accepted_items=self._buffer_accepted_items,
                flushed_items=self._buffer_flushed_items,
                dropped_items=dropped_items,
                capacity_dropped_items=self._buffer_capacity_dropped_items,
                ingestion_failed_items=self._buffer_ingestion_failed_items,
                pending_items=len(self._telemetry_buffer),
                degraded=dropped_items > 0 or self._buffer_error_message is not None,
                error_message=self._buffer_error_message,
            )

    def snapshot(self) -> MetadataSnapshot:
        """Return the current collected metadata snapshot."""
        return self.backend.snapshot()

    def validate(self) -> None:
        """Validate the currently collected backend state."""
        self.backend.validate()


def _build_ingestion_degradation_result(
    *,
    policy: MetadataBufferPolicy,
    capacity_dropped_items: int,
    ingestion_failed_items: int,
    error_message: str | None,
) -> MetadataProviderResult:
    """Build metadata-owned operational status for degraded telemetry ingestion."""
    dropped_items = capacity_dropped_items + ingestion_failed_items
    facts: dict[str, MetadataValue] = {
        "buffer_identifier": policy.buffer_identifier,
        "capacity": policy.capacity,
        "drop_policy": policy.drop_policy,
        "dropped_items": dropped_items,
        "capacity_dropped_items": capacity_dropped_items,
        "ingestion_failed_items": ingestion_failed_items,
    }
    if error_message is not None:
        facts["error_message"] = error_message
    return MetadataProviderResult.from_sequences(
        provider_id="metadata.runtime.telemetry_buffer",
        schema_version=METADATA_PAYLOAD_VERSION,
        events=(
            MetadataEvent(
                event_type="metadata_ingestion_degraded",
                identity=MetadataIdentity(
                    entity_type="metadata_ingestion_buffer",
                    identifier=policy.buffer_identifier,
                ),
                producer="metadata.runtime.telemetry_buffer",
                facts=facts,
            ),
        ),
    )
