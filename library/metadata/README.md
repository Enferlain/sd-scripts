# Metadata README

`library/metadata/` is the repo-owned metadata system.

This file is the active guide for how metadata is supposed to work in the codebase.

## Short Version

- Runtime code owns live state and source truth.
- `library/metadata` owns accepted metadata item schemas, validation, routing, backend accumulation, and export/projection.
- Normal runtime code should file typed metadata items. It should not build records, events, projections, or export dictionaries directly.

## Public Runtime Model

The public model is intentionally small:

1. `library/metadata/dataclasses/<concern>.py` defines accepted typed metadata items.
2. Local code files those items through `MetadataRuntime.file(item)`, or uses
   the bounded telemetry buffer for explicitly high-frequency facts.
3. `library/metadata` validates the item, routes it to the right emitter, and ingests the result into a backend.
4. Backends expose snapshots that projections/storage/export can consume later.

In code, the intended runtime-facing shape is:

```python
from library.metadata import LoggedArtifactFacts, MetadataRuntime, RunLifecycleFacts

metadata = MetadataRuntime()

metadata.file(
    RunLifecycleFacts(
        run_identifier=run_name,
        event_type="run_started",
        run_name=run_name,
        status="running",
    )
)

metadata.file(
    LoggedArtifactFacts(
        path=path,
        kind=kind,
        metadata={"format": "json"},
    )
)
```

That is the system local code should think in.

### High-Frequency Telemetry

`MetadataRuntime.file(item)` remains the direct path for low-volume facts.
High-frequency producers may instead use:

- `buffer(item)` to validate and enqueue without performing backend/storage I/O
- `flush_buffer(max_items=...)` to flush bounded batches at an appropriate
  orchestration boundary
- `buffer_report()` to inspect accepted, flushed, pending, capacity-dropped,
  and ingestion-failed counts

The buffer has an explicit capacity and `drop_oldest` or `drop_newest`
retention policy. Drops and prior ingestion failures are emitted as
`metadata_ingestion_degraded` events on the next successful flush. A failed
flush drops that bounded batch and records degraded state rather than raising
through the telemetry path.

Telemetry flushes are serialized without holding the producer queue lock.
`flush_buffer(...)` remains a synchronous storage operation, so orchestration
must not call it from a latency-critical training section.

`file_many(items)` is the synchronous batch path. It lets the backend/store use
one batch transaction, but callers should not use it as a supposedly
non-blocking hot-path API.

## Core Split

Treat config and metadata as parallel systems with different jobs:

- Config = desired intent
- Metadata = observed truth, provenance, emitted artifacts, compatibility/export facts

The metadata layer should know how metadata is recorded and exported.
Normal domain code should know when facts exist and when they should be filed.

## Ownership Rules

### Central in `library/metadata/`

Keep these here:

- Shared typed metadata item dataclasses
- Validation
- Routing from typed items to metadata records/events
- Backends and storage
- Projections/export shaping
- Shared metadata key helpers

### Local to the origin domain

Keep these with the runtime that owns them:

- Live runtime objects
- State that changes during the run
- Lifecycle boundaries where facts become available
- Domain-specific facts that are not truly shared

### Explicit local exceptions

Local metadata modules are allowed for explicit plugin-like or family-local metadata.

Current likely examples:

- adapter method / family metadata
- possibly data/cache-specific metadata if it remains highly local

The default should still be central shared item schemas plus local filing call sites.

## Dataclass Rule

`library/metadata/dataclasses/<concern>.py` is the canonical home for shared metadata item types.

These dataclasses should stay close to schema-only:

- field definitions
- tiny constructors like `from_*` or `for_*` when they genuinely help
- light field-local normalization/validation when it belongs to the item itself

Avoid putting record/event export logic on the dataclasses.

The important distinction is:

- dataclasses define the accepted metadata items
- they are not the same thing as long-lived local runtime state

## Concerns

A concern is just the metadata topic.

Examples:

- `observability`
- `optimization`
- `checkpoint`

The concern groups related item types under one metadata topic, but it is not the main runtime abstraction.

The main runtime abstraction is still:

- file a typed metadata item

So local code should not need to think in terms of concern-specific provider or collector objects unless a future case truly needs one as an internal implementation detail.

## Validation

`library/metadata/validation.py` is the central home for validation.

Today that includes two different validation layers:

- item validation before or while filed metadata enters the system
- backend/snapshot validation for required facts before projection/export boundaries

The point is that validation should stay central even when the metadata flow grows.

## Versioning

The metadata system currently uses two different versioning layers:

- storage schema version
- metadata payload version

The storage schema version is the database/store layout version in
`library/metadata/storage.py`.

The metadata payload version is the version stamped onto metadata
identities/records/events by the metadata package itself.

Normal runtime code should not carry or manage either of these directly.
Filed metadata item dataclasses should stay focused on the facts themselves.

## Internal Flow

The internal flow should look like this:

```text
[origin runtime state]
        ->
[typed metadata item]
        ->
[MetadataRuntime.file(item) / buffer(item)]
        ->
[validation]
        ->
[central emitter routing]
        ->
[MetadataProviderResult]
        ->
[backend snapshot]
        ->
[projection / storage / export]
```

The important boundary is:

- runtime code knows the live facts
- metadata code knows the metadata system

## Emitters

Emitters are internal metadata-system assembly helpers.

Their job is to turn accepted typed metadata items into:

- `MetadataRecord`
- `MetadataEvent`
- `MetadataProviderResult`

They are not the main public runtime API.

That means:

- local code should not manually construct records/events
- local code should not need to know which emitter to call
- routing from item type to emitter should stay inside `library/metadata`

## Backends

Backends are the accumulation boundary.

They ingest `MetadataProviderResult` objects and expose collected state through a `MetadataSnapshot`.

That makes them the bridge between:

- runtime filing
- later projection/storage/export work

`MetadataRuntime` uses a backend internally so local code can stay at the level of `file(item)`.

## Relationships

Metadata edges describe queryable relationships between accepted facts and
their scopes or evidence. Shared graph vocabulary and construction helpers live
in `library/metadata/graph.py` so observation, structural, profile, accounting,
artifact, checkpoint, and other metadata concerns can use the same relationship
language instead of growing concern-specific edge strings.

Emitters should create relationship edges only from identities provided by the
filed fact. If a runtime identity is unavailable, omit that edge rather than
inventing a placeholder.

`MetadataGraphIndex` builds reusable record, incoming-edge, and outgoing-edge
lookups for one public metadata snapshot. Domain read views such as
`ResourceRunView` use that index so repeated scope and evidence queries do not
rescan the complete telemetry graph.

Source evidence remains broader than a run view's own fact membership. A
profile or accounting record may explicitly reference accepted records from
another run or another metadata concern, including artifacts; resolving that
evidence does not make it a resource fact of the viewed run.

## Projections

Typed metadata items and collected records/events are the source of truth.

Exported key dictionaries are projections from that source of truth.

Current projection families include:

- legacy `ss_*`
- `modelspec.*`
- repo-owned `kuro.*`
- resource-monitor flat compatibility events used by JSONL and transitional report input

Resource compatibility events are projected from accepted observation-frame
facts or a `ResourceRunView`; JSONL is an export/fallback consumer of that
projection, not the report source of truth.

Those shapes belong in projection code, not in runtime call sites.

## What To Avoid

Avoid these patterns in normal runtime code:

- building metadata records/events directly
- building projection/export dictionaries directly
- passing large runtime objects deep into metadata code when a small typed item would do
- inventing per-file metadata mini-frameworks

## Practical Notes

- The canonical filing method is `MetadataRuntime.file(item)`. Extra local helper functions are optional convenience seams, not new filing APIs.
- Prefer direct `metadata_runtime.file(TypedFacts(...))` calls when the local conversion is small and obvious.
- If a local helper is useful, it should primarily clarify local-to-typed-facts conversion. Favor names like `build_*_facts(...)` over introducing concern-specific `file_*` APIs unless the helper is truly about one local boundary.
- Observability now has multiple real producers on the same path:
  - logging observer lifecycle/artifact events
  - resource monitor events
  - run report records
  - analytics/debug snapshot records
- That shared path is intentional. Future observability slices should prefer filing more typed items into the existing runtime/backend snapshot instead of inventing parallel observability collection flows.
- The first live analytics snapshot is the benchmark-report payload summary filed from `write_run_report()`. It intentionally keeps backend-neutral run/resource/config summary data while leaving the bulky `full_config_yaml` in the report artifact itself instead of duplicating it into metadata storage.
- Lifecycle events should prefer richer structured context when the trainer already knows it. The current live path now carries run identifier, mode, strategy, optimizer, config name, step/epoch where meaningful, duration, and failed-run error messages rather than treating lifecycle metadata as a minimal placeholder.
- Another live analytics snapshot producer now exists at training startup: the observer files the structured startup summary through the same runtime path so debug/export work can use one metadata-backed source instead of scraping console-only output.
- Benchmark-report analytics snapshots now also carry the lightweight runtime trace summary. That trace is owned by the trainer/logging path rather than the metadata system itself, but the existing benchmark-report payload snapshot is the current metadata-backed place where launch-to-phase timings and milestone deltas become comparable across runs.

## What This README Is For

This file is for the settled metadata system shape that should stay true even while implementation details continue to move.

If a future change no longer matches this README, update the README together with the code so the system remains navigable.
