## Context

The active resource path has a useful runtime spine:

```text
Trainer / phases
    -> ResourceMonitor lifecycle hooks
    -> snapshots and samples
    -> resource event dictionary
       -> JSONL writer
       -> ResourceMonitorFacts -> MetadataRuntime.file(...)
    -> reports.py reparses JSONL and derives summaries
```

This works for the current narrow schema, but it has four architectural
pressures:

1. the resource event dictionary and `ResourceMonitorFacts` mirror each other
   manually
2. JSONL is treated as the report source of truth while metadata receives a
   second representation of the same facts
3. collection, orchestration, console output, event construction, persistence,
   and report derivation are tightly coupled
4. the current shapes describe observations but do not provide a coherent model
   for structural facts, reusable profiles, or accounting

The metadata backbone is already the repo-owned system for accepted typed
metadata items, validation, central emitter routing, backend accumulation,
durable storage, identities, relationships, and projections. Resource
intelligence must extend that architecture rather than create a competing
warehouse or parallel canonical event schema.

## Goals / Non-Goals

**Goals:**

- Design the complete target resource-intelligence system rather than a
  throwaway monitoring-only intermediate.
- Preserve the current small trainer-facing resource-monitor lifecycle API.
- Represent observations, structural facts, profiles, and accounting as
  distinct but related resource concepts.
- Make metadata the durable fact-system boundary without moving resource
  collection or interpretation into metadata.
- Provide one accepted typed fact flow from which JSONL, reports, profile
  artifacts, dashboard inputs, and future analytics can be projected.
- Support low-cost normal operation and explicit higher-cost diagnostic
  collection.
- Preserve enough identity and relationships to compare runs and explain
  resource behavior across ranks, processes, devices, phases, components, and
  artifacts.

**Non-Goals:**

- Do not promise precise per-allocation ownership from aggregate counters.
- Do not treat phase-local deltas as proof of ownership.
- Do not make metadata emitters infer resource meaning.
- Do not replace the existing logging/metadata systems with an external
  observability platform.
- Do not add every available psutil, NVML, or CUDA counter without a defined
  question, cost, and retention policy.

## Decisions

### Decision: Build one resource-intelligence domain with four semantic fact classes

The target system distinguishes:

```text
Observation
  A measured resource value at a known time and scope.

Structural fact
  A known resource-bearing object or state size, such as parameter bytes,
  optimizer tensor bytes, cache artifact bytes, or worker-process residency.

Profile
  A derived reusable description of a run or workload shape, such as peaks,
  steady-state ranges, phase signatures, and regression-comparison features.

Accounting statement
  An explicit explanation of how observed or structural resource values relate
  to known owners or operations, including an honest accounting gap.
```

Every persisted resource fact must preserve its semantic class. Measured,
estimated, derived, and accounted values must never become interchangeable
because they share a unit.

Alternative considered: keep adding fields to `ResourceMonitorFacts`.

Why not: one flat event shape cannot express the different truth semantics,
lifecycles, identities, and relationships required by the target system.

### Decision: Produce and retain co-collected observations as observation frames

One resource-monitor boundary commonly collects several measurements with the
same timestamp, runtime identity, collection policy, and quality context. The
canonical production and retention unit for this telemetry is an observation
frame containing shared context plus individually identified measurements.

The frame is not a fifth semantic fact class. Each nested value remains an
observation with its own resource kind, measurement kind, value, unit, source,
and concrete scope. Profiles and accounting statements must be able to
reference an individual measurement, while ingestion, retention, and
compatibility projections may operate on the containing frame.

The first-slice scalar `ResourceObservationFacts` remains valid for isolated
observations and as the accepted measurement semantics. Before migrating the
current bundled resource events, the metadata contract must add a frame or
equivalent batch representation that avoids duplicating shared context and
relationships for every scalar.

See `reconciliation.md` for the disposition of every current event field,
startup estimate, metadata fact, and report-derived value.

Alternative considered: expand every current bundled event directly into
independent scalar resource records.

Why not: that loses co-collection identity, repeats shared context and edges,
and can multiply active-run metadata retention before buffering and retention
policy are defined.

### Decision: Resource-domain code owns production and interpretation; metadata owns accepted durable facts

Runtime resource code owns:

- collectors and live state
- sampling and collection-cost policy
- runtime context attachment
- derivation of resource profiles
- accounting logic and accounting-gap calculation
- decisions about when a resource fact becomes available

`library/metadata/` owns:

- shared accepted resource fact dataclasses
- validation
- identities and relationships
- routing to metadata events/records
- backend accumulation and durable storage
- projections and exports

Metadata must store accepted accounting statements, but it must not invent
owners, infer causes, or derive profiles by itself.

Alternative considered: make resource intelligence a separate database and
keep metadata as a secondary sink.

Why not: that would duplicate identities, storage, validation, and query
semantics already owned by the metadata backbone.

### Decision: Use one typed fact flow; JSONL becomes a projection

Resource runtime code will produce typed resource-domain facts that can be
filed as accepted metadata items. JSONL remains a useful portable/debug
artifact, but it is generated from the accepted fact representation rather than
being authored independently.

Reports, profiles, comparisons, and accounting views will consume a queryable
resource-run dataset assembled from metadata snapshots/storage or from the same
typed facts during a live run. They will not require JSONL to be the canonical
database.

The current JSONL shape may remain available as a compatibility projection
during migration.

Alternative considered: keep JSONL canonical because reports already parse it.

Why not: JSONL is a useful export format but a weak canonical model for typed
relationships, validation, durable querying, and multiple derived fact classes.

### Decision: Extend MetadataRuntime for telemetry-volume facts instead of bypassing it

The current `MetadataRuntime.file(item)` path is appropriate for low-volume
lifecycle facts but may be inefficient for sampled resource telemetry. The
resource-intelligence system will define ingestion policy explicitly:

- low-volume records/events may continue through single-item filing
- sampled observations may use metadata-runtime-owned batch filing, buffering,
  and retention policies
- collection must not block training indefinitely
- dropped or degraded telemetry must itself be observable

Resource-domain code must not call metadata backends or storage directly.
`MetadataRuntime` may internally optimize ingestion into its configured backend,
but it remains the only runtime-facing durable filing boundary. This is an
extension of the metadata runtime/storage contract, not permission for resource
code to create a separate canonical store or parallel filing API.

The first bounded-ingestion contract uses:

- direct `file(...)` for low-volume facts
- synchronous `file_many(...)` for explicit batch boundaries
- `buffer(...)` for high-frequency producer paths that must not perform
  backend/storage I/O
- bounded `flush_buffer(...)` calls at orchestration-selected boundaries
- explicit `drop_oldest` or `drop_newest` retention when capacity is exhausted
- metadata-owned `metadata_ingestion_degraded` events on the next successful
  flush after drops or ingestion failure

A failed buffered flush drops only the selected bounded batch and records
degraded state instead of raising through the telemetry path. This protects
training from storage pressure while preserving an honest observability gap.

The first monitor-migration seam is
`library.logging.resource_monitor.fact_production`. Current lifecycle and
sampled paths still produce the compatibility `ResourceMonitorFacts` shape,
but conversion now belongs to the resource domain and filing remains a separate
metadata-runtime concern. Canonical observation-frame production replaces this
compatibility builder in the next migration task without changing trainer
lifecycle hooks.

The first canonical-frame migration keeps the compatibility facts during the
transition but produces `ResourceObservationFrameFacts` from the same event
boundary. Compatibility facts and canonical frames are filed together in one
metadata-runtime batch so JSONL/report migration can later switch to projections
without re-collecting or re-interpreting the monitor counters.

The first JSONL migration slice projects the existing flat resource-monitor
artifact from produced observation-frame facts for identified runs. It preserves
the current artifact shape as a compatibility export while making the canonical
frame the preferred source for JSONL fields. A narrow fallback to the
transitional compatibility fact remains only for event boundaries that produce no
observation frame, and JSONL-only runs without a durable run identifier keep the
pre-existing raw-event write path until run identity is mandatory or a separate
non-durable projection context exists.

The first report-input migration slice moves the shared flat compatibility-event
projection into metadata ownership and uses it from both live JSONL writing and
the metadata-backed resource-run view. Benchmark reports obtain the observer's
public metadata snapshot after `ResourceMonitor.end_session()`, build a
`ResourceRunView`, and prefer projected canonical frames even when a JSONL
artifact also exists. JSONL parsing remains only as an explicit compatibility
fallback for callers without projected observation frames; an empty resource
view does not suppress an existing compatibility artifact during migration.
Existing report
session, phase, peak, per-device, and debug calculations continue to consume the
flat compatibility shape during this migration; the later report projection
slice will own the final report/export schema itself.

The first console migration slice keeps console output explicitly
presentation-only. Normal session, phase, and step resource lines are rendered
from resource-domain console summary objects rather than directly from monitor
internals, preserving current user-facing text without making console output an
accepted fact source. Startup component memory remains a separate structural
estimate surface until structural facts are migrated.

The first cross-surface equivalence check proves one lifecycle run keeps JSONL,
metadata compatibility events, canonical observation frames, nested
measurements, and console summaries aligned. JSONL repeats compatibility context
such as `run_identifier`; metadata events preserve run identity through their
record identity; canonical frames carry run identity and collection context as
accepted resource facts.

After report and export equivalence was established, canonical observation
frames became the sole normal durable representation for identified monitor
boundaries. `ResourceMonitorFacts` is now constructed and filed only when a
boundary produces no observation frame; JSONL-only runs without durable run
identity retain the raw-event fallback. This removes the bundled compatibility
mirror without removing the explicitly bounded degraded/no-frame fallback.

The first relationship-definition slice introduces a shared metadata graph
vocabulary rather than leaving resource edges as scattered string literals or
creating resource-specific graph terms. Observation frames and scalar
observations relate to runs, hosts, processes, ranks, phases, steps, collectors,
events, and devices through reusable verbs such as `observed_during`,
`observed_on`, `observed_in`, and `produced_by`; the target entity type carries
the scope specificity. Frame measurements remain individually addressable
through containment edges. Structural facts relate to their run, declared owner,
component, and group through shared graph terms. Profiles, accounting
statements, accounting gaps, and artifact-linked facts continue to use explicit
source-reference edges as their evidence trail. Missing identities are omitted
rather than invented.

### Decision: Preserve the ResourceMonitor lifecycle API as the orchestration facade

The trainer-facing API remains intentionally small:

```python
start_session()
end_session()
phase_start(name)
phase_end(name)
step_end(global_step, epoch)
emit_startup_component_memory(...)
```

The implementation behind that facade may be decomposed into collectors,
context assembly, fact production, buffering, and projections. New training
hooks should only be added when the target resource model cannot obtain an
important fact through existing lifecycle boundaries.

Alternative considered: expose collectors and sinks directly to training code.

Why not: that would spread observability mechanics through orchestration and
make future evolution harder.

### Decision: Collectors declare capabilities, cost, scope, and availability

Each collector must declare:

- which fact kinds it can produce
- supported process/device/resource scopes
- expected collection cost or collection class
- availability and degraded/fallback behavior
- whether it is safe on lifecycle boundaries, background sampling, or explicit
  diagnostics only

Modes or future collection profiles select collector capabilities and cadence.
The existing `off/basic/sampled/deep` behavior may remain as a compatibility
surface, but internal policy must not depend on one monolithic conditional
collector.

### Decision: Resource identity is relational, not phase-only

Resource facts may relate to:

- run
- process and rank
- host
- device
- phase or runtime event
- training step/epoch
- model component or optimization group
- cache/checkpoint/artifact
- collector and measurement source

Phases remain important context, but they are not the only resource identity.
This avoids forcing structural facts and device/process observations into a
phase-only model.

### Decision: Accounting is first-class but evidence-constrained

Accounting belongs in the target system now, not as an undefined future layer.
An accounting statement must include:

- the resource scope and quantity being explained
- the owner or operation being accounted
- whether its basis is measured, structural, estimated, or derived
- the source facts or relationships supporting it
- the accounting window or validity boundary
- any unresolved accounting gap

The system must allow partial accounting. An accounting gap is a result, not an
owner. Reports must not silently force unexplained values into broad labels.

Declared owner scopes may provide bounded operation/window evidence when the
runtime code genuinely knows which work is occurring. Structural accounting
and operation/window accounting remain distinct: an operation-local movement
does not prove persistent ownership. Heavy scope diagnostics must remain
explicitly configured or bounded.

### Decision: Profiles are durable derived facts, not report formatting

A resource profile is a queryable, versioned derived product suitable for:

- run comparison
- regression detection
- capacity planning
- future recommendations
- dashboard and warehouse consumption

Profiles are produced by resource-domain derivation code and filed into
metadata with links to their source facts and derivation version. Reports are
projections of profiles and facts; they are not the only place profiles exist.

### Decision: Reports query a resource-run view

The system will provide a resource-run query/view boundary that assembles the
facts needed by reports and analysis without exposing metadata backend internals
or requiring raw JSONL parsing. The view consumes metadata-owned public
snapshot/query/projection APIs; it must not read raw metadata storage directly.

This view must preserve original observed facts alongside derived profiles and
accounting statements so consumers can distinguish them.

The view builds one metadata-owned graph index per snapshot. The index maps
records by identity and edges by source and target identity so frame, scope,
and evidence queries do not repeatedly scan the complete telemetry edge set.
Generic graph helpers remain usable directly for one-off snapshot queries.

Frame-measurement lookup verifies the frame's exact run relationship, resolves
its incoming containment edges through that index, and restores the frame's
declared `measurement_identifiers` order. This keeps ordering independent of
backend edge iteration while avoiding a complete run-observation traversal per
frame. A directional local 1,000-frame report projection improved from roughly
10.6 seconds on the repeated-scan path to roughly 0.03 seconds on the indexed
path; full cross-policy acceptance thresholds remain task 10.1 work.

Evidence lookup follows each record's explicit `source_fact_references` rather
than constraining sources to the viewed run or to resource entity types. A
profile or accounting statement may therefore resolve accepted cross-run,
run-level, artifact, or other metadata records when those records were declared
as evidence; this does not make them members of the viewed run's resource fact
set.

### Decision: Metadata projections own resource export shaping

JSONL, report payload, profile artifact, and accounting artifact schemas are
metadata projections from accepted facts. Resource-domain code may decide when
an export is requested and logging/observability code may register the produced
artifact, but runtime resource code must not independently author export
schemas.

Resource document projections use a metadata-owned `ResourceExportProjection`
contract carrying an export schema name/version, source run identity, projected
payload, and the accepted record identities used by the projection. Export
schema versions are independent from metadata payload-schema and profile or
accounting derivation versions. Compatibility JSONL retains its existing flat
line shape, while report, profile, and accounting documents expose an explicit
schema envelope.

The report projection may pair lifecycle frames and calculate presentation-only
peaks, deltas, and debug rows. It must not turn those calculations into durable
profiles or accounting statements. Profile and accounting exports serialize
only already-accepted profile, statement, and gap records, preserving declared
and resolved evidence identities and keeping gaps separate from owners.

Produced resource/report artifacts are registered through the shared logging
observer. Artifact registrations, run-report records, and analytics snapshots
with a real run identity receive a metadata graph `derived_from` edge to that
run. Missing run identity omits the edge rather than inventing a placeholder.
The resource JSONL artifact is registered after the monitor closes its session,
and report JSON/Markdown artifacts carry their projection schema metadata.

Once compatibility and directional performance checks passed, the duplicate
resource report schema builders were removed from `library.logging.reports`.
Report composition and Markdown rendering consume the metadata-owned export;
they no longer carry a second implementation of phase pairing, peaks, deltas,
or debug-row shaping.

## Risks / Trade-offs

- **[Metadata becomes overloaded by high-frequency telemetry]** -> add explicit
  batch/buffer/retention APIs and benchmark ingestion overhead before routing
  default sampled telemetry durably.
- **[Central metadata package becomes a resource-domain god object]** ->
  metadata owns schemas and storage only; resource code owns collection,
  derivation, and accounting decisions.
- **[Fact model becomes too abstract before real use cases]** -> evolve it
  through concrete current facts and required queries, with versioned additive
  schemas and focused end-to-end tests.
- **[Accounting output overstates causality]** -> require basis/source/window
  semantics and support explicit accounting gaps.
- **[JSONL compatibility breaks existing workflows]** -> retain a compatibility
  projection until typed-fact and report migrations are verified.
- **[Collectors add training overhead or instability]** -> require cost classes,
  bounded queues, degraded behavior, and explicit diagnostic gating.
- **[Profiles or accounting statements become detached from their evidence]** ->
  version derivations, preserve source-fact relationships, and validate those
  relationships centrally before accepting the derived facts.

## Migration Plan

1. Reconcile current surfaces against the target fact model and settle
   observation-frame semantics.
2. Define the accepted resource fact catalog and identities in metadata,
   including observation, structural, profile, and accounting item types.
3. Add bounded metadata ingestion and retention behavior for realistic
   observation frames.
4. Introduce a resource-domain fact producer and resource-run query/view seam
   while preserving current monitor behavior.
5. Convert the current snapshot/sample event path to produce typed facts once,
   then project the current JSONL compatibility shape and metadata events.
6. Move reports from raw JSONL parsing to the resource-run view while preserving
   report output and JSONL artifact registration. During migration, typed facts
   filed through `MetadataRuntime` remain authoritative; compatibility JSONL and
   current event dictionaries must not become alternate report sources.
7. Decompose current collection into capability/cost-aware collectors.
8. Migrate startup component estimates into structural facts.
9. Add initial durable profiles and accounting statements grounded in existing
   measured and structural facts.
10. Retire compatibility-only parallel schema assembly after end-to-end
   equivalence and performance verification.

Rollback remains possible during migration because the current event/JSONL path
stays available until typed-fact projections and report queries are proven.

## Open Questions

- Which sampled observations should be retained durably by default versus
  summarized and discarded?
- Should the first resource-run query view read directly from a metadata
  snapshot, SQLite storage, or a backend-neutral query protocol?
- Which current modes should remain user-facing once collector capabilities and
  cost policies exist?
- What is the minimum trustworthy first accounting output beyond structural
  parameter/component facts?
- Which identities belong in shared metadata records versus resource-specific
  relationship facts?
