# Metadata System Design

Date: 2026-05-14
Status: design draft before OpenSpec proposal

This document turns the metadata inventory, coverage map, research inputs, and
design discussion into an implementation-oriented direction. It is meant to be
the source material for the later OpenSpec change.

Companion notes:

- `docs_design/metadata_system_inventory.md`
- `docs_design/metadata_system_coverage.md`
- `docs_design/metadata_system_research_inputs.md`

## Core Thesis

The metadata system should become the repo-owned counterpart to the config
system.

```text
Config:    desired intent
Metadata: observed truth, provenance, compatibility, and produced artifacts
```

The current config system uses discoverable Python dataclasses, centralized
registration, explicit validation, and YAML projection through Hydra. The
metadata system should follow the same general spirit, but with a different
purpose:

- Config says what the user asked the run to do.
- Metadata says what actually existed, happened, was produced, and can be reused.

The backbone should not try to implement async data, sharded caches, streaming
training, or run warehouse behavior immediately. Instead, it should be shaped so
those domains can add metadata surfaces later without rewriting the core.

## Settled Setup Goal

The metadata system is one central backbone, not a `metadata.py` per normal
domain. `library/metadata/` owns the recorded dataclass catalog, emitters and
providers, projections, backend/storage, validation, and key helpers.
Normal domain code owns the runtime objects and lifecycle call sites that hand
facts to the central system. Only explicit plugin/facet areas get local
metadata modules; adapter methods are the current clear exception, and model
families remain an explicit future decision.

## Key Decisions

### Decision 1: Use Typed Python Dataclasses As The Authoring Shape

Metadata surfaces should be authored as Python dataclasses first.

Rationale:

- This matches the repo's config precedent.
- Dataclasses are easy to inspect, test, document, and refactor.
- They avoid letting storage schemas or exported string dictionaries dictate the
  in-code architecture.
- They are lighter than adding a Pydantic-style dependency and stricter than
  unrestricted dictionaries.

Implication:

- Storage, safetensors metadata, report JSON, tracker payloads, and legacy keys
  are projections from typed internal records.
- The source of truth is not `dict[str, str]`, even when the final artifact
  format requires strings.
- Recorded metadata dataclasses should live in
  `library/metadata/dataclasses/<concern>.py`, following the same discoverable
  catalog idea as `library/config/dataclasses/`.
- Metadata assembly code should live in the central metadata package by default.
  Domain code owns the source objects and the lifecycle call sites that pass
  those objects into the metadata system.

### Decision 2: Add A Central `library/metadata/` Package

The concern is large enough to have a central package.

Proposed central responsibilities:

- Recorded metadata dataclasses and protocols.
- Metadata backend / composition service.
- Emitter/builder functions that turn repo runtime objects into metadata
  records.
- Provider contracts and registration where a provider object is useful.
- Validation.
- Projection serializers.
- Storage interfaces and future SQLite store.
- Key namespace helpers.

The central package owns recorded metadata schemas, assembly helpers,
projection, validation, and infrastructure. Domain code owns source truth: the
runtime objects, family/method implementations, and lifecycle moments where
metadata should be emitted.

### Decision 3: Central Emitters Are The Default Metadata Code Home

Normal domain packages should not grow their own metadata modules. Instead, the
default home for metadata-related code is `library/metadata/`:

```text
library/metadata/dataclasses/   # what recorded metadata exists
library/metadata/emitters/      # how repo objects become metadata records
library/metadata/projections/   # how records become exported formats
library/metadata/storage/       # how records are persisted or exported
```

Domain implementations call central emitters at lifecycle boundaries and pass in
the domain-owned objects or narrow context needed to build records. This keeps
the source truth local while keeping metadata logic discoverable and consistent.

Examples:

- Training code calls central run/checkpoint emitters with config, manifests,
  runtime state, strategy output, and save-boundary facts.
- Data/cache code calls central data/cache emitters with manifests, entries,
  buckets, cache namespaces, readiness state, and shard facts.
- Optimization code calls central optimization emitters with optimizer plans,
  parameter groups, scheduler runtime facts, and optimizer runtime facts.
- Logging/reporting code calls central observability emitters with report,
  artifact, resource, and tracker-boundary facts.

The central backend should not need to know SDXL, LoRA, sharded cache, or
optimizer internals directly. It should collect and compose typed recorded
schemas produced by central emitters and explicit plugin/family hooks.

Explicit exceptions must be named rather than implied. Adapter methods are the
current clear exception because the adapter method tree behaves like a plugin
facet; method-local persistence facts such as VeRA reconstruction metadata may
live locally. Model-family metadata may later be treated similarly only if model
families are explicitly designed as plugin-like implementations.

### Decision 4: Use Metadata Emitters And Provider Contracts

The architecture should use central emitters/builders, provider contracts where
stateful or plugin-like emission is useful, a central backend, and
domain/family lifecycle call sites.

```text
[domain lifecycle boundary]
        ->
[metadata emitter / provider contract]
        ->
[main metadata backend]
        ->
[projection / storage boundary]
```

In this design, "adapter" should not be used as the normal code term because it
collides with PEFT adapter training. Use `emitter` for central functions that
turn known inputs into records, and `provider` for an object/protocol that emits
records, events, relationships, or required-fact declarations.

The emitter/provider layer facilitates interaction. It should not contain all
family details itself, and it should not make domain decisions such as optimizer
construction, data readiness transitions, or model-family loading.

### Decision 5: Use A Hybrid State/Event/Projection Model

The metadata backbone should support three related but separate concepts:

- State/snapshot metadata: what is true now, or what was true at a checkpoint or
  report boundary.
- Event metadata: what happened over time.
- Projection metadata: what gets exported to a specific external shape.

Rationale:

- Checkpoints, model identity, compatibility, and cache reuse need snapshots.
- Async data, readiness transitions, failures, artifact registration, and
  observability need events.
- Safetensors, `ss_*`, `modelspec.*`, `kuro.*`, reports, and trackers need
  projections.

The backend should be able to compose linked run-level records while preserving
queryability by recorded schema and producer.

### Decision 6: Introduce Repo-Owned `kuro.*` Exported Keys

The eventual repo-owned exported metadata namespace should use `kuro.*`.

Rationale:

- `ss_*` keys are legacy Kohya compatibility keys.
- `modelspec.*` keys are model-spec compatibility keys, mainly relevant to
  Stability-family artifacts.
- Kuro needs its own namespace for repo-owned exported facts.

Implication:

- `ss_*` remains a compatibility projection.
- `modelspec.*` remains a family/export projection where applicable.
- `kuro.*` becomes the repo-owned exported namespace for durable artifact facts.
- Internal dataclass names do not need to mirror the string key prefix.

### Decision 7: Validate Required Producer-Owned Facts Fail-Fast

Metadata validation should fail when active producers omit required facts.

Rationale:

- Missing metadata should be fixed when encountered, not silently normalized
  away.
- Required facts are only meaningful if the producer owns them and the consumer
  can trust them.
- This repo is already in refactor mode and does not need broad backwards
  compatibility compromises.

Implication:

- Providers should declare required facts for active surfaces.
- The backend should validate required facts before checkpoint/export/report
  boundaries.
- Optional facts can remain absent.
- Compatibility projections can degrade only by explicit policy.

### Decision 8: SQLite Is The Intended Durable Local Store

The design should assume SQLite as the likely canonical local operational store
for future durable run/sample/cache metadata.

Rationale:

- It matches the research notes around sample registry and sharded cache lookup.
- It supports incremental updates, point lookups, state transitions, and local
  workflows.
- It does not lock the repo into a remote service.
- It can be introduced behind a storage interface.

Related formats:

- JSON/YAML: human/debug/export projections, not the large-scale canonical store.
- Parquet: analytics/export companion for large scans and warehouse-style uses.
- Zarr: possible future tensor payload format, not a metadata-core store. Metadata
  may describe Zarr payloads if that path is chosen later.

### Decision 9: Keep The Core Toolset Small

The first design target should avoid heavy new dependencies unless a later slice
proves they are needed.

Initial tool choices:

- Python dataclasses for in-code schema.
- Repo-owned validators instead of Pydantic-style runtime models.
- Canonical JSON serialization for signatures, tests, and debug exports.
- The existing hash utilities where possible for compatibility signatures.
- Python's `sqlite3` module as the likely first SQLite implementation path.

Future optional tools:

- `pyarrow` or another Parquet stack for analytics/export if and when Parquet
  exports become active work.
- Zarr only if tensor payload storage moves in that direction.
- A richer SQL layer only if handwritten SQLite access becomes a real burden.

Non-goal:

- Do not introduce a database ORM or schema framework just to define metadata
  concepts. The metadata dataclasses should remain the repo-facing authoring
  surface.

## Proposed Package Shape

The exact file names can change, but the package should likely be shaped around
these responsibilities:

```text
library/metadata/
  __init__.py
  backend.py
  registry.py
  validation.py
  keys.py
  serialization.py

  dataclasses/
    identity.py
    provenance.py
    compatibility.py
    run.py
    artifact.py
    model.py
    data.py
    cache.py
    optimization.py
    observability.py

  emitters/
    run.py
    data.py
    model.py
    optimization.py
    checkpoint.py
    observability.py

  providers/
    base.py
    registry.py

  projections/
    kuro.py
    kohya_ss.py
    modelspec.py
    safetensors.py
    reports.py
    trackers.py

  storage/
    base.py
    memory.py
    sqlite.py
    json_export.py
    parquet_export.py
```

Normal domain folders should not receive local `metadata.py` modules by
default. If metadata is treated as a backbone, normal metadata code lives in the
central package. Domain files keep the lifecycle call sites and pass known
runtime objects or narrow context into central emitters/providers.

Default examples:

```text
library/metadata/emitters/run.py
library/metadata/emitters/data.py
library/metadata/emitters/optimization.py
library/metadata/emitters/checkpoint.py
library/metadata/projections/kohya_ss.py
library/metadata/projections/modelspec.py
library/metadata/projections/kuro.py
```

Non-default examples that should not appear just because a normal domain needs
metadata:

```text
library/training/metadata.py
library/data/metadata.py
library/optimization/metadata.py
library/logging/metadata.py
```

The current training metadata module is a transitional assembly seam, not the
target pattern for future domains. It should shrink or move into central
emitters as the backbone stabilizes.

Local metadata modules are justified only for explicit plugin/facet
exceptions, not as a convenience pattern for ordinary domains.
Current likely exceptions:

```text
library/adapters/methods/peft/vera/metadata.py  # method-local reconstruction facts
library/models/<family>/metadata.py             # only if model families are explicitly treated as facets
```

Those local modules may expose plugin/family hooks, but recorded metadata that
is stored by the repo backend still belongs in the central dataclass catalog
unless the exception explicitly includes local typed schemas.

## Core Conceptual Types

The exact dataclass names can be settled during implementation, but the design
needs these concepts.

### Metadata Identity

Every important record should know what it is about.

Likely fields:

- Entity type.
- Stable ID where available.
- Run-local ID where needed.
- Domain namespace.
- Human label where useful.
- Version or schema version.

Examples of entity types:

- Run.
- Dataset.
- Sample.
- Sample view.
- Model family.
- Loaded component.
- Training mode.
- Adapter method.
- Adapter artifact.
- Optimizer plan.
- Cache namespace.
- Cache entry.
- Shard.
- Checkpoint.
- Report.
- Sample image.

### Metadata Record

A record is a typed snapshot of facts about one entity.

Examples:

- Run facts.
- Model component facts.
- Artifact facts.
- Cache namespace facts.
- Optimizer plan facts.
- Sample readiness facts.

Records should be typed dataclasses, not anonymous dictionaries.

### Metadata Event

An event records something that happened.

Examples:

- Run started.
- Component loaded.
- Cache namespace selected.
- Artifact written.
- Sample became ready.
- Sample failed and was quarantined.
- Report generated.

Events are important for async/streaming/warehouse futures, but even v1 can use
them for artifact registration and run lifecycle.

### Metadata Edge

The backend should represent relationships between records.

Useful relationship examples:

- Artifact was produced by run.
- Cache entry was derived from sample.
- Adapter trainable targets model component.
- Checkpoint contains model family metadata.
- Report summarizes run.
- Sample belongs to dataset view.
- Shard contains cache entry.

This can begin as in-memory links and later map naturally to SQLite tables.

### Metadata Projection

A projection turns internal metadata into an external representation.

Projection examples:

- `ss_*` compatibility keys.
- `modelspec.*` keys.
- `kuro.*` artifact keys.
- Safetensors metadata dicts.
- Benchmark report JSON.
- Tracker config/metrics payloads.
- Future warehouse exports.

Projections are allowed to be lossy and format-constrained. Internal metadata
should stay richer than exported metadata.

## Backend Responsibilities

The main metadata backend should own composition and policy, not domain
decision-making.

Responsibilities:

- Register providers.
- Start and finish metadata collection scopes.
- Accept typed records and events.
- Maintain run-level composition.
- Maintain identity and relationship links.
- Validate required facts at declared boundaries.
- Produce projections for consumers.
- Route records/events to storage when a store is configured.
- Preserve domain queryability.

The backend should not:

- Discover facts by rummaging through domain internals.
- Know model-family internals.
- Know PEFT method internals.
- Execute data workers or cache writes.
- Own optimizer construction.
- Render human console output directly.
- Treat exported `dict[str, str]` as the source of truth.

## Emitter Responsibilities

A metadata emitter is central metadata code that turns already-known repo
objects into typed metadata records, events, relationships, or provider results.
Emitters are the default place for metadata assembly code.

Emitter responsibilities:

- Accept explicit domain-owned inputs or narrow context objects.
- Construct central recorded metadata dataclasses.
- Convert central dataclasses into records/events for the backend.
- Apply metadata-specific validation and normalization.
- Avoid changing domain behavior or owning domain decisions.
- Keep compatibility mapping visible by calling projection helpers instead of
  scattering exported keys through normal domain modules.

Emitter examples:

- A run emitter builds run/session/objective facts from trainer config, runtime
  state, manifests, and objective identity supplied by training code.
- A checkpoint emitter builds artifact/save-boundary facts from checkpoint name,
  step, epoch, format, and metadata policy.
- A data emitter builds sample/view/readiness/cache facts from manifests,
  entries, buckets, and cache namespaces supplied by data code.
- An optimization emitter builds plan/group/runtime facts from optimization
  plans supplied by optimization code.

Emitters should not:

- Construct optimizers, schedulers, models, datasets, or caches.
- Read broad global trainer state when narrow inputs are available.
- Become a dumping ground for arbitrary dictionaries.
- Replace plugin/family hooks where the metadata surface is genuinely local to
  that plugin/family.

## Provider Responsibilities

A metadata provider is the producer-side integration surface between code that
has facts and the metadata backend. Providers are useful when emission is
stateful, registered, or plugin-like. They are not a reason for every normal
domain package to grow a local metadata module.

Provider responsibilities:

- Declare provider identity and schema version.
- Emit central recorded metadata dataclasses or records built from them.
- Emit events for important lifecycle transitions.
- Declare required facts.
- Declare compatibility signatures where applicable.
- Optionally provide projections only when the projection is genuinely
  producer/plugin-specific.
- Validate local invariants before handing facts to the backend.

Provider examples:

- Central training/checkpoint emitters may wrap their output in providers for
  backend ingestion.
- Model-family hooks may emit central component/model/export records unless
  model families are explicitly treated as plugin-like implementations.
- Adapter-method producer may emit method-local persistence and target records
  from local plugin-like schemas.
- Data/cache emitters may expose provider objects for central
  sample/view/readiness/cache records.
- Optimization emitters may expose provider objects for central
  plan/group/runtime records.
- Observability emitters may expose provider objects for central
  artifact/report/resource records.

The provider layer should not become a dumping ground for arbitrary dictionaries.
Extensions should be typed and namespaced. If they are recorded by the repo
metadata backend, their dataclasses belong in `library/metadata/dataclasses/`
unless covered by an explicit plugin-like exception.

## State, Events, And Snapshots

The design should support all three without forcing every domain to use all of
them immediately.

### State/Snapshot Use Cases

- Checkpoint metadata.
- Final artifact metadata.
- Model/component identity.
- Config identity.
- Cache compatibility.
- Dataset view membership.
- Optimizer plan identity.
- Run report context.

### Event Use Cases

- Run lifecycle.
- Artifact registration.
- Sample lifecycle transitions.
- Cache write completion.
- Failure/quarantine/retry.
- Resource monitor events.
- Future queue/backpressure events.
- Future warehouse timelines.

### Snapshot Boundary Examples

- Run startup.
- Checkpoint save.
- Final artifact save.
- Benchmark report generation.
- Epoch/view creation.
- Cache namespace creation.
- Future registry export.

## Storage Strategy

### Immediate Shape

The first implementation can keep storage minimal if needed, but the architecture
should be designed around durable storage from the start.

Useful first stores:

- In-memory store for tests and local composition.
- JSON export for debugging and early review.
- Safetensors projection for artifact metadata.

### Intended Durable Store

SQLite should be the intended local canonical store for operational metadata
that needs durability and point lookup.

Likely future SQLite responsibilities:

- Run records.
- Artifact records.
- Event timeline.
- Sample registry.
- Cache namespace registry.
- Shard membership/index summaries.
- Compatibility signatures.
- Export snapshots.

Likely table concepts:

- `metadata_records`
- `metadata_edges`
- `metadata_events`
- `metadata_artifacts`
- `metadata_signatures`
- `metadata_exports`
- Future data/cache tables for sample registry and shard lookup.

This is not a final schema. It is a direction so the in-code model does not
fight future persistence.

### Analytics And Exports

Parquet should be treated as a future analytics/export companion, especially for
large sample registries and run-history analysis.

JSON/YAML should remain useful for:

- Debugging.
- Small sidecars.
- Human inspection.
- Golden tests.

Zarr should be treated as tensor payload research. Metadata should be able to
describe Zarr payloads, but Zarr should not drive the metadata-core design.

## Projection Strategy

The metadata backend should project internal facts into specific external
formats.

### `kuro.*`

Repo-owned durable artifact keys.

Examples of likely categories:

- `kuro.schema_version`
- `kuro.run.*`
- `kuro.repo.*`
- `kuro.model.*`
- `kuro.component.*`
- `kuro.adapter.*`
- `kuro.objective.*`
- `kuro.cache.*`
- `kuro.artifact.*`

The exact key list should be designed later. The important decision is that
repo-owned exported keys use the `kuro.*` namespace.

### `ss_*`

Legacy Kohya compatibility projection.

Rules:

- Keep compatibility where existing tools expect these keys.
- Do not treat `ss_*` as the internal source of truth.
- Generate `ss_*` from typed records.
- Preserve the existing no-metadata/minimum-metadata behavior through explicit
  projection policy.

### `modelspec.*`

Model-spec projection for supported model families and artifact types.

Rules:

- Keep it as a projection, not a universal metadata shape.
- Let model-family providers decide when it applies.
- Learn from its structure for repo-owned model-card exports, but do not force
  non-Stability families into it.

### Safetensors Metadata

Safetensors metadata remains a string-only storage boundary.

Rules:

- Convert typed internal metadata into `dict[str, str]`.
- Validate projection values before writing.
- Keep richer internal facts in backend/store/sidecar records when needed.

## Validation Strategy

Validation should happen at emitter/provider boundaries and export boundaries.

### Provider Validation

Emitters/providers validate metadata facts before handing records to the
backend. Domain code remains responsible for the correctness of the source
objects it passes in.

Examples:

- Adapter method metadata contains required reconstruction keys.
- Model family can provide component identity.
- Cache namespace signature has required inputs.
- Optimizer plan labels match runtime learning-rate groups.

### Backend Validation

The backend validates cross-domain and boundary requirements.

Examples:

- A checkpoint artifact must link to a run.
- An adapter artifact must link to adapter method identity.
- A compatibility projection must have required source facts.
- A `kuro.*` projection must include schema version.
- A required provider for the active mode must have emitted its facts.

### Severity

Default behavior:

- Required facts missing from an active emitter/provider: fail.
- Invalid required facts: fail.
- Optional facts missing: allow.
- Compatibility projection degraded: only allow through explicit policy.
- Diagnostic/report-only missing facts: warning or omission by explicit policy.

## Versioning And Migration

Metadata must be versioned from the start.

Versioned areas:

- Internal dataclass schema versions.
- Provider schema versions.
- `kuro.*` projection version.
- SQLite schema version.
- Cache namespace/signature versions.
- Artifact projection versions.

Migration considerations:

- Old `ss_*` metadata should remain loadable as legacy artifact metadata.
- Existing `modelspec.*` metadata should remain loadable where applicable.
- New metadata should not require rewriting old artifacts.
- Future SQLite migrations should be explicit and testable.
- Projection compatibility tests should lock exported key behavior.

## How Existing Surfaces Settle

### Training Metadata Emitter Path

Current role:

- `library/metadata/emitters/run.py` builds typed full/minimum
  `RunMetadataFacts` for active training runs.
- `library/metadata/emitters/checkpoint.py` composes run/model/artifact facts
  through the backend and projection layer for checkpoint exports.
- `library/training/metadata.py` is only a compatibility import wrapper.
- Deprecated PEFT script copies are fenced with replacement `train.py` preset
  guidance instead of retaining stale training metadata helper imports.

Future direction:

- Keep trainer code as the lifecycle call site that supplies config, manifests,
  runtime state, strategy output, and save-boundary facts.
- Eventually stop assembling flat artifact dictionaries directly.

### `library/utils/model_metadata.py`

Current role:

- Builds SAI Model Spec metadata and minimal adapter metadata.

Future direction:

- Move model-spec projection logic under metadata projections.
- Let model-family hooks or providers supply family facts where the family is an
  explicit facet.
- Keep compatibility helpers until callers migrate.

### Strategy Checkpoint Hooks

Current role:

- `update_metadata(...)` and `get_model_metadata(...)`.

Future direction:

- Strategy/model-family hooks or providers should replace or back these hooks.
- Checkpoint code should request artifact projections from the metadata backend.

### Adapter Method State Dicts

Current role:

- Pass artifact metadata through, with method-local additions such as VeRA.

Future direction:

- Adapter methods emit typed method-local metadata.
- State dict writers project method-local facts into artifact metadata when
  needed.

### Data Manifest And Cache Metadata

Current role:

- Manifest and cache metadata are scattered across structures, safetensors
  metadata, and validation helpers.

Future direction:

- Central data/cache emitters build sample, view, cache namespace, and derived
  payload facts from manifest/cache objects supplied by data code.
- Future SQLite registry becomes the durable source for large-scale state.

### Optimization Metadata

Current role:

- Already has typed runtime metadata and controlled leakage into optimizer
  payloads.

Future direction:

- Keep this as a model for typed runtime metadata that does not leak into
  artifact projections by default.
- Add central optimization emitter/provider output for optimizer plan and
  scheduler runtime facts.

### Observability Metadata

Current role:

- `LoggedArtifact`, `TrainingObserver`, `RunReportContext`, and resource monitor
  facts.

Future direction:

- Logging remains presentation/sink-oriented.
- Metadata backend provides artifact/run/event records that logging/reporting can
  project.

## First Implementation Slice

The first implementation should prove the backbone without trying to implement
all future data/cache/warehouse functionality.

Recommended first slice:

1. Add `library/metadata/` contracts and core dataclasses.
2. Add in-memory backend and validation scaffold.
3. Add projection scaffolds for `kuro.*`, `ss_*`, and `modelspec.*`.
4. Add central run/model/artifact emitters or providers for the active
   checkpoint path.
5. Route checkpoint metadata construction through the backend while preserving
   existing exported keys.
6. Add focused tests proving:
   - required facts fail when missing;
   - `ss_*` projection remains compatible;
   - `modelspec.*` projection remains compatible for SD/SDXL;
   - `kuro.*` projection includes repo-owned schema identity;
   - no raw domain dict becomes the internal source of truth.

This slice gives the metadata system a real path into current behavior without
blocking on sample registry, sharded caches, or streaming.

## Later Implementation Slices

Likely follow-up slices:

- Observability/artifact registration integration.
- Optimizer/plan emitter/provider integration.
- Adapter method metadata provider integration.
- Durable SQLite store for run/artifact/event records.
- Data/cache emitter/provider integration for sample readiness, cache
  namespaces, and future shard lookup.
- Analytics/export companions such as JSON debug exports and future Parquet
  snapshots.

## Implementation Note: First Backbone Slice

Status as of 2026-05-15:

- `library/metadata/` now contains the first repo-owned backbone: typed
  identities, records, events, edges, provider results, required-fact validation,
  backend/store protocols, an in-memory backend, and projection contracts.
- The first projections cover `kuro.*`, legacy `ss_*`, `modelspec.*`, and a
  composite safetensors metadata export boundary.
- Active checkpoint metadata now routes through the backbone and projects back
  into the artifact metadata dictionary expected by current save paths. The
  current `library/training/metadata.py` seam is transitional; the target shape
  is central metadata emitters plus local trainer call sites.
- The old active training metadata helper was deleted after its behavior moved
  into the backbone-facing training metadata module. Exported compatibility
  keys remain projections/facts, not a reason to keep old helper modules alive.
- `output.saving.no_metadata` keeps the existing minimum legacy metadata policy;
  full metadata exports additionally include the first `kuro.*` keys.
- Data/cache metadata provider for current manifests and caches.
- SQLite store introduction for run/artifact records.
- Sample registry design for async/sharded-cache work.
- Parquet export for analytics.
- Run warehouse/dashboard integration.

## Open Considerations

### Central Package Versus Domain Source Ownership

Decision direction:

- `library/metadata/` owns recorded metadata schemas, contracts, backend,
  validation, emitters/builders, projections, storage, and key helpers.
- Domain code owns source truth and lifecycle call sites: it knows when a run,
  checkpoint, cache entry, optimizer plan, report, or artifact has reached a
  metadata boundary.
- Normal local `metadata.py` modules are not the default pattern.
- Local recorded metadata schemas are allowed only for explicit plugin/facet
  exceptions.

This keeps metadata discoverable in the same spirit as config dataclasses while
still keeping the real runtime decisions near the code that owns them.

### How Strict To Make Extension Payloads

Decision direction:

- Do not allow arbitrary unlabeled dicts as a normal pattern.
- Allow extension payloads only when they are typed, namespaced, versioned, and
  emitted by an identified producer.

This keeps extension easy without recreating metadata soup.

### How Much SQLite To Implement In V1

Decision direction:

- Design for SQLite now.
- Do not require SQLite for the first checkpoint-metadata migration if it slows
  the first slice too much.
- Keep storage interfaces compatible with future SQLite from day one.

### How To Avoid A Metadata God Object

Decision direction:

- Backend composes and links records.
- `library/metadata/dataclasses/` owns recorded schemas.
- `library/metadata/emitters/` owns normal metadata assembly.
- Domain code owns source objects and call sites.
- Consumers request projections or schema/producer-specific queries.
- Avoid one giant flattened run metadata dict.

### How To Keep Exported Metadata Small

Decision direction:

- Export only the projection needed for the artifact.
- Keep richer metadata in backend/store/sidecars when needed.
- Treat safetensors metadata as a constrained projection, not the whole truth.

## Non-Goals For The First OpenSpec Change

- Do not implement async data.
- Do not implement sharded cache storage.
- Do not implement streaming ingest.
- Do not implement exposure/deficit scheduling.
- Do not implement a full run warehouse.
- Do not migrate every existing metadata-like dict in one pass.
- Do not choose final SQLite tables for sample registry and shard lookup.

Those futures should influence the backbone shape, not overload the first
implementation.

## Summary

The metadata system should be a dataclass-first, emitter/provider-driven,
centrally composed backbone under `library/metadata/`.

It should record observed truth, provenance, compatibility, events, and produced
artifacts. It should keep domain source truth and lifecycle decisions local
while making metadata schemas, normal assembly code, projections, validation,
and storage discoverable under `library/metadata/`. It should export legacy
`ss_*`, family-specific `modelspec.*`, and repo-owned `kuro.*` metadata through
explicit projection layers. It should validate required facts fail-fast. It
should be designed with SQLite durability, JSON/debug exports, Parquet
analytics, and future warehouse/dashboard consumers in mind.

The first implementation should prove the backbone by routing current checkpoint
metadata through typed providers and projections while preserving existing
artifact compatibility.
