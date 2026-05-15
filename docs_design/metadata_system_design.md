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

### Decision 2: Add A Central `library/metadata/` Package

The concern is large enough to have a central package.

Proposed central responsibilities:

- Shared metadata dataclasses and protocols.
- Metadata backend / composition service.
- Provider registration.
- Validation.
- Projection serializers.
- Storage interfaces and future SQLite store.
- Shared key namespace helpers.

Domain-specific metadata facts should still be owned by their domains. The
central package owns the common language and infrastructure, not every fact.

### Decision 3: Domains Own Their Metadata Providers

Domain implementations should produce their own metadata through provider
surfaces.

Examples:

- Model families own family/component/model-spec facts.
- Adapter methods own method-local reconstruction and compatibility facts.
- Data/cache code owns sample, cache, readiness, and shard facts.
- Optimization owns plan, group, scheduler, and optimizer-runtime facts.
- Logging/reporting owns observability and report event facts.

The central backend should not need to know SDXL, LoRA, sharded cache, or
optimizer internals directly. It should collect and compose typed surfaces
declared by those domains.

### Decision 4: Use A Metadata Adapter Layer

The architecture should use a central backend, an adapter/provider layer, and
domain/family implementations.

```text
[main metadata backend]
        <->
[metadata adapter/provider layer]
        <->
[domain/family/method implementations]
```

In this design, "adapter" means the metadata integration layer, not PEFT
adapter training. To avoid ambiguity in code, the likely code term should be
`provider` or `metadata adapter`, not plain `adapter`.

The adapter/provider layer facilitates interaction. It should not contain all
family details itself.

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
domain-owned queryability.

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
    adapter.py
    optimization.py
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

Domain-owned provider implementations can live in their own packages and import
the central contracts:

```text
library/training/metadata.py
library/models/sdxl/metadata.py
library/adapters/methods/peft/vera/metadata.py
library/data/metadata.py
library/optimization/metadata.py
library/logging/metadata.py
```

This keeps domain knowledge local while keeping the backbone discoverable.

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

The main metadata backend should own composition and policy, not every domain
fact.

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

- Know model-family internals.
- Know PEFT method internals.
- Execute data workers or cache writes.
- Own optimizer construction.
- Render human console output directly.
- Treat exported `dict[str, str]` as the source of truth.

## Provider Responsibilities

A metadata provider is the domain-owned integration surface between a domain and
the backend.

Provider responsibilities:

- Declare provider identity and schema version.
- Emit typed records for the domain it owns.
- Emit events for important lifecycle transitions.
- Declare required facts.
- Declare compatibility signatures where applicable.
- Optionally provide domain-specific projections.
- Validate local invariants before handing facts to the backend.

Provider examples:

- Training provider emits run/session/objective/training-loop facts.
- Model-family provider emits component/model/export facts.
- Adapter-method provider emits method-local persistence and target facts.
- Data provider emits sample/view/readiness facts.
- Cache provider emits namespace/payload/shard facts.
- Optimization provider emits plan/group/runtime facts.
- Observability provider emits artifact/report/resource facts.

The provider layer should not become a dumping ground for arbitrary dictionaries.
Domain extensions should be typed and namespaced.

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

Validation should happen at producer boundaries and export boundaries.

### Provider Validation

Providers validate local facts they own.

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

- Required producer-owned facts missing: fail.
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

### `library/training/training_metadata.py`

Current role:

- Builds broad `ss_*` training metadata dictionaries.

Future direction:

- Become a training metadata provider plus compatibility projection source.
- Eventually stop assembling flat artifact dictionaries directly.

### `library/utils/model_metadata.py`

Current role:

- Builds SAI Model Spec metadata and minimal adapter metadata.

Future direction:

- Move model-spec projection logic under metadata projections.
- Let model-family providers supply family facts.
- Keep compatibility helpers until callers migrate.

### Strategy Checkpoint Hooks

Current role:

- `update_metadata(...)` and `get_model_metadata(...)`.

Future direction:

- Strategy/model-family providers should replace or back these hooks.
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

- Data/cache providers emit sample, view, cache namespace, and derived payload
  facts.
- Future SQLite registry becomes the durable source for large-scale state.

### Optimization Metadata

Current role:

- Already has typed runtime metadata and controlled leakage into optimizer
  payloads.

Future direction:

- Keep this as a model for domain-owned runtime metadata.
- Add provider output for optimizer plan and scheduler runtime facts.

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
4. Add training/model/artifact providers for the active checkpoint path.
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
- Optimizer/plan provider integration.
- Adapter method metadata provider integration.
- Durable SQLite store for run/artifact/event records.
- Data/cache provider integration for sample readiness, cache namespaces, and
  future shard lookup.
- Analytics/export companions such as JSON debug exports and future Parquet
  snapshots.

## Implementation Note: First Backbone Slice

Status as of 2026-05-15:

- `library/metadata/` now contains the first repo-owned backbone: typed
  identities, records, events, edges, provider results, required-fact validation,
  backend/store protocols, an in-memory backend, and projection contracts.
- The first projections cover `kuro.*`, legacy `ss_*`, `modelspec.*`, and a
  composite safetensors metadata export boundary.
- Active checkpoint metadata now routes through training-owned provider wrappers
  in `library/training/metadata_providers.py`, then projects back into the
  artifact metadata dictionary expected by current save paths.
- Existing training/model metadata builders remain as compatibility seams during
  migration. They feed provider wrappers instead of being deleted or rewritten
  in this slice.
- `output.saving.no_metadata` keeps the existing minimum legacy metadata policy;
  full metadata exports additionally include the first `kuro.*` keys.
- Data/cache metadata provider for current manifests and caches.
- SQLite store introduction for run/artifact records.
- Sample registry design for async/sharded-cache work.
- Parquet export for analytics.
- Run warehouse/dashboard integration.

## Open Considerations

### Central Package Versus Local Ownership

Decision direction:

- Central package owns contracts, backend, validation, projections, storage.
- Domains own provider implementations and domain-specific facts.

This balances discoverability with local ownership.

### How Strict To Make Extension Payloads

Decision direction:

- Do not allow arbitrary unlabeled dicts as a normal pattern.
- Allow extension payloads only when they are typed, namespaced, versioned, and
  owned by a provider.

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
- Providers own fact production.
- Consumers request projections or domain-specific queries.
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

The metadata system should be a dataclass-first, provider-driven, centrally
composed backbone under `library/metadata/`.

It should record observed truth, provenance, compatibility, events, and produced
artifacts. It should preserve domain ownership while enabling cross-domain
queries and projections. It should export legacy `ss_*`, family-specific
`modelspec.*`, and repo-owned `kuro.*` metadata through explicit projection
layers. It should validate required producer-owned facts fail-fast. It should be
designed with SQLite durability, JSON/debug exports, Parquet analytics, and
future warehouse/dashboard consumers in mind.

The first implementation should prove the backbone by routing current checkpoint
metadata through typed providers and projections while preserving existing
artifact compatibility.
