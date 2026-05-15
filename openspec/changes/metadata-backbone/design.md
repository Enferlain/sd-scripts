## Context

Metadata is currently produced and consumed through several unrelated surfaces:
training-run dictionaries, safetensors headers, `ss_*` compatibility keys,
`modelspec.*` helpers, strategy checkpoint hooks, adapter state, data/cache
records, and logging/report artifacts. Those surfaces each carry useful facts,
but there is no shared typed backbone that can compose them, validate required
facts, preserve relationships, or project them into compatibility formats.

The intended architecture is:

```text
main metadata backend <-> metadata adapter/provider layer <-> domain and family metadata surfaces
```

The central metadata system owns composition, validation, relationships,
projection, and storage seams. Domains, model families, training modes, adapter
families, and data/cache systems own the facts they know how to produce.

This change establishes the backbone for that architecture without trying to
implement every future metadata consumer at once.

## Goals / Non-Goals

Goals:

- Create a typed, dataclass-first metadata package under `library/metadata/`.
- Treat metadata as the observed counterpart to config: config describes desired
  intent, metadata records observed truth, provenance, compatibility, and
  artifacts.
- Support domain-owned providers that emit typed facts into a central backend.
- Add projection boundaries for repo-owned `kuro.*` keys and compatibility
  projections for existing `ss_*` and `modelspec.*` exports.
- Route active checkpoint artifact metadata through the backbone while
  preserving existing exported compatibility keys.
- Add validation that fails fast when producer-owned required facts are missing.
- Add storage interfaces and an in-memory/test backend shaped for a future
  SQLite durable store.

Non-goals:

- Do not implement a full SQLite warehouse in the first slice.
- Do not implement async data ingestion, streamed training input, sharded cache
  accounting, or exposure accounting in this change.
- Do not replace all logging/reporting/data/cache metadata producers at once.
- Do not require new external dependencies for the first slice.
- Do not remove existing `ss_*` or `modelspec.*` compatibility keys.

## Decisions

1. Metadata uses a dataclass-first schema.

   The first implementation should resemble the repository's existing config
   style: transparent dataclasses, small validators, explicit conversion helpers,
   and focused tests. This keeps the metadata surface approachable and avoids
   introducing an ORM or validation framework before the shape has stabilized.

2. The backbone lives in `library/metadata/`.

   Metadata is broad enough to deserve its own package instead of being hidden
   inside training, adapters, or config. The package should contain shared
   records, identities, provider contracts, backend interfaces, validation,
   projection helpers, and storage seams.

3. Domains own their own metadata facts.

   The central backend must not learn every adapter-family or model-family
   detail. Instead, domains provide typed metadata through provider surfaces.
   The backend composes those records, validates declared requirements, and
   projects them for artifacts or downstream consumers.

4. The model is a hybrid of state, events, and projections.

   Some metadata is current state, such as artifact identity and model family.
   Some is event-like, such as cache creation or checkpoint save events. Some is
   exported projection, such as safetensors headers. The backbone should keep
   these concepts separate enough that future streamed or durable systems can
   extend them without rewriting the first implementation.

5. Exported metadata is projected, not authored ad hoc.

   `kuro.*` is the repo-owned exported namespace. Existing `ss_*` keys remain
   compatibility projections. Existing `modelspec.*` keys remain relevant for
   SD/SDXL-style artifacts and should be preserved through a projection seam.

6. Validation fails fast for required producer-owned facts.

   Required facts should fail at provider or export boundaries rather than being
   silently omitted. Optional facts may remain absent, but required omissions
   should surface as actionable errors while the producing area is still known.

7. SQLite is the intended durable local store, but not the first blocker.

   The first slice should define storage interfaces and provide an in-memory
   implementation for tests and active integration. SQLite remains the intended
   durable backend because it fits local indexing, runs without service
   dependencies, and matches the future research notes' need for queryable
   provenance. Parquet is a future analytics/export format. Zarr is only
   relevant for tensor-like payload storage and should not shape the core
   metadata API.

8. The first integration target is checkpoint artifact metadata.

   Checkpoint metadata is the active, user-visible surface where compatibility
   matters most. Routing it through the backbone proves the architecture while
   keeping the first implementation bounded.

## Risks / Trade-Offs

- A central package can become a "god object" if it owns domain facts directly.
  The provider pattern is meant to prevent that by keeping fact production close
  to the domain that understands it.
- A new projection layer can accidentally change exported metadata. Compatibility
  tests for `ss_*`, `modelspec.*`, and active checkpoint headers should be added
  before broad migration.
- Dataclasses provide less automatic validation than libraries such as Pydantic.
  The trade-off is intentional for now: small repo-owned validators are easier to
  audit and keep dependency-free.
- Designing storage seams before implementing SQLite may leave some interface
  details abstract. The first backend should still encode IDs, relationships,
  events, and projections explicitly enough to avoid a throwaway shape.
- Pulling too much from the future research notes into v1 would stall the
  backbone. Async data, streaming, sharding, and accounting should remain design
  pressure, not immediate scope.

## Migration Plan

1. Add `library/metadata/` with typed records, identifiers, provider contracts,
   validation errors, backend interfaces, projection interfaces, and an in-memory
   implementation.
2. Add projection helpers for repo-owned `kuro.*` keys and compatibility
   projections for `ss_*` and `modelspec.*`.
3. Bridge existing training and model metadata producers behind provider or
   projection surfaces without changing their public behavior.
4. Route the active checkpoint metadata path through the metadata backend and
   projected export dictionaries.
5. Add tests for required fact validation, provider composition, projection
   output, and compatibility with existing active checkpoint metadata.
6. Keep old helper functions as wrappers during migration where that lowers
   risk, then remove or narrow them in follow-up changes once consumers settle.

Rollback path:

- Preserve existing metadata helpers until checkpoint projection tests pass.
- Keep exported dictionaries compatible with the old shape.
- If backbone integration fails late, checkpoint code can temporarily call the
  old helpers while retaining the new package and tests for follow-up.

## Open Questions

- Which `kuro.*` keys should be emitted in the first implementation versus
  reserved for later domains?
- Should provider registration start as explicit assembly code, a registry, or a
  hybrid that mirrors the config system's discoverability?
- How much of `modelspec.*` should be modeled as first-class typed facts versus
  kept as a projection-only compatibility layer?
- When SQLite is added, should it store every event by default or only selected
  durable records from provider-declared policies?
