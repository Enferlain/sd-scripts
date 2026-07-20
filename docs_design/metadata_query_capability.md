# Metadata Query Capability Exploration

Date: 2026-07-20
Status: pre-OpenSpec exploration; not implemented or accepted architecture

## Purpose

Record the metadata-system and model-metadata capability gap discovered after
the first model-family metadata slice. This note is intentionally outside the
module README: it contains open design work, not facts about how the production
metadata system already behaves.

Durable work is tracked in Beads:

- `sd-scripts-8ck`: establish stable source-model identity and run provenance;
- `sd-scripts-edl`: research a typed metadata query capability with optional
  source resolution;
- `sd-scripts-b25`: research and add a queryable model structure/tensor catalog;
- `sd-scripts-d79`: first link resource component facts to accepted qualified
  model-component identities.

The model catalog depends on source-model identity, the query design, and
qualified component identity. A researched OpenSpec should settle the query
design before implementation.

The model-specific source/provenance audit, structural evidence constraints,
and recommended milestone boundary are developed in
`docs_design/model_metadata_continuation_research.md`.

## Implemented Current State

The central metadata system is producer-driven:

```text
domain lifecycle boundary
    -> build or construct known typed facts
    -> MetadataRuntime.file(...) / file_many(...) / buffer(...)
    -> emitter
    -> backend/store
    -> snapshot
    -> graph index, concern view, or projection
```

Current read capabilities operate only on already accepted state:

- `MetadataSnapshot.records_for(...)` and `record_for(...)` filter collected
  records;
- `MetadataGraphIndex` indexes identities and edges in one snapshot;
- `ResourceRunView` provides rich resource-specific queries over accepted
  resource records;
- SQLite preserves records/events/edges but its public read operation returns a
  complete snapshot;
- `MetadataProvider.collect_metadata()` takes no query and backend provider
  ingestion immediately retains its complete result.

There is no general metadata operation that accepts a typed question, returns a
bounded answer, or obtains a missing fact from live state or an artifact.

The current system also lacks an authoritative cross-run source-model identity.
Run metadata records configured base-model and VAE name strings, while model
realizations are deliberately run-qualified. There is no accepted source-model
entity, resolved revision or content identity, or source-to-realization edge.
Consequently, the system can describe which configured name and family a run
used, but cannot prove that two runs used the same underlying model source.

That provenance is a base metadata requirement, independent of the proposed
query API. Querying determines how a consumer requests facts; it does not create
the durable identities and relationships that make those facts authoritative.

## Existing Direct Domain Capabilities

Absence of a metadata query service does not mean the repository cannot inspect
runtime objects. Code that owns a live object can ask it directly. For models,
current paths can enumerate parameters, buffers, and modules, read tensor shape
and dtype, create component-qualified parameter references, and aggregate
parameter counts and logical bytes.

Execution-owning code still needs live domain objects when it is performing an
operation on those objects or observing mutable runtime state. An optimizer
should not query metadata merely to recover the live `Parameter` objects it
already owns.

That does not make direct re-inspection the default for stable entity facts.
When an authoritative accepted fact already describes an identified source,
artifact, or realization and remains valid, execution and non-execution code
should be able to reuse it. The metadata system should not repeatedly derive an
unchanged model fact merely because the caller also happens to hold a live
model object.

The gap appears when a consumer knows the identity or kind of information it
needs but should not depend on one concrete representation. Examples include
inspection tools, historical analysis, artifact comparison, reports, resource
intelligence, and future recommendation or diagnostic systems.

## Corrected Query And Resolution Terms

The earlier linear wording `resolve -> query -> retain -> project` was wrong.
Resolution does not already provide a query capability, and a stored fact may
answer a query without any new resolution.

The candidate relationship is:

```text
typed query request
    -> look for an applicable accepted/cached/stored fact
    -> if unavailable and explicitly permitted:
         dispatch to a source-specific resolver
    -> typed result or explicit unavailable/stale outcome
         -> optionally retain
         -> optionally project
```

Working terminology for research:

- **Query** describes the bounded information an internal consumer requests.
  It is consumer-facing code, not necessarily an end-user UI or public API.
- **Resolver** is a possible internal source adapter for answering a particular
  query from explicit live state or an artifact when accepted facts are not
  sufficient.
- **Retention** decides whether a returned fact remains transient, is cached,
  or is filed into the metadata backend.
- **Projection** renders accepted or returned facts for an external format,
  report, dump, or analytics consumer.

The first design must not assume that all queries invoke a resolver, that all
resolved facts are retained, or that all direct domain reads should be replaced
by metadata calls.

## Why A Shared Query Capability May Be Worthwhile

A direct helper is preferable for one caller, one live source, and one known
fact. A central query capability earns its cost only where it provides one or
more of these properties:

- the caller has a durable identity rather than the concrete runtime object;
- the source may be an accepted snapshot, historical store, artifact, or live
  object;
- the requested fields or filters are selected dynamically;
- multiple consumers need the same identity, size, alias, availability, and
  validation semantics;
- the answer may be unavailable or stale and needs an explicit typed outcome;
- the caller needs introspection rather than an execution object.

It must not become a repository crawler, service locator for arbitrary runtime
objects, generic dictionary interface, or mandatory wrapper around ordinary
domain code.

## Source Model Identity And Run Provenance

Knowing which model a run used is base-level metadata, not an optional query or
caching optimization. The design needs to distinguish at least:

- a logical source reference, such as a repository plus resolved revision or a
  local artifact reference;
- content or artifact identity when it can be established by a defined digest
  or equivalent immutable evidence;
- the run-scoped loaded realization, including family/version and component
  composition;
- produced model artifacts derived from that realization.

The expected relationship is approximately:

```text
run -> loaded realization -> source model/artifact identity
                         \-> realized component sources
loaded realization -> produced checkpoint/adapter artifacts
```

A configured basename alone is not sufficient identity. The system must retain
the supplied reference for provenance, record how it was resolved, preserve the
evidence used to identify it, and avoid claiming exact cross-run equality when
that evidence is unavailable. Optional VAE, text-encoder replacements, merged
adapters, quantization, or other component substitutions must be representable
as part of the realized composition rather than hidden behind the base-model
name.

This foundation enables later reuse of stable model facts, but reuse policy and
cache invalidation remain separate concerns.

## Model Structure Proving Case

The completed model-family metadata slice records family declaration,
run-scoped realization, ordered top-level components, family contributions, and
artifact semantics. It does not represent nested modules, parameters, buffers,
tensor shapes, or storage identities.

The repository currently rediscovers related information in separate paths:

- model inspection renders named modules, parameters, buffers, shapes, dtypes,
  trainability, and buffer persistence;
- optimization grouping creates component-qualified live parameter references;
- startup diagnostics separately calculate component parameter counts and
  logical bytes and derive gradient/optimizer-state estimates.

A possible canonical structural descriptor needs research around:

- qualified realization, component, module, and tensor identity;
- parameter versus buffer kind;
- shape, dtype, element count, and logical byte size;
- trainability and buffer persistence;
- shared/tied parameter and storage alias semantics;
- component aggregation without double counting;
- live-model versus artifact/state-dict evidence;
- logical structure versus physical residency, quantization, sharding, offload,
  allocator overhead, and temporal runtime observations.

The capability should be able to answer one tensor question, a filtered query,
a component summary, or an explicitly requested complete inventory. Routine
training should not have to record every tensor for that capability to exist.

## Reuse Of Unchanged Facts

Avoiding repeated work for facts that have not changed is potentially valuable,
but it is not a near-term requirement. Reliable reuse requires answers to harder
questions:

- what identity and realization version make a fact stable;
- which mutations invalidate module, tensor, trainability, placement, or
  artifact-derived facts;
- whether validity is tied to a lifecycle phase, source revision, checkpoint,
  or process lifetime;
- whether the cached value is canonical evidence or only an optimization;
- how stale results are surfaced rather than silently reused.

The first query design should leave room for caching and validity scopes without
making a cache or invalidation framework necessary for its initial useful
slice.

## Open Questions For Research

- Should the first query API cover accepted snapshots only and add source
  resolution later, or prove both with one narrowly scoped model query?
- Should source resolution extend the current no-argument `MetadataProvider`
  contract or use a separate typed resolver contract?
- How does a caller supply access to a live source without turning central
  metadata into the owner of domain runtime objects?
- What result type distinguishes absent, unavailable, stale, invalid, and
  unsupported queries?
- Which query fields are safe to derive from an artifact header without loading
  tensor contents?
- Which existing read views should remain domain-specific even if query routing
  becomes shared?
- Where should query cost limits and permissions live for potentially complete
  inventories?
