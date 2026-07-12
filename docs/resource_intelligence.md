# Resource Intelligence

The resource-intelligence system records resource observations, known
resource-bearing state, reusable run profiles, and evidence-constrained
accounting without treating those values as interchangeable.

## Ownership

The system follows one direction of flow:

```text
training lifecycle context
    -> resource monitor facade
    -> resource-domain collection and interpretation
    -> typed metadata filing
    -> metadata validation, storage, relationships, queries, and projections
    -> reports and compatibility artifacts
```

- Training code owns when session, phase, step, component, and artifact
  boundaries occur.
- `library/logging/resource_monitor/` owns collectors, collection policy, live
  monitor state, profile derivation, and accounting decisions.
- `library/metadata/` owns accepted fact schemas, validation, emitter routing,
  identities, relationships, storage, query views, and export projections.
- `library/logging/reports.py` owns report orchestration and rendering. It does
  not define a second resource schema or derive durable profiles/accounting.

Resource-domain code files typed items through `MetadataRuntime`; it does not
call metadata backends or storage directly. Metadata stores resource meaning
provided by the resource domain but does not infer causes or owners from raw
counters.

## Fact Semantics

Four semantic classes are kept distinct:

- Observation: a measured value at a known time and scope.
- Structural fact: known or explicitly estimated resource-bearing state, such
  as component parameter memory or optimizer-state estimates.
- Profile: a versioned derived description of a run, linked to its accepted
  source evidence.
- Accounting statement or gap: an evidence-backed owner claim, or an explicit
  unexplained quantity when accepted evidence cannot support ownership.

An observation frame groups measurements collected at one boundary while each
measurement remains individually identifiable and referenceable. Phase deltas,
aggregate GPU counters, and operation-local measurements must not be presented
as persistent ownership without stronger owner-scope evidence.

## Collection Modes

`output.logging.resource_monitor` exposes four policies:

- `off`: no resource collection.
- `basic`: low-cost lifecycle/step snapshots such as process memory and CUDA
  allocator state.
- `sampled`: basic collection plus a background sampler for visible GPU memory
  and process memory.
- `deep`: sampled collection plus explicitly bounded allocator diagnostics.

Collector capabilities declare cost, cadence, scope, availability, and
degraded behavior. Optional collector failure must degrade resource visibility,
not fail training. High-frequency metadata can use the bounded metadata buffer;
drops and ingestion failures remain observable.

## Reports And Exports

Benchmark reports prefer `ResourceRunView`, built from the observer's public
metadata snapshot. The view preserves observations, collector-status evidence,
profiles, accounting statements, accounting gaps, and source relationships.

Metadata projections own the schemas for:

- flat resource-monitor JSONL
- resource report documents
- profile exports
- accounting exports

JSONL remains useful as a portable/debug artifact, but it is not the canonical
resource database. Reports parse JSONL only when an accepted metadata view or
projected observation frame is unavailable, and the report records which input
source it used.

## Remaining Compatibility Boundaries

Two compatibility paths are deliberately retained:

1. `ResourceMonitorFacts` is filed only when an event boundary cannot produce
   a canonical observation frame, normally because it contains no resource
   measurements. Identified boundaries that produce a frame do not also file
   the bundled compatibility fact.
2. Reports may read the flat JSONL artifact only when no usable metadata-backed
   frame view is available.

These paths are quarantined fallbacks, not alternate sources of truth. They may
be removed only after every supported lifecycle boundary has a canonical typed
representation and every supported report caller supplies a metadata-backed
resource view. Export-shape compatibility by itself is not a reason to restore
parallel runtime schema assembly.

## Failure Cleanup

Resource cleanup is best effort. Accepted facts filed before an OOM or cleanup
failure remain the latest authoritative context. Final collection, sampled/deep
shutdown, JSONL closure, profile/accounting derivation, and artifact/report
registration must not replace the original training failure.
