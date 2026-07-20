## Why

The current metadata system can describe a model family and one run-scoped loaded realization, but it cannot durably recognize the source model across runs, explain how source artifacts became that realization, finalize composition after deferred loading, or describe/query structure below top-level components. Those gaps prevent model metadata from serving as reusable repo knowledge and leave resource, optimization, artifact-lineage, and future intelligent configuration work without stable model evidence.

## What Changes

- Introduce portable catalog identity for known and previously unseen models, with explicit evidence strength/trust, versioned recognition policy, durable relationship/lineage records, and exact artifact hashing that deterministically identifies unseen source representations.
- Preserve distinct identities for logical catalog lineages, immutable model revisions, serialized source representations, selected states within those representations, run-scoped realizations, components, and produced descendants instead of treating a filename or one file hash as all of them.
- Extend every model family through one uniform typed loading-result schema containing ordered sources, source selections, component bindings, loading decisions, transformations, and limitations.
- Record realization composition as append-only successful observations and materialization attempts as separate events so an initially loaded realization can be finalized after deferred materialization without rewriting accepted history or treating failure as model state.
- Add model structure metadata for components, revision/realization path bindings, observation-local live objects/storage groups, parameters, persistent and non-persistent buffers, serialized/unknown source keys, tensor descriptors, and dimension-scoped coverage claims.
- Add versioned structural and state fingerprint evidence that can progressively recognize exact artifacts, converted representations, and related model revisions without claiming general functional equivalence.
- Integrate model descriptors with the shared typed metadata query capability and qualified resource/optimization identities once those prerequisite changes are complete; this change will not create a model-only query subsystem.
- Preserve compatibility projections as projections of accepted facts and leave `tools/` and the low-priority textual-inversion path outside the migration scope.

## Capabilities

### New Capabilities

- `model-catalog-identity`: Portable catalog identity, source-representation and source-selection identity, persistent evidence resolution, recognition policy, typed relationships, immutable revisions, and lineage.
- `model-source-provenance`: Uniform typed source/selection/component-binding evidence, materialization-attempt events, transformations, and append-only successful realization composition.
- `model-structure-metadata`: Scoped structural binding/object/storage inventories, dimension-specific coverage claims, tensor descriptors, fingerprints, and query integration.

### Modified Capabilities

- `loaded-model-components`: Model loading returns source/materialization evidence alongside the authoritative family-declared component surface.
- `model-family-metadata`: Run realizations link to catalog/revision/source-representation/source-selection identities and separate successful composition observations from attempt events.
- `optimization-target-refs`: Durable model structure identities become available alongside live execution references without changing grouping policy.
- `resource-intelligence`: Structural resource facts reference accepted model/component, structural path-binding, and observation-local object/storage identities when those identities exist.

## Impact

This affects central metadata dataclasses, builders, emitters, validation, storage/indexing, snapshots/views, and projections; model-loading contracts and SD/SDXL/SD3 loaders; trainer model-loading and deferred-preparation boundaries; safetensors and checkpoint inspection; model parameter inspection; optimization target references; and structural resource accounting. Persistent catalog resolution will build on accepted metadata records and the SQLite store rather than introducing an unrelated source of truth. The structural/query milestones are explicitly gated on the system-wide typed metadata query capability and qualified resource-component identity work tracked in Beads.
