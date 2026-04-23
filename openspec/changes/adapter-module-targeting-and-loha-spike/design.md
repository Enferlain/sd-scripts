## Context

The completed adapter-system rework established the ownership model for the
repo-owned adapter path:

- strategy exposes targetable model structure
- optimization owns training policy and target selection
- adapter runtimes realize concrete adapter state against those resolved targets
- optimization groups the returned trainable parameter refs
- `PeftMode` orchestrates the round trip between those layers

The current target-resolution path still resolves only component-root targets.
That is sufficient for the built-in migration slice, but it is too coarse for
the richer model-defined targeting the repo already wants to support.
Repo-native absorbed LyCORIS methods such as `loha` are the first consumer that
need concrete module targets while still returning parameter-native trainable
refs for grouping. This slice focuses on module-resolved methods first; it does
not redefine the shared targeting model as module-only forever.

The key constraint for this design is that grouping ownership is already in the
right place. The repo's optimization/grouping layer works at the parameter
level and does not need to learn adapter-method-specific module semantics. The
missing piece is explicit module targeting in optimization, not a new grouping
concept.

## Goals / Non-Goals

**Goals:**

- Extend the shared model-based target-resolution path to concrete module
  targets
- Extend the repo-owned adapter target payload so adapter runtimes can consume
  concrete module targets with stable provenance
- Keep the resolved-target model broad enough that later absorbed methods can
  bind at finer parameter granularity without redesigning ownership
- Implement a first repo-native `loha` runtime under `library/adapters/methods/`
  as the first step toward a fully repo-owned method implementation
- Keep grouping parameter-native by continuing to consume
  `AdapterTrainableParameterRef` instances with provenance
- Reuse the existing repo-owned loaded-runtime, merge, and export seams for the
  first absorbed method
- Make the intended end state explicit: absorbed methods should not permanently
  depend on vendor algorithm module classes at runtime once they are fully
  brought into the repo

**Non-Goals:**

- Generalize a broad shared LyCORIS abstraction layer before the first method
  has exercised the path end-to-end
- Move adapter lifecycle ownership out of `PeftMode`
- Recenter the adapter path around vendor APIs such as `create_lycoris(...)`,
  presets, or vendor-owned regex target discovery
- Treat a repo-owned wrapper around vendor module classes as the final absorbed
  method shape
- Change optimizer grouping to operate on modules instead of parameters
- Define mixed-method target assignment, overlap handling, or conflict
  resolution semantics in this slice
- Define the final configuration surface for every future absorbed method

## Decisions

### 1. Optimization will resolve concrete module targets from shared model structure

The optimization layer will extend the repo's shared target-resolution path to
concrete module targets and will hand downstream consumers concrete module
targets rather than only component-root selections.

Rationale:

- This keeps training policy ownership in optimization
- It keeps target resolution grounded in the original model structure rather
  than inventing separate target worlds for different training consumers
- It avoids making adapter runtimes rediscover target modules through vendor
  scanning logic
- It preserves the clean split that strategy exposes structure and optimization
  decides what is in scope

Alternatives considered:

- Let adapter runtimes rediscover targets from component roots
  - Rejected because it shifts target selection back into adapter-owned code
- Let strategy directly decide adapter targets
  - Rejected because it would blur structure exposure and training policy

### 2. The shared adapter target payload will stay minimal and module-first

The repo-owned adapter target payload will carry the concrete module plus the
minimum stable provenance needed to connect it back to optimization policy:
`component`, `component_key`, `path`, `module`, and optional tags/metadata.

Rationale:

- Adapter methods need the concrete module they are attaching to
- Grouping does not need module-type metadata because it consumes returned
  parameter refs, not target descriptors
- The first absorbed method can be expressed cleanly from module-resolved
  targets, while later parameter-granular methods can drive follow-up
  extensions if they prove additional shared fields are needed
- A smaller contract keeps the first absorbed-method pass focused and avoids
  inventing shared concepts before multiple methods prove they are needed

Alternatives considered:

- Add a first-class shared `target_kind` or `module_type` field immediately
  - Rejected for now because the first method can inspect `target.module`
    directly without teaching optimization/grouping about extra categories
- Add parameter-binding fields immediately for every future absorbed method
  - Rejected for now because this first slice is proving richer resolved
    targets through `loha`, not standardizing every later binding mode up front
- Keep the existing component-root-only target payload
  - Rejected because it is too coarse for module-attached absorbed methods

### 3. The first absorbed method will be a concrete repo-native `loha` runtime

The first absorbed LyCORIS method will be implemented directly as
`library/adapters/methods/peft/loha/runtime.py` rather than through a generic
shared LyCORIS runtime layer.

Rationale:

- `loha` is representative enough to pressure-test build/from-weights/merge and
  trainable-ref seams
- A concrete first method reveals what is genuinely shared before extracting
  common LyCORIS helper layers
- This avoids inventing a fake shared abstraction around vendor code too early

Alternatives considered:

- Build a generic shared LyCORIS runtime layer first
  - Rejected because the common surface is still hypothetical
- Start with a more unusual method such as `ia3` or `diag-oft`
  - Rejected because `loha` is the cleaner first representative spike

### 4. Vendor LyCORIS wrapper APIs will not become the repo contract

Vendor APIs such as `create_lycoris(...)`, presets, and wrapper-owned target
discovery will remain implementation details to avoid, not the repo contract.

Rationale:

- The vendor wrapper owns preset parsing, regex/name scanning, and the older
  adapter-owned optimizer hook model
- Reusing that wrapper as the repo boundary would reintroduce the same
  ownership confusion the adapter-system rework just removed
- The repo-owned path already has explicit build/from-weights/merge seams that
  the absorbed method should fit into instead
- The intended destination is a fully repo-owned method implementation, so even
  direct dependence on vendor algorithm classes should be treated as
  transitional rather than the permanent absorbed-method shape

Alternatives considered:

- Wrap `create_lycoris(...)` behind the new runtime boundary
  - Rejected because it would still hide adapter-owned target discovery inside
    the build step

### 5. Absorbed methods are expected to become fully repo-owned implementations

The long-term absorbed-method goal is not only repo-owned orchestration around
vendor algorithm code. Absorbed methods are expected to move toward fully
repo-owned implementations of the algorithm module behavior, state-dict
reconstruction, and merge/export semantics.

Rationale:

- Otherwise the repo still depends on vendor algorithm classes for the actual
  method behavior even after the runtime/orchestration seams have been brought
  in-house
- The term "absorbed method" becomes misleading if the algorithm core remains a
  permanent vendor runtime dependency
- Full repo-owned implementations make future refactors, diagnostics,
  compatibility work, and method-specific evolution easier to reason about

Trade-off:

- A small repo-owned runtime around vendor modules can still be a useful
  migration waypoint, but it should be documented as incomplete rather than as
  the fully realized absorbed-method end state

### 6. Grouping remains parameter-native and consumes returned trainable refs

Optimization grouping will continue to consume repo-owned
`AdapterTrainableParameterRef` instances with provenance and will not gain
adapter-method-specific module grouping behavior in this change.

Rationale:

- The repo's grouping layer is already built around parameter groups, which is
  the correct lowest granularity for optimizer construction
- Adapter methods may care about modules when realizing adapter state, but the
  grouping layer only needs the parameters they returned and enough provenance
  to label/train them
- Different training consumers may need different target-resolution depth, but
  they are still grounded in the same original model structure
- This keeps module-targeting work separate from optimizer-grouping work

Alternatives considered:

- Teach grouping to operate on module targets directly
  - Rejected because modules are an upstream targeting concern, not the final
    optimizer granularity

### 7. The first absorbed method will reuse existing export and loaded-runtime seams

The `loha` runtime will participate in the existing repo-owned adapter export,
loaded-runtime, and merge request seams instead of defining its own separate
save/load/merge boundary.

Rationale:

- The current repo-owned seams already capture the desired ownership split
- Reusing them validates that the absorbed-method path fits the architecture
  without special lifecycle contracts
- It keeps `PeftMode` as the training-side owner and leaves method behavior in
  the runtime implementation

Alternatives considered:

- Add a new LyCORIS-specific save/load/merge surface
  - Rejected because it would fork the adapter lifecycle contract without a
    clear need

## Risks / Trade-offs

- [Module-target selection may be designed too narrowly for only `loha`] →
  Keep the shared target payload minimal and prove it with one concrete method
  before extracting broader shared concepts
- [The first method may need naming conventions that are not yet explicit] →
  Introduce one repo-owned target-to-adapter naming helper and make both build
  and from-weights use it
- [Vendor module reconstruction may expose awkward edge cases] → Keep vendor
  wrapper APIs out of the repo contract and adapt only the module-level
  capabilities the repo-owned runtime actually needs
- [A migration spike could be mistaken for a fully absorbed method] → State
  explicitly that a repo-owned runtime around vendor module classes is only a
  migration waypoint and add follow-up tasks for replacing vendor algorithm
  dependencies
- [Config direction could drift back toward adapter-owned target policy] →
  Keep module-target configuration under optimization ownership and leave
  adapter-method settings focused on method behavior
- [The first slice could be misread as defining all absorbed methods as
  module-only] → State explicitly that `loha` is the first module-resolved
  consumer and leave finer-grained binding semantics for later slices

## Migration Plan

1. Extend the adapter target payload and optimization-side adapter target
   resolution from component roots to concrete module targets.
2. Implement a repo-native `loha` runtime that consumes those resolved targets
   and returns repo-owned trainable parameter refs.
3. Register `loha` as a repo-owned adapter method and wire `PeftMode` through
   the existing build/from-weights/merge seams.
4. Replace the transitional vendor `LohaModule` runtime dependency with a
   fully repo-owned `loha` implementation once the seam has been proven.
5. Add focused tests for module targeting, `loha` runtime construction,
   trainable-ref provenance, from-weights reconstruction, and merge/export
   behavior.
6. Use the `loha` spike to evaluate what, if anything, should later become a
   shared absorbed-LyCORIS helper layer.

Rollback strategy:

- Revert the new module-targeting path and `loha` registration while preserving
  the already-settled adapter-system runtime seams from the completed
  adapter-system rework

## Open Questions

- The shared target-resolution path is becoming module-capable, with `loha` as
  the first consumer that exercises that richer shape end-to-end. How much of
  that evolution should land in one pass versus follow-up cleanups after the
  first method spike?
- When the next absorbed method needs parameter-granular binding rather than
  only module-resolved realization, what additional shared target fields are
  actually proven necessary?
- Does the first repo-owned `loha` path need a new user-facing method-selection
  config surface right away, or can it temporarily ride the existing
  compatibility shell while the broader method-config direction is still being
  settled?
- What naming helper shape is sufficient for the first build/from-weights
  round-trip without prematurely standardizing all future absorbed methods?
- For `loha` specifically, what parts of the current vendor `LohaModule`
  behavior should be copied directly into a repo-owned implementation versus
  intentionally reshaped to better fit the repo's runtime contracts?
