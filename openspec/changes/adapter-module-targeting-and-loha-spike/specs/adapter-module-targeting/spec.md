## ADDED Requirements

### Requirement: Optimization-owned adapter module targeting
The adapter training architecture SHALL treat optimization as the owner of
concrete module targeting derived from shared model structure for adapter runs.

#### Scenario: Resolving adapter module targets
- **WHEN** an adapter training run is prepared
- **THEN** optimization MUST resolve the concrete target modules that are in
  scope for the adapter method
- **AND** `PeftMode` MUST orchestrate passing those already-resolved targets
  into adapter runtime construction
- **AND** the adapter runtime MUST consume those already-resolved targets

#### Scenario: Preventing adapter-owned target discovery
- **WHEN** an absorbed adapter method such as `loha` is instantiated
- **THEN** the repo-owned runtime MUST NOT depend on vendor-owned target
  discovery, presets, or regex/module scanning as the repo contract

#### Scenario: Absorbed methods move toward fully repo-owned algorithm implementations
- **WHEN** an absorbed adapter method such as `loha` is brought into the repo
- **THEN** the intended end state MUST be a fully repo-owned implementation of
  the method behavior rather than a permanent runtime dependency on vendor
  algorithm module classes
- **AND** any intermediate repo-owned runtime that still wraps vendor algorithm
  classes MUST be treated as transitional work rather than the finished
  absorbed-method state

#### Scenario: First slice does not define all absorbed methods as module-only
- **WHEN** this first module-resolved slice lands
- **THEN** it MUST be treated as proving richer resolved targets through
  `loha`
- **AND** it MUST NOT be read as forbidding later absorbed methods from needing
  finer parameter-granular binding

### Requirement: Repo-owned adapter target provenance for module-resolved methods
Module-resolved adapter runtimes SHALL receive repo-owned target provenance
that ties each resolved target back to optimization-owned policy decisions.

#### Scenario: Passing module-resolved targets into adapter runtime
- **WHEN** optimization constructs the adapter build request
- **THEN** each resolved target MUST include the concrete module object and
  stable provenance needed to identify its component and path within the model

#### Scenario: Preserving provenance through runtime realization
- **WHEN** an adapter runtime realizes adapter state against resolved targets
- **THEN** it MUST retain enough provenance to tie returned trainable parameter
  refs back to the original resolved module targets

### Requirement: Parameter-native optimization grouping remains the training boundary
Optimization grouping SHALL remain parameter-native even when adapter targeting
is resolved at the module level.

#### Scenario: Building optimizer groups from adapter runtime output
- **WHEN** optimization groups adapter trainables for optimizer construction
- **THEN** it MUST consume repo-owned trainable parameter refs with provenance
- **AND** it MUST NOT require module-level grouping primitives as the optimizer
  boundary
- **AND** `PeftMode` MUST remain the training-side orchestrator that passes the
  returned trainable refs back into optimization-owned grouping

#### Scenario: Adapter methods expose trainables after module realization
- **WHEN** an adapter runtime finishes realizing adapter state against module
  targets
- **THEN** it MUST expose trainable parameter refs that optimization can group
  without reading adapter-method-specific module internals

### Requirement: First absorbed `loha` method fits the repo-owned adapter runtime seams
The first absorbed `loha` method SHALL fit the existing repo-owned adapter
runtime, loaded-runtime, merge, and export seams.

#### Scenario: Building a repo-native `loha` runtime
- **WHEN** the selected adapter method is `loha`
- **THEN** the repo-owned runtime MUST build `loha` adapter state only from the
  resolved adapter targets supplied in the build request

#### Scenario: Fully absorbed `loha` implementation
- **WHEN** `loha` is considered fully absorbed into the repo-owned adapter
  system
- **THEN** the method implementation MUST own its algorithm module behavior,
  state-dict reconstruction, and merge/export behavior in repo-owned code
- **AND** it MUST NOT require the vendored LyCORIS `LohaModule` class as the
  steady-state runtime dependency

#### Scenario: Reconstructing `loha` from weights
- **WHEN** adapter weights are loaded for rank discovery, merge, or
  inference-style setup
- **THEN** the repo-owned `loha` runtime MUST reconstruct itself against the
  resolved adapter targets supplied in the from-weights request
- **AND** it MUST participate through `LoadedAdapterRuntime` rather than a
  vendor-owned wrapper contract

#### Scenario: Export and merge use repo-owned seams
- **WHEN** the repo-native `loha` runtime saves weights or merges loaded
  weights into model context
- **THEN** it MUST use the repo-owned adapter export and merge request seams
- **AND** `PeftMode` MUST remain the training-side owner of that lifecycle

### Requirement: Mixed-method overlap semantics remain out of scope for this slice
The first module-targeting and `loha` spike SHALL NOT define the final behavior
for assigning different methods across overlapping target groups.

#### Scenario: Later mixed-method targeting behavior
- **WHEN** future absorbed methods need per-group method assignment or overlap
  resolution
- **THEN** that behavior MUST be specified in a later change rather than being
  implicitly standardized by this slice
