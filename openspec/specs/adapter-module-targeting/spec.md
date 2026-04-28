# adapter-module-targeting Specification

## Purpose
TBD - created by archiving change adapter-module-targeting-and-loha-spike. Update Purpose after archive.
## Requirements
### Requirement: Optimization-owned adapter module targeting
The adapter training architecture SHALL treat optimization as the owner of
concrete module targeting derived from shared model structure for adapter runs.

#### Scenario: Resolving adapter module targets
- **WHEN** an adapter training run is prepared
- **THEN** optimization MUST resolve the concrete target modules that are in
  scope for the adapter method
- **AND** `AdapterMode` MUST orchestrate passing those already-resolved targets
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
that ties each resolved target back to optimization-owned policy decisions and
the shared optimization target-ref model.

#### Scenario: Passing module-resolved targets into adapter runtime
- **WHEN** optimization constructs the adapter build request
- **THEN** each resolved target MUST include the concrete module object and
  stable provenance needed to identify its component and path within the model
- **AND** each module-resolved target MUST expose or wrap a shared optimization
  target ref with kind `module`
- **AND** the shared target ref MUST include the target module type

#### Scenario: Preserving provenance through runtime realization
- **WHEN** an adapter runtime realizes adapter state against resolved targets
- **THEN** it MUST retain enough provenance to tie returned trainable parameter
  refs back to the original resolved module targets
- **AND** repo-owned adapter trainable refs MUST preserve source target
  provenance when the adapter runtime can identify the source target

### Requirement: Parameter-native optimization grouping remains the training boundary
Optimization grouping SHALL remain parameter-native even when adapter targeting
is resolved at the module level.

#### Scenario: Building optimizer groups from adapter runtime output
- **WHEN** optimization groups adapter trainables for optimizer construction
- **THEN** it MUST consume repo-owned trainable parameter refs with provenance
- **AND** it MUST NOT require module-level grouping primitives as the optimizer
  boundary
- **AND** `AdapterMode` MUST remain the training-side orchestrator that passes the
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
- **AND** `AdapterMode` MUST remain the training-side owner of that lifecycle

### Requirement: Mixed-method overlap semantics remain out of scope for this slice
The first module-targeting and `loha` spike SHALL NOT define the final behavior
for assigning different methods across overlapping target groups.

#### Scenario: Later mixed-method targeting behavior
- **WHEN** future absorbed methods need per-group method assignment or overlap
  resolution
- **THEN** that behavior MUST be specified in a later change rather than being
  implicitly standardized by this slice

### Requirement: Method-local adapter config owns runtime settings translation
The forward adapter config surface SHALL select an adapter method explicitly and
translate the matching method-local config subtree into repo-owned runtime
settings.

#### Scenario: Building runtime settings from the active method subtree
- **WHEN** a PEFT run selects an adapter method with `peft.method`
- **THEN** the matching `peft.<method>` config subtree MUST be translated into
  `AdapterRuntimeSpec(adapter_type, settings)`
- **AND** the adapter runtime MUST consume those normalized settings rather
  than raw Hydra/dataclass config objects

#### Scenario: Method config lives with the method implementation
- **WHEN** a repo-owned adapter method defines user-facing method settings
- **THEN** its config dataclass and runtime-settings translator MUST live with
  that method implementation
- **AND** the central `PeftConfig` MUST remain a thin typed shell that wires
  first-class method subtrees into the PEFT config surface

#### Scenario: Dynamic legacy adapter args are not a forward settings surface
- **WHEN** a forward PEFT config supplies method settings
- **THEN** those settings MUST be expressed through the active method subtree
- **AND** `peft.adapter_args` MUST NOT be accepted as a runtime settings source

### Requirement: Continuation intent defaults to strict continuation
PEFT continuation config SHALL treat `peft.continue_from` as strict
continuation unless the user explicitly requests another supported continuation
mode.

#### Scenario: Continuing from an adapter artifact without an explicit mode
- **WHEN** `peft.continue_from` is set
- **AND** `peft.continue_mode` is unset
- **THEN** the run MUST behave as strict continuation from that artifact

#### Scenario: Initializing from an artifact with changed settings
- **WHEN** a user wants the current method config to define the run while
  loading values from an existing adapter artifact
- **THEN** the user MUST explicitly select the non-strict continuation mode

### Requirement: Adapter target refs remain compatible during migration
Adapter runtime callers SHALL be able to consume existing adapter target fields
while shared target refs are introduced.

#### Scenario: Existing adapter runtime reads target fields
- **WHEN** an adapter runtime reads `component`, `component_key`, `path`, or
  `module` from an adapter resolved target
- **THEN** those fields MUST remain available during the shared target-ref
  migration
- **AND** new target-ref-backed behavior MUST NOT require adapter runtimes to
  inspect optimization grouping internals

