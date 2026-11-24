## MODIFIED Requirements

### Requirement: Optimization-owned adapter module targeting
The adapter training architecture SHALL treat optimization as the owner of
concrete target-selection policy derived from accepted model structure.
Accepted adapter behavior SHALL consume already-resolved targets when realizing
adapter state and relationships.

#### Scenario: Resolving adapter module targets
- **WHEN** an accepted adapter participant is materialized for training
- **THEN** optimization MUST resolve the concrete target modules in scope under the authored selection and current authority-qualified model projection
- **AND** the accepted adapter realization request MUST pass those resolved targets to the adapter runtime

#### Scenario: Preventing adapter-owned target discovery
- **WHEN** an absorbed adapter method such as `loha` is instantiated
- **THEN** the repo-owned runtime MUST NOT depend on vendor-owned target discovery, presets, or regex/module scanning as the repo contract

#### Scenario: Absorbed methods move toward fully repo-owned algorithm implementations
- **WHEN** an absorbed adapter method such as `loha` is brought into the repo
- **THEN** the intended end state MUST be a fully repo-owned implementation rather than a permanent dependency on vendor algorithm module classes
- **AND** any intermediate wrapper MUST remain explicitly transitional

#### Scenario: First slice does not define all absorbed methods as module-only
- **WHEN** the first module-resolved slice lands
- **THEN** it MUST prove richer resolved targets through `loha`
- **AND** it MUST NOT forbid later methods from requiring parameter-granular binding

### Requirement: Parameter-native optimization grouping remains the training boundary
Optimization grouping SHALL remain parameter-native even when adapter targeting
is module-resolved. Accepted adapter realization SHALL return repo-owned
trainable parameter refs with provenance for optimization to group.

#### Scenario: Building optimizer groups from adapter runtime output
- **WHEN** optimization groups adapter trainables
- **THEN** it MUST consume repo-owned trainable parameter refs with provenance
- **AND** it MUST NOT require module-level grouping primitives or adapter-method internals as the optimizer boundary

#### Scenario: Adapter methods expose trainables after module realization
- **WHEN** adapter state has been realized against module targets
- **THEN** the adapter result MUST expose trainable parameter refs that optimization can group without reading method-specific internals

### Requirement: First absorbed `loha` method fits the repo-owned adapter runtime seams
The first absorbed `loha` method SHALL fit the repo-owned adapter runtime,
loaded-runtime, merge, export, and accepted capability seams.

#### Scenario: Building a repo-native `loha` runtime
- **WHEN** the selected adapter method is `loha`
- **THEN** the repo-owned runtime MUST build `loha` state only from the resolved targets supplied in the accepted realization request

#### Scenario: Fully absorbed `loha` implementation
- **WHEN** `loha` is considered fully absorbed
- **THEN** the implementation MUST own algorithm behavior, state reconstruction, merge, and export in repo-owned code
- **AND** it MUST NOT require the vendored LyCORIS `LohaModule` class as its steady-state runtime

#### Scenario: Reconstructing `loha` from weights
- **WHEN** weights are loaded for continuation, merge, or inference-style setup
- **THEN** the repo-owned runtime MUST reconstruct against accepted resolved targets
- **AND** it MUST participate through `LoadedAdapterRuntime` or its accepted successor rather than a vendor-owned wrapper contract

#### Scenario: Export and merge use repo-owned seams
- **WHEN** the runtime exports or merges adapter state
- **THEN** it MUST use the repo-owned adapter product and merge/fold transition seams
- **AND** Trainer/pipeline orchestration MUST own lifecycle timing rather than `AdapterMode`
