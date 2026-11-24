## ADDED Requirements

### Requirement: Governed PEFT realization resolves authored adapter target intent
An authored strategy that selects PEFT treatment SHALL declare semantic target
intent through the maintained PEFT integration. Governed PEFT realization SHALL
resolve that accepted intent against a coherent, authority-qualified projection
of current host structure. The selected adapter method SHALL consume the
resulting scoped targets rather than independently decide which host structure
the strategy meant to affect.

#### Scenario: Resolving adapter module targets
- **WHEN** an accepted adapter participant is materialized for training
- **THEN** governed PEFT realization MUST resolve the concrete target modules in scope under the authored intent and current authority-qualified host projection
- **AND** the adapter realization request MUST pass those resolved targets to the selected method
- **AND** adapter-derived optimization subjects MUST come from the trainable parameter references returned after adapter realization

#### Scenario: Preventing method-owned target discovery
- **WHEN** an absorbed adapter method such as `loha` is instantiated
- **THEN** the method MUST NOT use vendor-owned discovery, presets, regex traversal, or independent model scanning as the repository targeting contract
- **AND** method-specific constraints MAY reject an otherwise resolved target but MUST NOT silently replace the authored target intent

#### Scenario: Absorbed methods move toward fully repo-owned algorithm implementations
- **WHEN** an absorbed adapter method is brought into the repository
- **THEN** its intended end state MUST be a fully repo-owned implementation rather than a permanent dependency on vendor algorithm module classes
- **AND** any intermediate wrapper over vendor algorithm classes MUST remain explicitly transitional

#### Scenario: Future methods may require finer targets
- **WHEN** a supported adapter method requires parameter-granular or another richer target form
- **THEN** the shared target model and governed PEFT realization MUST permit that form without redefining all adapter methods as module-only

### Requirement: Overlapping adapter target assignments require explicit support
The maintained PEFT integration SHALL NOT silently define how several adapter
participants or methods compose over overlapping host targets. A strategy that
requests such overlap SHALL select accepted coordination behavior covering the
applicable composition, ordering, activation, lifecycle, persistence, and
failure semantics.

#### Scenario: Strategy requests unsupported mixed-method overlap
- **WHEN** an authored strategy assigns several adapter methods to overlapping host targets without accepted coordination behavior
- **THEN** strategy fulfillment MUST reject that arrangement before adapter realization
- **AND** ordinary target resolution MUST NOT invent precedence or merge the assignments

## MODIFIED Requirements

### Requirement: Repo-owned adapter target provenance for module-resolved methods
Module-resolved adapter realizations SHALL receive repo-owned target provenance
that ties each resolved target to the authored target intent, accepted host and
component identity, source revisions, and the shared policy-neutral target-ref
model.

#### Scenario: Passing module-resolved targets into adapter runtime
- **WHEN** governed PEFT realization constructs an adapter realization request
- **THEN** each resolved target MUST include the scoped live module projection and stable provenance needed to identify its participant, component, and component-local path
- **AND** each module-resolved target MUST expose or wrap a shared target ref with kind `module`
- **AND** the shared target ref MUST include the target module type and source revisions

#### Scenario: Preserving provenance through runtime realization
- **WHEN** an adapter method realizes adapter state against resolved targets
- **THEN** it MUST retain enough provenance to tie returned trainable parameter refs back to the original resolved module targets
- **AND** repo-owned adapter trainable refs MUST preserve source target provenance when the method can identify the source target

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
loaded-runtime, merge, export, and accepted PEFT-integration and operation
seams.

#### Scenario: Building a repo-native `loha` runtime
- **WHEN** the selected adapter method is `loha`
- **THEN** the repo-owned runtime MUST build `loha` state only from the resolved targets supplied in the accepted realization request

#### Scenario: Fully absorbed `loha` implementation
- **WHEN** `loha` is considered fully absorbed
- **THEN** the implementation MUST own algorithm behavior, state reconstruction, merge, and export in repo-owned code
- **AND** it MUST NOT require the vendored LyCORIS `LohaModule` class as its steady-state runtime

#### Scenario: Reconstructing `loha` from weights
- **WHEN** weights are loaded for adapter-artifact continuation, initialization, merge, or inference-style setup
- **THEN** the repo-owned runtime MUST reconstruct against accepted resolved targets
- **AND** it MUST participate through `LoadedAdapterRuntime` or its accepted successor rather than a vendor-owned wrapper contract

#### Scenario: Export and merge use repo-owned seams
- **WHEN** the runtime exports or merges adapter state
- **THEN** it MUST use the repo-owned adapter product and merge/fold transition seams
- **AND** Trainer/pipeline orchestration MUST own lifecycle timing rather than `AdapterMode`

### Requirement: Method-local adapter config owns runtime settings translation
Authored strategy construction SHALL select an adapter method from the
strategy's declared supported set. The forward PEFT config surface MAY provide
that explicit selection as strategy-construction input and SHALL translate the
matching method-local subtree into repo-owned runtime settings. Configuration
SHALL NOT independently expand method support or remain a runtime selection
authority after fulfillment.

#### Scenario: Building runtime settings from the active method subtree
- **WHEN** strategy construction selects a supported adapter method using `peft.method`
- **THEN** the matching `peft.<method>` config subtree MUST be translated into `AdapterRuntimeSpec(adapter_type, settings)` or its accepted successor
- **AND** the adapter method MUST consume normalized settings rather than raw Hydra/dataclass config objects

#### Scenario: Method config lives with the method implementation
- **WHEN** a repo-owned adapter method defines user-facing method settings
- **THEN** its config dataclass and runtime-settings translator MUST live with that method implementation
- **AND** the central `PeftConfig` MUST remain a thin typed shell that wires first-class method subtrees into strategy construction

#### Scenario: Dynamic legacy adapter args are not a forward settings surface
- **WHEN** a forward PEFT config supplies method settings
- **THEN** those settings MUST be expressed through the active method subtree
- **AND** `peft.adapter_args` MUST NOT be accepted as a runtime settings source

### Requirement: Continuation intent defaults to strict continuation
PEFT continuation config SHALL treat `peft.continue_from` as strict
adapter-artifact continuation unless the user explicitly requests another
supported adapter initialization mode. Strict adapter-artifact continuation
SHALL NOT imply exact same-run restoration or reuse of the source run's
participant identity. Strategy construction SHALL translate these config inputs
into authored continuation intent before fulfillment; they SHALL NOT act as a
later runtime capability request.

#### Scenario: Continuing from an adapter artifact without an explicit mode
- **WHEN** `peft.continue_from` is set
- **AND** `peft.continue_mode` is unset
- **THEN** the run MUST enforce strict adapter-domain compatibility with that artifact
- **AND** a new run MUST establish a new adapter participant incarnation with lineage to the loaded artifact and its source state

#### Scenario: Initializing from an artifact with changed settings
- **WHEN** a user wants the current method config to define the run while loading values from an existing adapter artifact
- **THEN** the user MUST explicitly select the non-strict initialization mode
- **AND** the result MUST preserve provenance without claiming runtime continuity

#### Scenario: Restoring the exact same run
- **WHEN** the system resumes the same logical run and must preserve its adapter participant identity and continuation state
- **THEN** it MUST use the exact runtime-restoration contract rather than infer restoration from `peft.continue_from`

### Requirement: Adapter target refs remain compatible during migration
During the bounded migration to shared authority-qualified target references,
adapter realization callers MAY expose existing adapter target fields alongside
the shared refs. That compatibility surface SHALL remain transitional and SHALL
NOT establish semantic targeting ownership, durable identity, or permission to
inspect optimization grouping internals.

#### Scenario: Existing adapter runtime reads target fields
- **WHEN** a transitional adapter runtime reads `component`, `component_key`, `path`, or `module` from an adapter resolved target
- **THEN** those fields MUST remain consistent with the accompanying accepted target reference and revision-pinned projection
- **AND** the runtime MUST NOT treat those fields as an independent canonical binding or targeting policy

### Requirement: Adapter component scope derives from declared loaded components
Governed adapter target resolution SHALL derive candidate top-level component
scope from family-declared loaded components and accepted participant context
rather than from the fixed `text_encoders/vae/denoiser` loader tuple.

#### Scenario: Resolving targetable top-level components
- **WHEN** accepted adapter target intent is resolved
- **THEN** governed PEFT realization MUST evaluate the family-declared loaded components and their targeting-relevant semantics to determine which top-level components are in scope
- **AND** it MUST NOT assume the only possible top-level component buckets are text encoders, a VAE, and one denoiser

#### Scenario: Future family declares non-standard targetable components
- **WHEN** a model family declares targetable top-level components beyond the current diffusion backbone examples
- **THEN** adapter target intent and resolution MUST represent and expand those components without adding universal Trainer or loader slots

## REMOVED Requirements

### Requirement: Mixed-method overlap semantics remain out of scope for this slice

**Reason**: The governing architecture must state what happens when overlap is
requested rather than inherit an old spike-local deferral. It need not make
mixed-method overlap part of the ordinary maintained PEFT profile.

**Migration**: Reject overlapping adapter assignments unless the strategy
selects explicit accepted coordination behavior defining their composition,
ordering, activation, lifecycle, persistence, and failure semantics.

### Requirement: Optimization-owned adapter module targeting

**Reason**: Adapter attachment targets express strategy-authored PEFT intent and
must be resolved before the resulting adapter parameters exist as optimization
subjects. Making optimization the semantic targeting owner mixes adapter
realization with later parameter grouping.

**Migration**: Author semantic target intent through the maintained PEFT
integration, resolve it through governed PEFT realization against a coherent
authority-qualified host projection, and pass the returned trainable parameter
references to Trainer-owned optimization grouping.
