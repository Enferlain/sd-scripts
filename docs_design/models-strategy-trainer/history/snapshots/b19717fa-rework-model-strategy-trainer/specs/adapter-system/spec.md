## MODIFIED Requirements

### Requirement: Adapter-system realization layer
The adapter system SHALL realize accepted adapter behavior against already
resolved original-model targets and SHALL return typed participant,
relationship, trainable-state, persistence, and lifecycle results required by
the accepted arrangement and Trainer-owned mechanisms. Repository-maintained
PEFT integration SHALL define the behavior shared across supported methods;
each selected method SHALL provide only its method-specific implementation,
settings, state representation, constraints, and specialized operations.

#### Scenario: Instantiating an adapter type
- **WHEN** the pipeline materializes an authored adapter participant with resolved targets
- **THEN** the adapter system MUST instantiate the selected adapter implementation in the context of those targets
- **AND** construction MUST NOT make the adapter system the run binding or optimization authority

#### Scenario: Returning adapter-training information
- **WHEN** adapter realization succeeds
- **THEN** it MUST expose the concrete adapter state and typed transition/projection information needed for authority acceptance and optimization realization

#### Scenario: Method uses the shared PEFT integration
- **WHEN** an authored strategy selects a repository PEFT method from its declared supported set
- **THEN** common participant, relationship, targeting-input, training-subject, preparation, persistence, and restoration meanings MUST come from the maintained PEFT integration
- **AND** the method implementation MUST supply only the behavior and constraints that differ for that method

### Requirement: Explicit adapter-type configuration
The adapter architecture SHALL allow method-specific configuration and MUST NOT
require all adapter methods to fit a fake universal shared config surface.
Semantic target intent and optimization grouping policy SHALL remain outside
method-local settings.

#### Scenario: Adapter types with different settings
- **WHEN** two adapter methods require materially different settings
- **THEN** the configuration model MUST express those settings explicitly through their method implementations

#### Scenario: Keeping optimization policy separate from adapter config
- **WHEN** adapter targeting or grouping behavior is configured
- **THEN** semantic target intent MUST be authored through the maintained PEFT integration
- **AND** optimization grouping policy MUST remain owned by Trainer optimization
- **AND** neither concern may be smuggled into method-local settings

### Requirement: Optimization-to-adapter boundaries stay contract-based
Trainer optimization SHALL consume repo-owned adapter-facing trainable
references, provenance, and results rather than concrete adapter-method
internals. This boundary SHALL NOT make optimization the owner of semantic
adapter target intent or host-target resolution.

#### Scenario: Building optimizer groups for adapter training
- **WHEN** optimization consumes an accepted adapter realization result
- **THEN** it MUST operate on explicit adapter-facing trainable references, provenance, and metadata
- **AND** it MUST NOT depend on concrete built-in adapter class names, method-local attribute names, or method-specific naming quirks as its grouping interface

#### Scenario: Preserving future adapter breadth in optimization
- **WHEN** a new adapter method is introduced
- **THEN** optimization MUST NOT require new branches over that method's internal runtime shape unless the adapter-facing trainable contract itself changes

## ADDED Requirements

### Requirement: Authored adapter intent becomes accepted participants and relationships
An ordinary adapter training strategy SHALL declare adapter participants,
intended host relationships, semantic target intent, continuation intent, and
selected behavior before runtime materialization. Governed PEFT realization
SHALL resolve target intent against current accepted host structure. Adapter
construction, attachment, activation, replacement, merge/fold, and persistence
SHALL use explicit accepted transitions and results.

An authored strategy that adopts repository-maintained PEFT integration SHALL
declare the method set it currently supports. When strategy construction
includes an adapter participant for a run, it SHALL select that participant's
method from the declared set. Repository availability alone SHALL NOT silently
add support to an existing strategy.

#### Scenario: Injected adapter is attached
- **WHEN** adapter state is materialized and its host targets resolve
- **THEN** attachment MUST transition the accepted adapter/host relationship
- **AND** it MUST NOT create the adapter participant implicitly or hide the effect as arbitrary module mutation

#### Scenario: Undeclared PEFT method is selected
- **WHEN** strategy construction requests a repository PEFT method outside that strategy's declared supported set
- **THEN** strategy fulfillment MUST reject the selection before adapter realization

### Requirement: Adapter products, artifact loading, and exact restoration are distinct
The adapter domain SHALL distinguish publishing an adapter artifact, loading an
adapter artifact to initialize a participant, and restoring exact continuation
state for the same logical run. Shared weight formats, serializers, or storage
infrastructure SHALL NOT collapse these into one save/load lifecycle.

#### Scenario: Publishing an adapter product
- **WHEN** an accepted arrangement requests an adapter artifact product
- **THEN** the product implementation MUST resolve augmentation state, dependencies, relationships, representation, and serialization contributions from coherent accepted state
- **AND** persistence infrastructure MUST own timing, consistency, destination, retention, publication, and the actual artifact result

#### Scenario: Loading an adapter artifact into a new run
- **WHEN** a new run initializes adapter state from an existing artifact
- **THEN** it MUST establish a new adapter participant incarnation and preserve lineage to the artifact and source state
- **AND** it MUST use adapter materialization and initialization semantics rather than claim exact runtime restoration

#### Scenario: Restoring adapter state for the same run
- **WHEN** exact same-run restoration requires adapter-owned continuation state
- **THEN** the adapter runtime MUST contribute and restore that state through the runtime-restoration contract
- **AND** successful adapter product loading alone MUST NOT imply complete runtime restoration

#### Scenario: Product choice is independent of training treatment
- **WHEN** direct participant parameters and PEFT state are trained together
- **THEN** adapter, merged-model, or other artifact support MUST follow explicit product declarations rather than a mutually exclusive adapter-training path

### Requirement: Pipeline coordination replaces mode ownership
Trainer/pipeline orchestration SHALL own when adapter materialization,
attachment, preparation, lifecycle actions, optimization participation,
persistence, and restoration occur. Adapter-domain implementations SHALL own
their specialized mechanics and return typed results. The accepted arrangement
SHALL NOT contain a replacement `TrainingMode` authority or a mode value that
Trainer uses to select training behavior. Adapter-domain implementations SHALL
receive only the accepted inputs and authority required by their own operations.

#### Scenario: Adapter has a specialized post-step action
- **WHEN** a selected adapter requires maximum-norm or another declared post-step behavior
- **THEN** Trainer MUST invoke the accepted operation at its declared lifecycle point, or issue its named capability request when it is genuinely a pipeline-recognized capability
- **AND** the behavior MUST return explicit state/metrics effects rather than mutate unrelated Trainer state

#### Scenario: Adapter runtime contributes resume state
- **WHEN** exact restoration requires adapter-owned continuation state
- **THEN** the accepted adapter runtime MUST contribute that state through the restoration contract
- **AND** it MUST NOT own overall run checkpoint coordination

## REMOVED Requirements

### Requirement: Optimization-owned adapter targeting

**Reason**: Model-side adapter targets express authored PEFT treatment and are
resolved during governed PEFT realization before optimization receives the
resulting trainable parameters. Optimization ownership incorrectly merges two
separate concerns.

**Migration**: Move semantic target intent to strategy authoring through the
maintained PEFT integration, resolve concrete host targets through governed PEFT
realization, and retain parameter trainability, grouping, and advancement under
Trainer optimization.

### Requirement: Adapter persistence belongs to the adapter-training path

**Reason**: Artifact publication, artifact-based initialization, and exact
runtime restoration have different identity, completeness, and result semantics.
The old combined save/load path also incorrectly ties products to a mutually
exclusive training mode.

**Migration**: Use declared adapter product capabilities for publication,
adapter materialization with lineage for new-run artifact loading, and the
runtime-restoration contract for exact same-run continuation.

### Requirement: AdapterMode owns the training-side adapter path

**Reason**: The governing architecture dissolves `TrainingMode`; permanent
mode ownership would preserve the parallel runtime authority this change
removes.

**Migration**: Move authored treatment into the strategy's maintained PEFT
integration, adapter mechanics into accepted domain realization and operations,
and generic lifecycle, preparation, optimization, persistence, and restoration
timing to the Trainer/pipeline owners defined above.

### Requirement: AdapterMode remains explicit orchestration

**Reason**: Explicit lifecycle semantics remain required, but `AdapterMode` is
not their target owner.

**Migration**: Preserve each meaningful lifecycle transition through accepted
participant/relationship operations and typed capability results coordinated by
Trainer.
