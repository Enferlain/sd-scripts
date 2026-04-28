# adapter-system Specification

## Purpose
TBD - created by archiving change adapter-system-rework. Update Purpose after archive.
## Requirements
### Requirement: Optimization-owned adapter targeting
The adapter-training architecture SHALL treat optimization as the owner of
model-side targeting policy for adapter training.

#### Scenario: Selecting adapter effect targets
- **WHEN** an adapter training run is prepared
- **THEN** optimization MUST decide which original-model targets are in scope

#### Scenario: Preventing adapter-owned targeting policy
- **WHEN** an adapter type is instantiated for training
- **THEN** the adapter system MUST consume already-resolved target information
- **AND** it MUST NOT become the owner of targeting policy or selector grammar

### Requirement: Adapter-system realization layer
The adapter system SHALL realize adapter types against resolved original-model
targets and expose the runtime/training information needed for adapter
training.

#### Scenario: Instantiating an adapter type
- **WHEN** `AdapterMode` prepares adapter training with resolved targets
- **THEN** the adapter system MUST instantiate the selected adapter type in the
  context of those targets

#### Scenario: Returning adapter-training information
- **WHEN** the adapter system finishes instantiation
- **THEN** it MUST expose the concrete adapter runtime and the information
  needed for training-side orchestration and optimization handoff

### Requirement: AdapterMode owns the training-side adapter path
The adapter-training architecture SHALL keep `AdapterMode` as the training-side
owner for adapter runs.

#### Scenario: Adapter training orchestration
- **WHEN** an adapter training run is executed
- **THEN** `AdapterMode` MUST remain the owner of the training-side adapter
  lifecycle

#### Scenario: Separating optimization from training ownership
- **WHEN** optimization participates in an adapter run
- **THEN** optimization MUST NOT become the training-side owner of adapter
  orchestration

### Requirement: Explicit adapter-type configuration
The adapter-training architecture SHALL allow adapter-type-specific
configuration and MUST NOT require all adapter types to fit a fake universal
shared config surface.

#### Scenario: Adapter types with different settings
- **WHEN** two adapter types require materially different settings
- **THEN** the configuration model MUST allow those settings to be expressed
  explicitly without smuggling them through unrelated shared fields

#### Scenario: Keeping optimization policy separate from adapter config
- **WHEN** targeting or grouping behavior is configured
- **THEN** that policy MUST remain owned by optimization rather than being
  folded into adapter-type settings

### Requirement: Adapter persistence belongs to the adapter-training path
The architecture SHALL treat adapter save/load behavior as part of the
adapter-training path rather than as a variant of full fine-tune model
persistence.

#### Scenario: Saving adapter training outputs
- **WHEN** an adapter training run saves learned state
- **THEN** the save behavior MUST be owned on the training side by the adapter
  path through `AdapterMode`

#### Scenario: Distinguishing adapter persistence from fine-tune persistence
- **WHEN** the system defines adapter save/load behavior
- **THEN** it MUST treat the learned subject as augmentation state relative to
  a model context rather than as the model itself

### Requirement: Broad adapter-system applicability
The adapter-training architecture SHALL be designed broadly enough that current
LyCORIS or built-in diffusion adapter shapes do not define the full meaning of
an adapter.

#### Scenario: Supporting future adapter shapes
- **WHEN** a future adapter type differs substantially from current diffusion
  adapter patterns
- **THEN** the architecture MUST evaluate it against the general adapter
  framework rather than assuming current adapter shapes are universal

#### Scenario: Avoiding current-shape lock-in
- **WHEN** runtime or config boundaries are designed
- **THEN** they MUST avoid hard-coding assumptions that only fit today's
  LyCORIS or built-in adapter examples

### Requirement: Optimization-to-adapter boundaries stay contract-based
Optimization-owned adapter policy SHALL depend on repo-owned adapter-facing
contracts and returned metadata rather than concrete adapter-method internals.

#### Scenario: Building optimizer groups for adapter training
- **WHEN** optimization consumes adapter-training runtime information
- **THEN** it MUST operate on explicit adapter-facing contracts, provenance, or
  metadata
- **AND** it MUST NOT depend on concrete built-in adapter class names,
  method-local attribute names, or method-specific naming quirks as its
  grouping interface

#### Scenario: Preserving future adapter breadth in optimization
- **WHEN** a new adapter method is introduced
- **THEN** optimization MUST NOT require new branches over that method's
  internal runtime shape unless the adapter-facing contract itself changes

### Requirement: Grouping code integrates as grouping
Optimization grouping implementation SHALL treat grouping as a real owned
concern, but it MUST NOT become a holding area for unrelated config parsing,
compatibility-policy translation, or adapter-method internals that bypass the
repo-owned grouping boundary.

#### Scenario: Adding adapter grouping behavior
- **WHEN** adapter-specific grouping behavior is added under optimization
  ownership
- **THEN** the implementation MAY live in grouping code if it is genuinely part
  of grouping policy or group construction
- **AND** it MUST NOT be expressed through compatibility fields or
  adapter-layer-owned optimization logic

#### Scenario: Evaluating grouping-module changes
- **WHEN** new behavior is placed in grouping code
- **THEN** it MUST integrate through explicit grouping concepts and contracts
- **AND** it MUST NOT park loosely related config-loading or adapter-specific
  implementation details there just because the file is nearby

### Requirement: Production helpers stay typed and production-first
Typed production helpers SHALL continue to accept the specific runtime/config
objects they are designed for, and test convenience SHALL NOT redefine those
interfaces into loose duck-typed contracts.

#### Scenario: Testing typed optimization helpers
- **WHEN** tests exercise typed production helpers
- **THEN** the tests MUST prefer realistic typed inputs or focused test
  builders over weakening production interfaces to accept generic attribute
  bags

#### Scenario: Evolving adapter-path helper signatures
- **WHEN** a helper already has a typed repo-owned input contract
- **THEN** production code MUST NOT loosen that contract solely to accommodate
  lightweight fixtures

### Requirement: AdapterMode remains explicit orchestration
`AdapterMode` SHALL remain the training-side orchestration owner for adapter runs,
and its lifecycle steps MUST stay explicit rather than hiding adapter-state
mutation inside unrelated preparation stages.

#### Scenario: Preparing optimizer parameters for adapter training
- **WHEN** `AdapterMode` builds optimizer inputs for an adapter run
- **THEN** it MUST keep adapter-trainability and lifecycle transitions explicit
- **AND** it MUST NOT rely on hidden mutation inside optimizer-construction
  steps when that mutation is logically part of an earlier or separate phase

#### Scenario: Reviewing mode-layer responsibilities
- **WHEN** mode-layer code changes adapter state during orchestration
- **THEN** that state change MUST be visible as a deliberate lifecycle action
  rather than as a side effect of a helper whose primary job is something else

