## ADDED Requirements

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
- **WHEN** `PeftMode` prepares adapter training with resolved targets
- **THEN** the adapter system MUST instantiate the selected adapter type in the
  context of those targets

#### Scenario: Returning adapter-training information
- **WHEN** the adapter system finishes instantiation
- **THEN** it MUST expose the concrete adapter runtime and the information
  needed for training-side orchestration and optimization handoff

### Requirement: PeftMode owns the training-side adapter path
The adapter-training architecture SHALL keep `PeftMode` as the training-side
owner for adapter runs.

#### Scenario: Adapter training orchestration
- **WHEN** an adapter training run is executed
- **THEN** `PeftMode` MUST remain the owner of the training-side adapter
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
  path through `PeftMode`

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
