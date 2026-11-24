## MODIFIED Requirements

### Requirement: Adapter-system realization layer
The adapter system SHALL realize accepted adapter behavior against already
resolved original-model targets and SHALL return typed participant,
relationship, trainable-state, persistence, and lifecycle results required by
the accepted arrangement and Trainer-owned mechanisms.

#### Scenario: Instantiating an adapter type
- **WHEN** the pipeline materializes an authored adapter participant with resolved targets
- **THEN** the adapter system MUST instantiate the selected adapter implementation in the context of those targets
- **AND** construction MUST NOT make the adapter system the run binding or optimization authority

#### Scenario: Returning adapter-training information
- **WHEN** adapter realization succeeds
- **THEN** it MUST expose the concrete adapter state and typed transition/projection information needed for authority acceptance and optimization realization

### Requirement: Adapter persistence belongs to the adapter-training path
The architecture SHALL treat adapter artifact save/load behavior as an
adapter-domain product capability rather than as a variant of full-model
persistence or a responsibility of `AdapterMode`. Trainer/persistence
infrastructure SHALL own timing, consistency boundaries, destination, retention,
and publication coordination.

#### Scenario: Saving adapter training outputs
- **WHEN** an accepted adapter product is requested
- **THEN** the adapter capability MUST resolve its semantic state and serialization contribution
- **AND** persistence infrastructure MUST report the actual artifact result

#### Scenario: Distinguishing adapter persistence from fine-tune persistence
- **WHEN** the system defines adapter save/load behavior
- **THEN** it MUST treat the learned subject as augmentation state relative to its declared dependencies and relationships rather than as the host model itself

## ADDED Requirements

### Requirement: Authored adapter intent becomes accepted participants and relationships
An ordinary adapter training strategy SHALL declare adapter participants,
intended host relationships, continuation intent, and selected behavior before
runtime materialization. Adapter construction, target resolution, attachment,
activation, replacement, merge/fold, and persistence SHALL use explicit
accepted transitions and results.

#### Scenario: Injected adapter is attached
- **WHEN** adapter state is materialized and its host targets resolve
- **THEN** attachment MUST transition the accepted adapter/host relationship
- **AND** it MUST NOT create the adapter participant implicitly or hide the effect as arbitrary module mutation

### Requirement: Pipeline coordination replaces mode ownership
Trainer/pipeline orchestration SHALL own when adapter materialization,
attachment, preparation, lifecycle actions, optimization participation,
persistence, and restoration occur. Adapter-domain implementations SHALL own
their specialized mechanics and return typed results. No `TrainingMode`
replacement SHALL receive unrestricted whole-Trainer mutation access.

#### Scenario: Adapter has a specialized post-step action
- **WHEN** a selected adapter requires maximum-norm or another declared post-step behavior
- **THEN** Trainer MUST invoke that capability at its accepted lifecycle point
- **AND** the behavior MUST return explicit state/metrics effects rather than mutate unrelated Trainer state

#### Scenario: Adapter runtime contributes resume state
- **WHEN** exact restoration requires adapter-owned continuation state
- **THEN** the adapter capability MUST contribute that state through the restoration contract
- **AND** it MUST NOT own overall run checkpoint coordination

## REMOVED Requirements

### Requirement: AdapterMode owns the training-side adapter path

**Reason**: The governing architecture dissolves `TrainingMode`; permanent
mode ownership would preserve the parallel runtime authority this change
removes.

**Migration**: Move authored treatment into the strategy, adapter mechanics
behind accepted adapter capability exchanges, and generic lifecycle,
preparation, optimization, persistence, and restoration timing to the
Trainer/pipeline owners defined above.

### Requirement: AdapterMode remains explicit orchestration

**Reason**: Explicit lifecycle semantics remain required, but `AdapterMode` is
not their target owner.

**Migration**: Preserve each meaningful lifecycle transition through accepted
participant/relationship operations and typed capability results coordinated by
Trainer.
