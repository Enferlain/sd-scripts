## Purpose

Define the authored meaning retained after strategy fulfillment and the
verified, prepared executable state consumed by one Trainer engine.

## ADDED Requirements

### Requirement: Semantic acceptance and executable readiness remain distinct
Successful strategy fulfillment SHALL establish the accepted meaning and
run-specific obligations without claiming that unavailable concrete evidence
or preparation results already exist. Training or capability execution SHALL
begin only after every obligation required by its readiness checkpoint is
fulfilled against current authority state and the resulting executable state
is published.

#### Scenario: Accepted participant is deliberately unbound
- **WHEN** the contract permits a participant to materialize after strategy fulfillment
- **THEN** the arrangement MAY be semantically accepted while that participant remains explicitly unready
- **AND** no operation requiring it may execute until its materialization and preparation obligations are fulfilled

#### Scenario: Prepared candidate fails final verification
- **WHEN** preparation returns candidate runtime state that violates the current run-specific obligations
- **THEN** the candidate MUST NOT become executable or visible as current prepared state

### Requirement: Accepted execution survives the authoring object
The accepted run arrangement SHALL retain the selected implementations,
configuration, dependencies, operations, capability behavior, state authority,
and permissions required to execute the authored training meaning without an
ordinary runtime dependency on the authored strategy object.

#### Scenario: Authoring object is released
- **WHEN** strategy fulfillment succeeds and the authored strategy object is no longer retained
- **THEN** the arrangement MUST still support materialization, preparation, training, selected capabilities, persistence, and restoration
- **AND** ordinary execution MUST NOT call back into the authoring object to recover static choices

### Requirement: Accepted execution has explicit structure and effects
The arrangement SHALL describe executable operations or regions, their
dependencies, inputs, outputs, readiness requirements, declared effects,
observations, failure meanings, and authority. It SHALL NOT be only an
all-purpose lifecycle object whose methods hide those meanings.

#### Scenario: Maintained operations are composed
- **WHEN** a maintained strategy requires representation, conditioning, objective, predictor, and loss behavior
- **THEN** the accepted arrangement MUST preserve their required order and data dependencies
- **AND** the Trainer MUST NOT reconstruct that model-family anatomy

#### Scenario: Execution representation changes internally
- **WHEN** the same accepted meanings are implemented as a graph, regions, a schedule, lowered Python, or a hybrid
- **THEN** the observable dependencies, effects, authority, and results MUST remain conformant

### Requirement: Static choices leave the hot path
Implementation selection, static configuration, wiring, dependency resolution,
and authority approval SHALL be established before ordinary repeated execution.
The hot path SHALL receive only genuinely changing inputs, execution
coordinates, and current prepared runtime state.

#### Scenario: Ordinary training step executes
- **WHEN** the Trainer advances a prepared standard run
- **THEN** the step MUST NOT rediscover selected features or ask the authoring object which implementation to call

### Requirement: The standard profile retains generic Trainer mechanics
Under the standard execution/ownership profile, the Trainer engine SHALL retain
time, accumulation, synchronization, backward, clipping, optimizer and
scheduler advancement, zeroing, triggers, observation, interruption, and
cleanup. Accepted operations SHALL perform only their declared computation and
effects.

#### Scenario: Standard operation produces optimization input
- **WHEN** accepted model/objective operations produce the information required for optimization
- **THEN** the Trainer MUST apply the accepted standard optimization policy and mechanics
- **AND** neither the operation nor the authored strategy may also perform the same retained action

### Requirement: Execution results are not diffusion-shaped
The execution contract SHALL distinguish computation outputs, values consumed
by optimization, accounting values, observations, and declared state effects
without requiring one loss producer, one differentiable tensor, one
advancement sequence, or diffusion timestep fields.

#### Scenario: Diffusion objective reports timesteps
- **WHEN** a selected diffusion objective produces timestep observations
- **THEN** those values MUST be available to accepted objective, logging, or adaptive-state consumers
- **AND** non-diffusion arrangements MUST NOT provide placeholder timesteps

#### Scenario: Accounting and backward values differ
- **WHEN** the value reported for accounting differs from the value consumed by accepted loss modification or backward
- **THEN** the execution result MUST preserve the distinction explicitly

### Requirement: Custom structured execution preserves the accepted boundary
An author SHALL be able to replace maintained internal decomposition with a
custom structured operation while still satisfying the active contract's
inputs, outputs, effects, readiness, failures, and standard Trainer ownership.

#### Scenario: Custom computation uses standard optimization
- **WHEN** a research operation changes model/objective computation but not backward or advancement ownership
- **THEN** it MUST be accepted and executed through the standard profile without whole-Trainer mutation access

### Requirement: Imperative regions receive only accepted authority
An imperative runtime region SHALL declare the state and actions it requires.
Strategy fulfillment SHALL grant only authority supported by the selected
execution/ownership profile. Undeclared actions and effects SHALL be rejected
or prevented.

#### Scenario: Region requests backward authority
- **WHEN** an imperative region requests control over backward or optimizer advancement
- **THEN** strategy fulfillment MUST require a profile or extension that explicitly grants and constrains that authority
- **AND** the standard Trainer MUST stop owning exactly the transferred action for that region

#### Scenario: Region requests unsupported access
- **WHEN** a region requests unrestricted run state or a retained Trainer action not granted by its profile
- **THEN** strategy fulfillment MUST reject the arrangement before execution

### Requirement: Execution uses current prepared state
An operation or region SHALL execute only while every required participant
route, relationship, optimization runtime, and capability state satisfies its
accepted readiness and freshness dependencies.

#### Scenario: Required route becomes stale
- **WHEN** an accepted transition invalidates a route required by the execution structure
- **THEN** execution MUST stop using that route until an accepted current replacement is published
