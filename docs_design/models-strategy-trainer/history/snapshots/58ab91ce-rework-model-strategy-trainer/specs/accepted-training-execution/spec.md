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
configuration, dependencies, operations, dynamic policies, runtime-state
contributors, capability behavior, state authority, and permissions required
to execute the authored training meaning without depending on the authored
strategy object in its authoring role. Runtime code MAY depend on selected
executable implementations only through their accepted runtime seams and
authority.

#### Scenario: Authoring object is released completely
- **WHEN** strategy fulfillment succeeds and the authored strategy object is not retained in any capacity, including as a selected runtime implementation
- **THEN** the arrangement MUST still support materialization, preparation, training, selected capabilities, persistence, and restoration
- **AND** ordinary execution MUST NOT call back into the authoring interface to recover choices already established during fulfillment

#### Scenario: One object supplies separate authoring and runtime seams
- **WHEN** a selected implementation object participated in strategy authoring and also implements accepted runtime behavior
- **THEN** its runtime seam MUST receive only accepted inputs and scoped state projections and produce only declared outputs, effects, or transition proposals
- **AND** any requested arrangement change MUST be evaluated and published through the applicable run-authority transition path
- **AND** retaining that implementation MUST NOT make the strategy's authoring interface a runtime peer

### Requirement: Accepted execution has explicit structure and effects
The arrangement SHALL describe executable operations or regions, their
dependencies, inputs, outputs, readiness requirements, declared effects,
observations, owned-state definitions and transitions, permitted
authority-governed transitions, failure meanings, and authority. It SHALL NOT
be only an all-purpose lifecycle object whose methods hide those meanings.

#### Scenario: Maintained operations are composed
- **WHEN** a maintained strategy requires representation, conditioning, objective, predictor, and loss behavior
- **THEN** the accepted arrangement MUST preserve their required order and data dependencies
- **AND** the Trainer MUST NOT reconstruct that model-family anatomy

#### Scenario: Execution representation changes internally
- **WHEN** the same accepted meanings are implemented as a graph, regions, a schedule, lowered Python, or a hybrid
- **THEN** the observable dependencies, effects, authority, and results MUST remain conformant

### Requirement: Accepted execution preserves authored dynamic behavior
Completing strategy authoring SHALL NOT freeze behavior whose accepted meaning
depends on changing inputs, coordinates, randomness, observations, or owned
runtime state. The accepted arrangement SHALL retain the selected schedules,
decision policies, owned-state definitions, accepted initialization sources or
rules, state transitions, observation inputs, and restoration contributions
needed to execute that behavior within its accepted bounds.

#### Scenario: Stateful behavior starts without restored state
- **WHEN** an accepted stateful operation becomes ready without state restored from an earlier execution session
- **THEN** its state MUST be initialized from the source or rule preserved by the accepted arrangement
- **AND** initialization that depends on concrete runtime evidence MUST occur only after that evidence satisfies its applicable readiness obligation
- **AND** neither the Trainer nor the former authoring role may invent a different initial value

#### Scenario: Schedule changes behavior during a run
- **WHEN** an accepted operation derives its current behavior from step, epoch, phase, or another accepted coordinate
- **THEN** its current behavior MUST be determined by the accepted schedule semantics for that coordinate
- **AND** the Trainer MUST NOT require the authored strategy to remain active to choose the current behavior

#### Scenario: Observations update adaptive state
- **WHEN** an accepted stateful algorithm uses prior observations to update its owned state and later decisions
- **THEN** those observations and state transitions MUST remain available to that accepted runtime behavior
- **AND** its owned state MUST participate in applicable persistence or exact-restoration exchanges

#### Scenario: Runtime decision stays within accepted bounds
- **WHEN** an operation selects among alternatives using current inputs or owned state
- **THEN** every selectable alternative, required authority, and possible declared effect MUST already be permitted by the accepted arrangement
- **AND** a choice outside those bounds MUST be rejected rather than treated as ordinary dynamic execution

### Requirement: Static choices leave the hot path
Implementation selection, static configuration, wiring, dependency resolution,
and authority approval SHALL be established before ordinary repeated execution.
The hot path SHALL receive only genuinely changing inputs, execution
coordinates, current prepared runtime state, and accepted operation-owned
state. It SHALL NOT be construed to require precomputing, or to prohibit,
accepted schedules, adaptive updates, bounded runtime decisions, or declared
transitions.

#### Scenario: Ordinary training step executes
- **WHEN** the Trainer advances a prepared standard run
- **THEN** the step MUST NOT rediscover selected features or ask the authoring object which implementation to call
- **AND** it MAY invoke accepted behavior whose result and owned state depend on the current runtime inputs and observations

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

#### Scenario: Behavior requests an authority-governed transition
- **WHEN** accepted runtime behavior requests a permitted change to participants, relationships, bindings, preparation, optimization, or capability state
- **THEN** the request MUST pass through the applicable run-authority transition and republication path
- **AND** the operation MUST NOT mutate canonical run state merely because its ordinary computation is dynamic
