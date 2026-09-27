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

#### Scenario: Operation returns an invalid output structure
- **WHEN** a selected operation returns an output that fails its declared result check
- **THEN** execution MUST report a post-invocation failure with possible effects
- **AND** it MUST NOT classify the call as a harmless pre-execution rejection

### Requirement: Accepted execution covers the whole run
The accepted structure SHALL retain work with different lifetimes and
coordinates, including repeated training actions, selected input production,
requested capability work, and permitted run transitions. It SHALL expose the
cross-owner dependencies, handoffs, readiness, effects, and failure boundaries
needed for coordination or restoration. Pre-resolved action-local computation
MAY be part of that structure but SHALL NOT substitute for its run-level
meaning. Selected implementations MAY keep their internal mathematics,
workers, and backend scheduling private.

#### Scenario: Input production advances while no training action runs
- **WHEN** a selected input producer progresses before or independently of a due training action
- **THEN** the run MUST retain its lifecycle, dependency, readiness, handoff, and failure relationship to the consuming action
- **AND** the action's operation wiring MUST NOT be treated as the producer's lifecycle or continuation record

#### Scenario: Capability work follows an accepted trigger
- **WHEN** accepted progress makes a requested capability due
- **THEN** run-level coordination MUST check its current readiness, route its request, and retain its result or failure separately from the training action
- **AND** the selected capability implementation MAY keep its internal work outside the common execution structure

### Requirement: Participant identity and individual uses remain distinct
The accepted structure SHALL distinguish one participant's identity from each
invocation of that participant. Each use SHALL preserve its selected operation,
required current view or route, input/output meaning, ordering dependency, and
gradient requirement where another accepted owner relies on those facts. A
role label or repeated call SHALL NOT create a new participant or silently
change a binding.

#### Scenario: One base is used with and without an adapter
- **WHEN** one accepted base participant supplies a no-gradient teacher use and a differentiable adapted student use in one action
- **THEN** both uses MUST retain the same base participant identity and their distinct view and gradient requirements
- **AND** realization MUST verify that current preparation can provide both uses without unsafe overlapping view changes

#### Scenario: Two views toggle shared mutable adapter state
- **WHEN** accepted uses select incompatible views by changing mutable state on one realization
- **THEN** execution MUST serialize those uses and restore the required view after failure or stop dependent work when restoration is uncertain
- **AND** restoring the view MUST NOT itself mark the failed action or its other state effects as recovered
- **AND** ordinary per-use view selection MUST NOT by itself revise the participant binding

#### Scenario: Frozen base remains on a side-network gradient path
- **WHEN** a frozen base is invoked to propagate loss to a trainable side network
- **THEN** execution MUST preserve that required gradient path without making the base an optimization subject

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

#### Scenario: A state dependency changes
- **WHEN** a participant, binding, route, relationship, or observation-source meaning on which accepted operation-owned state depends changes
- **THEN** dependent execution MUST remain unavailable until the state owner applies an accepted preserve-with-evidence, migrate, reinitialize, or retire rule
- **AND** an ordinary new observation MUST NOT by itself be confused with a change to the observation producer's meaning

#### Scenario: Stateful operation fails after a possible update
- **WHEN** an operation may have consumed an observation or changed its owned state before failing
- **THEN** dependent work MUST stop unless an accepted owner-specific recovery rule establishes valid current state
- **AND** a missing successful result MUST NOT be treated as proof that the state is unchanged

#### Scenario: Stateful operation contributes to exact restoration
- **WHEN** exact same-run continuation depends on accepted operation-owned state
- **THEN** its owner MUST contribute the state, definition and dependency revisions, and relevant RNG, progress, or unfinished-effect status to a coherent runtime snapshot
- **AND** an ordinary trained-model product MUST NOT be treated as that continuation state unless its declared product semantics explicitly include it

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
training-action time and progress, accumulation, synchronization, backward,
clipping, optimizer and scheduler advancement, zeroing, triggers, observation,
interruption, and cleanup. Accepted operations SHALL perform only their
declared computation and effects.

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

#### Scenario: Selected computation has no latent or image batch
- **WHEN** an accepted pixel, video/audio, packed-token, or other non-latent operation supplies its own input and output meaning
- **THEN** the core execution exchange MUST NOT require a VAE, one image tensor, fixed modality axes, or diffusion timestep fields
- **AND** the selected behavior MUST preserve any representation, absence, mask, or logical-example boundaries its accepted consumers need

#### Scenario: Accounting and backward values differ
- **WHEN** the value reported for accounting differs from the value consumed by accepted loss modification or backward
- **THEN** the execution result MUST preserve the distinction explicitly

### Requirement: Input handoff retains identity and admission meaning
The run SHALL coordinate accepted input-provider lifecycle, readiness,
admission, delivery, and failure independently of an action's internal model
computation. Selected input behavior MAY progress on its own clock and own its
live-source traversal, selection, packing, buffers, and continuation state;
cache-production traversal remains with caching coordination. Each
handoff SHALL preserve the work and logical-example identity, provenance,
dependency revisions, consumer representation, and boundaries required by
the selected computation, accounting, or restoration policy. The core exchange
SHALL NOT require universal image, token, epoch, or fixed-batch fields.

#### Scenario: Encoded inputs finish out of order
- **WHEN** an independent producer returns a ready value for an earlier or later work request
- **THEN** training MUST check its exact work identity and current producer and participant dependencies before admitting it
- **AND** readiness alone MUST NOT pair it with another request or imply action consumption

#### Scenario: Input policy changes within accepted bounds
- **WHEN** a selected input policy changes its source mixture or packing stage within alternatives accepted during fulfillment
- **THEN** the provider MUST preserve its owned continuation state and deliver the example identities and boundaries required by the current consumer
- **AND** the run MUST NOT rediscover the authored strategy or rebuild the action merely because the bounded policy advanced

#### Scenario: Provider fails or produces incompatible work
- **WHEN** input is missing, failed, stale, misidentified, or incompatible with the accepted consumer representation
- **THEN** the handoff MUST fail or follow an explicitly accepted wait, recompute, or fallback policy before computation
- **AND** an unchecked blocking read inside an ordinary model operation MUST NOT substitute for run-level failure coordination

#### Scenario: Stateful input provider contributes to exact restoration
- **WHEN** exact same-run continuation depends on a selected input provider that can progress ahead of training
- **THEN** its owner MUST contribute its continuation state, dependency revisions, and the status of produced, ready, handed-off, and unfinished work required by its accepted replay or skip policy
- **AND** an incomplete contribution MUST NOT be reported as an exact recoverable input position

### Requirement: Cross-owner action facts remain correlatable
The run SHALL retain enough accepted work and attempt identity to relate an
input handoff, selected action, operation effects, per-unit optimization
outcomes, progress recording, and snapshot durability without assigning all
of those facts to the input provider or claiming that one action is an atomic
optimizer transaction. G3 defines the result/reporting meaning; G5 chooses
its concrete Python form.

#### Scenario: One input leads to two different unit outcomes
- **WHEN** one admitted input feeds an action whose first optimization unit returns and second unit has an uncertain outcome
- **THEN** the run MUST keep the input and action attempt associated with both separate unit outcomes
- **AND** it MUST NOT mark the entire action as an all-or-nothing update or infer that either unit's parameters numerically changed

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
execution/ownership profile. Each grant SHALL be scoped to its accepted region,
action phase, affected state, and transferred mechanics; the Trainer SHALL
retain every mechanic not transferred for that scope. Fulfillment SHALL reject
undeclared requested authority, and runtime seams SHALL expose only accepted
scoped access and reject undeclared effects observable at their boundaries.
Declarations alone SHALL NOT be treated as proof that arbitrary selected Python
cannot perform a hidden mutation.

#### Scenario: Standard and imperative work coexist in one run
- **WHEN** an accepted run contains ordinary actions and one region with a supported imperative grant
- **THEN** only that region's accepted scope MAY perform the transferred mechanics
- **AND** unrelated actions MUST retain their accepted standard owners without calling back into the authored strategy

#### Scenario: Region requests backward authority
- **WHEN** an imperative region requests control over backward or optimizer advancement
- **THEN** strategy fulfillment MUST require a profile or extension that explicitly grants and constrains that authority
- **AND** the standard Trainer MUST stop owning exactly the transferred action for that region

#### Scenario: Region requests unsupported access
- **WHEN** a region requests unrestricted run state or a retained Trainer action not granted by its profile
- **THEN** strategy fulfillment MUST reject the arrangement before execution

#### Scenario: Two-pass region requests a bounded authority transfer
- **WHEN** a supported extension accepts a region that owns two backward passes, a scoped temporary parameter edit, and intermediate gradient handling while Trainer retains final advancement
- **THEN** each action and phase MUST have one owner, including any intermediate gradient reset
- **AND** fulfillment MUST retain its parameter scope, effects, accumulation constraints, state contributors, clean-handback requirement, and recovery claim
- **AND** governed realization MUST check the current route and backend evidence before the region becomes executable

#### Scenario: Temporary edit cannot be restored with evidence
- **WHEN** the region fails while parameters are temporarily edited or its clean handback cannot be established
- **THEN** conflicting execution and snapshots MUST NOT use the affected live state
- **AND** dependent work MUST stop until an accepted recovery rule establishes valid current state or a coherent prior runtime snapshot is restored
- **AND** neither generic rollback nor blind in-process replay may be claimed

#### Scenario: Exact snapshot is requested during an imperative interval
- **WHEN** the active profile has no accepted mid-region restoration protocol
- **THEN** exact same-run snapshots MUST be limited to coherent quiescent boundaries with unperturbed parameters and all required owner contributions

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

#### Scenario: Stage change affects several owners
- **WHEN** an accepted stage policy changes selected execution, participant or relationship state, optimization runtime, and operation-owned state together
- **THEN** dependent work MUST wait for affected obligations and state mappings to be checked and one coherent current view to be published
- **AND** copying weights or constructing a new optimizer object alone MUST NOT determine participant identity or publish a completed transition

#### Scenario: Old and new paths intentionally overlap
- **WHEN** an authored transition deliberately keeps old and new execution paths active during an accepted interval
- **THEN** the authority MUST publish that coexistence as one coherent current arrangement with each path's applicable dependencies and readiness
- **AND** the interval MUST NOT be treated as a partially published replacement merely because both paths are present
