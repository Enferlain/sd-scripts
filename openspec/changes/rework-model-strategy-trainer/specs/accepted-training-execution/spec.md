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
needed for coordination or restoration in a hierarchical whole-run structure.
Pre-resolved action-local computation
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

#### Scenario: Owned state advances without an optimization offer
- **WHEN** an accepted activity updates its state or emits an observation without producing optimization input
- **THEN** its effects, completion, and dependent decisions MUST retain their accepted meaning
- **AND** coordination MUST NOT force that work through an optimizer action or one universal run clock

### Requirement: Equivalent execution preserves required interactions
Prepared execution SHALL preserve the accepted input associations and admission,
computation and gradients under the applicable numerical policy, dynamic
decisions and required observation order, owned-state and external effects,
authority, current-use dependencies, completion and failure facts, and
restoration relationships. Fusion, partitioning, chunking, or another backend
schedule MAY change private work only while those obligations remain satisfied.
Required interactions SHALL NOT have to remain individual runtime dispatches.
Generating Trainer-owned mechanics into an executable SHALL NOT transfer their
authority to selected computation.

#### Scenario: Combined execution preserves a feedback boundary
- **WHEN** preparation combines work whose later decision depends on an earlier observation
- **THEN** the combined execution MUST deliver that observation before the dependent decision and preserve all other required owner interactions
- **AND** equal final values MUST NOT substitute for those interactions

#### Scenario: Combined work loses a known partial outcome
- **WHEN** a proposed lowering cannot preserve a known unit outcome or an accepted capability, transition, admission, or snapshot interaction
- **THEN** that lowering MUST fail readiness for the accepted profile rather than substitute weaker completion or recovery semantics

### Requirement: Differentiated work preserves routing and numerical lifetime
Where another accepted owner relies on it, the description SHALL preserve
outputs and derivative demands, seed information where applicable, subject or
input-path routing, intentional gradient cuts, contribution destinations and
windows, numerical policy, and phase ownership. Governed preparation SHALL
establish the required retained state or supported recomputation, completion,
and safe-release conditions. This SHALL NOT require a project-owned autodiff
engine, public tensor-operation graph, or continuation object for every call.

#### Scenario: A frozen participant transmits a required gradient
- **WHEN** a frozen participant lies on a derivative path to an accepted training subject
- **THEN** prepared execution MUST preserve that path while leaving the frozen participant outside optimization membership
- **AND** unsupported routing MUST fail readiness rather than silently detach the path

#### Scenario: One derivative is ready while another needs earlier state
- **WHEN** a first contribution is ready but another required derivative, recomputation, or device use still relies on earlier numerical state
- **THEN** conflicting mutation MUST wait until that required use completes or is satisfied by a supported isolated representation
- **AND** earlier advancement MUST still satisfy all accepted window, synchronization, clipping, and ordering obligations
- **AND** a revision label alone MUST NOT be treated as retention of numerical contents

#### Scenario: Derivative work crosses an independently coordinated boundary
- **WHEN** another owner requests or completes derivatives outside a closed execution region
- **THEN** the handoff MUST preserve originating invocation, required state, remaining requests, and completion, cancellation, and release rules
- **AND** cancellation or failure MUST NOT release still-used resources or certify uncertain gradients as usable
- **AND** replay or recomputation MUST follow the accepted state, RNG, and effect rule

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
An optimization offer under Trainer-owned mechanics SHALL identify its accepted
source, destination units, and gradient requirements without itself advancing
an optimizer or certifying contribution completion.

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
Production provenance SHALL identify the source states actually used, with
the associations required by the selected admission policy. It MAY describe
one protected state or snapshot, or structured provenance over multiple
states and portions of the produced work; the exchange SHALL NOT require one
state/version per product. Such provenance SHALL NOT by itself authorize
mixed-state production or establish its admissibility.

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

#### Scenario: One consumer joins independently produced inputs
- **WHEN** accepted computation needs several inputs whose producers finish out of order
- **THEN** admission MUST preserve each requested and produced work identity, actual dependencies, and association with that consumer
- **AND** a missing, swapped, or inadmissible member MUST prevent joined computation or follow an explicitly accepted wait, reissue, or fallback policy
- **AND** partial handoffs MUST NOT be silently duplicated or discarded

#### Scenario: Producer dependency changes while work is in flight
- **WHEN** a selected producer finishes work after its source state changes
- **THEN** the result MUST preserve actual production-state provenance rather than merely repeat the dependency requested when work began
- **AND** admission MUST apply the accepted freshness or lag policy before consumer use
- **AND** a detached produced value MUST NOT silently substitute for a required live derivative path

#### Scenario: Produced work spans several source states
- **WHEN** accepted production uses different source states for portions of one product
- **THEN** the handoff MUST preserve structured provenance associating those portions with the states actually used to the extent required by the admission policy
- **AND** admission MUST evaluate that provenance rather than assume one launch or completion version describes the whole product
- **AND** mixed-state work MUST NOT become admissible merely because its provenance is available

### Requirement: Cross-owner action facts remain correlatable
Run coordination SHALL preserve the relationships among accepted work
addresses and invocation/attempts, admitted inputs and actual dependencies,
operation/region effects, addressed unit incarnations and definition revisions,
unit-local outcomes, progress records, and snapshot durability, including after
partial failure. Each owner SHALL supply the facts it owns without transferring
its state ownership. These relationships MAY span owner-held records and SHALL
NOT require one universal result object, event log, step counter, or fixed
input-action-optimizer sequence.

#### Scenario: One input leads to two different unit outcomes
- **WHEN** one admitted input feeds an action whose first optimization unit returns and second unit has an uncertain outcome
- **THEN** the run MUST keep the input and action attempt associated with both separate unit outcomes
- **AND** it MUST NOT mark the entire action as an all-or-nothing update or infer that either unit's parameters numerically changed

#### Scenario: Several input handoffs lead to separate outcomes
- **WHEN** an action consumes several admitted inputs and later only some addressed mechanics complete
- **THEN** run coordination MUST preserve every input-to-attempt association and the known, uncertain, and unattempted outcomes supplied by the responsible owners
- **AND** correlation MUST NOT depend on one input being the action's identity or on every owner sharing one progress clock

### Requirement: Completion and failure follow the accepted activity relationships
Produced, ready, admitted, handed-off, consumed, contributed, advanced,
reported, and durably captured SHALL remain distinct facts wherever accepted
consumers depend on them. Before invocation, rejection SHALL NOT imply selected
work ran. Once selected or backend work begins, the responsible owner SHALL
report known effects and outcomes separately from uncertainty; dependent
unsafe work SHALL stop until an accepted recovery establishes validity.
Missing completion SHALL NOT prove unchanged state, automatic rollback,
all-or-nothing advancement, or replay safety.

#### Scenario: Production completes before a consuming action exists
- **WHEN** an independent activity makes work ready before its consumer is due
- **THEN** its production completion MUST remain meaningful independently of action or optimization completion
- **AND** readiness MUST NOT imply admission, consumption, or update contribution

#### Scenario: Shutdown fails after partial execution failure
- **WHEN** selected work or retained mechanics fail and cleanup also fails
- **THEN** reporting MUST preserve the primary failure, cleanup failures, and separately known effects and outcomes
- **AND** cleanup failure MUST NOT overwrite evidence needed to gate dependent work or recover

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

#### Scenario: A two-pass region returns a produced gradient
- **WHEN** the supported split gives a region backward and temporary-edit authority while Trainer retains final clipping and advancement
- **THEN** Trainer MUST verify restoration of the required current view, accepted gradient readiness, and accounted region effects against the same attempt and current dependencies
- **AND** Trainer MUST NOT perform another standard backward on that handback
- **AND** a failed or uncertain handback MUST NOT authorize retained advancement

#### Scenario: The selected two-pass profile requires a quiescent window
- **WHEN** that profile's entry is requested while the scoped unit has pending contributions from earlier work
- **THEN** entry MUST be rejected before the region's mutation rather than clear those contributions to make the profile appear applicable
- **AND** a different accumulation treatment MUST require its own accepted policy and backend support

#### Scenario: Region fails after restoring its temporary edit
- **WHEN** a failed region restores parameters but other reached effects or its contribution remain incomplete or uncertain
- **THEN** restoration alone MUST NOT make the action successful or authorize advancement
- **AND** continuation MUST require the accepted recovery rule covering all relevant effects

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

### Requirement: Current-use protection spans owners and actual use
Admission, execution, conflicting mutations and publication, and capability
access SHALL share a protocol protecting the required current-state use.
Protection SHALL cover backward, recomputation, transfers, outstanding device
work, and handback verification through retained mechanics wherever those uses
require the same state. Conflicting work SHALL wait, be rejected, or use a
supported isolation policy. Freshness checks alone SHALL NOT substitute for
protection, and this requirement SHALL NOT prescribe a particular lock.

#### Scenario: A conflicting publication follows a successful freshness check
- **WHEN** an admitted use still needs current state and another owner proposes an incompatible edit or publication
- **THEN** coordination MUST prevent that conflict for the required use lifetime
- **AND** checking revisions before invocation MUST NOT permit the state to change unsafely afterward

#### Scenario: Region protection crosses the handback boundary
- **WHEN** retained Trainer mechanics need a region's restored view and produced gradient
- **THEN** protected access MUST continue or transfer without a gap through handback verification and those retained uses
- **AND** capability readers MUST NOT observe an excluded intermediate state

#### Scenario: Partial acquisition or cancellation leaves uncertain state
- **WHEN** acquisition, execution, cancellation, or cleanup may leave affected state uncertain
- **THEN** releasing access MUST NOT make dependent work usable before accepted recovery establishes valid state
- **AND** known outcomes and primary and cleanup failures MUST remain available
- **AND** withholding ordinary use MUST NOT prohibit authorized restoration or replacement needed for recovery

### Requirement: Conformance and execution cost are checked against accepted meaning
Production acceptance SHALL cover materially different standard and granted
arrangements using observable values, identities, state, gradients, effects,
outcomes, and required order rather than prototype class names or private call
sequences. Each applicable case SHALL include positive and rejected or failed
variants; local evidence SHALL NOT certify untested backend or restoration
guarantees. Performance checks SHALL compare semantically matched execution
with equivalent direct Python and, where comparable, current-loop scope, isolate
framework overhead with cheap workloads, and apply documented regression and
scaling budgets for the compared scope and environment. Required dynamic
decisions and safety checks SHALL remain in the measured scope. New required
coordination or guards SHALL be present in both compared paths and receive
a separately justified budget rather than be omitted to fit an earlier scope.
Dispatch cost SHALL scale with due work and its relevant dependencies rather
than with unrelated dormant arrangement size. Numerical tolerances and
reference fixtures SHALL be acceptance-policy choices for the applicable
change, not permanent semantic requirements of accepted execution.

#### Scenario: A faster lowering omits a required runtime guard
- **WHEN** a candidate benchmark removes admission, freshness, protected access, or a dynamic decision required by that arrangement
- **THEN** the timing MUST NOT qualify as a conformant matched comparison
- **AND** additional required coordination MUST have an equivalent reference and separately justified budget

#### Scenario: A cost comparison adds required coordination
- **WHEN** a candidate performs required coordination not covered by an earlier benchmark scope
- **THEN** acceptance MUST use an equivalent reference and a separately justified budget for that added scope
- **AND** the result MUST NOT claim that the earlier measurement covers the added coordination

#### Scenario: Dormant work and retained history increase
- **WHEN** a prepared arrangement has more dormant work but unchanged due work and relevant dependencies
- **THEN** ordinary dispatch MUST NOT scan the whole arrangement by convention
- **AND** fixed queue, history, and gradient-window bounds MUST yield bounded retained framework state over elapsed attempts
- **AND** intentional durable-history retention MUST be measured as its own accepted policy

#### Scenario: A cost result is used as production evidence
- **WHEN** performance measurements support acceptance
- **THEN** they MUST record matched scope, source revisions, environment, warmup, iteration counts, retention policy, spreads, and paired excess across at least three independent process runs with multiple samples and alternating comparison order
- **AND** allocation diagnostics MUST distinguish peak bytes, retained state, and cumulative allocation rate rather than treat them as interchangeable
- **AND** dispatch-only or CPU-local results MUST NOT claim full Trainer, granted-region, or backend performance
