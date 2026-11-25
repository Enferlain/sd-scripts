## Purpose

Define semantic optimization intent, identity, realization, ownership, and
publication independently of concrete parameters, optimizer objects, and
backend wrappers.

## ADDED Requirements

### Requirement: Strategy declares semantic training subjects
The authored strategy SHALL declare intended training subjects and constraints
through accepted participant and substructure meanings. It SHALL NOT directly
make current parameter objects, `requires_grad` flags, optimizer groups, or
backend handles authoritative.

#### Scenario: Adapter and base parameters are selected
- **WHEN** an authored strategy declares combined base-model and adapter training
- **THEN** fulfillment MUST preserve both semantic selections and constraints
- **AND** Trainer optimization MUST resolve their current parameter realizations

#### Scenario: A unit's members resolve to no parameters
- **WHEN** a unit's members resolve to no parameters against the loaded structure
- **THEN** resolution MUST refuse the unit, naming it and the declaration that lets it be empty, unless the strategy declares that the unit may be empty
- **AND** a unit declared as possibly empty that resolves to no parameters MUST step nothing and advance no progress

### Requirement: Optimization units have run-scoped semantic identity
An optimization unit SHALL distinguish its non-positional address, run-scoped
incarnation, accepted-definition revision, concrete runtime realization, and
mutable optimizer/scheduler state.

#### Scenario: Backend replaces optimizer objects
- **WHEN** preparation replaces concrete parameters, optimizer, scheduler, or wrappers without changing the accepted unit definition
- **THEN** the unit MUST retain its identity and accepted-definition revision

#### Scenario: Accepted membership changes
- **WHEN** replanning changes semantic parameter membership, optimizer-significant grouping, policy, advancement policy, or semantic dependencies
- **THEN** the same independently managed unit MUST advance its definition revision

#### Scenario: Unit is split or merged
- **WHEN** an independently managed optimization responsibility is split, merged, retired, or recreated
- **THEN** the new responsibilities MUST receive new unit identities

#### Scenario: Ordinary state evolves within an accepted definition
- **WHEN** optimizer state advances or a selected schedule or due policy chooses within its accepted bounds
- **THEN** the unit MUST retain its identity and definition revision
- **AND** the system MUST NOT rebuild preparation solely because that ordinary state changed

### Requirement: Semantic membership is authority-qualified
Optimization membership SHALL identify accepted participants and parameter
substructure rather than current `nn.Parameter` object identity. A binding or
route change SHALL invalidate affected runtime realization and trigger
re-realization or replanning.

#### Scenario: New module objects realize the same definition
- **WHEN** compatible participant replacement recreates parameters while accepted unit membership and policy remain unchanged
- **THEN** optimization MAY realize the same unit identity and revision with the new concrete handles

### Requirement: Standard ownership does not overlap
Under the standard profile, each selected semantic parameter SHALL belong to
one optimization unit and one execution group within that unit. Tied or shared
aliases SHALL count as the same parameter. Deliberate overlap SHALL require a
recognized capability or explicit extension with accepted coordination rules.

#### Scenario: Tied parameter is selected twice
- **WHEN** several accepted target paths resolve to one tied parameter within the same unit and execution group
- **THEN** resolution MUST consolidate the aliases into one optimizer member while preserving their provenance
- **AND** repeated references alone MUST NOT be treated as competing ownership

#### Scenario: Aliases claim competing owners
- **WHEN** resolved aliases assign the same parameter to different units or execution groups under the standard profile
- **THEN** resolution MUST reject the conflicting ownership before execution

#### Scenario: Deliberate overlap is requested
- **WHEN** an experiment requires one parameter in multiple advancement responsibilities
- **THEN** fulfillment MUST require accepted rules for ordering, cadence, gradients, clipping, zeroing, restoration, and backend support

### Requirement: Logical and execution groups remain distinct
The optimization exchange SHALL preserve semantic logical groups separately
from optimizer/backend-facing execution groups. Group learning-rate and policy
behavior SHALL be derived from the accepted plan rather than incidental list
position.

#### Scenario: Backend combines logical groups
- **WHEN** a backend or optimizer materializes several logical groups into an execution-specific shape
- **THEN** logical identity and diagnostic meaning MUST remain available independently from that concrete grouping

### Requirement: Accepted optimization meaning precedes physical construction
The accepted optimization plan SHALL retain unit identities and definition
revisions, semantic membership, logical groups, trainability,
optimizer/scheduler and advancement policies, semantic dependencies, and
constraints independently from current parameter and backend objects.
Provisional candidates and published runtimes SHALL remain distinguishable
from that accepted meaning.

#### Scenario: Two backends require different construction orders
- **WHEN** the same accepted plan is realized with optimizer-before-wrapper or wrapper-before-optimizer ordering
- **THEN** both realizations MUST preserve the accepted unit, member, group, policy, and dependency meanings
- **AND** changing physical construction order MUST NOT itself revise the unit definition

### Requirement: Trainer owns standard realization and mechanics
Under the standard profile, Trainer optimization SHALL realize
trainability, current parameters, logical/execution groups, clipping members,
synchronization/accumulation members, optimizer/scheduler state, and ordinary
backward, advancement, and zeroing mechanics. Specialized behavior SHALL NOT
transfer those responsibilities merely because it is selected.

#### Scenario: Auxiliary learned capability is selected
- **WHEN** a capability contributes additional learned state
- **THEN** its accepted contract MUST state how that state participates in optimization
- **AND** selection alone MUST NOT grant it imperative backward or advancement authority

### Requirement: Runtime lifecycle projections are coherent and separate
Trainability, clipping, synchronization/accumulation, module runtime-mode
needs, and unit-local optimizer transitions SHALL be derived as coherent
views rather than independent mode flags or repeated queries. A module's
runtime mode SHALL be set for each call through its participant's route
from what that call's own work needs, never by a run-wide switch: train mode
only for the modules of a unit whose loss the call reaches, and evaluation
mode otherwise.

#### Scenario: Validation runs beside training
- **WHEN** validation calls a participant while training continues in another activity
- **THEN** each call MUST run the participant's modules in the mode its own work needs
- **AND** no switch made for one activity's calls MAY change the mode of another activity's calls

#### Scenario: A frozen network lies on a gradient path
- **WHEN** a call's path from a trained module to its unit's loss passes through a frozen network
- **THEN** the frozen network MUST run in evaluation mode while the trained modules run in train mode
- **AND** module mode MUST follow trainability rather than the flow of gradients

#### Scenario: A later backward pass recomputes the forward
- **WHEN** a call's forward will be recomputed inside a later backward pass, as checkpointing does
- **THEN** the modes the call needed MUST stay in place until its results' backward pass no longer needs them
- **AND** a call that needs another mode of the same modules MUST wait until then

#### Scenario: An optimizer switches to its evaluation point
- **WHEN** a unit's optimizer policy has an evaluation point, such as averaged weights or a schedule-free optimizer's
- **THEN** reading it MUST go through a route on the participant whose dependencies are that unit state
- **AND** the switch MUST belong to the unit and that route, never to a module mode set for a call

### Requirement: Realization and preparation form one publication attempt
The system SHALL first resolve an accepted semantic optimization plan and SHALL
publish a complete current runtime only after all required trainability,
parameter resolution, optimizer/scheduler construction, backend preparation,
rank agreement, and validation succeed. Backend constraints MAY determine the
internal construction order.

#### Scenario: Backend needs optimizer before model preparation
- **WHEN** a backend requires concrete optimizer state before a joint preparation call
- **THEN** the realization job MAY construct it in that order while keeping Trainer semantic ownership
- **AND** no partial unit runtime may become current

#### Scenario: Backend needs prepared parameters first
- **WHEN** another backend requires participant transformation before optimizer construction
- **THEN** the same exchange MUST permit that order without changing the accepted semantic plan

### Requirement: Binding changes invalidate affected optimization runtime
Every optimization runtime SHALL have authority-verifiable dependency evidence
covering the applicable participant, relationship, binding, route, substructure,
unit-definition, obligation, and backend/preparation revisions. An accepted
change that breaks those dependencies SHALL remove the runtime from current use
until re-realization or replanning succeeds.

#### Scenario: Adapter attachment changes parameter structure
- **WHEN** an accepted relationship transition changes the available optimization substructure
- **THEN** affected units MUST become non-current and be re-resolved before another advancement

#### Scenario: A change affects an inseparable backend group
- **WHEN** an accepted change invalidates one member of an inseparable backend group
- **THEN** every covered route and unit runtime depending on that group MUST lose prepared usability
- **AND** an unrelated precise projection MAY remain usable only when its dependencies exclude the affected state and group

#### Scenario: Parameter surgery preserves the accepted selector
- **WHEN** an allowed structure-preserving transition changes physical parameters without changing the accepted selector, grouping, policy, or semantic dependencies
- **THEN** the unit MUST retain its definition revision while its members, aliases, trainability, and backend runtime are re-resolved
- **AND** an arbitrary runtime trainability flag MUST NOT redefine accepted membership

### Requirement: Optimization dependency evidence may be a snapshot or a narrower set
Optimization runtime dependency evidence MAY be established through an authority
snapshot, a prepared generation's dependency evidence, or a justified narrower
dependency set; it SHALL NOT require every runtime object to store an explicit
tuple of those revisions. Without a justified narrower dependency set the
runtime SHALL depend conservatively on its source authority snapshot.

#### Scenario: A runtime declares no narrower dependencies
- **WHEN** an optimization runtime records no justified narrower dependency set
- **THEN** it MUST depend on its whole source authority snapshot
- **AND** it MUST NOT be required to store an explicit tuple of every relevant revision

### Requirement: Reuse under a changed obligation revision needs evidence
Reuse under a changed obligation revision SHALL require authority-backed
evidence that the applicable obligations remain satisfied.

#### Scenario: The obligation revision changes while bindings do not
- **WHEN** an accepted change advances the applicable obligation revision while the runtime's participant bindings stay the same
- **THEN** the runtime MUST be reused only with authority-backed evidence that its applicable obligations remain satisfied

### Requirement: Unit identity continuity does not imply state continuity
Before affected work resumes after replacement, surgery, replanning, or a
stage transition, the responsible owners SHALL establish the accepted
preserve-with-evidence, migrate, reset, or unavailable treatment of mutable
optimizer/scheduler, gradient-window, and backend state. Pending contributions
SHALL be completed, preserved, or discarded only under an accepted rule.

#### Scenario: Compatible replacement recreates the same unit
- **WHEN** a compatible participant replacement preserves unit identity and definition revision but recreates its parameter objects
- **THEN** readiness MUST require evidence for the selected optimizer-state continuation treatment
- **AND** matching identity or parameter-list position MUST NOT silently reuse momentum, gradients, or scheduler state

#### Scenario: Transition occurs during an accumulation window
- **WHEN** an accepted structural or policy change affects a unit with pending contributions
- **THEN** dependent work MUST wait for the accepted window-continuation rule and complete current publication
- **AND** old contributions MUST NOT silently enter the new runtime

### Requirement: Advancement authority is profile-defined
The active ownership profile SHALL explicitly define supported
advancement policies and which executor owns backward, gradient manipulation,
advancement, and zeroing. A policy outside the standard Trainer-owned set SHALL
require an accepted extension and SHALL NOT create dual ownership.

#### Scenario: Unusual deterministic cadence remains supported
- **WHEN** a declared cadence fits a supported standard advancement policy
- **THEN** Trainer MUST execute it without an imperative ownership transfer

#### Scenario: Research executor performs advancement
- **WHEN** an extension grants an imperative region optimizer advancement authority
- **THEN** Trainer MUST not also advance the same unit for that accepted action
- **AND** checkpoint and failure semantics MUST identify the actual owner

### Requirement: Standard advancement supports independently and jointly due units
The standard profile SHALL support one due unit, independently due units, and
jointly due disjoint units through accepted policies rather than one fixed
training-step sequence. Due decisions MAY use live coordinates and accepted
owned state without consulting the authoring interface. Executable readiness
SHALL require backend support for the selected policy.

#### Scenario: Independent units contribute on different clocks
- **WHEN** a bounded policy selects nonconsecutive contributions for independently due units
- **THEN** Trainer MUST preserve each unit's own gradient window and progress
- **AND** work on one unit MUST NOT clear another unit's pending gradients or imply its advancement

#### Scenario: Several units receive one shared source
- **WHEN** an accepted action offers one source to several disjoint units
- **THEN** Trainer MAY use shared backward work only when it satisfies all accepted gradient, window, synchronization, and ordering requirements
- **AND** each unit MUST retain its own contribution and advancement outcome

### Requirement: Advancement policies identify routes, windows and mechanics
Each standard advancement policy SHALL identify addressed sources and gradient
routes, contribution and accumulation-window completion rules, synchronization
and scaling requirements, clipping scope and timing, ordered eligible
advancement, scheduler triggers, zeroing/discard rules, lifecycle effects, and
failure and continuation ownership.

#### Scenario: Distinct sources could introduce cross-unit gradients
- **WHEN** several sources can influence parameters outside their accepted destination units
- **THEN** execution MUST preserve the accepted per-unit gradient routing through a supported lowering
- **AND** readiness MUST reject unsupported routing rather than silently sum sources or detach required paths

#### Scenario: Clipping requires synchronized window completion
- **WHEN** a unit or accepted joint scope becomes eligible for clipping and advancement
- **THEN** Trainer MUST apply the accepted synchronization, unscaling, completion, and clipping boundaries before invoking advancement
- **AND** a backend unable to isolate or synchronize the selected windows MUST fail readiness before executing that policy

#### Scenario: Scheduler depends on a backend-reported advancement
- **WHEN** the accepted scheduler trigger distinguishes an action, an attempted optimizer call, and a confirmed non-skipped advancement
- **THEN** runtime MUST use the required backend-supported distinction
- **AND** readiness MUST reject a policy whose required distinction the backend cannot report

### Requirement: A window's gradient is that of one batch of its contributions
Under the standard profile, a unit's window SHALL produce the gradient that
one batch of all its contributions would give under the loss's own
reduction, however the contributions split that batch and across every
rank. Each contribution's loss SHALL come with its mass, what that loss's
own reduction divides by, and the window SHALL divide once, by the total
mass.

#### Scenario: Contributions carry different masses
- **WHEN** a window's contributions carry different masses, such as a short final batch
- **THEN** the window's gradient MUST equal the gradient of one batch holding all of them under the loss's reduction
- **AND** clipping, the non-finite check and the reported loss MUST read that normalized result

#### Scenario: A window of no mass
- **WHEN** a window's contributions add to no mass, whether their masses are on the host or on the device
- **THEN** the unit MUST NOT step its optimizer or advance its progress for that window
- **AND** a contribution of no mass MUST contribute nothing to its window's gradient

#### Scenario: A weighted loss divides by its sample count
- **WHEN** a loss weights its samples but divides by their count, as a min-SNR weighting does
- **THEN** its mass MUST be the count its reduction divides by, as the loss declares it
- **AND** the mass MUST NOT be inferred from the loss's weights, mask or inputs

#### Scenario: A window spans several ranks
- **WHEN** a unit's window is synchronized across ranks
- **THEN** synchronization MUST sum the gradients and the masses and divide once, by the total mass
- **AND** an averaging all-reduce MUST be compensated rather than stacked on that division, and per-rank averages MUST NOT be averaged

#### Scenario: A loss's terms divide by different masses
- **WHEN** a contribution's loss adds terms whose reductions divide by different kinds of mass
- **THEN** each term MUST be divided by its own window total, and masses of different kinds MUST NOT be added
- **AND** a window whose terms' masses depend on the data MUST know those totals before the backward pass

### Requirement: Optimization reports contributions and reached mechanics separately
The optimization owner SHALL report addressed unit incarnations and definition
revisions, contribution/window and gradient readiness, reached mechanics,
backend-supported returned, skipped, uncertain, and not-attempted outcomes,
and relevant scheduler, zeroing, and progress facts. A returned call SHALL NOT
be treated as proof of numerical parameter change. A joint action SHALL NOT
imply atomic advancement, automatic rollback, or safe replay.

#### Scenario: First unit returns and second unit fails inside its backend
- **WHEN** the first due unit's optimizer call returns and a second call fails after entering the backend
- **THEN** reporting MUST preserve the first call's known outcome and mark the second uncertain unless backend evidence narrows that outcome
- **AND** later unattempted mechanics MUST remain distinguishable
- **AND** dependent unsafe work MUST stop until an accepted recovery establishes validity

#### Scenario: Scheduler or final zeroing fails after an optimizer returns
- **WHEN** a later retained mechanic fails after an optimizer call has returned
- **THEN** reporting MUST preserve the known optimizer outcome separately from the scheduler or zeroing failure
- **AND** missing overall completion MUST NOT erase reached effects or certify gradients safe for reuse

### Requirement: A non-finite gradient norm or loss fails the run unless an accepted policy skips its step
Under the standard profile, a unit's window whose gradient norm or loss is
not finite SHALL fail the run, unless an accepted policy skips that step
without applying it and reports the skip. The check SHALL read only values
the window computes anyway and SHALL add no host wait. It MAY lag the
advancement that used the gradient, SHALL be read before a product of the
unit is saved and when the run's work ends, and the unit SHALL NOT advance
once the failure is found.

#### Scenario: A window's gradient norm is not finite
- **WHEN** a unit's window ends with a gradient norm that is not finite and no accepted policy skips that step
- **THEN** the run MUST fail, at the latest when its work ends, naming the unit and the first window whose value was not finite
- **AND** that window's step, and those of later windows that ended before the check was read, MAY already have been taken, and reporting MUST keep their known outcomes
- **AND** the unit MUST NOT advance after the failure is found

#### Scenario: A unit does not clip
- **WHEN** nothing in a unit's window measures its gradient norm, because its policy does not clip
- **THEN** the check MUST read the window's loss rather than add a pass over the gradients
- **AND** a gradient that is not finite behind a finite loss MAY go undetected

#### Scenario: A save or the run's end comes before the check is read
- **WHEN** a product of the unit is about to be saved, or the run's work ends, while the unit's check has not been read
- **THEN** the check MUST be read first
- **AND** a failed check MUST fail the run before the product is written or the run reports its outcome

### Requirement: Exact restoration preserves unit state only within the same run
Exact restoration SHALL restore accepted unit identities, revisions, mutable
optimizer/scheduler state, and advancement coordinates when continuing the
same logical run. A new run or fork SHALL establish new unit identities and
retain source provenance separately.

#### Scenario: Same run resumes with recreated optimizer objects
- **WHEN** a complete snapshot restores the same logical run in a new process
- **THEN** unit identities and revisions MUST remain stable despite new Python objects

#### Scenario: Exact continuation needs an unfinished gradient window
- **WHEN** exact same-run continuation requires pending contributions or backend scaling/synchronization state
- **THEN** the optimization contributor MUST supply that state and its accepted member, policy, revision, and progress associations to the coordinated snapshot
- **AND** restoration MUST verify correspondence to restored semantic members before continuation
- **AND** an implicit reset MUST NOT be reported as exact restoration

#### Scenario: Snapshot uses a quiescent optimization boundary
- **WHEN** the accepted restoration policy avoids capturing pending gradients
- **THEN** snapshot coordination MUST establish a boundary with no required unfinished gradient window and all other owner contributions coherent
