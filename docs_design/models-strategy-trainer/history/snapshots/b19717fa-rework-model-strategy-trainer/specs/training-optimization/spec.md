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
- **THEN** the strategy acceptance check MUST preserve both semantic selections and constraints
- **AND** Trainer optimization MUST resolve their current parameter realizations

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
- **THEN** the strategy acceptance check MUST require accepted rules for ordering, cadence, gradients, clipping, zeroing, restoration, and backend support

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
Under the standard profile, Trainer/optimization infrastructure SHALL realize
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
needs, and unit-local optimizer transitions SHALL be published as coherent
views rather than independent mode flags or repeated queries.

#### Scenario: Validation enters evaluation mode
- **WHEN** Trainer coordinates a validation boundary
- **THEN** it MUST use accepted lifecycle projections for affected modules and unit runtimes
- **AND** optimization membership alone MUST NOT own the timing of the transition

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
unit-definition, obligation, and backend/preparation revisions. That evidence
MAY be established through an authority snapshot, a prepared generation's
dependency evidence, or a justified narrower dependency set; it SHALL NOT
require every runtime object to store an explicit tuple of those revisions.
Without a justified narrower dependency set the runtime SHALL depend
conservatively on its source authority snapshot.
An accepted change that breaks those dependencies SHALL remove the runtime
from current use until re-realization or replanning succeeds. Reuse under a
changed obligation revision SHALL require authority-backed evidence that the
applicable obligations remain satisfied.

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
The active execution/ownership profile SHALL explicitly define supported
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
training-step sequence. Each policy SHALL identify addressed sources and
gradient routes, contribution and accumulation-window completion rules,
synchronization and scaling requirements, clipping scope and timing, ordered
eligible advancement, scheduler triggers, zeroing/discard rules, lifecycle
effects, and failure and continuation ownership. Due decisions MAY use live
coordinates and accepted owned state without consulting the authoring interface.
Executable readiness SHALL require backend support for the selected policy.

#### Scenario: Independent units contribute on different clocks
- **WHEN** a bounded policy selects nonconsecutive contributions for independently due units
- **THEN** Trainer MUST preserve each unit's own gradient window and progress
- **AND** work on one unit MUST NOT clear another unit's pending gradients or imply its advancement

#### Scenario: Several units receive one shared source
- **WHEN** an accepted action offers one source to several disjoint units
- **THEN** Trainer MAY use shared backward work only when it satisfies all accepted gradient, window, synchronization, and ordering requirements
- **AND** each unit MUST retain its own contribution and advancement outcome

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
