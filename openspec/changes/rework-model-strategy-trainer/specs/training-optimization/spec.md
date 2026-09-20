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
- **WHEN** two target paths resolve to one tied parameter under the standard profile
- **THEN** resolution MUST reject or consolidate duplicate ownership rather than create two optimizer owners

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
Every optimization runtime SHALL declare the participant, relationship,
substructure, and preparation revisions on which it depends. An accepted change
that breaks those dependencies SHALL remove the runtime from current use until
re-realization or replanning succeeds.

#### Scenario: Adapter attachment changes parameter structure
- **WHEN** an accepted relationship transition changes the available optimization substructure
- **THEN** affected units MUST become non-current and be re-resolved before another advancement

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

### Requirement: Exact restoration preserves unit state only within the same run
Exact restoration SHALL restore accepted unit identities, revisions, mutable
optimizer/scheduler state, and advancement coordinates when continuing the
same logical run. A new run or fork SHALL establish new unit identities and
retain source provenance separately.

#### Scenario: Same run resumes with recreated optimizer objects
- **WHEN** a complete snapshot restores the same logical run in a new process
- **THEN** unit identities and revisions MUST remain stable despite new Python objects
