## Purpose

Define runtime preparation as part of governed realization so that it preserves
accepted identities, evaluates its products against the authoritative
run-specific obligations, separates ownership surfaces, and never publishes
stale or partial state.

## ADDED Requirements

### Requirement: Preparation derives one coherent attempt-scoped job
Each preparation attempt SHALL be derived atomically from one coherent accepted
authority state and obligation revision. The job SHALL identify source
revisions, requested routes and views, every required participant, constraints,
joint groups, mutation permissions, freshness guarantees, and typed
infrastructure items. Preparation SHALL produce candidate state and evidence;
it SHALL NOT define new compatibility meaning or certify its own result.

Training-side preparation coordination SHALL derive the job from the accepted
requirements and current authority projection, then coordinate final
installation of its verified result across the authority-owned routes and
views and Trainer-owned optimization and backend state. It SHALL NOT keep a
competing binding map or independently define conformance rules. Each owner
retains its own state, and none may publish a partial current result.

#### Scenario: Frozen participant requires backend preparation
- **WHEN** a frozen participant is required by accepted execution and needs movement, casting, wrapping, sharding, or compilation
- **THEN** the preparation job MUST include it despite its absence from optimization membership

#### Scenario: Backend needs ordered calls
- **WHEN** one preparation job requires multiple ordered or joint backend operations
- **THEN** it MAY perform several calls while remaining one attempt with one source dependency set

#### Scenario: Preparation request is already incompatible
- **WHEN** current authoritative facts already prove that a requested preparation cannot satisfy its run-specific obligations
- **THEN** the attempt MUST be rejected before expected-expensive backend work
- **AND** the incompatibility MUST NOT be deferred merely because the backend could also fail

### Requirement: Backend coordination does not replace semantic identity
Backend replicas, wrappers, and composite handles SHALL remain runtime
infrastructure state. They SHALL NOT replace participant references, merge
participant identities, or become execution routes merely because a backend
returns them.

Prepared usability SHALL require matching current authority-owned route and
view guarantees together with the Trainer-owned backend coordination state for
every participant that state covers. Neither ownership surface alone SHALL
certify that a participant is ready for execution.

#### Scenario: Composite covers several participants
- **WHEN** a backend returns one composite coordination handle for multiple participants
- **THEN** every participant MUST retain its own reference and accepted route meaning
- **AND** the composite MUST remain Trainer/runtime state

#### Scenario: Backend handle outlives a route guarantee
- **WHEN** a backend coordination handle still exists but a covered participant's required route is no longer current
- **THEN** that participant MUST NOT be reported as prepared for execution solely because the handle exists

### Requirement: Preparation results separate ownership surfaces
A preparation result SHALL distinguish authority-owned route/view proposals,
Trainer-owned optimization candidates, Trainer-owned backend coordination
state, exact source revisions, satisfied constraints, freshness guarantees,
runtime facts, and structured failures.

#### Scenario: Model and optimizer are prepared jointly
- **WHEN** one backend operation returns model wrappers, an optimizer, a scheduler, and a synchronization handle
- **THEN** the result MUST preserve which values belong to the run authority, optimization runtime, and backend infrastructure
- **AND** joint creation MUST NOT imply joint semantic ownership

### Requirement: Fallible work precedes final publication
All ordinarily fallible backend work, component preparation, candidate
assembly, rank agreement, evidence collection, obligation evaluation, and
freshness validation SHALL finish before final publication. The complete
candidate result SHALL be evaluated against the exact obligation revision from
which its attempt was derived. Final publication SHALL install only complete,
already-validated authority and Trainer state without invoking expected-fallible
external work or callbacks.

#### Scenario: Candidate fails validation
- **WHEN** a completed backend result violates an accepted route, constraint, group, or freshness requirement
- **THEN** final publication MUST reject it without exposing any mixed current state

#### Scenario: Observation fails after publication
- **WHEN** reporting or history routing fails after the prepared state is current
- **THEN** the accepted current state MUST remain committed
- **AND** observation MUST NOT become a rollback authority

### Requirement: Replacement-only failure preserves still-valid prior state
Preparation that cannot mutate authoritative inputs in place SHALL keep its
candidate unpublished until success. Failure SHALL discard that candidate and
preserve prior routes and views whose guarantees remain valid.

#### Scenario: Backend wrapper creation fails
- **WHEN** replacement-only preparation fails before publication
- **THEN** no candidate route or backend state may become current
- **AND** the old current route MUST remain usable if no accepted transition invalidated it

### Requirement: Destructive preparation withdraws guarantees before mutation
An operation permitted to mutate an authoritative realization in place SHALL
first establish an authority-recognized destructive attempt and withdraw every
affected freshness guarantee. Failure SHALL leave those guarantees withdrawn
until accepted state is replaced or re-established.

The destructive attempt SHALL have an authority-recognized coordination
identity distinct from its source snapshot, so withdrawing the source
guarantees does not invalidate that attempt's own completion basis. An older
optimistic result SHALL NOT use the withdrawn snapshot to restore usability.

#### Scenario: Withdrawing guarantees advances the authority snapshot
- **WHEN** a destructive attempt withdraws its source route guarantees before mutation
- **THEN** the authority MUST record a new freshness revision that withdraws those guarantees from the source snapshot
- **AND** the accepted attempt MUST remain addressable by its distinct coordination identity for completion or failure
- **AND** an older optimistic result based only on the withdrawn snapshot MUST NOT become current

#### Scenario: In-place mutation fails
- **WHEN** destructive preparation mutates an authoritative object and then fails
- **THEN** the system MUST NOT advertise the prior route or views as usable solely because no replacement was published
- **AND** the failure MUST identify the state requiring recovery

#### Scenario: Older optimistic candidate completes later
- **WHEN** a replacement-only candidate derived before a destructive attempt finishes during or after that attempt
- **THEN** it MUST be rejected and MUST NOT restore withdrawn guarantees

### Requirement: Stale results never publish against newer state
Every result SHALL remain bound to its preparation attempt and source
dependencies. A relevant authority or owned preparation-state change before
publication SHALL reject the stale result or require explicit re-resolution.

#### Scenario: Relationship changes during preparation
- **WHEN** a relationship dependency changes after an optimistic job is resolved
- **THEN** the result MUST be rejected before publication

### Requirement: Inseparable backend groups constrain later preparation
A later preparation job that overlaps an inseparable current backend group
SHALL include every member of that group. It MAY replace, expand, or merge
complete groups but SHALL NOT omit an overlapped member.

#### Scenario: Job omits one current group member
- **WHEN** a proposed job overlaps an inseparable backend group but does not include all its members
- **THEN** the job MUST be rejected before destructive work begins

#### Scenario: Group member is retired
- **WHEN** one participant covered by an inseparable group is retired
- **THEN** the backend group MUST be evicted
- **AND** surviving members MUST lose prepared usability until a coherent group is rebuilt

### Requirement: Relationship-dependent preparation is invalidated consistently
Every relationship revision SHALL invalidate prepared routes or views whose
declared dependencies include that relationship, regardless of whether the
revision came from an explicit relationship transition, endpoint replacement,
or endpoint retirement. Relationship state SHALL NOT change while an endpoint
is under destructive preparation.

#### Scenario: Endpoint replacement changes an effect
- **WHEN** accepted endpoint replacement advances a relationship dependency
- **THEN** every dependent prepared view MUST lose freshness according to its declared dependency set

### Requirement: Process failure belongs to restoration
Unexpected process or rank termination during final installation SHALL abort
the in-process run. Recovery SHALL use the runtime restoration contract rather
than expose a partially current preparation transaction.

#### Scenario: Rank terminates at publication boundary
- **WHEN** a rank terminates before the run can confirm coherent publication
- **THEN** the current process group MUST NOT continue as though publication succeeded
- **AND** a later continuation MUST re-establish state through restoration
