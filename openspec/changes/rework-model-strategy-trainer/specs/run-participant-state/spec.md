## Purpose

Define one accepted run authority for the authoritative run-specific
obligations and for participant and relationship identity, current bindings,
routes, views, revisions, transitions, history, and lineage.

## ADDED Requirements

### Requirement: One authority owns canonical current run state
Each accepted run arrangement SHALL contain one contract-governed authority for
canonical participant, relationship, binding, execution-route, access-view,
arrangement, obligation, revision, dependency, and freshness state. Loaders,
capabilities, Trainer projections, metadata, and observability SHALL NOT retain
competing canonical copies or independently reinterpret the authored strategy.

#### Scenario: Loader produces a component candidate
- **WHEN** a loader creates a live component for a declared participant
- **THEN** the loader MUST submit a typed materialization proposal
- **AND** the candidate MUST NOT become current merely because a caller assigned it to a field

### Requirement: Initial realization and later transitions share one enforcement direction
Initial materialization, preparation publication, compatible replacement,
arrangement amendment, and restoration SHALL evaluate candidate state and
evidence against the authority's current run-specific obligations. Initial and
later operations MAY have different continuity and failure requirements, but
they SHALL NOT obtain validity from separate interpretations of the strategy.

#### Scenario: Initial participant materializes
- **WHEN** a declared participant supplies its first concrete realization and required evidence
- **THEN** the authority MUST evaluate the prospective state against the current obligation revision before publishing it

#### Scenario: Current participant is replaced
- **WHEN** a bound participant supplies a replacement and continuity evidence
- **THEN** the authority MUST apply the same current obligations together with the replacement-specific identity, revision, and invalidation rules

### Requirement: Authored address and runtime incarnation are distinct
An authored participant address SHALL identify a declaration in the strategy.
An authority-established participant reference SHALL identify exactly one
incarnation within one logical run authority. Role labels, source identity,
implementation type, and Python object identity SHALL NOT determine
incarnation equality.

#### Scenario: Teacher and student share a source
- **WHEN** teacher and student declarations initialize from the same artifact
- **THEN** they MUST receive distinct participant references
- **AND** their shared origin MAY be recorded separately as lineage

#### Scenario: Role label is reused
- **WHEN** two declared participants both fulfill a predictor or denoiser role
- **THEN** the shared role MUST NOT collapse their identities

### Requirement: Accepted state changes preserve or replace incarnation explicitly
Materialization, compatible replacement, preparation, route rebinding, wrapper
replacement, casting, and compilation SHALL preserve the participant reference.
Retirement SHALL close current use without erasing history. An incompatible
replacement, redeclaration, or new run SHALL establish a new reference.

#### Scenario: Compatible checkpoint replacement
- **WHEN** a replacement continues to satisfy the participant's complete accepted meaning
- **THEN** the authority MUST preserve its reference and advance the affected state revisions

#### Scenario: Incompatible semantic replacement
- **WHEN** a candidate would change the participant's accepted meaning
- **THEN** the authority MUST reject identity-preserving replacement
- **AND** the change MUST use an explicit retirement/declaration amendment with lineage where applicable

#### Scenario: Retired reference remains historical
- **WHEN** a participant is retired
- **THEN** its reference MUST remain meaningful for history and provenance
- **AND** it MUST never be revived or reassigned

### Requirement: Execution routes and access views remain distinct
Each participant and execution-route key pair SHALL have exactly one
authoritative current binding. An additional route SHALL require a materially
different callable or preparation representation and explicit freshness
semantics. Original, unwrapped, inspection, metadata, and artifact access SHALL
be typed views rather than competing routes. A state-only participant SHALL NOT
require a fabricated execution route.

#### Scenario: Injected adapter has no independent forward
- **WHEN** an adapter acts through an accepted host relationship without its own callable execution
- **THEN** its state MUST remain independently identifiable and persistable
- **AND** no fake route may be created for it

#### Scenario: Sampling needs a distinct compiled representation
- **WHEN** a selected sampling capability requires a separately prepared callable representation
- **THEN** it MAY use a named route with declared refresh dependencies
- **AND** that route MUST stop being current when its freshness guarantee fails

### Requirement: Participant, relationship, and optimization lifecycles are independent
Declaration, materialization, bound-state replacement, route rebinding,
relationship transition, arrangement amendment, and merge/fold SHALL remain
distinguishable operations. Participant lifecycle, optimization status, and
relationship lifecycle SHALL remain separate state axes.

#### Scenario: Participant is frozen
- **WHEN** optimization stops selecting a bound participant
- **THEN** the participant MUST NOT be retired and its relationships MUST NOT be detached solely because it is frozen

#### Scenario: Adapter relationship is deactivated
- **WHEN** an adapter/host effect becomes inactive
- **THEN** the relationship revision MUST change without creating or retiring either endpoint participant

### Requirement: Relationship identity is independently addressable
An authored relationship address SHALL establish one run-scoped relationship
incarnation whose endpoints, operational state, revision, history, and
persistence meaning can evolve independently from endpoint identity.

#### Scenario: One adapter has parallel effects
- **WHEN** one adapter participates in two independently managed effects
- **THEN** the effects MUST have distinct relationship identities even if their endpoint participants overlap

### Requirement: Transitions are revision-checked and atomic
Every state-changing proposal SHALL identify expected source and obligation
revisions. The authority SHALL evaluate the prospective complete state and
publish all accepted participant, relationship, route, view, obligation, and
invalidation changes together or none of them.

#### Scenario: Proposal is stale
- **WHEN** an expected participant, relationship, route, or dependency revision has changed
- **THEN** the authority MUST reject the proposal without partially changing canonical state

#### Scenario: Multi-participant amendment fails
- **WHEN** one part of an atomic amendment violates an accepted obligation
- **THEN** no new participant, relationship, or retirement from that amendment may become current

#### Scenario: Permitted amendment changes applicable obligations
- **WHEN** an arrangement amendment is permitted under the existing contract and profile
- **THEN** the authority MUST derive and evaluate the affected obligation revision as part of the same atomic transition
- **AND** consumers MUST NOT use evidence or projections derived from the superseded obligation revision

### Requirement: Freshness is explicit and conservative by default
Participant binding, relationship, and route state SHALL have independent
revisions. A derived projection that declares no narrower valid dependency set
SHALL depend on the complete authority snapshot from which it was resolved.

#### Scenario: Unrelated precision is unavailable
- **WHEN** a projection uses the conservative snapshot dependency and any snapshot-visible state changes
- **THEN** the projection MUST become stale even if a more precise future policy could have retained it

#### Scenario: Precise dependency remains valid
- **WHEN** a projection declares exact dependencies and an unrelated authority transition occurs
- **THEN** the authority MAY continue reporting it as fresh only if every declared dependency still holds

### Requirement: Consumers receive scoped coherent projections
The authority SHALL expose consumer-specific views rather than unrestricted
object lookup. A multi-participant consumer SHALL receive one coherent snapshot
or an explicitly valid fine-grained dependency set.

#### Scenario: Persistence requests state
- **WHEN** an artifact product resolves its current state
- **THEN** it MUST receive a product-specific semantic projection
- **AND** it MUST NOT default to the outermost prepared Python objects

#### Scenario: Metadata observes a transition
- **WHEN** metadata records participant or relationship history
- **THEN** it MUST consume accepted durable facts
- **AND** it MUST NOT receive authority to mutate live bindings

### Requirement: Runtime identity and lineage remain separate
Exact restoration of the same logical run SHALL preserve authority and
participant identities only when their accepted identity and revision state is
persisted and restored. A new run or a restoration lacking those facts SHALL
establish new identities and use explicit lineage or derivation relationships.

#### Scenario: Exact run restoration recreates Python objects
- **WHEN** a complete runtime snapshot restores the same logical authority in a new process
- **THEN** participant references and accepted revisions MUST remain the same
- **AND** recreated modules and wrappers MUST NOT create new participant incarnations

#### Scenario: Artifact initializes another run
- **WHEN** run B loads an artifact produced by run A
- **THEN** run B MUST establish its own participant references
- **AND** matching authored addresses MUST NOT be treated as shared runtime identity
