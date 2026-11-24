## Purpose

Define authoritative current participant state and preserve identity, historical meaning and provenance across run transitions.

## ADDED Requirements

### Requirement: One authority owns current participant state
Each accepted run arrangement SHALL have one authority for current participant bindings, relationships, execution routes, access views, revisions and freshness. Producers SHALL submit transitions to that authority. Trainer projections, loaders, capabilities and metadata SHALL NOT maintain competing canonical bindings.

#### Scenario: Compatible replacement
- **WHEN** a replacement satisfies the participant's accepted obligations
- **THEN** the authority preserves its participant reference, advances the applicable state revision and invalidates affected derived state
- **AND** consumers cannot continue using stale routes as current

### Requirement: Participant identity is distinct from object and address
An authority-established reference SHALL identify one participant incarnation. Materialization, compatible replacement and preparation SHALL preserve it. Retirement SHALL close current use while preserving historical meaning. Authored address, role label, shared source and Python object identity SHALL NOT determine incarnation equality.

#### Scenario: Teacher and student share an origin
- **WHEN** separately declared teacher and student participants originate from the same checkpoint
- **THEN** they have distinct participant references and can record shared lineage

#### Scenario: Retired reference
- **WHEN** a participant is retired
- **THEN** its reference remains meaningful for history and provenance
- **AND** it cannot be revived or reassigned to a new incarnation

### Requirement: Routes and views have distinct meanings
Each participant and execution-route pair SHALL have exactly one authoritative current binding. Additional routes SHALL require distinct execution representations and explicit freshness semantics. Inspection, original-object and artifact access SHALL be typed views rather than competing execution bindings. A state-bearing participant SHALL NOT require a fabricated executable route.

#### Scenario: Injected adapter state
- **WHEN** an adapter acts through a relationship to its host and has no independent forward route
- **THEN** its state remains independently identifiable without inventing an execution route

### Requirement: Arrangement transitions preserve independent lifecycle meanings
Declaration, materialization, replacement, route rebinding, relationship changes, arrangement amendment and merge/fold SHALL remain distinguishable transitions. Participant lifecycle, optimization status and relationship lifecycle SHALL remain independent. An atomic transition SHALL publish all of its accepted participant and relationship changes together or none of them.

#### Scenario: Freezing a participant
- **WHEN** optimization no longer selects a live participant
- **THEN** that change does not by itself retire the participant or detach its relationships

### Requirement: Persistence distinguishes identity and lineage
Artifact persistence SHALL select a coherent semantic product from accepted state. Artifacts SHALL have identities distinct from runtime participants. Exact same-run restoration SHALL preserve authority-established identities only when the required identity and revision state is restored; a new run SHALL establish new identities and retain provenance links.

#### Scenario: New run loads a prior artifact
- **WHEN** an artifact from run A initializes run B
- **THEN** run B establishes its own participant identities
- **AND** records derivation without treating matching authored addresses as shared runtime identity
