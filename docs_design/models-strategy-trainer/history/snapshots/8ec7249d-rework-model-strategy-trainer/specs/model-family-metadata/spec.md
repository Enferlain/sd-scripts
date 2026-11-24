## ADDED Requirements

### Requirement: Runtime participant identity is projected from the accepted run authority
Metadata SHALL record durable projections of accepted participant,
relationship, transition, artifact, and lineage facts when required for
history or provenance. Metadata SHALL NOT establish live participant identity,
infer it from Python objects, or become the runtime binding authority.

#### Scenario: Accepted participant is recorded
- **WHEN** an accepted run participant needs durable metadata identity
- **THEN** metadata MUST consume the authority's logical run identity, authored address, incarnation discriminator, and applicable revisions
- **AND** it MUST preserve those facts separately from catalog, source, model-revision, realization, and artifact identities

#### Scenario: New run loads an artifact
- **WHEN** metadata records that a participant in run B derives from an artifact produced by run A
- **THEN** it MUST record lineage between distinct identities
- **AND** it MUST NOT reuse run A's participant identity merely because addresses or sources match

#### Scenario: Exact runtime restoration is recorded
- **WHEN** a snapshot restores the same logical run authority and participant references in a new execution session
- **THEN** metadata MUST preserve the participant incarnations while allowing a distinct execution-session observation identity
