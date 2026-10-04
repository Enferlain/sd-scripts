## ADDED Requirements

### Requirement: Target refs can reference accepted structural identities
Shared policy-neutral target refs SHALL preserve applicable accepted
catalog/revision/realization/component and structural path-binding associations
separately from authority-qualified semantic target identity and revision-pinned
live consumer projections. Catalog resolution SHALL NOT be a prerequisite for
an otherwise valid live projection. Metadata SHALL NOT own semantic target
selection, trainability, grouping, or execution deduplication.

#### Scenario: Parameter target has accepted structural identity
- **WHEN** an accepted consumer resolves a parameter target represented in the accepted structural catalog
- **THEN** the target ref MUST expose or be relatable to that qualified revision- or realization-scoped parameter path-binding identity
- **AND** it MUST preserve component-qualified selector behavior and provide live parameter access only through a current revision-pinned projection

#### Scenario: Several paths reference one live parameter
- **WHEN** multiple structural path bindings resolve to the same observation-local live parameter object
- **THEN** the consumer MUST preserve the distinct structural bindings and their provenance
- **AND** execution resolution MUST account for their observation-local aliases under that consumer's accepted ownership/grouping policy rather than treat path identity as distinct physical state
- **AND** aliases within one accepted optimization unit/group MUST remain distinct from unsupported physical overlap across units/groups

#### Scenario: Accepted structural identity is unavailable
- **WHEN** an accepted consumer resolves a live target before structural metadata has been resolved
- **THEN** target construction MUST remain possible under its accepted authority and freshness contract
- **AND** it MUST omit the unavailable catalog/structural association rather than inventing one or omitting the required runtime participant identity

#### Scenario: Grouping consumes recorded immutable facts
- **WHEN** grouping or planning needs an immutable structural fact already accepted for the exact model revision and policy
- **THEN** it MAY consume that recorded fact through the shared typed query capability
- **AND** metadata MUST NOT take ownership of grouping or learning-rate policy
