## ADDED Requirements

### Requirement: Target refs can reference accepted structural identities
Shared optimization target refs SHALL preserve applicable accepted catalog/revision/realization/component and structural path-binding identities alongside their live execution references without making metadata responsible for optimization policy or deduplication.

#### Scenario: Parameter target has accepted structural identity
- **WHEN** optimization builds a target for a parameter represented in the accepted structural catalog
- **THEN** the target ref MUST expose or be relatable to that qualified revision- or realization-scoped parameter path-binding identity
- **AND** it MUST preserve its existing live parameter and selector behavior

#### Scenario: Several paths reference one live parameter
- **WHEN** multiple structural path bindings resolve to the same observation-local live parameter object
- **THEN** optimization MUST preserve the distinct metadata bindings
- **AND** it MUST continue to deduplicate execution targets by live-object semantics rather than path identity

#### Scenario: Accepted structural identity is unavailable
- **WHEN** optimization builds a live target before structural metadata has been resolved
- **THEN** target construction MUST remain possible under its existing execution contract
- **AND** it MUST omit the durable identity rather than inventing one

#### Scenario: Grouping consumes recorded immutable facts
- **WHEN** grouping or planning needs an immutable structural fact already accepted for the exact model revision and policy
- **THEN** it MAY consume that recorded fact through the shared typed query capability
- **AND** metadata MUST NOT take ownership of grouping or learning-rate policy
